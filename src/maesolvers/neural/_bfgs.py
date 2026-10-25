"""The dense BFGS optimizer: one iteration of SciPy's BFGS per step.

The only module that reads SciPy's private line search of
``minimize(method="BFGS")``; :meth:`maesolvers.neural.optimizers.BFGS.build`
imports it, so that a change of those private names breaks the ``BFGS``
part alone, not every neural solver.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from scipy.optimize._optimize import (  # SciPy BFGS's own line search, private
    _line_search_wolfe12,
    _LineSearchError,
)
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from maesolvers.neural._torch import flat_grad


class DenseBFGS(torch.optim.Optimizer):
    """BFGS with a dense inverse Hessian approximation: one iteration per ``step``.

    One iteration of SciPy's ``minimize(method="BFGS")``, with its line
    search. The approximation ``H`` starts as the identity; after each
    iteration, with ``s`` the change of the parameters and ``y`` that of the
    gradient, ``H <- (I - r s y') H (I - r y s') + r s s'``, ``r = 1 / y's``
    (1000 if ``y's = 0``), never skipped, as in SciPy. The
    step along ``d = -H g`` comes from SciPy's strong Wolfe line search
    (MINPACK's, with SciPy's second search as a fallback), started at
    ``min(1, 2.02 (f - f_prev) / g'd)``, the step that would repeat the last
    decrease of the loss, with ``f_prev = f + |g| / 2`` on the first
    iteration. When both searches fail, as when SciPy stops on a "precision
    loss", the step leaves the parameters as they are. ``H`` takes ``n^2``
    numbers for ``n`` parameters.

    Parameters
    ----------
    params
        The parameters, in one group.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        super().__init__(params, {})
        if len(self.param_groups) != 1:
            raise ValueError("DenseBFGS takes one parameter group")

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """Take one BFGS iteration.

        Parameters
        ----------
        closure
            Zeroes the gradients, evaluates the loss, back-propagates it and
            returns it.

        Returns
        -------
        torch.Tensor
            The loss at the parameters the step starts from.
        """
        evaluate = torch.enable_grad()(closure)
        params: list[torch.Tensor] = self.param_groups[0]["params"]
        state = self.state[params[0]]
        loss = evaluate()
        f, g = float(loss), flat_grad(params).cpu().numpy()
        if not g.any():  # a stationary point: no direction to search
            return loss
        start = parameters_to_vector(params).clone()
        x = start.cpu().numpy()
        if "h" not in state:
            state["h"] = np.eye(len(x))
            state["f"] = f + float(np.linalg.norm(g)) / 2
        else:  # SciPy's update, in SciPy's order of operations
            s, y = x - state["x"], g - state["g"]
            ys = float(y @ s)
            r = 1000.0 if ys == 0.0 else 1.0 / ys
            left = np.eye(len(x)) - r * np.outer(s, y)
            right = np.eye(len(x)) - r * np.outer(y, s)
            state["h"] = left @ (state["h"] @ right) + r * np.outer(s, s)
        d = -(state["h"] @ g)
        previous = state["f"]
        state["x"], state["g"], state["f"] = x, g, f
        cache: dict[bytes, tuple[float, np.ndarray]] = {}

        def at(point: np.ndarray) -> tuple[float, np.ndarray]:
            key = point.tobytes()
            if key not in cache:
                vector_to_parameters(torch.tensor(point).to(start), params)
                value = float(evaluate())
                cache[key] = (value, flat_grad(params).cpu().numpy())
            return cache[key]

        try:
            t = _line_search_wolfe12(
                lambda p: at(p)[0],
                lambda p: at(p)[1],
                x,
                d,
                g,
                f,
                previous,
                amin=1e-100,
                amax=1e100,
            )[0]
        except _LineSearchError:
            t = 0.0
        vector_to_parameters(torch.tensor(x + t * d).to(start), params)
        return loss
