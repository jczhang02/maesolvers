"""The uniform grid of the finite-difference schemes, in any dimension.

:class:`BoxGrid` is the grid of ``n`` cells per side over a ``d``-dimensional
:class:`~maesolvers.core.domains.Box` (``h = 1/n`` on the unit box,
ADR 0001): its nodes, the mask of the interior nodes that carry the unknowns
(the others form the boundary ring, which carries the Dirichlet data), the
neighbours of a node along a direction of index offsets, and the tensor
spline through values at the nodes, which evaluates a solution off the nodes
(``Result.u``, and so ``max_grid`` on the shared evaluation grid). The
operators on it are module-level pure functions: :func:`reach`, how far a
step along a direction stays in the box; :func:`stencil`, any linear
stencil of width 1; :func:`second_differences`, the centred second
differences as sparse matrices; :func:`centred_differences`, the same
differences of an array of node values by slicing, with no matrix, which a
matrix-free scheme evaluates; :func:`centred_hessian`, the discrete Hessian
at the nodes that a scheme reports as ``Result.node_hessian``;
:func:`determinant` and :func:`cofactors` of symmetric matrices given by
their entries, such as those of the discrete Hessian; and
:func:`tensor_spline`.

The node layout: the node with multi-index ``(i_0, ..., i_{d-1})``,
``0 <= i_k <= n``, lies at ``(axes[0][i_0], ..., axes[d-1][i_{d-1}])`` and
has the flat index ``sum_k i_k (n + 1)^(d - 1 - k)``: C order, the last axis
fastest, as ``numpy.meshgrid(*axes, indexing="ij")`` lists the nodes; in 2D
node ``(i, j)`` has flat index ``i (n + 1) + j``, as in maesolvers 0.1. So
``values.reshape(grid.shape)`` is the ``d``-dimensional array of node
values, and a step of index offsets moves the flat index by
:meth:`BoxGrid.offset`. For even ``n`` the grids nest: the nodes of
``BoxGrid(box, n // 2)`` are the nodes with even multi-indices, and
``values.reshape(grid.shape)[(slice(None, None, 2),) * d].ravel()`` holds the
values there in the coarse grid's flat order, the layout a multigrid
iteration restricts and prolongs on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import NdBSpline, RectBivariateSpline, make_interp_spline

from maesolvers.core._params import integer, points, positive
from maesolvers.core.domains import Box, Field

Entries = Mapping[tuple[int, int], np.ndarray]
"""The entries ``(k, l)``, ``k <= l``, of symmetric ``d x d`` matrices, one array each."""


@dataclass(frozen=True)
class BoxGrid:
    """The uniform grid of ``n`` cells per side over a box, in any dimension.

    The grid has ``(n + 1)^d`` nodes in the order of the module's node
    layout. The ``(n - 1)^d`` interior nodes carry the unknowns of a scheme,
    in flat order; the other nodes, those with some index 0 or ``n``, form
    the boundary ring, which lies on the faces of the box and carries the
    Dirichlet data ``g(grid.nodes[~grid.interior])``. The array properties
    are computed once and read-only.

    Parameters
    ----------
    box
        The box, of any dimension ``d >= 1``.
    n
        Cells per side, at least 2, so that there is an interior node.
    """

    box: Box
    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise TypeError(f"BoxGrid.box must be a Box, got {type(self.box).__name__}")
        object.__setattr__(self, "n", integer("n", self.n, 2))

    @property
    def dim(self) -> int:
        """Space dimension ``d``."""
        return self.box.dim

    @property
    def shape(self) -> tuple[int, ...]:
        """``(n + 1,) * d``: the shape of the array of node values."""
        return (self.n + 1,) * self.dim

    @cached_property
    def axes(self) -> tuple[np.ndarray, ...]:
        """The node coordinates along each axis, ``n + 1`` each, ends included."""
        return tuple(
            _read_only(np.linspace(a, b, self.n + 1))
            for a, b in zip(self.box.lo, self.box.hi, strict=True)
        )

    @property
    def h(self) -> tuple[float, ...]:
        """The spacing along each axis, ``axes[k][1] - axes[k][0]``."""
        return tuple(float(x[1] - x[0]) for x in self.axes)

    @cached_property
    def nodes(self) -> np.ndarray:
        """Shape ``((n + 1)^d, d)``: every node, in flat order."""
        mesh = np.meshgrid(*self.axes, indexing="ij")
        return _read_only(np.stack(mesh, axis=-1).reshape(-1, self.dim))

    @cached_property
    def interior(self) -> np.ndarray:
        """Shape ``((n + 1)^d,)``, boolean: True at the interior nodes."""
        return _read_only(_interior_mask(self.n, self.dim))

    @cached_property
    def interior_index(self) -> np.ndarray:
        """Shape ``((n - 1)^d, d)``, int: the multi-index of each interior node."""
        index = np.argwhere(self.interior.reshape(self.shape))
        return _read_only(index)

    def offset(self, direction: Sequence[int]) -> int:
        """Return the step of the flat index for a step of index offsets.

        The node ``direction`` away from the node with flat index ``k`` has
        flat index ``k + offset(direction)``, when it is in the grid.

        Parameters
        ----------
        direction
            ``d`` integer index offsets.

        Returns
        -------
        int
        """
        step = _direction(direction, self.dim)
        return int(step @ _strides(self.n, self.dim))

    def reach(self, direction: Sequence[int]) -> np.ndarray:
        """Return the fraction of a step that stays in the box, at each interior node.

        :func:`reach` at :attr:`interior_index`. The arms of a wide stencil,
        such as the 17-point stencil of
        :class:`~maesolvers.classical.monotone.Monotone17`, are steps of
        this kind.

        Parameters
        ----------
        direction
            ``d`` integer index offsets, not all zero.

        Returns
        -------
        numpy.ndarray
            Shape ``((n - 1)^d,)``: ``t`` at each interior node, in flat order.
        """
        return reach(self.interior_index, self.n, direction)

    def interpolant(self, values: np.ndarray, degree: int) -> Field:
        """Return the tensor spline through values at the nodes.

        Parameters
        ----------
        values
            Shape ``((n + 1)^d,)``: the values at the nodes, in flat order.
        degree
            Degree of the spline, 1 to 5 and at most ``n``.

        Returns
        -------
        Field
            :func:`tensor_spline` on :attr:`axes`, defined on the box; off
            the box it depends on the dimension (see :func:`tensor_spline`).
        """
        return tensor_spline(self.axes, values, degree)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _interior_mask(n: int, d: int) -> np.ndarray:
    """Shape ``((n + 1)^d,)``, boolean: True at the nodes off the boundary ring."""
    mask = np.zeros((n + 1,) * d, dtype=bool)
    mask[(slice(1, -1),) * d] = True
    return mask.ravel()


def _strides(n: int, d: int) -> np.ndarray:
    """Shape ``(d,)``: the step of the flat index along each axis."""
    return (n + 1) ** np.arange(d - 1, -1, -1)


def _direction(direction: Sequence[int], d: int) -> np.ndarray:
    step = np.asarray(direction)
    if step.shape != (d,) or not np.issubdtype(step.dtype, np.integer):
        raise ValueError(f"a direction is {d} integer index offsets, got {direction!r}")
    return step


def reach(index: np.ndarray, n: int, direction: Sequence[int]) -> np.ndarray:
    """Return the fraction of a step that stays in the box, at given nodes.

    From the node with multi-index ``i`` the step goes to
    ``i + t direction``; ``t`` is the largest value in ``[0, 1]`` for which
    every index stays in ``[0, n]``, positive at an interior node.
    ``t = 1`` where the node ``i + direction`` is in the grid (in the
    interior or on the boundary ring); otherwise the step leaves the box,
    through a face, at the point ``lo + (i + t direction) h`` of the
    boundary. ``t`` depends on the indices only, not on the box.

    Parameters
    ----------
    index
        Shape ``(m, d)``, int: multi-indices, each in ``[0, n]``.
    n
        Cells per side.
    direction
        ``d`` integer index offsets, not all zero.

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``: ``t`` at each node.
    """
    n = integer("n", n, 1)
    index = np.asarray(index)
    if index.ndim != 2 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError(
            f"index must be an int array of shape (m, d), got {index.shape}"
        )
    step = _direction(direction, index.shape[1])
    if not np.any(step):
        raise ValueError("the direction must not be zero")
    t = np.ones(len(index))
    for k, s in enumerate(step):
        if s > 0:
            t = np.minimum(t, (n - index[:, k]) / s)
        elif s < 0:
            t = np.minimum(t, index[:, k] / -s)
    return t


def stencil(n: int, offsets: np.ndarray, weights: Sequence[float]) -> sp.csr_matrix:
    """Return a linear stencil of width 1 at the interior nodes, as a matrix.

    The operator ``u -> sum_k weights[k] u(i + offsets[k])`` at every
    interior node ``i`` of the grid of ``n`` cells per side; an offset of
    width 1 stays in the grid from every interior node.

    Parameters
    ----------
    n
        Cells per side, at least 2.
    offsets
        Shape ``(k, d)``, ``d >= 1``, of an integer dtype, every entry -1, 0
        or 1: the index offsets of the stencil's points.
    weights
        ``k`` numbers: the weight of each point.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``((n - 1)^d, (n + 1)^d)``: from the values at all nodes to the
        stencil at the interior nodes, both in flat order.
    """
    n = integer("n", n, 2)
    offsets = np.asarray(offsets)
    if offsets.ndim != 2 or offsets.shape[1] < 1 or len(offsets) != len(weights):
        raise ValueError("stencil needs offsets of shape (k, d), d >= 1, and k weights")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(f"stencil offsets must be integers, got dtype {offsets.dtype}")
    if np.any(np.abs(offsets) > 1):
        raise ValueError("stencil offsets must be -1, 0 or 1")
    d = offsets.shape[1]
    interior = _interior_mask(n, d)
    index = np.flatnonzero(interior)
    rows = np.arange(index.size)
    shifts = offsets @ _strides(n, d)
    r = [rows] * len(shifts)
    c = [index + shift for shift in shifts]
    v = [np.full(rows.size, w) for w in weights]
    return sp.csr_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
        shape=(index.size, interior.size),
    )


def second_differences(
    n: int, h: Sequence[float]
) -> dict[tuple[int, int], sp.csr_matrix]:
    """Centred second differences on a grid of ``n`` cells per side.

    ``D_kk`` is the 3-point difference along axis ``k``,
    ``(u(i + e_k) - 2 u(i) + u(i - e_k)) / h_k^2``, and ``D_kl``, ``k < l``,
    the 4-point cross difference
    ``(u(i + e_k + e_l) - u(i + e_k - e_l) - u(i - e_k + e_l)
    + u(i - e_k - e_l)) / (4 h_k h_l)``. All are exact on quadratics; in 2D
    they are the nine points of the standard scheme, in 3D nineteen.

    Parameters
    ----------
    n
        Cells per side, at least 2.
    h
        The spacing along each axis: ``d >= 1`` positive, finite numbers.

    Returns
    -------
    dict
        ``D_kl`` for ``k <= l`` under the key ``(k, l)``: a
        :func:`stencil`, shape ``((n - 1)^d, (n + 1)^d)``.
    """
    if len(h) < 1:
        raise ValueError("h must hold the spacing along at least one axis")
    h = tuple(positive(f"h[{k}]", hk) for k, hk in enumerate(h))
    d = len(h)
    eye = np.eye(d, dtype=int)
    out = {}
    for k in range(d):
        scale = 1.0 / h[k] ** 2
        offsets = np.stack([eye[k], 0 * eye[k], -eye[k]])
        out[k, k] = stencil(n, offsets, [w * scale for w in (1.0, -2.0, 1.0)])
    for k in range(d):
        for m in range(k + 1, d):
            scale = 1.0 / (4.0 * h[k] * h[m])
            offsets = np.stack(
                [eye[k] + eye[m], eye[k] - eye[m], -eye[k] + eye[m], -eye[k] - eye[m]]
            )
            out[k, m] = stencil(n, offsets, [w * scale for w in (1.0, -1.0, -1.0, 1.0)])
    return out


def centred_differences(
    u: np.ndarray, h: Sequence[float]
) -> dict[tuple[int, int], np.ndarray]:
    """Return the centred second differences of node values at the interior nodes.

    The differences of :func:`second_differences`, computed by slicing the
    array of node values instead of by a matrix product: ``D_kk`` the
    3-point difference along axis ``k``, ``D_kl``, ``k < l``, the 4-point
    cross difference. They equal the entries of :func:`centred_hessian` at
    the interior nodes to rounding (the two sum the terms in another order).

    Parameters
    ----------
    u
        Shape ``(n + 1,) * d``, ``n >= 2``: the values at the nodes.
    h
        The spacing along each axis, ``d`` numbers.

    Returns
    -------
    dict
        ``D_kl`` for ``k <= l`` under the key ``(k, l)``, each of shape
        ``(n - 1,) * d``: at the interior nodes, in the grid's C order.
    """
    u = np.asarray(u, dtype=float)
    d = u.ndim
    if len(h) != d:
        raise ValueError(f"h must hold {d} spacings, got {len(h)}")

    def shifted(step: dict[int, int]) -> np.ndarray:
        index = tuple(
            slice(1 + step.get(k, 0), u.shape[k] - 1 + step.get(k, 0)) for k in range(d)
        )
        return u[index]

    centre = shifted({})
    out = {}
    for k in range(d):
        out[k, k] = (shifted({k: 1}) - 2.0 * centre + shifted({k: -1})) / h[k] ** 2
        for m in range(k + 1, d):
            out[k, m] = (
                shifted({k: 1, m: 1})
                - shifted({k: 1, m: -1})
                - shifted({k: -1, m: 1})
                + shifted({k: -1, m: -1})
            ) / (4.0 * h[k] * h[m])
    return out


def determinant(a: Entries, d: int) -> np.ndarray:
    """Return the determinant of symmetric ``d x d`` matrices given by their entries.

    Parameters
    ----------
    a
        The entries ``(k, l)``, ``k <= l``, arrays of one shape.
    d
        2 or 3.

    Returns
    -------
    numpy.ndarray
        The determinants, of the entries' shape.
    """
    if d == 2:
        return a[0, 0] * a[1, 1] - a[0, 1] ** 2
    if d == 3:
        return (
            a[0, 0] * a[1, 1] * a[2, 2]
            + 2.0 * a[0, 1] * a[0, 2] * a[1, 2]
            - a[0, 0] * a[1, 2] ** 2
            - a[1, 1] * a[0, 2] ** 2
            - a[2, 2] * a[0, 1] ** 2
        )
    raise ValueError(f"d must be 2 or 3, got {d}")


def cofactors(a: Entries, d: int) -> dict[tuple[int, int], np.ndarray]:
    """Return the cofactors of symmetric ``d x d`` matrices given by their entries.

    The cofactor ``C_kl`` is the derivative of :func:`determinant` with
    respect to the entry ``a_kl`` of a general matrix, so the derivative of
    the determinant of a symmetric matrix along a symmetric perturbation
    ``E`` is ``sum_k C_kk E_kk + 2 sum_{k<l} C_kl E_kl``.

    Parameters
    ----------
    a
        The entries ``(k, l)``, ``k <= l``, arrays of one shape.
    d
        2 or 3.

    Returns
    -------
    dict
        ``C_kl`` for ``k <= l`` under the key ``(k, l)``.
    """
    if d == 2:
        return {(0, 0): a[1, 1], (1, 1): a[0, 0], (0, 1): -a[0, 1]}
    if d == 3:
        return {
            (0, 0): a[1, 1] * a[2, 2] - a[1, 2] ** 2,
            (1, 1): a[0, 0] * a[2, 2] - a[0, 2] ** 2,
            (2, 2): a[0, 0] * a[1, 1] - a[0, 1] ** 2,
            (0, 1): a[0, 2] * a[1, 2] - a[0, 1] * a[2, 2],
            (0, 2): a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1],
            (1, 2): a[0, 1] * a[0, 2] - a[0, 0] * a[1, 2],
        }
    raise ValueError(f"d must be 2 or 3, got {d}")


def centred_hessian(grid: BoxGrid, values: np.ndarray) -> np.ndarray:
    """Return the centred discrete Hessian of node values, at every node.

    Parameters
    ----------
    grid
        The grid.
    values
        Shape ``((n + 1)^d,)``: the values at the nodes, in flat order.

    Returns
    -------
    numpy.ndarray
        Shape ``((n + 1)^d, d, d)``, symmetric: the
        :func:`second_differences` at the interior nodes and NaN on the
        boundary ring, as ``Result.node_hessian`` takes it;
        :func:`centred_differences` gives the same entries by slicing.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (len(grid.nodes),):
        raise ValueError(
            f"values must have shape ({len(grid.nodes)},), got {values.shape}"
        )
    hess = np.full((len(values), grid.dim, grid.dim), np.nan)
    for (k, m), operator in second_differences(grid.n, grid.h).items():
        hess[grid.interior, k, m] = hess[grid.interior, m, k] = operator @ values
    return hess


def tensor_spline(axes: Sequence[np.ndarray], values: np.ndarray, degree: int) -> Field:
    """Return the tensor-product interpolating spline through values on a grid.

    In 2D it is FITPACK's interpolating spline (scipy's
    ``RectBivariateSpline``), the interpolant of maesolvers 0.1, kept so
    that 2D results do not change. In any other dimension it is the tensor
    product of the 1D interpolating splines of
    ``scipy.interpolate.make_interp_spline`` (not-a-knot end conditions),
    fitted one axis after the other and evaluated by ``NdBSpline``, and it
    fits and evaluates the 21^4 points of the 4D shared evaluation grid in
    well under a second (``benchmarks/nd_grids.md``). On the box spanned by
    the axes the two are the same spline to rounding: on 2D axes they agree
    to about ``1e-15`` times the largest value (``tests/test_grid.py``).
    The spline is defined on that box only. Off it the two paths differ:
    FITPACK clamps each coordinate to the box, ``NdBSpline`` extends the
    polynomial pieces at the ends.

    Parameters
    ----------
    axes
        ``d >= 1`` increasing coordinate arrays.
    values
        The values at the grid points, of shape ``(len(axes[0]) * ... ,)``
        in C order (the last axis fastest) or of the grid's shape.
    degree
        Degree along every axis, 1 to 5, and less than the length of each
        axis.

    Returns
    -------
    Field
        Takes points of shape ``(..., d)``, returns values of shape ``(...)``.
    """
    degree = integer("degree", degree, 1)
    if len(axes) < 1:
        raise ValueError("axes must hold at least one coordinate array")
    shape = tuple(len(x) for x in axes)
    if degree > 5 or degree >= min(shape):
        raise ValueError(
            f"degree must be between 1 and 5 and below {min(shape)}, got {degree}"
        )
    values = np.asarray(values, dtype=float).reshape(shape)
    if len(axes) != 2:
        return _nd_spline(axes, values, degree)
    x, y = axes
    fit = RectBivariateSpline(x, y, values, kx=degree, ky=degree)

    def u(p: np.ndarray) -> np.ndarray:
        p = points(p, 2)
        out = fit(p[..., 0].ravel(), p[..., 1].ravel(), grid=False)
        return out.reshape(p.shape[:-1])

    return u


def _nd_spline(axes: Sequence[np.ndarray], values: np.ndarray, degree: int) -> Field:
    """Return the ``NdBSpline`` path of :func:`tensor_spline`, in any dimension.

    ``values`` has the grid's shape and ``degree`` is checked.
    """
    d = len(axes)
    coefficients, knots = values, []
    for k, x in enumerate(axes):
        fit = make_interp_spline(x, coefficients, k=degree, axis=k)
        coefficients = np.moveaxis(fit.c, 0, k)  # BSpline keeps its axis first
        knots.append(fit.t)
    spline = NdBSpline(tuple(knots), coefficients, degree)

    def u(p: np.ndarray) -> np.ndarray:
        p = points(p, d)
        return spline(p.reshape(-1, d)).reshape(p.shape[:-1])

    return u
