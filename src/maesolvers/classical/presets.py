"""Presets: named solvers for the common pairs of a scheme and an iteration.

A preset (ADR 0004) is an iteration whose scheme is built from flat
parameters, so the simple call ``FDPoisson(n=64).solve(problem)`` stays and
its :attr:`~maesolvers.core.solver.Solver.hparams` stay flat. The presets are
what :data:`maesolvers.registry.SOLVERS` and the comparison tables list, and
like every solver they declare their support on the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Generic, Literal, TypeVar

from maesolvers.classical.filtered import Filtered
from maesolvers.classical.hybrid import Hybrid
from maesolvers.classical.iterations import Newton, PoissonIteration, PolicyIteration
from maesolvers.classical.mixed import MixedHJB
from maesolvers.classical.monotone import Monotone17
from maesolvers.classical.schemes import Standard9
from maesolvers.core.problem import Problem
from maesolvers.core.solver import UnsupportedProblemError


@dataclass(frozen=True, kw_only=True)
class _Standard9Preset:
    """The fields and support of a preset on :class:`Standard9`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme from ``n`` and ``spline_degree`` before the iteration checks it.
    """

    domains: ClassVar[tuple[type, ...]] = Standard9.domains
    boundaries: ClassVar[tuple[type, ...]] = Standard9.boundaries
    dims: ClassVar[tuple[int, ...] | None] = Standard9.dims

    scheme: Standard9 = field(init=False, repr=False)
    n: int = Standard9.n
    spline_degree: int = Standard9.spline_degree

    def __post_init__(self) -> None:
        scheme = Standard9(n=self.n, spline_degree=self.spline_degree)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDNewton(_Standard9Preset, Newton):
    """The standard 9-point scheme solved by damped Newton.

    ``FDNewton(n=n, spline_degree=k, ...)`` is
    ``Newton(scheme=Standard9(n=n, spline_degree=k), ...)``. On a singular
    problem it may converge to a non-convex discrete root
    (``benchmarks/standard_fd.md``, Finding 2).

    Parameters
    ----------
    n
        Cells per side of the grid over the domain's bounding box
        (``h = 1/n`` on the unit square, ADR 0001), at least 2; on a box
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``; next to a
        curved boundary the residual is row-scaled
        (:meth:`Standard9.residual
        <maesolvers.classical.schemes.Standard9.residual>`), so the test
        there is looser by the inverse of the smallest step fraction.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes on a box, 1 to 5; a grid with fewer than ``spline_degree``
        cells per side takes degree ``n``. On a curved domain the solution
        off the nodes is the Clough-Tocher interpolant, whatever the degree.
    """

    start: None = field(default=None, init=False, repr=False)


@dataclass(frozen=True, kw_only=True)
class FDPoisson(_Standard9Preset, PoissonIteration):
    """The standard 9-point scheme solved by the Poisson iteration.

    ``FDPoisson(n=n, spline_degree=k, ...)`` is
    ``PoissonIteration(scheme=Standard9(n=n, spline_degree=k), ...)``, the
    "standard finite differences" column of Froese-Oberman 2011, Table 3.

    Parameters
    ----------
    n
        Cells per side of the grid over the domain's bounding box
        (``h = 1/n`` on the unit square, ADR 0001), at least 2; on a box
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns.
    max_iter
        Most iterations.
    tol
        Stop when successive iterates differ by at most ``tol`` at every node.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes on a box, 1 to 5; a grid with fewer than ``spline_degree``
        cells per side takes degree ``n``. On a curved domain the solution
        off the nodes is the Clough-Tocher interpolant, whatever the degree.
    """


_SquareSchemeT = TypeVar("_SquareSchemeT", Monotone17, Hybrid, Filtered, MixedHJB)


class _SquareBoxCheck(Generic[_SquareSchemeT]):
    """The ``check`` of a preset whose scheme needs a box with equal sides.

    Listed among a preset's bases before the iteration, whose ``check`` it
    extends.
    """

    domains: ClassVar[tuple[type, ...]]
    boundaries: ClassVar[tuple[type, ...]]
    dims: ClassVar[tuple[int, ...] | None]
    scheme: _SquareSchemeT

    def check(self, problem: Problem) -> None:
        """Raise if the problem is not a 2D box with equal sides and a Dirichlet boundary.

        Parameters
        ----------
        problem
            The problem to solve.

        Raises
        ------
        UnsupportedProblemError
            If the scheme does not support the problem; the message names the
            equal sides when they are what is missing.
        """
        if (
            isinstance(problem.domain, self.domains)
            and isinstance(problem.boundary, self.boundaries)
            and problem.dim in (self.dims or ())
            and not self.scheme.supports(problem)
        ):
            raise UnsupportedProblemError(
                f"{type(self).__name__} does not support {problem.name}: "
                f"{type(self.scheme).__name__} needs a Box with equal sides"
            )
        super().check(problem)  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class _Monotone17Preset(_SquareBoxCheck[Monotone17]):
    """The fields and support of a preset on :class:`Monotone17`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme from the flat fields before the iteration checks it.
    """

    domains: ClassVar[tuple[type, ...]] = Monotone17.domains
    boundaries: ClassVar[tuple[type, ...]] = Monotone17.boundaries
    dims: ClassVar[tuple[int, ...] | None] = Monotone17.dims

    scheme: Monotone17 = field(init=False, repr=False)
    n: int = Monotone17.n
    stencil: Literal[9, 17, 33] = Monotone17.stencil
    boundary: Literal["narrow", "interpolate"] = Monotone17.boundary
    jacobian_floor: float = Monotone17.jacobian_floor
    spline_degree: int = Monotone17.spline_degree

    def __post_init__(self) -> None:
        scheme = Monotone17(
            n=self.n,
            stencil=self.stencil,
            boundary=self.boundary,
            jacobian_floor=self.jacobian_floor,
            spline_degree=self.spline_degree,
        )
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "stencil", scheme.stencil)
        object.__setattr__(self, "jacobian_floor", scheme.jacobian_floor)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDMonotone(_Monotone17Preset, Newton):
    """The monotone wide-stencil scheme solved by damped Newton.

    ``FDMonotone(n=n, stencil=s, ...)`` is
    ``Newton(scheme=Monotone17(n=n, stencil=s, ...), ...)``, the monotone
    columns of Froese-Oberman 2011, Table 3 (SIAM J. Numer. Anal. and
    J. Comput. Phys.). Newton uses the scheme's generalised Jacobian, so it
    is a semi-smooth Newton method; ``max_iter`` is higher than ``Newton``'s,
    because T2 needs about 60 steps with the 9-point stencil
    (``docs/methods/monotone17.md``).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    stencil
        Points of the stencil: 9, 17 or 33.
    boundary
        ``"narrow"`` or ``"interpolate"``, as in :class:`Monotone17`.
    jacobian_floor
        The floor of the Jacobian's factors, times ``1 / (2 h^2)``, as in
        :class:`Monotone17`.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``.
    """

    start: None = field(default=None, init=False, repr=False)
    max_iter: int = 200


@dataclass(frozen=True, kw_only=True)
class _HybridPreset(_SquareBoxCheck[Hybrid]):
    """The fields and support of a preset on :class:`Hybrid`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme from the flat fields before the iteration checks it.
    """

    domains: ClassVar[tuple[type, ...]] = Hybrid.domains
    boundaries: ClassVar[tuple[type, ...]] = Hybrid.boundaries
    dims: ClassVar[tuple[int, ...] | None] = Hybrid.dims

    scheme: Hybrid = field(init=False, repr=False)
    start: FDMonotone = field(init=False, repr=False)
    n: int = Hybrid.n
    stencil: Literal[9, 17, 33] = Hybrid.stencil
    boundary: Literal["narrow", "interpolate"] = Hybrid.boundary
    jacobian_floor: float = Hybrid.jacobian_floor
    spline_degree: int = Hybrid.spline_degree
    epsilon: float | None = Hybrid.epsilon
    band: float = Hybrid.band

    def __post_init__(self) -> None:
        scheme = Hybrid(
            n=self.n,
            stencil=self.stencil,
            boundary=self.boundary,
            jacobian_floor=self.jacobian_floor,
            spline_degree=self.spline_degree,
            epsilon=self.epsilon,
            band=self.band,
        )
        start = FDMonotone(
            n=scheme.n,
            stencil=scheme.stencil,
            boundary=scheme.boundary,
            jacobian_floor=scheme.jacobian_floor,
            spline_degree=scheme.spline_degree,
            tol=self.tol,  # type: ignore[attr-defined]
        )
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "start", start)
        for name in (
            "n",
            "stencil",
            "jacobian_floor",
            "spline_degree",
            "epsilon",
            "band",
        ):
            object.__setattr__(self, name, getattr(scheme, name))
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDHybrid(_HybridPreset, Newton):
    """The hybrid scheme of Froese-Oberman 2011 solved by damped Newton.

    ``FDHybrid(n=n, stencil=s, epsilon=e, band=b, tol=t, ...)`` is
    ``Newton(scheme=Hybrid(n=n, stencil=s, epsilon=e, band=b, ...),
    start=FDMonotone(n=n, stencil=s, tol=t, ...), tol=t, ...)``: Newton on
    the hybrid scheme from the root of the monotone scheme, which the
    Poisson start does not reach on T2 (``benchmarks/hybrid.md``). It is
    the column "Hybrid" of Froese-Oberman 2011, Table 3 (J. Comput. Phys.):
    the standard 9-point scheme where the data are regular, the monotone
    wide-stencil scheme within ``band`` grid steps of the boundary and of
    where ``f`` is at most ``epsilon`` or at least ``1 / epsilon``.
    The start keeps ``FDMonotone``'s budget of 200 steps whatever
    ``max_iter``; ``Result.info["start_iterations"]`` and
    ``info["start_converged"]`` report it.

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    stencil
        Points of the monotone scheme's stencil: 9, 17 or 33.
    boundary
        ``"narrow"`` or ``"interpolate"``, as in :class:`Monotone17`.
    jacobian_floor
        The floor of the monotone Jacobian's factors, times ``1 / (2 h^2)``,
        as in :class:`Monotone17`.
    epsilon
        The threshold of the switching rule, or None for ``h``.
    band
        Radius in grid steps of the monotone neighbourhood, at least 0.
    max_iter
        Most Newton steps on the hybrid scheme.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``, on the hybrid
        scheme and on the start.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``.
    """

    max_iter: int = 200


@dataclass(frozen=True, kw_only=True)
class _FilteredPreset(_SquareBoxCheck[Filtered]):
    """The fields and support of a preset on :class:`Filtered`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme, :class:`Standard9` filtered with :class:`Monotone17` on one grid,
    from the flat fields before the iteration checks it. The support is
    Monotone17's, which Standard9's contains.
    """

    domains: ClassVar[tuple[type, ...]] = Monotone17.domains
    boundaries: ClassVar[tuple[type, ...]] = Monotone17.boundaries
    dims: ClassVar[tuple[int, ...] | None] = Monotone17.dims

    scheme: Filtered = field(init=False, repr=False)
    n: int = Monotone17.n
    stencil: Literal[9, 17, 33] = Monotone17.stencil
    boundary: Literal["narrow", "interpolate"] = Monotone17.boundary
    jacobian_floor: float = Monotone17.jacobian_floor
    epsilon: float | None = None
    spline_degree: int = Monotone17.spline_degree

    def __post_init__(self) -> None:
        monotone = Monotone17(
            n=self.n,
            stencil=self.stencil,
            boundary=self.boundary,
            jacobian_floor=self.jacobian_floor,
            spline_degree=self.spline_degree,
        )
        accurate = Standard9(n=monotone.n, spline_degree=monotone.spline_degree)
        scheme = Filtered(accurate=accurate, monotone=monotone, epsilon=self.epsilon)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", monotone.n)
        object.__setattr__(self, "stencil", monotone.stencil)
        object.__setattr__(self, "jacobian_floor", monotone.jacobian_floor)
        object.__setattr__(self, "epsilon", scheme.epsilon)
        object.__setattr__(self, "spline_degree", monotone.spline_degree)
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDFiltered(_FilteredPreset, Newton):
    """The filtered scheme of Froese and Oberman (2013) solved by Newton.

    ``FDFiltered(n=n, stencil=s, ...)`` is ``Newton(scheme=Filtered(
    accurate=Standard9(n=n), monotone=Monotone17(n=n, stencil=s, ...)))``
    with ``max_iter=200``: the standard 9-point scheme where it nearly agrees
    with the monotone wide-stencil scheme, the monotone scheme where they
    differ by more than twice the filter width (``docs/methods/filtered.md``).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2.
    stencil
        Points of the monotone scheme's stencil: 9, 17 or 33.
    boundary
        ``"narrow"`` or ``"interpolate"``, as in :class:`Monotone17`.
    jacobian_floor
        The floor of the monotone Jacobian's factors, times ``1 / (2 h^2)``,
        as in :class:`Monotone17`.
    epsilon
        The filter width; None takes Froese and Oberman's
        ``sqrt(h) + dtheta / 10``, as in :class:`Filtered`.
    max_iter
        Most Newton steps.
    tol
        Stop when ``max |residual| <= tol * max(1, max |f|)``.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``.
    """

    start: None = field(default=None, init=False, repr=False)
    max_iter: int = 200


@dataclass(frozen=True, kw_only=True)
class _MixedPreset(_SquareBoxCheck[MixedHJB]):
    """The fields and support of a preset on :class:`MixedHJB`.

    Listed first among a preset's bases, so its ``__post_init__`` builds the
    scheme from the flat fields before the iteration checks it.
    """

    domains: ClassVar[tuple[type, ...]] = MixedHJB.domains
    boundaries: ClassVar[tuple[type, ...]] = MixedHJB.boundaries
    dims: ClassVar[tuple[int, ...] | None] = MixedHJB.dims

    scheme: MixedHJB = field(init=False, repr=False)
    n: int = MixedHJB.n
    angles: int | None = MixedHJB.angles
    spline_degree: int = MixedHJB.spline_degree

    def __post_init__(self) -> None:
        scheme = MixedHJB(
            n=self.n, angles=self.angles, spline_degree=self.spline_degree
        )
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "n", scheme.n)
        object.__setattr__(self, "angles", scheme.angles)
        object.__setattr__(self, "spline_degree", scheme.spline_degree)
        super().__post_init__()  # type: ignore[misc]


@dataclass(frozen=True, kw_only=True)
class FDMixed(_MixedPreset, PolicyIteration):
    """The monotone mixed scheme of Chen, Wan and Lin (2018) solved by policy iteration.

    ``FDMixed(n=n, angles=M, ...)`` is
    ``PolicyIteration(scheme=MixedHJB(n=n, angles=M, ...), ...)``, the
    "proposed mixed stencil scheme" of Chen-Wan-Lin 2018, Tables 1 to 4:
    the standard 7-point stencil for the controls where it is monotone, the
    semi-Lagrangian wide stencil for the others (``docs/methods/chen_wan_lin2018.md``).

    Parameters
    ----------
    n
        Cells per side (``h = 1/n`` on the unit square, ADR 0001), so
        ``(n + 1)^2`` nodes and ``(n - 1)^2`` unknowns; at least 2. The
        paper's ``N`` is ``n - 1``.
    angles
        The number of angles of the wide-stencil search; None takes
        ``n - 1``, the paper's ``M = N``.
    max_iter
        Most policy iterations.
    tol
        Stop when the max residual of the HJB equation is at most ``tol``,
        the paper's ``1e-6`` by default.
    spline_degree
        Degree of the tensor spline that evaluates the solution off the
        nodes, 1 to 5; a grid with fewer than ``spline_degree`` cells per
        side takes degree ``n``.
    """
