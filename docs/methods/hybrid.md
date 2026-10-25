# Hybrid scheme

The hybrid scheme of Froese and Oberman (2011) averages two discretisations of $\det D^2 u = f$ on one grid: the standard 9-point scheme where the data say the solution is smooth, for its second-order accuracy, and the monotone wide-stencil scheme near where the solution may be singular, for its convergence to the viscosity solution.
maesolvers ships it as the scheme `Hybrid`, built from the parts `Standard9` and `Monotone17`, solved by `Newton` from the root of the monotone scheme (`start=FDMonotone(...)`), and as the preset `FDHybrid` for that combination (ADR 0004).
Reproduction: [`benchmarks/hybrid.md`](../../benchmarks/hybrid.md).

```python
from maesolvers import T2, FDHybrid, FDMonotone, Hybrid, Newton
from maesolvers.metrics import evaluate

result = FDHybrid(n=127).solve(T2)  # the preset
same = Newton(scheme=Hybrid(n=127), start=FDMonotone(n=127), max_iter=200).solve(T2)
print(evaluate(result)["max"])  # 2.00e-4; Froese-Oberman print 0.20e-3
```

## The scheme

The domain is a box with equal sides and $n$ cells per side ($h = 1/n$ on the unit square, ADR 0001); the unknowns are the values at the $(n - 1)^2$ interior nodes, and the boundary nodes carry the Dirichlet data.
At every interior node the discrete equation is (JCP 2011, Section 5)

$$
\mathrm{MA}_H[u] = w\thinspace \mathrm{MA}_N[u] + (1 - w)\thinspace \mathrm{MA}_M[u] = f,
$$

$$
\mathrm{MA}_N[u] = (D_{xx} u)(D_{yy} u) - (D_{xy} u)^2 \quad (\texttt{Standard9}),
$$

$$
\mathrm{MA}_M[u] = \min_{\text{orthogonal stencil pairs}} (D_{\nu\nu} u)^+ (D_{\nu'\nu'} u)^+ \quad (\texttt{Monotone17}).
$$

with a weight $w$ in $[0, 1]$ fixed before the solve from the data alone.
`docs/methods/standard9.md` and `docs/methods/monotone17.md` describe the two parts; `Hybrid` builds them from its own flat fields as its attributes `standard` and `monotone`, and its residual and Jacobian are the same averages of theirs:

$$
\texttt{residual} = w (\mathrm{MA}_N[u] - f) + (1 - w) (\mathrm{MA}_M[u] - f),
$$

$$
\texttt{jacobian} = \operatorname{diag}(w) J_N + \operatorname{diag}(1 - w) J_M.
$$

$\mathrm{MA}_M$ here is `Monotone17`'s, continued below zero with the slope $\kappa$ (`docs/methods/monotone17.md`, "Below zero"), and its stencil, boundary treatment and Jacobian floor are `Hybrid`'s fields of the same names.

### The switching rule

Froese and Oberman take the weight from the regularity theory of the Monge-Ampere equation: a $C^{2,\alpha}$ solution exists when the domain is strictly convex with a $C^{2,\alpha}$ boundary, $g$ is $C^{2,\alpha}$ and $f$ is $C^\alpha$ and strictly positive (their conditions (2)).
They name the set where one of these may fail (eq. (6)),

$$
\Omega^s = \lbrace x \in \Omega : f(x) \le \varepsilon \text{ or } f(x) \ge 1/\varepsilon\rbrace \cup \lbrace x \text{ on the boundary} : \text{flat there, or } g \text{ not } C^{2,\alpha}\rbrace ,
$$

take $\varepsilon = h$, and set $w = 0$ in an $h$-neighbourhood of $\Omega^s$ and $w = 1$ away from it, through "a differentiable function".
Eq. (6) as printed reads $\varepsilon < f(x) < 1/\varepsilon$, which is the regular set, not the singular one; the text around it, and the T21 remark that the scheme is the monotone one where $f$ is "either $0$ or very large", mean the complement written above.

`Hybrid.weights(problem, nodes, h)` computes this rule on the grid, through the pure function `hybrid_weights(f, epsilon, band)` in `maesolvers.classical.hybrid`:

- $\Omega^s$ on the grid is every boundary node (the sides of a box are flat) and every interior node where $f \le \varepsilon$ or $f \ge 1/\varepsilon$, or $f$ is not finite;
- $w = 0$ at every interior node within `band` grid steps of a node of $\Omega^s$ (Euclidean distance between nodes), and $w = 1$ at the others.

`epsilon = None` means $h$, $\varepsilon$ must lie in $(0, 1)$ (with $\varepsilon \ge 1$ every node would be singular), and `band = 1` is the paper's $h$-neighbourhood, so by default the first ring of interior nodes is always monotone, and so is every node within one step of where $f$ is at most $h$ or at least $1/h$.
The paper gives no formula for the transition from $0$ to $1$, so the rule is a step; `benchmarks/hybrid.md` tests wider bands.
The clause $g \text{ not } C^{2,\alpha}$ is not computable from a `Problem` and is left out: on a box the whole boundary is in $\Omega^s$ anyway.

A rule of your own is a subclass that overrides `weights(problem, nodes, h)`.
`discretize` passes the interior nodes in the order of the unknowns, shape $((n - 1)^2, 2)$ (node $(i, j)$, at $(i h, j h)$ on the unit square, in row $(i - 1) (n - 1) + j - 1$), and the grid step $h$; the rule returns one weight per row, and `discretize` checks the shape and that each weight lies in $[0, 1]$.
A rule that depends on position reads it from `nodes`:

```python
from dataclasses import dataclass

import numpy as np

from maesolvers import T1, FDMonotone, Hybrid, Newton


@dataclass(frozen=True, kw_only=True)
class MonotoneOnTheLeft(Hybrid):
    def weights(self, problem, nodes, h):
        return np.where(nodes[:, 0] < 0.5, 0.0, 1.0)


result = Newton(scheme=MonotoneOnTheLeft(n=32), start=FDMonotone(n=32)).solve(T1)
```

Results carry `info["monotone_fraction"]`, the mean of $1 - w$ over the unknowns: the share of monotone nodes for a rule of $0$ and $1$, and $0.5$ for $w = 0.5$ everywhere.

## Newton's method and its start

`Newton` solves the hybrid scheme as it solves its parts: damped Newton steps on the residual, halved until the max residual decreases, stopping when it is at most `tol` $\cdot \max(1, \max \lvert f \rvert)$ (`docs/methods/standard9.md`).

The start matters.
The standard part has non-convex roots, and on T2 Newton from its default start, $\Delta u_0 = 2 \sqrt{f}$, reaches one from $n = 63$ on: the error is $16$, $37$ and $425$ times the published one at $n = 63$, $127$ and $361$ (`benchmarks/hybrid.md`, Finding 1).
The paper starts Newton from one step of the semi-implicit (Poisson) iteration, convexified by the convex envelope and interpolated from a coarse grid (JCP Section 7.1).
maesolvers starts it from the root of the monotone scheme instead, whose roots are convex along every stencil direction.
`Newton`'s field `start` takes any solver of the same unknowns that is a `SchemeIteration`: `Newton(scheme=Hybrid(n), start=FDMonotone(n))` first runs `FDMonotone(n)` with its own `max_iter` and `tol` and steps from its final unknowns, and `FDHybrid` does this with a start built from its own fields.
The paper's start without the convex envelope is `start=PoissonIteration(scheme=Standard9(n), max_iter=1)`.
From the monotone root Newton converges in $3$ steps on T1 and T2 ($4$ on T1 at $N = 361$) and to the convex root on every problem tried.
It costs one monotone solve: $5$ to $43$ steps with $17$ points (`benchmarks/monotone17.md`).
`Result.info["iterations"]`, `converged`, `history` and the callbacks (so `MLflowLogger`) cover the steps on the hybrid scheme only; `info["start_iterations"]` and `info["start_converged"]` report the start, `time` includes it, and a start that did not converge also raises a `RuntimeWarning`.
`FDHybrid`'s start keeps `FDMonotone`'s budget of $200$ steps whatever `max_iter`.

## Source paper

- B. D. Froese, A. M. Oberman, "Fast finite difference solvers for singular solutions of the elliptic Monge-Ampere equation", J. Comput. Phys. 230 (2011) 818-834, doi 10.1016/j.jcp.2010.10.020: the hybrid scheme (Section 5), Newton's method with the averaged Jacobian and its start (Section 7), the failure of Newton on the standard scheme (Section 8.1), and Table 3, column "Hybrid", compared on T1, T2, T20 and the cone T21.

The parts' sources are in `docs/methods/standard9.md` and `docs/methods/monotone17.md`.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `Hybrid` | $n$ | $64$ | cells per side, at least $2$ |
| `Hybrid` | `stencil` | $17$ | points of the monotone stencil: $9$, $17$ (the paper's) or $33$ |
| `Hybrid` | `boundary` | `"narrow"` | the monotone scheme's wide pairs next to the boundary, as in `Monotone17` |
| `Hybrid` | `jacobian_floor` | $10^{-12}$ | the monotone Jacobian's floor, as in `Monotone17` |
| `Hybrid` | `spline_degree` | $3$ | degree of the spline off the nodes, $1$ to $5$; degree $n$ if $n$ is smaller |
| `Hybrid` | `epsilon` | None ($h$) | threshold of the switching rule, in $(0, 1)$: $f \le \varepsilon$ or $f \ge 1/\varepsilon$ is singular |
| `Hybrid` | `band` | $1.0$ | radius in grid steps of the monotone neighbourhood of the singular set, at least $0$ |
| `Newton` | `start` | None (`FDHybrid`: `FDMonotone` on its fields) | a `SchemeIteration` whose final unknowns are the first iterate; it keeps its own `max_iter` and `tol` |
| `Newton` | `max_iter` | $50$ (`FDHybrid`: $200$) | most Newton steps on the hybrid scheme |
| `Newton` | `tol` | $10^{-10}$ | largest residual, relative to $\max(1, \max \lvert f \rvert)$; `FDHybrid` passes it to its start too |

The preset takes the scheme's parameters flat: `FDHybrid(n=127, band=2)` is `Newton(scheme=Hybrid(n=127, band=2), start=FDMonotone(n=127), max_iter=200)`.

## Known limits

- Only a 2D `Box` with equal sides and a `Dirichlet` boundary, as for `Monotone17`; on a strictly convex domain the paper's rule would make the scheme the standard one, but `Monotone17` has no curved boundary yet.
- On a box the first ring of nodes is always monotone, so on a smooth problem the hybrid scheme is less accurate than the standard one (the paper's Remark 6): on T1 its error is $2.4$ to $3.0$ times `FDPoisson`'s.
  On smooth problems report the standard scheme; the hybrid scheme is for problems that may be singular.
- `Newton` from its default start can reach a non-convex root (above); give it a monotone `start`, as the preset does.
- Where $f$ vanishes or is huge everywhere (the cone T21) the weight is $0$ at every node and the scheme is `Monotone17`, with the same limits (`docs/methods/monotone17.md`).
- `Result.node_hessian` is the centred 9-point Hessian of the node values, as for both parts; on T20 it is indefinite at nodes of the flat disk (`benchmarks/monotone17.md`, Finding 6).

## Reproduction

At the levels of ADR 0002, as `benchmarks/hybrid.md` states them:

| Problem | Froese-Oberman 2011, JCP, Table 3, column "Hybrid" |
|---|---|
| T1 | mismatch: ours is $1.4$ to $2.9$ times smaller, and converges faster (order $1.91$ against $1.63$) |
| T2 | order of magnitude; exact from $N = 127$ on, and at every $N$ with $h = 1/(N - 1)$ |
| T20 | mismatch: ours is $2.4$ to $7$ times smaller, the same pattern as the monotone column |
| T21 (cone) | mismatch: the scheme is the monotone one there |

On T2 the hybrid scheme gains on the monotone one from $N = 89$ on and reaches $4.2 \times 10^{-5}$ at $N = 361$, order $1.5$, the paper's $O(h^{1.5})$.
No band of a fixed number of grid steps reproduces the T1 column; the T20 column inherits the open causes of the monotone column; Newton on the standard scheme from the exact solution of T2 converges to the convex root in our runs, so the paper's Section 8.1 is not confirmed as stated, while from the Poisson start Newton on the hybrid scheme reaches a non-convex root.

`tests/test_reproduction.py` pins the exact values; the full comparison and the findings are in [`benchmarks/hybrid.md`](../../benchmarks/hybrid.md).
