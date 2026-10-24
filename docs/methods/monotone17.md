# Monotone wide-stencil scheme

The monotone wide-stencil scheme of Froese and Oberman (2011) discretises `det D^2 u = f` so that the discrete equation is monotone, and its solutions converge to the viscosity solution even when the solution is singular.
maesolvers ships it as the scheme `Monotone17`, with the 9-, 17- and 33-point stencils in 2D and the 19-point stencil in 3D, solved by `Newton`, and as the preset `FDMonotone` for that pair (ADR 0004).
Reproduction: [`benchmarks/monotone17.md`](../../benchmarks/monotone17.md) in 2D, [`benchmarks/monotone19.md`](../../benchmarks/monotone19.md) in 3D.

```python
from maesolvers import T2, FDMonotone, Monotone17, Newton
from maesolvers.metrics import evaluate

result = FDMonotone(n=63, stencil=17).solve(T2)  # the preset
same = Newton(scheme=Monotone17(n=63, stencil=17), max_iter=200).solve(T2)  # its parts
print(evaluate(result)["max"])  # 5.72e-4; Froese-Oberman print 0.6e-3
```

## The scheme

The domain is a box with equal sides and `n` cells per side, spacing `h` (`h = 1/n` on the unit square, ADR 0001); the unknowns are the values at the `(n - 1)^2` interior nodes, and the boundary nodes carry the Dirichlet data `g`.

For a convex `C^2` function the determinant of the Hessian is the smallest product of second derivatives along two orthogonal directions (Froese-Oberman 2011, Lemma 2):

```text
det D^2 u = min over orthonormal (nu, nu') of u_nu nu * u_nu' nu'.
```

The scheme restricts the directions to those of a stencil on the grid and takes centred differences along them:

```text
D_nu nu u(x) = (u(x + nu h) - 2 u(x) + u(x - nu h)) / (|nu|^2 h^2)
MA_M[u](x)   = min over pairs (nu, nu') of (D_nu nu u)^+ (D_nu' nu' u)^+ = f(x).
```

A stencil of width `w` holds every grid direction `(a, b)` with coprime components and `max(|a|, |b|) <= w`; each direction is paired with its perpendicular `(-b, a)`.
Widths 1, 2 and 3 are the 9-, 17- and 33-point stencils, with 2, 4 and 8 orthogonal pairs (`stencil_frames`, which calls a pair a frame); the pairs are orthogonal in space only on square cells, hence the equal sides.

Each term is nondecreasing in the differences `u_j - u_i`, so the scheme is monotone (degenerate elliptic), and Barles-Souganidis theory gives convergence to the viscosity solution as `h` and the angular resolution `dtheta` of the stencil go to zero.
The consistency error is `O(h^2 + dtheta)`: for a fixed stencil the error stops decreasing once the angular error dominates, which is what the tables show for the 9-point stencil on T1.

### Below zero

The bare scheme is flat wherever a second difference is negative: `(D_nu nu u)^+ = 0` there, whatever its value.
Where `f = 0` this makes every grid function with a negative second difference in each pair a root: on T20 at `N = 31`, Newton on the bare scheme reaches two different non-convex roots from two starts, with errors `1.09e-3` and `1.56e-3` against the convex root's `0.80e-3` (17 points, `benchmarks/monotone17_checks.csv`, rows `bare`).
Where `f > 0` it leaves Newton without a direction out of a non-convex iterate: on T1 and T2 from the papers' start the line search of the bare scheme stalls at the first step (same rows).
`Monotone17` therefore continues each pair's value below zero with a positive slope `kappa`:

```text
F_pair(a, b) = a^+ b^+ + kappa (min(a, 0) + min(b, 0)),     kappa = sqrt(max(f, mean f, 1)).
```

This leaves the value of every discretely convex grid function unchanged, keeps the scheme monotone, and makes a pair's value at most `kappa a <= a` when it sees a negative second difference `a`.
So a root of `min F_pair = f` has no negative second difference in any pair that enters the minimum, and is a root of Froese and Oberman's `MA_M[u] = f` that is convex along every stencil direction, whatever `kappa`.
The lower bound 1 puts `kappa` on the scale of `Newton`'s stopping rule, `max |residual| <= tol * max(1, max |f|)`: a negative second difference `a` adds at least `|a|` to the residual, so the rule cannot accept a non-convex iterate.
(A bound below 1, such as the Jacobian floor, let `Newton` stop at the harmonic start where `f = 0` everywhere.)
Tests check that ten times `kappa`, and a non-convex start, reach the same solution on T20 and on `f = 0` with convex data.
The papers enforce convexity through their starting point instead (one Poisson step, convexified where needed, SIAM Section 5.2).

Froese and Oberman also define a smooth scheme `(MA)^delta`, with `max^delta(a, b) = (a + b + sqrt((a - b)^2 + delta^2))/2` and `min^delta` likewise in place of the positive parts and the minimum (SIAM Section 3.5), and prove that Newton converges for it (Theorem 9).
They print no `delta`, and their Section 5 and JCP Section 7 apply Newton to `MA_M` itself with Danskin's Jacobian, as `Monotone17` does; `(MA)^delta` moves the T20 error by a factor 4 between `delta = 1e-3` and `1e-2` at `N = 31`, and Newton on it fails at `N = 63` for both (`benchmarks/monotone17.md`, Finding 5).

### Next to the boundary

At a node next to the boundary, a wide direction's arm can leave the box.
`boundary` decides what happens then:

- `"narrow"` (the default) leaves that pair out of the minimum, so the stencil narrows to the pairs that fit; the axis pair always fits.
- `"interpolate"` shortens the arm to the boundary and takes the Dirichlet data there, with the first-order difference for unequal arms (Oberman 2008, Section 3.1).

The papers describe the second (SIAM p. 1700, JCP p. 823); their tables match the first, to four digits on T1 from `N = 181` on, and to three at `N = 63` and 127 (`benchmarks/monotone17.md`, Finding 1).
The 9-point stencil fits everywhere, so the two agree there.

## In three dimensions

Lemma 2 of Froese and Oberman holds in any dimension: for a convex `C^2` function the determinant of the Hessian is the smallest product of second derivatives along the `d` directions of an orthonormal frame.
`Monotone17(stencil=19)` is their 3D scheme (SIAM Section 7):

```text
MA_M[u](x) = min over frames (nu_1, nu_2, nu_3) of prod_k (D_nu_k nu_k u)^+ = f(x),
```

with the frames of the 19-point stencil (`stencil_frames(19)`): the axis frame, and each axis with the two diagonals of the plane perpendicular to it, `(e_1, e_2 + e_3, e_2 - e_3)` and its two rotations, four triples on the three axes and the six face diagonals.
A body diagonal such as `(1, 1, 1)` is orthogonal to no two directions of width 1, so the 27-point cube adds no frame; the paper's "19 point stencil" is exactly the set these frames use.
The stencil decides the dimension: a scheme with 9, 17 or 33 points supports 2D problems only, one with 19 points 3D problems only, on a cube with a `Dirichlet` boundary, so `FDMonotone(n=31, stencil=19).solve(T9)`.
Every arm has width 1 and ends at a node, so `boundary` changes nothing in 3D, and the Laplacian of `solve_laplace` is the 7-point one, the sum of the axis frame's differences.

Below zero the value continues as in 2D, `prod_k a_k^+ + kappa sum_k min(a_k, 0)`, with `kappa = max(f, mean f, 1)^(2/3)`, the scale of the derivative of a product of three second differences.
The Jacobian differentiates the active triple, `c_k = prod_{j != k} max(a_j, epsilon)` where `a_k >= 0` and `kappa` where `a_k < 0`, so `-J` is again an M-matrix.
Where `f = 0` two factors can vanish at once and the coefficient is `epsilon^2`.
Newton converges slowly on the 3D `C^1` example, 87 steps at `N = 31` against 6 to 10 on T9 and T22; the floor changes that (29 steps with the paper's `1e-8`, 52 with the coefficient floored once), but no floor is fastest at every `N`, and the cause is open (`benchmarks/monotone19.md`, issue #65).

The angular resolution of the 19-point stencil is coarse: the largest angle between a direction and the nearest stencil direction is `acos(sqrt(2/3))`, about 35 degrees, at the body diagonals (`angular_resolution(19)`), against 22.5 degrees for the 9-point stencil in 2D.
On T9 the error therefore stays above `1.2e-2` at every grid of the paper's table, as theirs does.

## Newton's method

`Newton` solves the scheme as it solves the standard one: damped Newton steps on the residual, from `laplace u_0 = 2 sqrt(f)` (`3 f^(1/3)` in 3D), halved until the max residual decreases, stopping when it is at most `tol * max(1, max |f|)`.
The scheme's `jacobian` is a generalised Jacobian (Danskin's theorem): at each node it differentiates the pair that attains the minimum,

```text
J = sum over the active pair of diag(c_a) D_aa + diag(c_b) D_bb,
c_a = max(b, epsilon) where a >= 0, kappa where a < 0,
```

so `Newton` on this scheme is a semi-smooth Newton method, and `-J` is an M-matrix (positive diagonal, nonpositive off-diagonal, diagonally dominant), so every step is well posed.
The floor `epsilon = jacobian_floor / (2 h^2)` follows Froese and Oberman (SIAM Section 5.3), who take `jacobian_floor = 1e-8`.
In 2D, with that floor Newton converges only linearly where `f = 0`: on T20 it takes 439 steps at `n = 181` with the 17-point stencil.
The default `1e-12` takes 27 there and reaches the same root (`benchmarks/monotone17_checks.csv`, rows `floor`), since the floor changes the steps, not the equation; in 3D neither floor is faster at every grid ("In three dimensions" above).

The papers start from `laplace u_0 = sqrt(2 f)` (`(d! f)^(1/d)` in `d` dimensions, SIAM eq. (8)) on a coarse grid, interpolated and convexified; Newton's start is a lower bound for the Laplacian of the convex solution and needs no convexification, because the scheme has no non-convex roots and a non-convex iterate cannot pass the stopping rule.
The start changes the number of steps, not the solution.

## Source papers

- B. D. Froese, A. M. Oberman, "Convergent finite difference solvers for viscosity solutions of the elliptic Monge-Ampere equation in dimensions two and higher", SIAM J. Numer. Anal. 49 (2011) 1692-1714, doi 10.1137/100803092: the scheme in any dimension (Sections 3.3 to 3.5), Newton with the floored Jacobian (Section 5), Table 3, the 9-, 17- and 33-point columns, compared on T1, T2, T20 and the cone T21, and Table 4, the 19-point stencil in 3D (Section 7), compared on T9, the 3D `C^1` example and T22.
- B. D. Froese, A. M. Oberman, "Fast finite difference solvers for singular solutions of the elliptic Monge-Ampere equation", J. Comput. Phys. 230 (2011) 818-834, doi 10.1016/j.jcp.2010.10.020: the same scheme with the 17-point stencil (Section 4), and Table 3, column "Monotone", compared on the same four problems.
- A. M. Oberman, "Wide stencil finite difference schemes for the elliptic Monge-Ampere equation and functions of the eigenvalues of the Hessian", Discrete Contin. Dyn. Syst. Ser. B 10 (2008) 221-238, doi 10.3934/dcdsb.2008.10.221: the stencils (Table 1) and the boundary treatment `"interpolate"` (Section 3.1).
  Its own scheme, the product of the smallest and largest directional second difference, is a different discretisation; maesolvers does not ship it.

`benchmarks/monotone17.md` transcribes the tables next to ours.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `Monotone17` | `n` | 64 | cells per side, at least 2 |
| `Monotone17` | `stencil` | 17 | points of the stencil: 9, 17 or 33 in 2D, 19 in 3D |
| `Monotone17` | `boundary` | `"narrow"` | wide pairs next to the boundary: left out (`"narrow"`) or with arms shortened to the boundary (`"interpolate"`) |
| `Monotone17` | `jacobian_floor` | 1e-12 | the Jacobian's factors are at least `jacobian_floor / (2 h^2)`; the papers take 1e-8 |
| `Monotone17` | `spline_degree` | 3 | degree of the spline off the nodes, 1 to 5; degree `n` if `n` is smaller |
| `Newton` | `max_iter` | 50 (`FDMonotone`: 200) | most steps |
| `Newton` | `tol` | 1e-10 | largest residual, relative to `max(1, max \|f\|)` |

The preset takes the scheme's parameters flat: `FDMonotone(n=127, stencil=33)` is `Newton(scheme=Monotone17(n=127, stencil=33), max_iter=200)`.
Its `max_iter` is higher than `Newton`'s because T2 with the 9-point stencil needs about 60 steps at `n = 127`.

## Known limits

- Only a `Box` with equal sides and a `Dirichlet` boundary, in the stencil's dimension: `solve` raises `UnsupportedProblemError` for a rectangle, a disk (T8), a transport boundary (T12), a 3D box with a 2D stencil (T9 with the default `stencil=17`), or a box in 4D (T10).
- In 3D only the 19-point stencil exists: its 35-degree angular resolution bounds the accuracy on smooth problems (T9 stays near `1.25e-2`), and a wider 3D stencil, whose frames would mix directions of several widths, is not implemented.
  The hybrid and filtered schemes stay 2D, because their standard part `Standard9` is 2D.
- The accuracy is limited by the stencil's angular resolution: on T1 the error stays near `1.6e-3` with 9 points, `4.4e-4` with 17 and `1.5e-4` with 33 however fine the grid, against `5.3e-7` for the standard scheme at `n = 361`.
  On smooth problems report the standard scheme (`FDPoisson`); the monotone scheme is for singular ones.
- `Result.node_hessian` is the centred 9-point Hessian of the node values (`centred_hessian` in `maesolvers.classical.grid`), not a quantity of the scheme.
  On T20 its smallest eigenvalue is negative, by far more than rounding (below `-1e-3` at `n = 16`, `tests/test_monotone.py`), at nodes of the flat disk where `f = 0`, because the centred `u_xy` (half the difference of the two diagonal second differences) makes the 2 by 2 form indefinite where `u_xx` and `u_yy` are near zero, although every stencil direction is convex.
  So `nonconvex_fraction` is 0.04 to 0.10 on T20 at every `n`, stencil and boundary treatment, and does not decrease with `n` (`benchmarks/monotone17.md`, Finding 6); on T2 it is 8e-5 to 1.6e-3, at nodes next to the corner.
- The cone T21, `f` a Dirac mass, is not reproduced: with the papers' `f = 4/h^2` at the tip, the scheme's minimum over pairs at the tip of the exact cone is `2/h^2` or less, so the discrete cone is steeper than the exact one and the error does not decrease with `n` (`benchmarks/monotone17.md`, Finding 4).
- The solves use a sparse LU at every step: at `n = 361` a step takes about 1.2, 4 and 15 seconds with 9, 17 and 33 points on one core of a100server.

## Reproduction

At the levels of ADR 0002, as `benchmarks/monotone17.md` states them:

| Problem | Froese-Oberman 2011, SIAM, Table 3 (9, 17, 33 points) | Froese-Oberman 2011, JCP, Table 3 (17 points) |
|---|---|---|
| T1 | order of magnitude; exact from `N = 127` | order of magnitude; exact from `N = 181` |
| T2 | exact | order of magnitude; exact from `N = 89`, and at `N <= 63` with `h = 1/(N - 1)` |
| T20 | 9 points order of magnitude; 17 and 33 points mismatch | mismatch |
| T21 (cone) | mismatch | mismatch |

The published tables match the narrow stencil next to the boundary, not the interpolation the papers describe; at coarse grids the published T1 errors are larger than the converged ones; on T20 they stay above the convex discrete solution's, within 13 percent of a non-convex root of the bare scheme at `N = 31`; the papers' discrete Dirac mass does not fit the scheme.
In 3D, against Froese-Oberman 2011, SIAM, Table 4 (19 points), with `h = 1/N` (ADR 0001):

| Problem | Level |
|---|---|
| T9 | order of magnitude; the printed digits at `N = 21` and 31 |
| `C^1` example (15), no T id | order of magnitude, only through `N = 7` (without it, a mismatch: orders 0.11 published, 0.37 ours) |
| T22 | mismatch (orders), only through `N = 7` (without it, order of magnitude: 0.29 and 0.21) |

The paper does not say whether its `N` counts cells or nodes; with `h = 1/(N - 1)` T9 matches the printed digits at `N = 11`, 15 and 21 and T22 is order of magnitude, while no convention matches every cell (`benchmarks/monotone19.md`, Finding 1).
On the `C^1` example our error is 1.3 to 1.9 times the published one at every `N`, and the bare scheme's non-convex roots are further off, not closer.
`tests/test_reproduction.py` pins the exact values; the full comparison and the findings are in [`benchmarks/monotone17.md`](../../benchmarks/monotone17.md) (2D) and [`benchmarks/monotone19.md`](../../benchmarks/monotone19.md) (3D).
