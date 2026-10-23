# Filtered scheme

The filtered scheme of Froese and Oberman (2013) combines an accurate scheme with a monotone one: it is the accurate scheme where the two nearly agree, as on a smooth solution, and the monotone scheme where they differ, near a singularity.
It stays within a small distance `eps` of the monotone scheme, so it is "nearly monotone", and its solutions still converge to the viscosity solution as `eps` goes to zero.
maesolvers ships it as the scheme `Filtered`, which takes any accurate and any monotone scheme as parts, solved by `Newton`, and as the preset `FDFiltered`, the standard 9-point scheme `Standard9` filtered with the monotone wide-stencil scheme `Monotone17` (ADR 0004).
Reproduction: [`benchmarks/filtered.md`](../../benchmarks/filtered.md).

```python
from maesolvers import T20, FDFiltered, Filtered, Monotone17, Newton, Standard9
from maesolvers.metrics import evaluate

result = FDFiltered(n=62, stencil=17).solve(T20)  # the preset
parts = Filtered(accurate=Standard9(n=62), monotone=Monotone17(n=62, stencil=17))
same = Newton(scheme=parts, max_iter=200).solve(T20)  # its parts
print(evaluate(result)["max"], result.info["monotone_fraction"])  # 1.73e-4, 0.0
```

## The scheme

Write `F_A[u]` and `F_M[u]` for the residuals of the two parts at the unknowns, here `MA_A[u] - f` and `MA_M[u] - f`.
The filtered scheme is (Froese-Oberman 2013, Section 4.1)

```text
F_H[u] = F_M[u] + eps S((F_A[u] - F_M[u]) / eps),

S(x) = x                  for |x| <= 1,
       sign(x) (2 - |x|)  for 1 <= |x| <= 2,
       0                  for |x| >= 2.
```

`S` (`filter_function`) is continuous and bounded by 1, so `|F_H - F_M| <= eps` for every grid function.
At a node where `|F_A - F_M| <= eps`, `F_H = F_A`; where `|F_A - F_M| >= 2 eps`, `F_H = F_M`; in the band between, `F_H = 2 F_M - F_A + 2 eps sign(F_A - F_M)`.
`f` cancels in `F_A - F_M`, so the filter compares the two discrete Monge-Ampere operators.

The paper's width is

```text
eps = sqrt(h) + dtheta / 10      (filter_width)
```

with `h` the grid spacing and `dtheta` the angular resolution of the monotone scheme's stencil.
The paper does not give `dtheta` a value.
maesolvers takes the largest angle between neighbouring stencil directions: `pi/4` for 9 points, `atan(1/2)` for 17 and `atan(1/3)` for 33.
That is twice Oberman's (2008, eq. (3.3)) angular resolution, the largest angle between a direction and the nearest stencil direction (`angular_resolution`, `pi/8`, `atan(1/2)/2`, `atan(1/3)/2`).
The evidence for the full gap is the paper's own T2 columns of Table 4.1: with it our last iterates print as the published values in 14 of 15 cells, with half of it in none of the six at `N = 255` and 361 (`benchmarks/filtered.md`, Finding 3); none of these solves converges.
`Filtered(epsilon=None)` computes the width from the monotone part's `n` and `angular_resolution`; a number fixes it.

Why it converges: the monotone scheme is consistent, monotone and stable, and the filtered scheme differs from it by at most `eps`, which goes to zero with `h`; Froese and Oberman extend the Barles-Souganidis argument to such nearly monotone schemes (their Theorem 1, Lemma 3).
Why it is accurate: on a smooth solution the two schemes differ by their consistency errors, `O(h^2)` and `O(h^2 + dtheta)`, and if these are smaller than `eps` the filter keeps the accurate scheme everywhere (their Lemma 4).
For a fixed stencil `dtheta` does not go to zero, so whether a node keeps the accurate scheme depends on the solution's Hessian: on the centred smooth example of the paper every node does, on T1 (`exp(|x|^2 / 2)`, whose Hessian is far from isotropic near the corner `(1, 1)`) some nodes do not.
`Result.info["monotone_fraction"]` reports the share of unknowns where `|F_A - F_M| > eps`, the quantity of the paper's Figure 4.3.

### The parts

`Filtered(accurate, monotone, epsilon)` needs:

- `accurate`: a scheme with `residual` and `jacobian`; it also gives the result's solution fields (node values, discrete Hessian, interpolant).
- `monotone`: a scheme with `residual`, `jacobian` and `solve_laplace`; `Newton` starts from its Laplacian.
- Both on the same unknowns in the same order: `discretize` checks that their right-hand sides agree, and construction that their `n` agree when both have one.

It supports a problem when both parts do; with `Standard9` and `Monotone17` that is a 2D box with equal sides and Dirichlet data.

## Newton's method

`Newton` solves the filtered equations as it solves any scheme's: damped Newton steps on the residual, halved until the max residual decreases, from `laplace u_0 = 2 sqrt(f)`, stopping when the max residual is at most `tol * max(1, max |f|)`.
The Jacobian is the paper's approximation (p. 438):

```text
J = diag(1 - S') J_M + diag(max(S', 0)) J_A,     S' = filter_slope((F_A - F_M) / eps),
```

the exact derivative where `S' = 1` (`J_A`) and where `S' = 0` (`J_M`), and `2 J_M` in the band, where the exact derivative is `2 J_M - J_A`.
The paper drops `-J_A` there because a negative weight on `J_A` "can lead to poorly conditioned or ill-posed linear systems".
So in the band Newton is a quasi-Newton method, and where a node in the band attains the max residual its direction need not decrease it; see "Known limits".
The paper takes the full step, with no line search; `Newton` halves it until the max residual decreases, and that is where the two part ways on T2 (Known limits).

## Source paper

- B. D. Froese, A. M. Oberman, "Convergent filtered schemes for the Monge-Ampere partial differential equation", SIAM J. Numer. Anal. 51(1) (2013) 423-444, doi 10.1137/120875065: the filter (eq. (1.3)), the convergence theory of nearly monotone schemes (Section 2), the filtered scheme for Monge-Ampere with its width and Jacobian (Section 4.1), and Tables 4.1 and 4.2 on the centred `C^2` example, T20, T2 and the cone T21.
- A. M. Oberman, "Wide stencil finite difference schemes for the elliptic Monge-Ampere equation and functions of the eigenvalues of the Hessian", Discrete Contin. Dyn. Syst. Ser. B 10 (2008) 221-238, doi 10.3934/dcdsb.2008.10.221: the angular resolution `dtheta` (eq. (3.3)).
- The parts: [`standard9.md`](standard9.md) and [`monotone17.md`](monotone17.md).

`benchmarks/sources.md`, section 5, transcribes both tables; `benchmarks/filtered.md` puts ours next to them.

## Parameters

| Part | Parameter | Default | Meaning |
|---|---|---|---|
| `Filtered` | `accurate` | (required) | the accurate scheme |
| `Filtered` | `monotone` | (required) | the monotone scheme |
| `Filtered` | `epsilon` | None | the filter width; None is `sqrt(h) + dtheta / 10` from the monotone part |
| `Newton` | `max_iter` | 50 (`FDFiltered`: 200) | most steps |
| `Newton` | `start` | None (`FDFiltered`: None, fixed) | another iteration whose final unknowns Newton starts from; on T2 neither `FDMonotone` nor `FDPoisson` as start makes it converge, on T20 the monotone start stalls and the standard root is already a filtered root (Known limits) |
| `Newton` | `tol` | 1e-10 | largest residual, relative to `max(1, max \|f\|)` |

The preset takes the parts' parameters flat: `FDFiltered(n=126, stencil=33, boundary="interpolate", jacobian_floor=1e-12, epsilon=None, spline_degree=3)` is `Newton(scheme=Filtered(accurate=Standard9(n=126, spline_degree=3), monotone=Monotone17(n=126, stencil=33, boundary="interpolate", jacobian_floor=1e-12, spline_degree=3), epsilon=None), max_iter=200)`.

## Known limits

- Only what both parts support: with `Standard9` and `Monotone17`, a 2D `Box` with equal sides and a `Dirichlet` boundary.
- `Newton` with the paper's approximate Jacobian often does not converge, and the solve ends with `converged` False (`benchmarks/filtered.md`, Finding 4; issue #56).
  With the default width and `max_iter = 200`, on the catalogue problems it supports:

  | Problem | `n = 16` | `n = 64` |
  |---|---|---|
  | T1 | converged | not converged |
  | T2 | converged | not converged |
  | T3 | converged | converged |
  | T4 | not converged | not converged |
  | T5 | not converged, 0 steps | not converged |
  | T7 | converged | not converged |
  | T20 | converged | not converged |

  On the paper's grids it ends that way on T2 at every `N` and stencil with `boundary="narrow"` (with `"interpolate"` one solve converges: 33 points, `N = 31`, error `1.01e-3` against the published `1.74e-3`), on T20 from `N = 127` on, on the cone, and on T1 at all `N` but one.
  Three situations stop it: a node of the filter band, where the paper's Jacobian is not the derivative, attains the largest residual, and the line search finds no decrease (T2, T1; residual `1e-2` to `2e-1`); the approximate Jacobian is nearly or exactly singular, so no step decreases the residual (T5 at `n = 16`, which returns the Laplace start; the cone at `n = 62`); or the largest residual sits at a node of the accurate piece, next to a jump of `f` (T4; also the cone at `n = 126`).
  The paper's own iteration takes the full Newton step, with no line search (p. 438), and on T2 it converges where `Newton` stalls: in 9 of the 15 cells of the paper's Table 4.1 (every `N` with 9 points, `N = 31` and 63 with 17 and 33), each to a root that prints as the published error, in 74 to 1141 steps; with 17 and 33 points from `N = 127` on it does not converge either.
  It also converges on T1 at `n = 62` and 126 and on T20 at `n = 62`, and diverges on T20 at `n = 126`.
  `Newton` keeps its line search, which halves the step until the max residual decreases, because the other presets rest on it; a small subclass of `Newton` gives the full step (`benchmarks/run_filtered.py`, `UndampedNewton`), and an opt-in option is the follow-up in issue #56.
  Tried on the damped `Newton` without success: the paper's Jacobian floor (`jacobian_floor=1e-8`, as in FO 2011, J. Comput. Phys., Section 7.2, which the paper cites for its regularisation), the smooth max `(MA)^delta` of FO 2011, SIAM J. Numer. Anal., Section 3.5 (`delta` from `1e-4` to `0.1`), other widths, `kappa = 1`, the exact Jacobian `2 J_M - J_A` in the band, and starts from the monotone or the standard root (on T2 neither start converges; on T20 the monotone start stalls, while the standard root is already a filtered root).
- Where the filter keeps the standard scheme, the filtered scheme has the standard scheme's non-convex roots, and Newton from `laplace u_0 = 2 sqrt(f)` can reach one: on T20 at `N = 63` it reaches exactly `FDNewton`'s root, error `1.73e-4`, against `1.34e-4` for the convex root, which is a root of the filtered scheme too (Finding 5).
  `nonconvex_fraction` is 0.07 to 0.11 there.
- The monotone part is `Monotone17`, not the paper's operator: below zero `Monotone17` continues each pair's product with the slope `kappa = sqrt(max(f, mean f, 1))` (`monotone17.md`, "Below zero"), the paper's `sum min{D, delta}` with slope 1, so `F_A - F_M`, and with it the filter's choice, differ at nodes with a negative second difference.
  And the paper regularises the max operator for Newton "as in [FO11b]" (p. 438), the JCP 2011 paper, which floors each second difference in the Jacobian at `1e-8 / (2 h^2)` (its Section 7.2): that is `FDFiltered(jacobian_floor=1e-8)`, where the default is `1e-12`; the floor changes Newton's steps, not the scheme, and does not make `Newton` converge (Finding 4).
- The default `dtheta` is the largest angle between neighbouring stencil directions, twice Oberman's angular resolution; the paper does not define it, and its T2 columns are the evidence for this choice (Finding 3).

## Reproduction

Against Froese-Oberman 2013, Tables 4.1 and 4.2, at the levels of ADR 0002, as `benchmarks/filtered.md` states them; the paper's `N` counts nodes, so ours run at `n = N - 1`, and its smooth example is T1 centred at `(1/2, 1/2)` (Finding 1):

| Problem | Filtered (Tables 4.1 and 4.2) | Monotone (Table 4.1) | Standard (Table 4.2) |
|---|---|---|---|
| centred `C^2` | exact against Table 4.2; order of magnitude against Table 4.1 (exact but at `N = 63`, where it prints `1.04e-5` for `1.06e-5`) | 17 points exact; 9 and 33 order of magnitude (exact from `N = 63`) | exact |
| T20 | mismatch: a non-convex root (Finding 5) | order of magnitude (last digit) | exact |
| T2 | with the paper's undamped Newton: exact in 9 of 15 cells of Table 4.1 (every `N` with 9 points, `N = 31` and 63 with 17 and 33) and at `N = 31` and 63 in Table 4.2, mismatch where it does not converge (17 and 33 points from `N = 127`); `FDFiltered`'s damped Newton converges in no cell (Finding 4) | exact | exact |
| T21 (cone) | mismatch | mismatch | exact |

`tests/test_reproduction.py` pins the converged values, on T2 at `N = 31` with a test-local undamped Newton; the full comparison and the findings are in [`benchmarks/filtered.md`](../../benchmarks/filtered.md).
