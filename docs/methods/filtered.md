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
print(evaluate(result)["max"], result.info["monotone_fraction"])  # 1.85e-4, 0.0021
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
The paper does not give `dtheta` a value; maesolvers takes Oberman's (2008, eq. (3.3)) definition, the largest angle between a direction and the nearest stencil direction: `pi/8` for 9 points, `atan(1/2)/2` for 17 and `atan(1/3)/2` for 33 (`angular_resolution`).
`Filtered(epsilon=None)` computes it from the monotone part's `n` and `angular_resolution`; a number fixes it.

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
| `Newton` | `start` | None (`FDFiltered`: None, fixed) | another iteration whose final unknowns Newton starts from; starts from the monotone or the standard root do not make it converge either (Known limits) |
| `Newton` | `tol` | 1e-10 | largest residual, relative to `max(1, max \|f\|)` |

The preset takes the parts' parameters flat: `FDFiltered(n=126, stencil=33, boundary="interpolate", jacobian_floor=1e-12, epsilon=None, spline_degree=3)` is `Newton(scheme=Filtered(accurate=Standard9(n=126, spline_degree=3), monotone=Monotone17(n=126, stencil=33, boundary="interpolate", jacobian_floor=1e-12, spline_degree=3), epsilon=None), max_iter=200)`.

## Known limits

- Only what both parts support: with `Standard9` and `Monotone17`, a 2D `Box` with equal sides and a `Dirichlet` boundary.
- `Newton` often does not converge.
  Where a node in the filter band attains the largest residual, the paper's Jacobian gives no direction of descent and the line search ends the solve with `converged` False and a residual of `1e-2` to `1e-1`: on T2 at every `n`, on T20 from `n = 126`, on the cone, and on T1 at every `n` (`benchmarks/filtered.md`, Finding 4).
  The last iterate is still a fair answer: on T2 it matches the paper's errors to the printed digits up to `N = 127`, and on T1 its error lies between the monotone and the standard scheme's.
  Undamped steps, the exact Jacobian `2 J_M - J_A` in the band, and starts from the monotone or the standard solution do not converge either.
- Where the filter keeps the standard scheme, the filtered scheme has the standard scheme's non-convex roots, and Newton from `laplace u_0 = 2 sqrt(f)` can reach one: on T20 its error is `1.85e-4` at `N = 63` against `1.34e-4` for the convex root, which is a root of the filtered scheme too (Finding 5).
  `nonconvex_fraction` is 0.08 to 0.11 there.
- The monotone part is `Monotone17`, not the paper's operator: below zero `Monotone17` continues each pair's product with the slope `kappa = sqrt(max(f, mean f, 1))` (`monotone17.md`, "Below zero"), the paper's `sum min{D, delta}` with slope 1, so `F_A - F_M`, and with it the filter's choice, differ at nodes with a negative second difference.

## Reproduction

Against Froese-Oberman 2013, Tables 4.1 and 4.2, at the levels of ADR 0002, as `benchmarks/filtered.md` states them; the paper's `N` counts nodes, so ours run at `n = N - 1`, and its smooth example is T1 centred at `(1/2, 1/2)` (Finding 1):

| Problem | Filtered (Tables 4.1 and 4.2) | Monotone (Table 4.1) | Standard (Table 4.2) |
|---|---|---|---|
| centred `C^2` | exact against Table 4.2; Table 4.1 prints `1.04e-5` for `1.06e-5` at `N = 63` | 17 points exact, 9 and 33 exact from `N = 63` | exact |
| T20 | mismatch: a non-convex root (Finding 5) | order of magnitude, last digit | exact |
| T2 | exact to `N = 127`, order of magnitude beyond (9 points: mismatch); not converged | exact | exact |
| T21 (cone) | mismatch | mismatch | exact |

`tests/test_reproduction.py` pins the converged values; the full comparison and the findings are in [`benchmarks/filtered.md`](../../benchmarks/filtered.md).
