# Standard 9-point scheme with Shortley-Weller boundaries: sanity checks

Method: `Standard9` on a curved domain, under the presets `FDNewton` and `FDPoisson` (issue #1, `docs/methods/standard9.md`, section "Curved boundaries").
No paper prints a table for the standard 9-point Monge-Ampere scheme with Shortley-Weller boundaries; Shortley-Weller 1938 treats Laplace's equation.
So none of the five levels of ADR 0002 applies: this file records sanity checks and observed orders, not a reproduction, and `tests/test_reproduction.py` pins nothing for it.

Raw data: `shortley_weller.csv` (one row per solve) and `shortley_weller_trace.csv` (Newton's residual after each step, Findings 2 and 3), both written by `run_shortley_weller.py` on a100server at commit b303900 (`just remote-bg shortley-weller env OMP_NUM_THREADS=16 python benchmarks/run_shortley_weller.py`), which also logs every solve to MLflow under `maesolvers-<solver>-<problem>`: `maesolvers-fdnewton-T8`, `maesolvers-fdpoisson-T8`, `maesolvers-concavestartnewton-T8`, `maesolvers-fdnewton-t1_exact_on_disk` and the like, with the per-step residual of every Newton run.
`n` is the number of cells per side of the grid over the domain's bounding box (ADR 0001), so `hx` and `hy` (both in the CSV) differ when the box is not a square: `hx = hy = 2/n` on T8's unit disk, `hx = hy = 1/n` on the inscribed disk and on T1, `hx = 1/n` and `hy = 0.6/n` on the ellipse, and `hx = (1 + sqrt 7)/(4n) = 0.911/n` and `hy = 1/n` on T6.
Error: max over the nodes with unknowns and the boundary points of `|u_h - u|`; the boundary points carry the exact data, so the max is reached at the nodes.
Defaults throughout: Newton `tol = 1e-10` relative to `max(1, max f)`, Poisson iteration `tol = 1e-12` on the update.

## T8: which branch each iteration selects

T8 is the unit disk with `f = 1` and `g = 0`; `h = 2/n`.
Its convex solution `(|x|^2 - 1)/2` and the concave `(1 - |x|^2)/2` both solve the equation, and the Shortley-Weller stencils are exact on quadratics, so both are exact roots of the discrete equation.
A run is "convex" or "concave" when its node values are within `1e-8` of that root.

| n | unknowns | boundary points | smallest step `t` | `FDNewton` | steps | `FDPoisson` | steps | Newton from `laplace u = -2` |
|---|---|---|---|---|---|---|---|---|
| 16 | 193 | 144 | 0.135 | convex, error 5.0e-16 | 1 | convex, error 9.1e-13 | 37 | concave |
| 32 | 793 | 296 | 0.136 | convex, 9.4e-16 | 1 | convex, 9.1e-13 | 37 | concave |
| 64 | 3205 | 608 | 0.037 | convex, 8.3e-15 | 1 | convex, 9.2e-13 | 37 | concave |
| 128 | 12849 | 1224 | 0.018 | convex, 3.0e-14 | 1 | convex, 9.7e-13 | 37 | concave |
| 256 | 51429 | 2464 | 0.0083 | convex, 1.0e-13 | 1 | convex, 1.1e-12 | 37 | concave |
| 512 | 205857 | 4936 | 0.0042 | convex, 3.3e-16 | 1 | convex, 1.7e-12 | 37 | concave |

Every run converged; the convex runs have `nonconvex_fraction` 0 and smallest Hessian eigenvalue 1 to the digits printed, the concave ones `nonconvex_fraction` 1 and eigenvalue -1.

- `FDPoisson` selects the convex branch at every `n`.
  It cannot do otherwise: every iterate solves `laplace u = r` with `r >= 0` and `g = 0`, and the Shortley-Weller Laplacian satisfies the discrete maximum principle, so every iterate is `<= 0`, while the concave root is positive inside.
- `FDNewton` selects the convex branch at every `n`, but T8 does not test its selection: its start `laplace u = 2 sqrt(f) = 2` is the convex root itself, so it stops after one step.
  From the mirrored start `laplace u = -2` (`ConcaveStartNewton` in the script) Newton stops on the concave root after one step; the start picks the branch, as on the box (`standard_fd.md`, Finding 2).
- Issue #1 expected "a zero initial guess tends to land on the concave branch".
  Newton from `u = 0` is undefined here: the Jacobian of `u_xx u_yy - u_xy^2` at `u = 0` is the zero matrix.
  We did not run it.

## Convergence order with T1's exact solution

`u = exp(|x|^2/2)`, `f = (1 + |x|^2) exp(|x|^2)` and the Dirichlet data from `u`, on two curved domains inside T1's square `(0, 1)^2`: the inscribed disk `Disk((0.5, 0.5), 0.5)` (problem `t1_exact_on_disk`) and `Ellipse((0.5, 0.5), (0.5, 0.3))` (`t1_exact_on_ellipse`), both defined in the script, not in the catalogue.
On the disk `h = 1/n`, the spacing of T1 on the box at the same `n`.
T6 (catalogue, Nystrom-Vestberg 2023, Example 4.6) adds a domain with corners and a straight cut.
`FDNewton` and `FDPoisson` reach the same discrete solution on all three (same errors to four digits), so one error column serves both.

| n | disk: max error | order | ellipse: max error | order | T6: max error | order | T1 on the box: max error |
|---|---|---|---|---|---|---|---|
| 16 | 2.230e-4 | | 7.751e-5 | | 3.866e-4 | | 2.664e-4 |
| 32 | 5.533e-5 | 2.01 | 1.934e-5 | 2.00 | 1.044e-4 | 1.89 | 6.696e-5 |
| 64 | 1.398e-5 | 1.98 | 4.872e-6 | 1.99 | 2.695e-5 | 1.95 | 1.678e-5 |
| 128 | 3.491e-6 | 2.00 | 1.218e-6 | 2.00 | 6.842e-6 | 1.98 | 4.197e-6 |
| 256 | 8.730e-7 | 2.00 | 3.047e-7 | 2.00 | 1.724e-6 | 1.99 | 1.049e-6 |
| 512 | 2.182e-7 | 2.00 | 7.617e-8 | 2.00 | 4.325e-7 | 1.99 | 2.623e-7 |

The T1 column is T1 itself on the box at the same `n` (rows `T1` of the CSV, `FDPoisson` and `FDNewton` agree to four digits), for scale; the disk's error is about 0.83 times the box's at equal `h`.
The order is 2 on all three domains: the first-order consistency of the cut stencils does not show in the error at the nodes, as for Shortley-Weller on Laplace's equation.
All discrete solutions are convex at every node (`nonconvex_fraction` 0, smallest Hessian eigenvalue 1.02 on the disk, 1.05 on the ellipse, 1.40 on T6).
Steps: `FDNewton` 3 to 6 where it converged (Finding 3), `FDPoisson` 40 to 50, at every `n`.

The interpolant's error on the shared grid (`max_err_grid` in the CSV) falls at about the same rate but is 7 to 8 times the node error on the disk, 11 to 18 times on the ellipse (whose cells are stretched, `hy = 0.6 hx`) and 7 to 10 times on T6, against about 1 on the box with the cubic spline; it is a secondary value (ADR 0007).
The factor is the Clough-Tocher interpolant's own error inside the hull of the nodes, not the treatment past the hull: on T8, whose node values are exact, `max_err_grid` is still `0.1 h^2` (1.7e-3 at `n = 16`), and on T6 at `n = 16, 32, 64` the largest grid error lies inside the hull, near the arc.
The interpolant's gradients are global estimates (scipy), accurate to `O(h)` only, so it is second order but with a large constant; it is the first candidate for a better one.
The first version of this receipt extrapolated past the hull with the value on the nearest hull edge only; at T6's two corners off the grid, where the domain reaches past the hull in a wedge of width `O(h)`, that is first order, and it gave `max_err_grid` 1.5e-2 and 2.1e-3 on T6 at `n = 16` and 32 (39 and 20 times the node error).
The first-order extrapolation with the estimated gradient (`docs/methods/standard9.md`, "Curved boundaries") brings these to 3.6e-3 and 9.9e-4 and leaves every other value of the CSV unchanged; `test_the_interpolant_reaches_corners_off_the_grid_at_second_order` checks it at points within `3h` of both corners.

## Finding 1: nodes on the boundary up to rounding

Nodes that lie on the boundary in exact arithmetic, such as `(15/17, 8/17)` on the unit circle at `n = 34`, can test as inside the open domain by rounding, with a crossing at a step fraction near `1e-15`.
In a first version their weights were of order `1e15 / h^2`, and their discrete Hessian was noise: `FDNewton` on T8 reported a non-convex fraction of 0.4 percent at `n = 34` while its values matched the exact solution to `1e-15`.
A node closer than `1e-6` steps to the boundary now joins the boundary and takes the data at itself, which moves the problem by at most `1e-6 h |grad u|`; the smallest step fraction in this file is `5.5e-5` (T6, `n = 256`).
A node outside the closed domain by rounding, such as `(0.6, 0.8)` on the unit circle at `n = 10`, came back as a crossing at a step fraction near 1 from each of its neighbours in the open domain: repeated boundary points, which Qhull drops as coplanar, leaving NaN in the Clough-Tocher gradients at their rows.
A crossing closer than `1e-6` steps to an outside neighbour now makes that neighbour a boundary node, so it appears once.
This removes exact duplicates only.
An unknown that stays above the `1e-6` snap but lies within about `1e-5` steps of the boundary still yields near-coincident points: on T6's domain at `n = 70` and `100` (not at the receipt's powers of two), two of its crossings fall within `1e-5` steps of each other or of it, Qhull drops one as coplanar, and its Clough-Tocher gradient is NaN.
This is harmless: Qhull leaves the point out of the triangulation, so it is not a vertex of the hull, and `_scattered` reads the gradients only at hull vertices; the interpolant stays finite and matches the values at every point to `1e-14`, the dropped ones included.
It predates round 2 of the review, and the receipt is unaffected.
In this receipt the ellipse's top vertex `(0.5, 0.8)` is such a node at every `n`; it had appeared three times, so the ellipse rows lose 2 `boundary_points`.
Their `rms_err_nodes` rises by up to 0.3 percent (`n = 16`), because the two copies with zero error left the mean; iterations are unchanged, and the other values (max errors, residuals, `eigmin`) change only at the level of rounding.
Every other row of both CSVs is bit-identical apart from `time_s`; the ellipse rows of the trace differ only in `max_err_nodes`, by 7e-16.
`test_a_node_on_the_circle_joins_the_boundary`, `test_a_node_outside_the_circle_by_rounding_joins_the_boundary_once` and `test_boundary_points_are_distinct_and_the_interpolant_is_finite` pin this.

## Finding 2: equations scaled by their smallest step

The weights of a cut stencil grow like `1 / t`, and with them the rounding error of its equation.
Without scaling, Newton stalls short of its tolerance on the ellipse and on T6 at `n = 128` (smallest steps 0.018 and 0.00094), with errors already on the order-2 line: no step length decreases the residual any further, so it stops unconverged.
`shortley_weller_trace.csv` holds the two runs (`row_scaled` False; `UnscaledStandard9` in the script, `Standard9` with `t = 1`), logged as `n=128-unscaled-trace` in `maesolvers-newton-t1_exact_on_ellipse` and `maesolvers-newton-T6`:

| Problem | n | residual after each step, unscaled | tolerance `1e-10 max f` | converged |
|---|---|---|---|---|
| `t1_exact_on_ellipse` | 128 | 6.27e-2, 1.17e-4, 1.53e-9 | 8.47e-10 | no |
| T6 | 128 | 7.70, 0.167, 9.42e-5, 1.52e-8 | 5.39e-9 | no |

`Standard9.residual` now multiplies each equation by its smallest step fraction `t` (1 at a full stencil and on every box), which changes neither the roots nor the Newton direction; the tolerance at the cut nodes is then effectively `1 / t` looser, and `residual_max` in the CSV is this scaled residual.
With it `FDNewton` converges on both at `n = 128`, in 3 and 4 steps (`shortley_weller.csv`); `test_newton_meets_its_tolerance_next_to_short_steps` pins the ellipse.

## Finding 3: `FDNewton` stops unconverged at `n = 512` on the disk and the ellipse

At `n = 512` `FDNewton` reports `converged = False` on `t1_exact_on_disk` (5 steps, residual 1.4e-9) and `t1_exact_on_ellipse` (7 steps, 2.1e-9), while its error stays on the order-2 line and equals the `FDPoisson` error to four digits.
Its tolerance there is `1e-10 max f`, about 1.1e-9 on the disk.
The cause is the rounding floor of the nine-point stencil at this `n`, not the boundary treatment.
`shortley_weller_trace.csv` holds Newton's residual after each step (`row_scaled` True), logged as `n=361-trace` and `n=512-trace` in `maesolvers-fdnewton-T1` and `maesolvers-fdnewton-t1_exact_on_disk`:

| Problem | n | residual after each step | tolerance `1e-10 max f` | converged |
|---|---|---|---|---|
| T1 on the box | 361 | 6.05, 0.164, 2.52e-4, 2.01e-9 | 2.18e-9 | yes |
| T1 on the box | 512 | 7.03, 0.304, 7.15e-4, 4.32e-9, 2.00e-9 | 2.19e-9 | yes, by 10 percent |
| `t1_exact_on_disk` | 512 | 8.66e-2, 1.62e-4, 1.62e-9, 1.51e-9, 1.38e-9 | 1.05e-9 | no |

The floors are near 2e-9 on the box and 1.4e-9 on the disk, but the disk's tolerance is half the box's, because `max f` at the nodes of the inscribed disk (10.5) is half its value on the square (21.9).
The tolerance scales with `max f`, the rounding floor with the solution and `1 / h^2`; from about `n = 512` the default `tol = 1e-10` is out of reach for this solution on either domain.
Not tuned away: the receipt keeps the default `tol`, and `FDPoisson`, whose stopping test is on the update, converges at every `n`.
