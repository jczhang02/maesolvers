# Reproduction: monotone mixed scheme (Chen-Wan-Lin 2018)

Source: Y. Chen, J. W. L. Wan, J. Lin, "Monotone mixed finite difference scheme for Monge-Ampere equation", J. Sci. Comput. 76 (2018) 1839-1867, Section 7, Tables 1 to 5 (Zotero attachments 3XWEWWT6, 9WII8MMM; every number below was checked against the PDF's page images).
The roadmap and the survey call the paper "Chen-Wan-Wang 2018"; its title page names Jessey Lin.

Problems (the paper's Examples 1 to 5, with Dirichlet data from the exact solution):

- Example 1, T1 on `[-1, 1]^2`, built in `run_chen_wan_lin2018.py` as `T1-square` (the catalogue's T1 is on `(0, 1)^2`);
- Example 2, T2 on `(0, 1)^2`, the catalogue's;
- Example 3, T20c: `f = (1 - 0.1/|x|)^+`, `u = ((|x| - 0.1)^+)^2 / 2` on `(-0.5, 0.5)^2`, the catalogue's (new here; the survey has it on `[-1, 1]^2`, the PDF on `(-0.5, 0.5)^2`);
- Example 4, the cone T21, `u = |x|`, `f = pi delta_0` on `(-0.5, 0.5)^2`, built per grid with a discrete Dirac mass at the origin (Finding 3);
- Example 5, `f = 1`, `g = 0` on `(-0.5, 0.5)^2`, built as `T7-half`: T7 under `x -> 2x`, `u -> (u - 1)/4`; the paper reports `u_h(0, 0)`.

Columns: "mixed" is `FDMixed(n=N)`, `PolicyIteration` on `MixedHJB` with the defaults; "wide only" is `PolicyIteration(scheme=MixedHJB(n=N, wide_search="only"))`, the pure semi-Lagrangian scheme of Feng-Jensen 2017 that the paper compares with; Table 5 adds "everywhere", `PolicyIteration(scheme=MixedHJB(n=N, wide_search="everywhere"))`, the monotone maximum of the paper's Section 5.2 (Finding 2), and its non-monotone column is the standard 9-point scheme, `FDPoisson(n=N, max_iter=300000)`.
`n = N` cells per side, `M = N` angles, the paper's tolerance `1e-6` on the residual of the HJB equation.
Each policy's linear system is solved by sparse LU, except in the columns where the wide stencil enters at many nodes (the pure semi-Lagrangian scheme, T21, Table 5) from `N = 256` on: there BiCGSTAB with an incomplete LU solves it to a relative residual of `1e-12` (`krylov_tol`), since the LU fills in to tens of GB and does not finish in hours at `N = 512`.
The paper uses GMRES with an incomplete LU throughout; where the LU is cheap it stays, because that tolerance moves the fourth digit of a 7-point error at `N = 512` (T1: `4.70448e-6` by BiCGSTAB, an MLflow run at commit 1f96a1b, against `4.70454e-6` by LU and the published `4.705e-6`).
The errors are the paper's: `||e||_inf`, the max at the nodes, and `||e||_2 = sqrt(h^2 sum e^2)` over the nodes (Finding 1).

Receipts: `chen_wan_lin2018.csv` (Tables 1 to 4), `chen_wan_lin2018_table5.csv` and `chen_wan_lin2018_checks.csv` (the evidence for the findings), all from `run_chen_wan_lin2018.py` on a100server at commit fb57cfb (Table 5 at 63b3dc1, which added the "everywhere" column; its mixed and standard rows are those of the earlier runs at 9c24bcb and 55da491 to every digit, apart from the run times) with a clean checkout; every solve is one run in the MLflow experiment `maesolvers-mixed-<problem>` (`T1-square`, `T2`, `T20c`, `T21`, `T7-half`); the receipts' runs carry the commit tags fb57cfb and 63b3dc1, and the same experiments also hold the runs of earlier attempts (commits 386c6ea to 55da491; some KILLED), which the receipts do not use.

## Verdict

| Table | Problem | Mixed scheme | Pure semi-Lagrangian |
|---|---|---|---|
| 1 | T1 on `[-1, 1]^2` | exact: every printed digit, both norms, every `N`; 4 policy iterations, as printed | order of magnitude: within 3.2%, orders within 0.02 |
| 2 | T2 | mismatch (ADR 0002 level 5): every printed digit at `N = 32` to 128, but at 256 and 512 policy iteration from the paper's start reaches a second root of the scheme, 10 to 65 times less accurate, with orders of -1.33 and 0.15 (Finding 5) | order of magnitude: within 3% |
| 3 | T20c | order of magnitude: 6 of the 10 printed values exact, the rest within 0.7% (Finding 4) | order of magnitude: within 3% |
| 4 | T21 | mismatch for the paper's `f = pi delta`: with the mass `pi/h^2` the error is `0.039` at every `N`; with `(4 sqrt 2 - 2)/h^2`, a mass fitted to the exact solution, within 19%, orders within 0.11 (Finding 3) | not printed |
| 5 | `f = 1`, `g = 0` | the mixed scheme: order of magnitude with `wide_search="everywhere"`, converged at every `N`, `u_h(0, 0)` within `1.0e-4` and the published value at `N = 256` and 512; the default rule gives the same values where it converges, but cycles at `N = 256` | the standard 9-point scheme: order of magnitude, within `2e-5`, the published value at `N = 256` and 512, every `N` converged |

### Table 1 (1), T1 on [-1, 1]^2, mixed scheme

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.201e-3 | 1.201e-3 | 9.598e-4 | 9.598e-4 |  |  | 4 / 4 |
| 64 | 3.009e-4 | 3.009e-4 | 2.404e-4 | 2.404e-4 | 2.00 / 2.00 | 2.00 / 2.00 | 4 / 4 |
| 128 | 7.526e-5 | 7.526e-5 | 6.013e-5 | 6.013e-5 | 2.00 / 2.00 | 2.00 / 2.00 | 4 / 4 |
| 256 | 1.882e-5 | 1.882e-5 | 1.504e-5 | 1.504e-5 | 2.00 / 2.00 | 2.00 / 2.00 | 4 / 4 |
| 512 | 4.705e-6 | 4.705e-6 | 3.759e-6 | 3.759e-6 | 2.00 / 2.00 | 2.00 / 2.00 | 4 / 4 |

### Table 1 (2), T1 on [-1, 1]^2, pure semi-Lagrangian

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.868e-2 | 1.887e-2 | 1.557e-2 | 1.565e-2 |  |  | 5 / 5 |
| 64 | 1.020e-2 | 1.040e-2 | 8.364e-3 | 8.499e-3 | 0.87 / 0.86 | 0.90 / 0.88 | 5 / 5 |
| 128 | 5.263e-3 | 5.432e-3 | 4.240e-3 | 4.361e-3 | 0.95 / 0.94 | 0.98 / 0.96 | 6 / 6 |
| 256 | 2.801e-3 | 2.849e-3 | 2.259e-3 | 2.293e-3 | 0.91 / 0.93 | 0.91 / 0.93 | 5 / 6 |
| 512 | 1.600e-3 | 1.629e-3 | 1.268e-3 | 1.290e-3 | 0.81 / 0.81 | 0.83 / 0.83 | 5 / 6 |

### Table 2 (1), T2, mixed scheme

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 6.450e-5 | 6.450e-5 | 2.359e-4 | 2.359e-4 |  |  | 4 / 5 |
| 64 | 1.628e-5 | 1.628e-5 | 8.211e-5 | 8.211e-5 | 1.99 / 1.99 | 1.52 / 1.52 | 5 / 6 |
| 128 | 4.084e-6 | 4.084e-6 | 2.882e-5 | 2.882e-5 | 2.00 / 2.00 | 1.51 / 1.51 | 5 / 15 |
| 256 | 1.022e-6 | 1.027e-5 | 1.015e-5 | 2.215e-4 | 2.00 / -1.33 | 1.51 / -2.94 | 5 / 21 |
| 512 | 2.557e-7 | 9.232e-6 | 3.583e-6 | 2.324e-4 | 2.00 / 0.15 | 1.50 / -0.07 | 5 / 100 |

### Table 2 (2), T2, pure semi-Lagrangian

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.493e-3 | 1.517e-3 | 5.799e-3 | 5.842e-3 |  |  | 5 / 5 |
| 64 | 9.634e-4 | 9.882e-4 | 4.394e-3 | 4.399e-3 | 0.63 / 0.62 | 0.40 / 0.41 | 4 / 6 |
| 128 | 5.166e-4 | 5.236e-4 | 2.697e-3 | 2.698e-3 | 0.90 / 0.92 | 0.70 / 0.71 | 5 / 7 |
| 256 | 3.153e-4 | 3.194e-4 | 1.824e-3 | 1.836e-3 | 0.71 / 0.71 | 0.56 / 0.56 | 5 / 6 |
| 512 | 1.583e-4 | 1.594e-4 | 1.120e-3 | 1.120e-3 | 0.99 / 1.00 | 0.70 / 0.71 | 5 / 7 |

### Table 3 (1), T20c, mixed scheme

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.270e-4 | 1.270e-4 | 4.298e-4 | 4.298e-4 |  |  | 4 / 4 |
| 64 | 4.273e-5 | 4.271e-5 | 1.520e-4 | 1.520e-4 | 1.57 / 1.57 | 1.50 / 1.50 | 6 / 6 |
| 128 | 1.835e-5 | 1.835e-5 | 6.907e-5 | 6.906e-5 | 1.22 / 1.22 | 1.14 / 1.14 | 7 / 7 |
| 256 | 1.544e-5 | 1.544e-5 | 5.959e-5 | 5.959e-5 | 0.25 / 0.25 | 0.21 / 0.21 | 9 / 11 |
| 512 | 3.396e-6 | 3.374e-6 | 1.513e-5 | 1.504e-5 | 2.18 / 2.19 | 1.98 / 1.99 | 20 / 16 |

### Table 3 (2), T20c, pure semi-Lagrangian

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.337e-3 | 1.368e-3 | 6.604e-3 | 6.677e-3 |  |  | 5 / 6 |
| 64 | 9.084e-4 | 9.175e-4 | 3.304e-3 | 3.328e-3 | 0.56 / 0.58 | 1.00 / 1.00 | 6 / 7 |
| 128 | 6.940e-4 | 7.014e-4 | 1.901e-3 | 1.914e-3 | 0.39 / 0.39 | 0.80 / 0.80 | 7 / 6 |
| 256 | 3.815e-4 | 3.848e-4 | 9.335e-4 | 9.391e-4 | 0.86 / 0.87 | 1.03 / 1.03 | 7 / 7 |
| 512 | 1.998e-4 | 2.009e-4 | 4.563e-4 | 4.582e-4 | 0.93 / 0.94 | 1.03 / 1.04 | 9 / 9 |

### Table 4, the cone T21, mixed scheme, f = (4 sqrt 2 - 2)/h^2 at the origin

| N | L2 paper | L2 ours | Linf paper | Linf ours | order L2 paper / ours | order Linf paper / ours | policy iterations paper / ours |
|---|---|---|---|---|---|---|---|
| 32 | 1.156e-3 | 1.372e-3 | 3.868e-3 | 4.114e-3 |  |  | 9 / 11 |
| 64 | 6.484e-4 | 7.141e-4 | 2.583e-3 | 2.629e-3 | 0.83 / 0.94 | 0.58 / 0.65 | 15 / 17 |
| 128 | 3.803e-4 | 4.121e-4 | 1.848e-3 | 1.878e-3 | 0.77 / 0.79 | 0.48 / 0.49 | 17 / 19 |
| 256 | 2.159e-4 | 2.295e-4 | 1.305e-3 | 1.325e-3 | 0.82 / 0.84 | 0.50 / 0.50 | 23 / 28 |
| 512 | 1.148e-4 | 1.202e-4 | 9.203e-4 | 9.250e-4 | 0.91 / 0.93 | 0.50 / 0.52 | 27 / 35 |

## Finding 1: the grid is `h = 1/N`, `N` cells per side

The paper's Section 4 sets up "an `N x N` square grid" with the unknowns at `i, j = 1, ..., N` and the boundary at `i, j = 0` and `N + 1`, which is `N + 1` cells per side.
Its tables match `N` cells per side instead, `h = 1/N` on the unit square (ADR 0001), as the Froese-Oberman papers do: with `n = N` every printed digit of Tables 1 and 2 is ours, and with `n = N + 1` none is (rows `grid` of the checks: on T1 at `N = 32`, `9.022e-4` against the published `9.598e-4`; on T20c `1.973e-4` against `4.298e-4`).
So at the paper's even `N` the origin is a node, which Examples 4 and 5 need.
The `L2` norm is the discrete `sqrt(h^2 sum e^2)`: on `[-1, 1]^2` it exceeds the max norm (Table 1), which an RMS error cannot; the RMS error times the square root of the area misses Table 1 by 3% (`1.240e-3` against `1.201e-3` at `N = 32`).

## Finding 2: the wide stencil competes only where no 7-point stationary point lies in its region

The paper's Section 5.2 describes the optimal control at a node as the best of six regional optima, the wide-stencil region among them, at every node.
Its cost estimate says otherwise: `O(1)` per node "if the standard 7-point stencil discretization can be applied monotonically on all or most of the grid points", which holds only if the wide-stencil search is skipped there.
The tables follow the cost estimate.
`wide_search="outside"`, the default, skips the search where the stationary point of a 7-point objective, eq. (30), lies in its region (there it maximises that objective over all controls), and matches Tables 1 to 3.
`wide_search="everywhere"` searches at every node, as Section 5.2 reads (rows `search`):

- on T1 the wide stencil never wins, and on T20c the same few nodes take it under either rule, so both solutions are unchanged;
- on T2 it takes the wide stencil at 0.7% of the nodes, next to the corner `(1, 1)`, where the Hessian of the solution is near the edge of diagonal dominance: the max error is `8.82e-4`, `7.77e-4`, `5.10e-4` at `N = 32`, 64, 128 against the published `2.359e-4`, `8.211e-5`, `2.882e-5`, and the order drops from 1.5 to 0.2 and 0.6;
- on the cone it fails, with max errors of 1.5 to 3.7 that grow with `N`: near the tip the wide stencil's arms of length `sqrt(h)` see smaller second differences of the cone than the 7-point stencil's, so the wide controls give the larger value of the objective, the maximum takes them, and the discrete solution there drops by `O(1)`;
- on Table 5's problem both rules give the same `u_h(0, 0)` to 6 digits wherever the default converges, so Table 5 does not show which rule the paper used (the "everywhere" column of `chen_wan_lin2018_table5.csv`); Tables 2 and 4 do.

The skip makes the scheme switch between two stencils at a node as `u` moves a stationary point across the edge of its region, so it is no longer the maximum of monotone operators everywhere that the paper's Section 6 proves convergent; Finding 5 shows the consequences, a second root and policy iteration that cycles.

## Finding 3: the Dirac mass of the cone

The paper writes `f = pi delta(0, 0)` and does not say how it discretises it.
With `f = pi / h^2` at the node at the origin, the error is `0.0389` at every `N` (rows `mass`): at the tip the 7-point stencil of the exact cone gives `u_xx = u_yy = 2/h` and `u_xy = (sqrt 2 - 2)/h` (stencil 1), so `det = (4 sqrt 2 - 2)/h^2 = 3.657/h^2`, and the scheme's discrete Monge-Ampere measure of a cone of slope `s` is `3.657 s^2`, not `pi s^2`; the discrete solution is a flatter cone, and its tip is `0.039` high.
Froese-Oberman's `4/h^2` (`monotone17.md`, Finding 4) gives `0.0246` at every `N`, the tip too low.
With `f = (4 sqrt 2 - 2)/h^2`, the mass at which the exact cone solves the 7-point equation at its tip, so a mass fitted to the exact solution, the tip error vanishes and the error, at the nodes around the tip where the wide stencil is used, falls like `h^0.5` in the max norm and `h^0.8` in `L2`, as the paper's does; `run_chen_wan_lin2018.py` uses this mass for Table 4.
The published errors lie within 19% of ours, the orders within 0.11, and the policy iterations grow alike (9, 15, 17, 23, 27 printed; 11, 17, 19, 28, 35 ours): this is likely, not certainly, the paper's discretisation.
Since the mass is fitted, the level for the problem the paper states, `f = pi delta`, is a mismatch; the fitted mass is the evidence for this finding, not a reproduction.

## Finding 4: small differences on T20c

On T20c six of the ten printed values are ours to every digit; the max error at `N = 128` is one unit off in the fourth digit (`6.906e-5` against `6.907e-5`), the `L2` error at `N = 64` two units (`4.271e-5` against `4.273e-5`), and both norms at `N = 512` differ by 0.6%.
T20c is the one example of Tables 1 to 3 where the wide stencil enters (at up to 0.7% of the nodes, next to the circle `|x| = 0.1`), so the difference lies in a detail of the wide stencil the paper leaves open: the placement of the `M` angles (`theta = -pi/4 + k pi/(2M)` here), or the bilinear interpolation at the ends of an arm.
The pure semi-Lagrangian column, which uses the wide stencil everywhere, differs from Tables 1 to 3 by up to 3.2% (T1, `L2` at `N = 128`), for the same reason: on T1 at `N = 32`, `M = N - 1`, `N + 1`, `N / 2` and `2N` angles give max errors of `1.89e-2`, `1.85e-2`, `1.65e-2` and `1.560e-2` against the published `1.557e-2` and our `1.565e-2` with `M = N` (rows `angles`).

## Finding 5: on T2 the scheme has a second root

From the paper's start, `laplace u = 2 sqrt(f)` (its eq. (50)), policy iteration on T2 reaches the published errors at `N = 32` to 128, but at `N = 256` and 512 it ends on another discrete solution, with the wide stencil at the nodes next to the top side near the corner (0.06% of the unknowns, `wide_fraction` in the CSV), and max errors `2.215e-4` and `2.324e-4` against the published `1.015e-5` and `3.583e-6`.
From the exact solution at the nodes it reaches, in 3 policy iterations, the root with the 7-point stencil at every node, whose errors are the published ones to every printed digit at every `N` (rows `start`): `1.022e-6` and `1.015e-5` at `N = 256`, `2.557e-7` and `3.583e-6` at `N = 512`.
A maximum of monotone operators has one root; the skip of Finding 2 breaks that, and which root policy iteration reaches depends on its path, here on how the early iterates near the singular corner fall on either side of the edge of a 7-point region.
The paper's 5 policy iterations from `N = 64` on suggest its path never left the 7-point stencil.
Ours from the same start already differs at `N = 128`, with 15 policy iterations against the printed 5 and the errors still exact, the first sign of the split; it takes 21 at `N = 256` and does not converge in 100 at 512, where the policy cycles with a residual near `0.09`.
Level for Table 2: a mismatch (ADR 0002 level 5), since the paper's method from the paper's start misses `N = 256` and 512 by 10 to 65 times.
The agreement at `N = 32` to 128, and the exact 7-point root from the exact solution, which is not the paper's start, are the evidence for this finding, not a reproduction.

## Table 5

| N | mixed, paper | everywhere, ours | policy iterations | default rule, ours | policy iterations | standard, paper | standard, ours (`-u_min` of `FDPoisson`) |
|---|---|---|---|---|---|---|---|
| 32 | -0.18380 | -0.18370 | 8 | -0.18370 | 8 | 0.18063 | 0.18065 |
| 64 | -0.18444 | -0.18447 | 10 | -0.18447 | 11 | 0.18312 | 0.18313 |
| 128 | -0.18461 | -0.18463 | 14 | -0.18463 | 15 | 0.18436 | 0.18437 |
| 256 | -0.18485 | -0.18485 | 18 | | 100, not converged | 0.18499 | 0.18499 |
| 512 | -0.18507 | -0.18507 | 26 | -0.18507 | 24 | 0.18530 | 0.18530 |

Table 5 prints `u_h(0, 0)`, not an error, so ADR 0002's level 2 is judged on the ratio ours / published of `u_h(0, 0)` at every printed `N`, and its clause on the convergence orders has nothing to compare.
With `wide_search="everywhere"` policy iteration converges at every `N` in 8 to 26 policy iterations, to residuals below the tolerance `1e-6`, and `u_h(0, 0)` is the published value at `N = 256` and 512 and within `1.0e-4` at the coarser grids, a ratio within `6e-4` of 1: order of magnitude, the level of the column, although the agreement is much closer than the level asks.
At every `N` the wide stencil enters at 7% to 17% of the nodes, next to the boundary and its corners, where the solution is not smooth, so Finding 4's open details of the wide stencil apply.
The default rule gives the same `u_h(0, 0)` to 6 digits at `N = 32`, 64, 128 and 512, so Table 5 does not show which rule the paper used (Finding 2).
At `N = 256` it cycles between policies with a residual near `0.07` and stops after 100 steps unconverged, the default rule's cycling of Finding 5; its last iterate prints the published value, but it is not a root of the scheme, so the cell is left empty, and with the default rule alone the column would be a mismatch at `N = 256`.
The non-monotone column is the standard 9-point scheme, whose equation is even in `u` when `g = 0`: its concave solution is minus its convex one, which `FDPoisson` computes, so the column prints `-u_h(0, 0)` of `FDPoisson`; it matches within `2e-5`, and to every digit at `N = 256` and 512.
The Poisson iteration takes 952, 3338, 11 593, 39 438 and 130 107 steps until successive iterates agree to `1e-12`, so `run_chen_wan_lin2018.py` gives it 300 000 (its default of 10 000 stopped it unconverged from `N = 128` on in an earlier run, with the same printed values).
`T7-half` is T7 scaled, so Benamou-Froese-Oberman's `u_min = 0.2682` of T7 at 61 nodes per side gives `(0.2682 - 1)/4 = -0.1830` here, consistent with both columns.

## Conventions that differ

- The paper's `N` is our `n` (Finding 1); the catalogue's T1 is on `(0, 1)^2`, the paper's on `[-1, 1]^2`.
- The paper solves each policy's system by GMRES with an incomplete LU; ours is the sparse LU, and BiCGSTAB with an incomplete LU where the wide stencil enters at many nodes from `N = 256` on.
- The paper's iteration counts count the policies evaluated, as `Result.info["iterations"]` does.
- `evaluate` reports the max and the RMS error at the nodes (ADR 0007); the paper's `L2` norm is computed in `run_chen_wan_lin2018.py`.
