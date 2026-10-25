# Semi-Lagrangian scheme: published versus ours

`FDSemiLagrangian` (`Howard` on `SemiLagrangian`) against Feng and Jensen, SIAM J. Numer. Anal. 55 (2017), Section 8, read from the journal PDF (`sources.md`, section 7).
**Levels (ADR 0002): figure only on T39 in $L^\inf ty$ and $H^1$**, against Figure 6, a table of the best relative errors in $L^2$, $L^\inf ty$ and $H^1$ over the stencil factor $m$ at each level under a figure caption, and the rates of p. 711; **mismatch on T39 in $L^2$**, where ours is $3.8$ to $6.0$ times the paper's at the same best $m$ and nearly the same rate (Finding 4); **mismatch on T40** in all three norms, which our structured grid solves to the rounding level (Finding 1).
The paper's nonsmooth behaviour is compared on T40 turned off the grid, a test-local problem without a T id (Finding 2).

Receipt: `feng_jensen2017.csv`, written by `run_feng_jensen2017.py --levels 8` on a100server at commit 4010174, clean checkout (tmux `maesolvers-feng-jensen-fe`), $6$ worker processes with one thread each: T39, T40 and T40 turned on $n = 10 \cdot 2^l$, $l = 0, \ldots, 7$, with $m = 2, 4, \ldots, 64$, $141$ solves, each with its $L^2$ and $H^1$ errors (`rel_l2_err`, `rel_h1_err`).
It replaces the receipt of the run at cf16a7c (tmux `maesolvers-feng-jensen3`), which had no $L^2$ and $H^1$ columns: in every one of the $141$ rows every column the two share but `time_s` is the same string, so the scheme commits between them (among them the on-hull check, b108f80, and the floor on the distance of an unknown to the hull, 2bf55af) change no solve.
The runner leaves out three solves at $n = 1280$ with small $m$ (`EXCLUDED`, with the reason in its docstring), where Howard's linear solves are slowest (Conventions, Linear solver; #72): T40 at $m = 2$, T40 turned at $m = 2$ and $4$.
None of them can be the best $m$ at its level, so the tables below do not depend on them: T40 is at the rounding level for every $m$, and the run at dde1578 below solved T40 at $m = 2$ to $8.0 \times 10^{-13}$ in $2$ steps (MLflow run 99394efcf4db4f45b2a37c24e874cb94, $2.5$ hours) and T40 turned at $m = 4$ to $3.954 \times 10^{-2}$ in $17$ steps (run 49e68b077354419b8f261c7f03b66ca0, $3.6$ hours), against $5.76 \times 10^{-3}$ at $m = 64$; T40 turned at $m = 2$ was never solved (more than $13$ hours, stopped).
The runs of cf16a7c for these three rows were stopped (4010174 never started them), and MLflow marks them `KILLED`, as it does the six runs of the first runner at 119b73f in `-T39` and the unfinished $m = 2$ run of dde1578 in `-T40_turned`.
A first run of the same scheme code at dde1578 (tmux `maesolvers-feng-jensen2`) gave the same errors, to the four printed digits, in all $141$ shared solves; its rows existed only in memory when it was stopped, which is why the runner now writes each row as it finishes.
Every solve in the receipt converged (Howard's step below $10^{-6}$).
Every solve is an MLflow run named `n=<n> m=<m>` in `maesolvers-feng_jensen2017-T39`, `-T40` and `-T40_turned`, status `FINISHED`; the tag `mlflow.source.git.commit` tells the three runs apart, and those of 4010174 log `rel_l2_err` and `rel_h1_err` besides `evaluate`'s metrics.
The receipt's `nonconvex_fraction` column measures the least-squares Hessian fit, not the solution (Finding 3).
Diagnostics behind Finding 3 and the timing of `Newton` (Howard's algorithm): `feng_jensen2017_checks.csv`, written by `run_feng_jensen2017.py --checks` on a100server at commit 87566d9, clean checkout (`just remote`, $4$ threads), each row also an MLflow run named `check=<kind> n=<n> m=<m>` in `maesolvers-feng_jensen2017-<problem>` (T20's in `-T20`).
`python benchmarks/run_feng_jensen2017.py --summary benchmarks/feng_jensen2017.csv` prints the tables below from the CSV.

## The comparison

The paper refines a quasi-uniform unstructured triangulation of the domain, from $91$ nodes to $1{,}214{,}465$ in $7$ uniform refinements, and for each level prints the smallest relative error over $m$ in $\lbrace 2, 4, 8, 16, 32, 64 \rbrace$ with the minimising $m$.
We refine the grid $n = 10, 20, \ldots, 1280$ cells per side of $[-1, 1]^2$, $h = 2/n$, which holds $1.14$ to $1.43$ times as many nodes as the paper's level of the same index ($130$ to $1.38$ million), over the same $m$, and pair the levels by their index.
The $L^\inf ty$ errors are at the nodes, relative to $\max \lvert u \rvert$ over the domain: $4$ for T39 (at the corner $(1, 1){}$), $1$ for T40.
The $L^2$ and $H^1$ errors are those of the piecewise linear solution over the triangulated domain, relative to the norms of $u$ there (Conventions, Error measure); each norm has its own best $m$, as in the paper.

## T39, the quartic problem: figure only in $L^\inf ty$ and $H^1$, mismatch in $L^2$

$L^\inf ty$, the paper's middle column:

| $n$ | nodes | ours | $m$ | steps | paper's DoFs | paper | paper's $m$ |
|---|---|---|---|---|---|---|---|
| $10$ | $130$ | $5.72 \times 10^{-2}$ | $2$ | $4$ | $91$ | $9.26 \times 10^{-2}$ | $2$ |
| $20$ | $459$ | $2.70 \times 10^{-2}$ | $2$ | $4$ | $329$ | $3.85 \times 10^{-2}$ | $2$ |
| $40$ | $1{,}618$ | $1.57 \times 10^{-2}$ | $4$ | $5$ | $1{,}249$ | $2.15 \times 10^{-2}$ | $2$ |
| $80$ | $5{,}952$ | $7.61 \times 10^{-3}$ | $4$ | $5$ | $4{,}865$ | $9.59 \times 10^{-3}$ | $4$ |
| $160$ | $22{,}666$ | $4.09 \times 10^{-3}$ | $8$ | $5$ | $19{,}201$ | $5.63 \times 10^{-3}$ | $4$ |
| $320$ | $88{,}329$ | $2.12 \times 10^{-3}$ | $8$ | $5$ | $76{,}289$ | $2.44 \times 10^{-3}$ | $8$ |
| $640$ | $348{,}562$ | $1.15 \times 10^{-3}$ | $16$ | $5$ | $304{,}129$ | $1.51 \times 10^{-3}$ | $8$ |
| $1280$ | $1{,}384{,}524$ | $6.71 \times 10^{-4}$ | $16$ | $5$ | $1{,}214{,}465$ | $6.59 \times 10^{-4}$ | $16$ |

Rates: ours $h^{0.92}$ end to end, the paper's measure ($h^{0.92}$ by least squares too), against $h^{1.02}$; in the best $k$, ours $k^{1.60}$ against $k^{1.78}$.

Verdict: figure only, and close.
At every level ours lies between $0.62$ and $1.02$ times the paper's error, the best $m$ follows the paper's to within a factor $2$ (equal at $5$ of $8$ levels), and the end-to-end rate is $0.10$ below the paper's.
By the numbers this would meet the order-of-magnitude level of ADR 0002 (ratio in $[1/2, 2]$, orders within $0.2$), but that level compares at the paper's printed grids, and our grids are not the paper's: they are paired with its levels by index, with $14$% to $43$% more nodes (Conventions).
Each column of the paper's Figure 7 is ours too: at a fixed $m$ the error stops falling once $h / k$ is no longer small (at $m = 2$ it levels off at $1.66 \times 10^{-2}$, the paper's at about $1.3 \times 10^{-2}$), and a larger $m$ wins at a finer grid.

$L^2$ and $H^1$, the paper's first and last columns, each with its own best $m$:

| $n$ | $L^2$ ours | $m$ | paper | paper's $m$ | $H^1$ ours | $m$ | paper | paper's $m$ |
|---|---|---|---|---|---|---|---|---|
| $10$ | $2.62 \times 10^{-1}$ | $2$ | $6.92 \times 10^{-2}$ | $2$ | $2.54 \times 10^{-1}$ | $2$ | $1.86 \times 10^{-1}$ | $2$ |
| $20$ | $1.25 \times 10^{-1}$ | $2$ | $2.99 \times 10^{-2}$ | $2$ | $1.29 \times 10^{-1}$ | $2$ | $9.08 \times 10^{-2}$ | $2$ |
| $40$ | $6.51 \times 10^{-2}$ | $4$ | $1.69 \times 10^{-2}$ | $4$ | $6.69 \times 10^{-2}$ | $4$ | $4.93 \times 10^{-2}$ | $4$ |
| $80$ | $3.33 \times 10^{-2}$ | $4$ | $7.12 \times 10^{-3}$ | $4$ | $3.44 \times 10^{-2}$ | $4$ | $2.28 \times 10^{-2}$ | $4$ |
| $160$ | $1.65 \times 10^{-2}$ | $8$ | $4.18 \times 10^{-3}$ | $8$ | $1.68 \times 10^{-2}$ | $8$ | $1.28 \times 10^{-2}$ | $8$ |
| $320$ | $9.13 \times 10^{-3}$ | $8$ | $1.78 \times 10^{-3}$ | $8$ | $8.88 \times 10^{-3}$ | $8$ | $5.82 \times 10^{-3}$ | $8$ |
| $640$ | $4.71 \times 10^{-3}$ | $16$ | $1.06 \times 10^{-3}$ | $16$ | $4.43 \times 10^{-3}$ | $16$ | $3.36 \times 10^{-3}$ | $8$ |
| $1280$ | $2.91 \times 10^{-3}$ | $16$ | $4.82 \times 10^{-4}$ | $16$ | $2.54 \times 10^{-3}$ | $16$ | $1.59 \times 10^{-3}$ | $16$ |

Rates: in $L^2$ ours $h^{0.93}$ end to end ($h^{0.94}$ by least squares) against $h^{1.02}$, and $k^{1.62}$ against $k^{1.79}$; in $H^1$ ours $h^{0.95}$ ($h^{0.96}$) against $h^{0.98}$, and $k^{1.66}$ against $k^{1.72}$.

Verdict in $H^1$: figure only, and close, as in $L^\inf ty$.
At every level ours is $1.31$ to $1.60$ times the paper's error, the best $m$ is the paper's at $7$ of $8$ levels ($16$ against $8$ at $n = 640$), and the rates differ by $0.03$ in $h$ and $0.06$ in $k$.

Verdict in $L^2$: mismatch.
Ours is $3.8$ to $6.0$ times the paper's error at every level, outside the factor $2$ of ADR 0002's order-of-magnitude level, while its best $m$ is the paper's at all $8$ levels and its rate is within $0.09$ of the paper's; Finding 4.

## T40, the nonsmooth problem: mismatch

Every solve, at every $n$ and $m$, ends within $10^{-12}$ of the exact solution at the nodes (largest relative error $5.4 \times 10^{-13}$, at $n = 1280$, $m = 4$; the run at dde1578 logged $8.0 \times 10^{-13}$ at $m = 2$), against $1.03 \times 10^{-1}$ to $4.51 \times 10^{-3}$ in the paper.
So do the $L^2$ and $H^1$ errors over the whole triangulated domain: at most $4.5 \times 10^{-13}$ ($n = 1280$, $m = 4$) and $2.2 \times 10^{-12}$ ($n = 640$, $m = 2$) over all solves, the best at each level $1.4 \times 10^{-14}$ and $9.0 \times 10^{-14}$ or less, against $4.50 \times 10^{-2}$ to $3.12 \times 10^{-4}$ and $2.02 \times 10^{-1}$ to $4.25 \times 10^{-2}$ in the paper.
That is a mismatch in the direction of a better result, and Finding 1 explains it: it is a property of the structured grid, not of the scheme.

## Finding 1: on a grid whose columns follow the kink, the interpolant of T40 solves the scheme

T40's kink is the line $x_1 = 0$.
Away from the boundary our triangulation is the lattice's, whose vertical lines through nodes are unions of triangle edges, and $x_1 = 0$ is either such a line (even $n$) or runs through the middle of a column of cells (odd $n$).
In both cases the piecewise linear interpolant $I u$ of $u = \lvert x_1 \rvert$ is there a convex function of $x_1$ alone, so along the stencil direction $(0, 1)$ its second difference is $0$, and along every other direction it is at least $0$.
The frame of $(0, 1)$ and $(1, 0)$ then gives $-\min (d_1, d_2) = 0$, every other frame gives at most $0$, and $I u$ solves the scheme, whose solution is unique.
The cells cut by the circle, whose triangles join lattice nodes to boundary points, are not covered by this argument; the receipt shows that they do not change the outcome: the error at the nodes is at the rounding level at every $n$ and $m$, and so are the $L^2$ and $H^1$ errors, so the piecewise linear solution is $u$ itself on the triangulated domain, which needs every triangle on one side of $x_1 = 0$.
`tests/test_semilagrangian.py` checks this at $n = 15$ and $16$.
The paper's mesh is unstructured, and the authors note that the kink "is not aligned with the computational mesh"; there $I u$ is not convex along $(0, 1)$ near the kink, and the scheme's error is $O(10^{-2})$.

## Finding 2: turned off the grid, the nonsmooth problem behaves like the paper's

T40 turned is T40 with the kink along the line through $0$ normal to $\nu = (\cos 0.3, \sin 0.3)$: $u = \lvert x \cdot \nu \rvert$, $f = 0$, the Dirichlet data from $u$, $\max \lvert u \rvert = \cos 0.3 + \sin 0.3$ at the corner $(1, 1)$.
No grid line and no stencil direction is parallel to the kink ($0.3 + \pi/2$ is not a multiple of $\pi/64$).

| $n$ | nodes | ours | $m$ | steps | paper (T40) | paper's $m$ |
|---|---|---|---|---|---|---|
| $10$ | $130$ | $5.25 \times 10^{-2}$ | $8$ | $4$ | $1.03 \times 10^{-1}$ | $4$ |
| $20$ | $459$ | $1.03 \times 10^{-2}$ | $16$ | $4$ | $5.69 \times 10^{-2}$ | $4$ |
| $40$ | $1{,}618$ | $9.92 \times 10^{-3}$ | $32$ | $4$ | $3.08 \times 10^{-2}$ | $8$ |
| $80$ | $5{,}952$ | $6.93 \times 10^{-3}$ | $64$ | $5$ | $2.03 \times 10^{-2}$ | $16$ |
| $160$ | $22{,}666$ | $6.11 \times 10^{-3}$ | $64$ | $6$ | $1.38 \times 10^{-2}$ | $32$ |
| $320$ | $88{,}329$ | $6.06 \times 10^{-3}$ | $64$ | $8$ | $9.12 \times 10^{-3}$ | $32$ |
| $640$ | $348{,}562$ | $5.85 \times 10^{-3}$ | $64$ | $8$ | $6.04 \times 10^{-3}$ | $64$ |
| $1280$ | $1{,}384{,}524$ | $5.76 \times 10^{-3}$ | $64$ | $8$ | $4.51 \times 10^{-3}$ | $64$ |

Rates: ours $h^{0.46}$ end to end and $h^{0.34}$ by least squares, against the paper's $h^{0.64}$ on T40: the end-to-end slope rests on the coarsest level, and from $n = 80$ on our best error falls by only $17$%.

At a fixed $m$ the error hardly falls with $h$ (at $m = 2$, $9.7 \times 10^{-2}$ from $n = 40$ on; at $m = 64$, from $6.9 \times 10^{-3}$ to $5.8 \times 10^{-3}$ between $n = 80$ and $1280$): the error near a kink is set by the stencil size, of order $h^2 / k^2$ times a second derivative that the kink makes of order $1 / h$.
So, as in the paper, a finer grid helps only with a wider stencil, and the best $m$ grows with the refinement, from $8$ to $64$; the set of $m$ stops at $64$, which is best from $n = 80$ on, so our best error stalls near $6 \times 10^{-3}$ where the paper's still falls.
At the finest two levels, where the paper's best $m$ is $64$ too, ours is within a factor $1.3$ of its T40 error ($5.85 \times 10^{-3}$ against $6.04 \times 10^{-3}$, $5.76 \times 10^{-3}$ against $4.51 \times 10^{-3}$); at the first six levels ours is $1.5$ to $5.5$ times smaller; there the wide stencils, clipped at the boundary, already do well on our grids, where the paper's errors at $m = 64$ are about $0.24$ on its first three levels (its Figure 7).

$L^2$ and $H^1$, each with its own best $m$, against the paper's T40:

| $n$ | $L^2$ ours | $m$ | paper (T40) | paper's $m$ | $H^1$ ours | $m$ | paper (T40) | paper's $m$ |
|---|---|---|---|---|---|---|---|---|
| $10$ | $3.58 \times 10^{-2}$ | $8$ | $4.50 \times 10^{-2}$ | $4$ | $2.52 \times 10^{-1}$ | $8$ | $2.02 \times 10^{-1}$ | $2$ |
| $20$ | $1.13 \times 10^{-2}$ | $16$ | $1.62 \times 10^{-2}$ | $4$ | $1.87 \times 10^{-1}$ | $16$ | $1.51 \times 10^{-1}$ | $4$ |
| $40$ | $4.25 \times 10^{-3}$ | $32$ | $7.11 \times 10^{-3}$ | $8$ | $1.31 \times 10^{-1}$ | $32$ | $1.21 \times 10^{-1}$ | $8$ |
| $80$ | $1.71 \times 10^{-3}$ | $64$ | $3.35 \times 10^{-3}$ | $16$ | $9.39 \times 10^{-2}$ | $64$ | $9.80 \times 10^{-2}$ | $16$ |
| $160$ | $8.70 \times 10^{-4}$ | $64$ | $1.70 \times 10^{-3}$ | $32$ | $6.99 \times 10^{-2}$ | $64$ | $7.91 \times 10^{-2}$ | $32$ |
| $320$ | $6.89 \times 10^{-4}$ | $64$ | $9.63 \times 10^{-4}$ | $32$ | $5.88 \times 10^{-2}$ | $64$ | $6.33 \times 10^{-2}$ | $32$ |
| $640$ | $6.15 \times 10^{-4}$ | $64$ | $5.10 \times 10^{-4}$ | $64$ | $5.45 \times 10^{-2}$ | $64$ | $5.12 \times 10^{-2}$ | $64$ |
| $1280$ | $5.81 \times 10^{-4}$ | $64$ | $3.12 \times 10^{-4}$ | $64$ | $5.26 \times 10^{-2}$ | $64$ | $4.25 \times 10^{-2}$ | $64$ |

Rates: in $L^2$ ours $h^{0.85}$ end to end and by least squares, against the paper's $h^{1.02}$; in $H^1$ ours $h^{0.32}$ ($h^{0.34}$ by least squares), against $h^{0.33}$.
The $H^1$ error, the paper's plainest sign of the kink, is within a factor $1.25$ of the paper's T40 error at every level ($0.88$ to $1.25$ times it) and falls at the paper's rate: the gradient of $u_h$ is wrong by $O(1)$ in a strip of triangles along the kink, as on the paper's mesh.
The $L^2$ error is $0.51$ to $0.80$ times the paper's at the first six levels and stalls with the best $m$ at $64$, as the $L^\inf ty$ error does, so at the finest two levels it is $1.2$ and $1.9$ times the paper's.

This is a comparison with a different problem, not a reproduction; it shows that the nonsmooth behaviour of Figures 6 and 7 comes from a kink across the mesh, which T40 on our grid lacks.

## Finding 3: the receipt's non-convex fraction measures the Hessian fit, not the solution

`node_hessian`, from which `evaluate` takes `nonconvex_fraction` and `eigmin`, is the least-squares fit of a symmetric matrix to the second differences along all $64$ stencil directions (`fit_hessian`).
At a root of the scheme every second difference is nonnegative (the scheme's root conditions), but a fit to nonnegative data can be indefinite: one unit difference and the others $0$ fit the eigenvalues $-1/D$ and $3/D$ for $D$ directions (`tests/test_semilagrangian.py`).
The fit is worst where the differences are nearly all $0$, in a flat region or next to a kink, so the receipt's `nonconvex_fraction` is $0.46$ to $0.76$ on T40 for solutions exact to $10^{-12}$ ($0.31$ to $0.78$ on T40 turned, $0$ to $0.014$ on T39).
`run_feng_jensen2017.py --checks` measures it at the root (`feng_jensen2017_checks.csv`, rows `fit`, commit 87566d9, $4$ threads):

| problem | $n$ | $m$ | max error (absolute) | nonconvex_fraction | eigmin | nodes with all $d \ge -10^{-8}$ | of those, fitted eigenvalue $\lt -10^{-6}$ | smallest $d$ |
|---|---|---|---|---|---|---|---|---|
| T20 | $361$ | $4$ | $1.06 \times 10^{-3}$ | $8.9$% | $-2.52 \times 10^{-2}$ | $95.2$% | $4.0$% (down to $-2.49 \times 10^{-2}$) | $-8.1 \times 10^{-4}$ |
| T39 | $320$ | $8$ | $8.47 \times 10^{-3}$ | $0$ | $1.1 \times 10^{-4}$ | $99.999$% | $0$ | $-5.4 \times 10^{-7}$ |
| T40 | $80$ | $16$ | $1.0 \times 10^{-15}$ | $65$% | $-2.34$ | $100$% | $25$% (down to $-2.34$) | $-1.2 \times 10^{-10}$ |
| T40 | $320$ | $16$ | $5.8 \times 10^{-15}$ | $64$% | $-9.25$ | $100$% | $7.8$% (down to $-9.25$) | $-1.9 \times 10^{-9}$ |
| T40 turned | $320$ | $16$ | $2.16 \times 10^{-2}$ | $58$% | $-4.67$ | $100$% | $16$% (down to $-4.67$) | $-9.3 \times 10^{-10}$ |

The second differences are nonnegative up to rounding on T39 and T40, and on T20 down to $-8.1 \times 10^{-4}$, a trace of Howard's stopping tolerance ($10^{-6}$ in $u$, divided by $k^2 / 2$); the fit's negative eigenvalues are $30$ times larger there and of order $1$ on T40.
The Hessian of the maximising frame, $d_1 s_1 s_1^T + d_2 s_2 s_2^T$, has the eigenvalues $d_1$, $d_2$, so its smallest eigenvalue is the smallest $d$ of that frame: the same values as the last column here.
Reporting it as `node_hessian` (#73) would change the `nonconvex_fraction` and `eigmin` columns of this receipt and of `comparison.csv`, so it is left for that issue.
The T20 row's $8.9$% differs from the comparison receipt's $8.8$%, solved with one thread: the rounding of the Krylov solves depends on the thread count, and these two metrics are sensitive to it (the max error agrees to six digits).

## Finding 4: on T39 our $L^2$ error is $4$ to $6$ times the paper's, at the same best $m$

On T39 our $L^\inf ty$ error is $0.62$ to $1.02$ times the paper's and our $H^1$ error $1.31$ to $1.60$ times, but our $L^2$ error is $3.8$ to $6.0$ times the paper's, although its best $m$ is the paper's at every level and its rate ($h^{0.93}$) is within $0.09$ of the paper's.
The shapes of the errors differ.
With $\lVert u \rVert_{L^2} = 1.146$ and $\max \lvert u \rvert = 4$ over the domain, our absolute $L^2$ error is $1.16$ to $1.33$ times our absolute max error at the nodes at the best $m$ of each level, near the bound $\sqrt{\lvert \Omega \rvert} = 1.83$ of a function that is as large everywhere as at its maximum: ours is a broad bias.
The paper's, read the same way, is $0.20$ to $0.23$ times its max error at every level, an error concentrated near where it is largest.
The paper's relative $H^1$ error, in turn, is $2.7$ to $3.3$ times its relative $L^2$ error, ours $0.87$ to $1.03$ times.
A reading that would narrow the gap, which the paper does not state: the paper's $L^2$ values come within a factor $1.5$ of $\lVert u - u_h \rVert_{L^2} / \lVert u \rVert_{H^1}$, with $\lVert u \rVert_{H^1} = 4.659$, $4.06$ times $\lVert u \rVert_{L^2}$ for $\lvert x \rvert^4$ on the domain; our $L^2$ error divided so is $0.93$ to $1.49$ times the paper's ($0.93$, $1.03$, $0.95$, $1.15$, $0.97$, $1.26$, $1.09$, $1.49$ from the coarsest level), a ratio that grows with the refinement, so a different normalisation alone does not explain it.
We keep the measure the paper states, $\lVert u - u_h \rVert / \lVert u \rVert$ in each norm (the axis label of its Figure 7), and record the mismatch; the paper gives no code or further detail to settle it.

## Howard's algorithm

Fig. 8 of the paper gives $4$ to $16$ steps to a step below $10^{-6}$ over all levels and $m$, $5$ to $8$ (quartic) and $5$ to $9$ (nonsmooth) at the best $m$.
Ours, over all levels and $m$ of the receipt: $4$ to $7$ on T39 ($4$ or $5$ at the best $m$), $2$ to $9$ on T40, and $4$ to $27$ on T40 turned, the most at $m = 2$ and $4$, where the narrow stencil sees the kink; at the best $m$, $4$ to $8$.

`Newton(scheme=SemiLagrangian(n))` reaches the same root, but `Newton` solves its linear systems by LU, which fills in on wide stencils: on T39 it took $0.85$ s against $0.37$ s for `FDSemiLagrangian` at $n = 64$, $m = 4$, and $64.1$ s against $1.3$ s at $n = 128$, $m = 8$, with the same error to $10^{-10}$ (`feng_jensen2017_checks.csv`, rows `newton` and `howard`, $4$ threads).
#72 proposes to let the scheme choose its linear solve.

## Conventions that differ

- **Grid.** The paper's nodes are those of a quasi-uniform unstructured triangulation refined uniformly; ours are the grid nodes in the open domain and the boundary points of `shortley_weller_operators`, triangulated by Delaunay (`docs/methods/feng_jensen2017.md`). Our grids hold $1.14$ to $1.43$ times the paper's number of nodes at the same level index.
- **Mesh size.** The paper's $h$ is the "(average) mesh size" and $k = m h$; ours is the grid spacing $2/n$. The paper does not print its $h$, so $k$ cannot be compared directly.
- **Stencil size near the boundary.** The paper shrinks the stencil where $m h$ exceeds the distance to the boundary of the triangulated domain, and its Figure 5(b) shows a size that depends on the direction; ours is $\min (m h, \mathrm{distance})$ at each node, the same for every direction.
- **Split.** The paper discretises the rotations of the frames by $i \pi / 64$ and says nothing about the split $a$ of $\operatorname{diag}(a, 1 - a)$; we take the supremum over $a$ in closed form, which any fine discretisation of $a$ approaches.
- **Error measure.** Our $L^\inf ty$ error is the max error at the nodes relative to $\max \lvert u \rvert$ over the domain; the paper's is that of the finite element function, whose maximum over the domain is at most the nodal one plus the interpolation error, $O(h^2)$ for T39.
  Our $L^2$ and $H^1$ errors are those of the piecewise linear function $u_h$ on the Delaunay triangulation the scheme interpolates on, over the polygon $\Omega_h$ it covers, relative to the norms of $u$ over $\Omega_h$ (`fe_errors` in the runner): every integral by a collapsed Gauss rule of $25$ points per triangle, exact to degree $8$, so exact for T39, with the triangles split along the kink of T40 and T40 turned, so that the rule integrates functions that are smooth on each piece.
  The paper says neither how it integrates nor which $H^1$ norm it takes; we take the full norm $(\lVert v \rVert^2 + \lVert \nabla v \rVert^2)^{1/2}$, which FEniCS's `errornorm(..., "H1")` computes, and in which the gradient term dominates.
  The paper's $\Omega_h$ is its triangulated domain, inscribed in the same domain as ours.
- **Linear solver.** The paper used FEniCS without naming its linear solver; ours is BiCGSTAB with diagonal scaling, then GMRES, then LU (`Howard`). The discrete solution is the one an LU factorisation gives, but at small $m$ on fine grids the Krylov methods need many steps, and their budget of $10 N$ steps each keeps the LU fallback out of reach (Rerun; #72).

## Not compared

- Figure 4 (the solutions on the coarsest mesh).

## Rerun

```sh
just push
just remote-bg feng-jensen-fe env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/run_feng_jensen2017.py --processes 6 --levels 8 \
    --out /jczhang02/maesolvers-data/data/feng_jensen2017-fe.csv
# when the log ends:
scp a100server:/jczhang02/maesolvers-data/data/feng_jensen2017-fe.csv benchmarks/feng_jensen2017.csv
uv run python benchmarks/run_feng_jensen2017.py --summary benchmarks/feng_jensen2017.csv
# the checks of Finding 3 and of Newton's LU, about 3 minutes:
just remote env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python benchmarks/run_feng_jensen2017.py --checks \
    --out /jczhang02/maesolvers-data/data/feng_jensen2017_checks.csv
scp a100server:/jczhang02/maesolvers-data/data/feng_jensen2017_checks.csv benchmarks/
```

The $141$ solves end; those at $n = 1280$ take $7$ to $47$ minutes each at $m \ge 4$ ($7$ to $31$ at 4010174) and $1.3$ to $2$ hours for T39 at $m = 2$; each worker at $n = 1280$ holds about $10$ GB.
The three excluded solves take hours (Receipt), because BiCGSTAB needs thousands of steps and often breaks down into GMRES, whose budget of $10 N$ steps keeps the LU fallback out of reach (#72).
The $141$ rows took $2.5$ hours with $6$ workers at cf16a7c and $1.3$ hours at 4010174, the time of the T39 solve at $n = 1280$, $m = 2$, which starts first.
A rerun with the same `--out` solves only the rows it lacks, so a rerun that adds columns needs a fresh `--out`.
The checks run their solves one after the other, in about $3$ minutes; each is an MLflow run named `check=<kind> n=<n> m=<m>` in `maesolvers-feng_jensen2017-<problem>`, T20's in `maesolvers-feng_jensen2017-T20`.
