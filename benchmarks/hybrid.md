# Hybrid scheme: published versus ours

Problems: T1 (smooth, $u = \exp(\lvert x \rvert^2 / 2){}$), T2 (corner blow-up, $u = -\sqrt{2 - \lvert x \rvert^2}$), T20 ($C^1$, $u = ((\lvert x - x_0 \rvert - 0.2)^+)^2 / 2$, $f = 0$ on a disk) and T21 (the cone $\lvert x - x_0 \rvert$, $f$ a Dirac mass), on $(0, 1)^2$ with Dirichlet data from the exact solution, $x_0 = (0.5, 0.5)$.
Published numbers: Froese-Oberman, J. Comput. Phys. 230 (2011), Table 3, column "Hybrid" (JCP below), checked against the page image; `sources.md` transcribes the whole table.

Ours: `FDHybrid(n=N)`, the preset `Newton(scheme=Hybrid(n=N), start=FDMonotone(n=N), max_iter=200)` (ADR 0004): the 17-point monotone stencil, `boundary="narrow"` unless a column says otherwise, the paper's switching rule with $\varepsilon = h$ and `band = 1` (`docs/methods/hybrid.md`), residual tolerance $10^{-10}$ relative to $\max(1, \max f)$, float64, scipy sparse LU.
Every run of `hybrid.csv` converged.
Error: max over all grid nodes of $\lvert u_h - u \rvert$, $h = 1/N$ (ADR 0001) unless a column says otherwise.
T21 is not in the catalogue; `run_hybrid.py` builds it per grid with `cone(n)` of `run_monotone17.py` ($f = 4/h^2$ at the nodes within $h/2$ of $x_0$), with $n = N - 1$ so that $x_0$ is a node.
Raw data: `hybrid.csv` and `hybrid_checks.csv`, written on a100server by `run_hybrid.py` at commit dd03dd0, one after the other (`just remote-bg hybrid-r1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/run_hybrid.py --processes 8`, then `hybrid-checks-r1` with `--checks --processes 16`).
The commits this file cites are on the branch of PR #53, which was squash-merged into `next` as fd52e07, so they are not in its history; GitHub keeps them under `refs/pull/53/head`.
Every solve of these two files is one run in the MLflow experiment `maesolvers-hybrid-<problem>` (T1, T2, T20, T21), tagged with commit dd03dd0 and a clean checkout; at that commit `run_hybrid.py` created the experiments before its worker processes started; since then `tracking.run` names its experiment by id, which settles the race of issue #54 for every script.
Earlier generations are in the same experiments: `hybrid.csv` at 9f6df9d and the checks at 9f6df9d, 9c1db06 and ad53e5b, with the same values in every row they share with the current files.
Four solves of the 9f6df9d generation went to the experiment `Default` instead, through a race between the worker processes that created the experiments: the runs `N=31 h=1/N interpolate` of T1, T2 and T20, and `N=31 h=1/(N-1) narrow` of T2.

"As printed" rounds our value to the digits the paper prints; a bold value differs from the published one there.
The levels follow ADR 0002; the convergence order of a column is the least-squares slope of $\log \mathrm{error}$ against $\log N$ over its printed $N$, written "orders: published X, ours Y"; a ratio is ours divided by published.

## Summary

| Problem | JCP Table 3, column "Hybrid" |
|---|---|
| T1 | mismatch (Finding 3) |
| T2 | order of magnitude; exact from $N = 127$ on, and at every $N$ with $h = 1/(N - 1)$ (Finding 2) |
| T20 | mismatch (Finding 4) |
| T21 | mismatch (`monotone17.md`, Finding 4): the hybrid scheme is the monotone one there |

`tests/test_reproduction.py` pins T2 at $N = 127$ and $181$, each to the printed digits.
Section 8.1 of the paper, Newton on the standard scheme failing on T2 from the exact solution, is not confirmed (Finding 5); Newton on the hybrid scheme from the Poisson start does reach a non-convex root (Finding 1).

## T1, smooth example

In units of $10^{-5}$:

| $N$ | Published | Ours | As printed | `boundary="interpolate"` | Standard (published) |
|---|---|---|---|---|---|
| $31$ | $24.45$ | $16.904$ | **$16.90$** | $6.869$ | $7.14$ |
| $45$ | $15.29$ | $8.569$ | **$8.57$** | $2.892$ | $3.39$ |
| $63$ | $9.06$ | $4.560$ | **$4.56$** | $1.667$ | $1.73$ |
| $89$ | $5.32$ | $2.369$ | **$2.37$** | $0.913$ | $0.87$ |
| $127$ | $3.02$ | $1.201$ | **$1.20$** | $0.478$ | $0.43$ |
| $181$ | $1.61$ | $0.606$ | **$0.61$** | $0.246$ | $0.21$ |
| $255$ | $0.87$ | $0.311$ | **$0.31$** | $0.128$ | $0.11$ |
| $361$ | $0.46$ | $0.157$ | **$0.16$** | $0.065$ | $0.05$ |

Ratio $0.34$ to $0.69$, falling with $N$; orders: published $1.63$, ours $1.91$ ($1.86$ with `"interpolate"`).
Level: mismatch, as the ratio is below $1/2$ from $N = 89$ on and the orders differ by $0.28$ (Finding 3).
As the paper says (its Remark 6), the hybrid scheme is less accurate than the standard one on this smooth solution because the first ring of nodes is monotone: ours by a factor $2.4$ at $N = 31$ to $3.0$ at $N = 361$, the published by $3.4$ to $9.2$.
The ring is $4 (N - 2)$ of the $(N - 1)^2$ unknowns (`monotone_fraction` $0.13$ at $N = 31$, $0.011$ at $N = 361$).

## T2, example with blow-up

In units of $10^{-3}$:

| $N$ | Published | Ours | As printed | $h = 1/(N - 1)$ | Monotone, ours (`monotone17.csv`) |
|---|---|---|---|---|---|
| $31$ | $1.74$ | $1.6543$ | **$1.65$** | $1.7375$ | $1.6547$ |
| $45$ | $0.98$ | $0.9470$ | **$0.95$** | $0.9794$ | $0.9471$ |
| $63$ | $0.59$ | $0.5721$ | **$0.57$** | $0.5860$ | $0.5721$ |
| $89$ | $0.35$ | $0.3409$ | **$0.34$** | $0.3467$ | $0.3745$ |
| $127$ | $0.20$ | $0.2001$ | $0.20$ | $0.2025$ | $0.3485$ |
| $181$ | $0.12$ | $0.1176$ | $0.12$ | $0.1186$ | $0.3356$ |
| $255$ | $0.07$ | $0.0704$ | $0.07$ | $0.0708$ | $0.3294$ |
| $361$ | $0.04$ | $0.0418$ | $0.04$ | $0.0419$ | $0.3262$ |

Ratio $0.95$ to $1.04$; orders: published $1.53$, ours $1.50$, the paper's $O(h^{1.5})$.
Level: order of magnitude, exact from $N = 127$ on; with $h = 1/(N - 1)$ every printed value is ours to the printed digits (Finding 2).
`boundary="interpolate"` gives the same values to four digits.
The hybrid scheme gains on the monotone one from $N = 89$ on, where the error moves from the corner, monotone in both, into the region where the hybrid scheme is the standard one; at $N = 361$ it is $8$ times smaller.
The centred Hessian of our solution has a negative eigenvalue at a few nodes next to the corner (`nonconvex_fraction` $10^{-4}$ to $1.5 \times 10^{-3}$), as for the monotone scheme (`monotone17.md`, T2).

## T20, $C^1$ example with a flat disk

In units of $10^{-4}$:

| $N$ | Published | Ours | As printed | `boundary="interpolate"` | Monotone, ours | Standard (published) | Newton steps |
|---|---|---|---|---|---|---|---|
| $31$ | $12.2$ | $1.753$ | **$1.8$** | $1.693$ | $8.03$ | $2.6$ | $6$ |
| $45$ | $5.9$ | $1.153$ | **$1.2$** | $1.123$ | $6.78$ | $1.8$ | $5$ |
| $63$ | $4.2$ | $1.110$ | **$1.1$** | $1.094$ | $6.83$ | $1.5$ | $7$ |
| $89$ | $2.6$ | $0.739$ | **$0.7$** | $0.731$ | $6.64$ | $0.9$ | $9$ |
| $127$ | $2.0$ | $0.581$ | **$0.6$** | $0.577$ | $6.55$ | $0.6$ | $6$ |
| $181$ | $1.2$ | $0.446$ | **$0.4$** | $0.444$ | $6.55$ | $0.4$ | $20$ |
| $255$ | $1.0$ | $0.390$ | **$0.4$** | $0.390$ | $6.59$ | -- | $26$ |
| $361$ | $0.7$ | $0.297$ | **$0.3$** | $0.297$ | $6.60$ | -- | $50$ |

Ratio $0.14$ to $0.42$; orders: published $1.12$, ours $0.71$.
Level: mismatch (Finding 4).
The monotone scheme covers the flat disk, a ring of one step around it and the boundary ring: $31$ percent of the unknowns at $N = 31$, $14$ percent at $N = 361$.
The centred Hessian is indefinite at nodes of the flat disk (`nonconvex_fraction` $0.03$ to $0.10$), as for the monotone scheme (`monotone17.md`, Finding 6).

## T21, the cone

$f$ is $0$ or $4/h^2$ at every node, so the switching rule puts every node in the monotone band (`monotone_fraction` $1$), and the hybrid scheme is `Monotone17`, as the paper says (its Section 8.5).
The monotone start already meets the tolerance, so Newton takes no step (`info["iterations"]` is $1$: the check of the start), and the errors are those of `monotone17.csv` ($17$ points, $h = 1/(N - 1){}$) to the last digit: $0.691$ at every $N$, against the published $3 \times 10^{-3}$ to $4 \times 10^{-3}$.
Level: mismatch, for the reason of `monotone17.md`, Finding 4: the papers' discrete Dirac mass does not fit the scheme.

## Finding 1: Newton from the Poisson start reaches a non-convex root on T2; a convex start selects the convex one

`Newton(scheme=Hybrid(n=N))` from its default start, $\Delta u_0 = 2 \sqrt{f}$, converges in the residual on T2, but from $N = 63$ on to a root whose error is $16$, $37$ and $425$ times the published value at $N = 63$, $127$ and $361$ (`hybrid_checks.csv`, rows `section81`, scheme `Hybrid`, start `poisson`):

| $N$ | Poisson start | Exact start | Monotone start (`FDHybrid`) | Published |
|---|---|---|---|---|
| $31$ | $1.654 \times 10^{-3}$ | $1.654 \times 10^{-3}$ | $1.654 \times 10^{-3}$ | $1.74 \times 10^{-3}$ |
| $63$ | $9.577 \times 10^{-3}$ | $5.721 \times 10^{-4}$ | $5.721 \times 10^{-4}$ | $0.59 \times 10^{-3}$ |
| $127$ | $7.387 \times 10^{-3}$ | $2.001 \times 10^{-4}$ | $2.001 \times 10^{-4}$ | $0.20 \times 10^{-3}$ |
| $361$ | $1.698 \times 10^{-2}$ | $4.177 \times 10^{-5}$ | $4.177 \times 10^{-5}$ | $0.04 \times 10^{-3}$ |

That root is non-convex: from $N = 63$ on the centred Hessian is indefinite at $0.3$ to $0.8$ percent of the nodes, against $0.008$ to $0.07$ percent at the convex root.
The monotone band around the corner, where $f \ge 1/h$, does not prevent it: the standard scheme just outside the band admits the non-convex root.
From the exact solution, or from the root of the monotone scheme, Newton reaches the convex root in $3$ steps.
The paper starts Newton from one Poisson step convexified by the convex envelope of the result (its Section 7.1), which is the part of the method that selects the convex root here; `FDHybrid` takes the root of its own monotone part as a convex start instead (`docs/methods/hybrid.md`), and `tests/test_hybrid.py` checks both behaviours at $N = 63$.

## Finding 2: the T2 column matches $h = 1/(N - 1)$ at every $N$

With $n = N - 1$ cells every published T2 value is ours to the printed digits ($1.7375$, $0.9794$, $0.5860$, $0.3467$ for $1.74$, $0.98$, $0.59$, $0.35$, and the four values from $N = 127$ on).
With $n = N$ the values from $N = 127$ on also agree, but not at $N = 31$ to $89$.
At $N = 31$ to $63$ our hybrid and monotone errors agree to three digits (the maximum is at the corner, which both solve with the monotone scheme), and the published columns "Monotone" and "Hybrid" print the same three values there; `monotone17.md`, Finding 3, found the same convention for that column at these $N$.
At $N = 89$ the two columns part: the monotone column matches $h = 1/N$ ($0.37$) and not $1/(N - 1)$ ($0.38$), while the hybrid column matches $1/(N - 1)$ ($0.35$) and not $1/N$ ($0.34$).
So the hybrid runs on T2 probably used $h = 1/(N - 1)$ throughout; for the standard scheme every printed value follows $h = 1/N$ (`standard_fd.md`, Finding 1), and why the T2 runs of the two newer columns would not is not known.
ADR 0001 keeps $h = 1/N$, and the level is stated at that convention.

## Finding 3: on T1 the published errors are $1.4$ to $2.9$ times ours and converge more slowly

The published hybrid errors on T1 are above ours at every $N$, by a factor that grows from $1.45$ at $N = 31$ to $2.9$ at $N = 361$; their order is $1.63$ against our $1.91$, while the paper's text says the accuracy "approaches $O(h^2)$".
T1 is the cleanest test of the switching rule: $f$ is between $1$ and $22.2$ on the square, never at most $h$ or at least $1/h$ for $N \ge 31$, so the hybrid scheme differs from the standard one only through the boundary.
Three candidate causes, tested in `hybrid_checks.csv`:

- The width and shape of the monotone band, which the paper does not state beyond "zero in an $h$-neighborhood" and "goes to $1$ elsewhere".
  No fixed band reproduces the column (rows `band`, $N = 31$, $63$, $127$, in units of $10^{-5}$):

  | band | $N = 31$ | $N = 63$ | $N = 127$ |
  |---|---|---|---|
  | $0$ (the standard scheme) | $7.14$ | $1.73$ | $0.43$ |
  | $1$ (default) | $16.90$ | $4.56$ | $1.20$ |
  | $2$ | $18.57$ | $4.73$ | $1.29$ |
  | $3$ | $26.84$ | $7.49$ | $2.22$ |
  | published | $24.45$ | $9.06$ | $3.02$ |

  The ratio ours / published falls with $N$ for every band, so a band whose width is a fixed number of grid steps cannot match it; a weight whose transition widens relative to $h$ as the grid is refined could, but the paper gives none, and we did not fit one.
- The boundary treatment of the monotone part: `"interpolate"` lowers our errors further (ratio $0.14$ to $0.28$), so it moves away from the column.
- An early stop: Table 1 reports $3$ to $6$ Newton steps on T1.
  From our start the error is settled to three digits after the first step at $N = 31$, $45$ and $63$ (rows `steps`), so stopping early from our start would not raise it; the paper's start (a coarse-grid Poisson step, convexified and interpolated) is farther from the root, and its iterates may not have been converged, as on the monotone column at coarse grids (`monotone17.md`, Finding 2), but we cannot repeat that start.

The level is mismatch, recorded and not tuned: the default rule is the paper's text, and the evidence does not single out a cause.

## Finding 4: on T20 the published errors are $2.4$ to $7$ times ours, above the published standard column

On T20 our hybrid error is below the published one at every $N$, by a factor $7.0$ at $N = 31$ falling to $2.4$ at $N = 361$, and it is even below the published standard column ($2.6 \times 10^{-4}$ at $N = 31$) up to $N = 127$, while the published hybrid column lies $2.8$ to $4.7$ times above that standard column.
Our hybrid solution is the convex one: the monotone part covers the flat disk where $f = 0$, and its continuation below zero (`docs/methods/monotone17.md`, "Below zero") leaves it no non-convex root there.
The published monotone column on T20 shows the same pattern against ours (`monotone17.md`, Finding 5), and the published hybrid, which uses the monotone scheme in the same disk, inherits its candidate causes: a non-convex root of the bare scheme in the flat disk, reached from the paper's start, or a regularisation with an unstated $\delta$.
Widening the band raises our error but not to the published one (rows `band`: at $N = 31$, $4.39 \times 10^{-4}$ with `band = 2` and $5.74 \times 10^{-4}$ with $3$, against $12.2 \times 10^{-4}$), and `band = 0` (monotone only where $f \le h$) gives $4.22 \times 10^{-4}$.
Level: mismatch.

## Finding 5: Section 8.1 is not confirmed: Newton on the standard scheme from the exact solution converges to the convex root

The paper reports that Newton on the standard 9-point scheme, started from the exact solution of T2, oscillates and becomes non-convex after two iterations (its Section 8.1 and Fig. 2), at a grid size it does not give.
`Newton(scheme=Standard9(n=N))` from the exact solution converges in $5$ steps at $N = 31$, $63$, $127$ and $361$, with and without its line search (every step is a full one), to the convex root, whose errors are the published standard column (rows `section81`, scheme `Standard9`, starts `exact` and `exact, undamped`):

| $N$ | Exact start | Exact start, full steps | Poisson start | Published "Standard" |
|---|---|---|---|---|
| $31$ | $1.7147 \times 10^{-2}$ | $1.7147 \times 10^{-2}$ | $1.7453 \times 10^{-2}$ | $1.715 \times 10^{-2}$ |
| $63$ | $1.2531 \times 10^{-2}$ | $1.2531 \times 10^{-2}$ | $1.9309 \times 10^{-2}$ | $1.253 \times 10^{-2}$ |
| $127$ | $9.004 \times 10^{-3}$ | $9.004 \times 10^{-3}$ | $1.8791 \times 10^{-2}$ | $0.900 \times 10^{-2}$ |
| $361$ | $5.4095 \times 10^{-3}$ | $5.4095 \times 10^{-3}$ | $2.2248 \times 10^{-2}$ | $0.541 \times 10^{-2}$ |

From the Poisson start Newton reaches a non-convex root instead (`nonconvex_fraction` $0.1$ to $1.1$ percent), as `standard_fd.md`, Finding 2, found for `FDNewton`.
The iterates on the way do not reproduce Fig. 2 either (rows `exact, 1 step` and `exact, 2 steps`): after the first step the centred Hessian is indefinite at $0.22$, $0.10$, $0.025$ and $0.005$ percent of the nodes ($N = 31$, $63$, $127$, $361$; errors $2.35 \times 10^{-2}$, $1.76 \times 10^{-2}$, $1.27 \times 10^{-2}$, $7.7 \times 10^{-3}$), and after the second it is convex again at $N = 31$ to $127$ (errors $1.78 \times 10^{-2}$, $1.31 \times 10^{-2}$, $9.48 \times 10^{-3}$) and indefinite at $2$ nodes at $N = 361$, so the first iterate is briefly non-convex and the second is (nearly) convex, while the paper shows a non-convex second iterate.
So the failure the paper describes is real for a start away from the convex root, but not from the exact solution in our runs; the instability it attributes to the standard scheme is one of branch selection, which the start decides (Finding 1).

## Conventions

- Eq. (6) of the paper, the possible singular set, prints $\lbrace x \in \Omega : \varepsilon < f(x) < 1/\varepsilon\rbrace$, which is the set where $f$ is regular; the text ("a neighborhood of the possible singular set"), conditions (2), and the remark that the scheme is the monotone one where $f$ is "either $0$ or very large" (Section 8.5) mean the complement, $f \le \varepsilon$ or $f \ge 1/\varepsilon$, which `Hybrid` uses.
- The weight is a step on the grid ($0$ within `band` steps of the singular set, $1$ elsewhere); the paper's "differentiable function" has no stated form (Finding 3).
- The start differs: the paper's convexified coarse-grid Poisson step against our monotone root (Finding 1).

## Iterations and time

From the monotone root, Newton on the hybrid scheme takes $3$ steps on T1 and T2 at every $N$ ($4$ on T1 at $N = 361$), and $5$ to $50$ on T20, against the paper's $3$ to $6$, $4$ to $8$ and $4$ to $20$ (its Table 1); the monotone start itself takes $5$ to $14$, $10$ to $30$ and $8$ to $43$ steps (`start_iterations` in `hybrid.csv`, `boundary="narrow"`, $h = 1/N$; $9$ to $83$ on T21), which the paper's start avoids; every start converged (`start_converged`).
On T21 Newton takes no step after the start (the paper: $9$ to $34$), since the scheme is the monotone one there and the start is its root; `iterations` is $1$, the check of the start.
A solve at $N = 361$ takes about $1$ minute on T1, $2$ on T2 and $4$ on T20 on one core of a100server, start included; CPU times are 2011 MATLAB hardware and are not compared.

## Not compared

- Table 4 (3D hybrid, 19-point stencil): 3D boxes come with 0.2.
- Table 1's CPU times and the Poisson and Gauss-Seidel columns of Tables 1 and 2 (2011 hardware; the Poisson iteration is `FDPoisson`, compared in `standard_fd.md`).
