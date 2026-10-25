# Multigrid: published versus ours

Problems, with Dirichlet data from the exact solution: in 2D on $(0, 1)^2$, T1 ($u = \exp(\lvert x \rvert^2 / 2)$, Example 4.1), T3 ($u = (2 \sqrt{2} / 3) \lvert x \rvert^{3/2}$, Example 4.2) and T2 ($u = -\sqrt{2 - \lvert x \rvert^2}$, Example 4.3); in 3D, T9 ($u = \exp(\lvert x \rvert^2 / 2)$ on $(0, 1)^3$, Example 4.4), T46 ($u = \lvert x \rvert^2 / 2 - \sin x - \sin y - \sin z$ on $(0, \pi)^3$, Example 4.5), T28 ($u = \lvert x \rvert^{3/2} / 3$ on $(0, 1)^3$, Example 4.6) and T22 ($u = -\sqrt{3 - \lvert x \rvert^2}$ on $(0, 1)^3$, Example 4.7).
Published numbers: J. Liu, B. D. Froese, A. M. Oberman, M. Xiao, Int. J. Comput. Math. 94 (2017) 1850-1866, Tables 1 and 2 (2D, red-black and lexicographic Gauss-Seidel) and Tables 3, 5, 6 and 7 (3D, lexicographic), each checked against the page images; the text layer of the PDF garbles the first rows of Tables 5 to 7 and prints Example 4.5's $\lvert x \rvert^2 / 2$ as $(x + y + z) / 2$.
Table 4 reprints numbers of other papers and is not a target; its wide-stencil column is cross-checked against the 19-point monotone scheme (section "Table 4").

Ours: `Multigrid(scheme=StandardBox(n=N), ordering=...)`, which the preset `FDMultigrid(n=N, ordering=...)` is (ADR 0004; `docs/methods/multigrid.md`), with the paper's settings, which are the defaults: one FMG cycle, then FAS V-cycles with $2 + 2$ nonlinear Gauss-Seidel sweeps and one sweep on the grid of $2$ cells per side; float64.
The columns compared with the paper stop as the paper does, at the first cycle whose relative residual is at most $10^{-6}$ (`PaperStop` in `run_multigrid.py`, a subclass whose `converged` tests the relative residual); every run stops so after $1$ or $2$ cycles.
`FDMultigrid` itself stops on its residual bound (`tol` $= 10^{-6}$ times $\max(1, \max \lvert f \rvert)$ on the residual's RMS); its cycles, its error and whether it converged are the `default_*` columns of `multigrid.csv` (Finding 5).
Error: max over all grid nodes of $\lvert u_h - u \rvert$, `evaluate`'s `max`, $h = 1/N$ (ADR 0001; $\pi/N$ on T46).
"Converged" is the error of the discrete solution itself: `FDMultigrid` (red-black) for ten cycles, the FMG cycle and nine V-cycles, whatever the residual, after which the relative residual is at most $3 \times 10^{-16}$ (except on T2, $1.4 \times 10^{-11}$, and T22 at $N = 128$, Finding 4).
The Gauss-Seidel columns repeat the paper's stand-alone solver: lexicographic sweeps of `StandardBox.relax` from $1.01\,u$ at the unknowns until the residual's discrete L2 norm is $10^{-6}$ times that of the start (`gauss_seidel` in `run_multigrid.py`, which `tests/test_reproduction.py` pins).
Raw data: `multigrid.csv` ($82$ rows), written on a100server by `run_multigrid.py` at commit 3d31695, clean checkout (`just remote-bg multigrid-r3 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/run_multigrid.py --workers 8 --out /jczhang02/maesolvers-data/data/multigrid-r3.csv`, then copied here), `multigrid_checks.csv` by `run_multigrid.py --checks --workers 5` and `multigrid_table4.csv` by `run_multigrid.py --table4 --workers 3`, both at the same commit.
The columns of the earlier generation (64fe66a, with the paper's rule as the library's stopping rule) are repeated to all digits but the times and the last bits of the relative residuals, which now take the RMS over the unknowns instead of over all nodes, the ring counting $0$; the `default_*` columns and the checks of the FMG interpolation and the coarse start are new.
Every row is one run in the MLflow experiment `maesolvers-multigrid-<problem>` (T1, T2, T3, T9, T22, T28, T46), named `table<k>-<ordering>-n<N>`, `check-<variant>-<ordering>-n<N>` or `check-<variant>-<ordering>-n<N>-cycles<k>`, or `table4-n<N>`, tagged with its commit and a clean checkout; the experiments also hold the earlier generations (85bcd7c, 64fe66a, f11539d, 2b0f5ad, and the T2 checks at fe47d9b).
`run_multigrid.py --report` prints the tables below from the CSV.

"As printed" rounds our value to the digits the paper prints; a bold value differs from the published one there.
The levels follow ADR 0002; the convergence order of a column is the least-squares slope of $\log \text{error}$ against $\log N$ over its printed $N$, written "orders: published X, ours Y"; a ratio is ours divided by published.

## Summary

| Table | Problem | FMG-FAS | Gauss-Seidel |
|---|---|---|---|
| $1$ (red-black) | T1, T3, T2 | order of magnitude (ratios $0.94$ to $1.08$, orders equal to $0.02$; "Iter" equal in all $21$ cells) | |
| $2$ (lexicographic) | T1, T3, T2 | order of magnitude (ratios $0.92$ to $1.05$, orders equal to $0.01$; "Iter" equal in $19$ of $21$ cells) | |
| $3$ | T9 | order of magnitude (ratios $1.14$ to $1.99$, $0.75$ to $1.38$ with the 4-point FMG interpolation; Finding 1) | exact; same sweeps |
| $5$ | T46 | order of magnitude (ratios $1.09$ to $1.60$, $0.95$ to $1.24$ with the 4-point FMG interpolation; Finding 1) | exact; same sweeps |
| $6$ | T28 | order of magnitude (ratios $0.80$ to $0.84$) | exact; same sweeps at $N = 8$, $16$, one fewer at $32$ |
| $7$ | T22 | mismatch (Finding 4: ratio $0.25$ at $N = 32$; $0.97$ to $1.24$ at the other four $N$) | exact; same sweeps at $N = 8$, $16$, three fewer at $32$ |

`tests/test_reproduction.py` pins the Gauss-Seidel errors to the printed digits and the sweep counts at $N = 8$ on the four 3D problems and at $N = 16$ on T9.

## Tables 1 and 2: 2D

Max error, red-black (Table 1):

| $N$ | T1 published | T1 ours | T3 published | T3 ours | T2 published | T2 ours | T2 Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| $128$ | $4.5 \times 10^{-6}$ | $4.447 \times 10^{-6}$ | $3.8 \times 10^{-5}$ | $3.626 \times 10^{-5}$ | $8.9 \times 10^{-3}$ | $8.959 \times 10^{-3}$ | $2$, $2$ |
| $256$ | $1.1 \times 10^{-6}$ | $1.111 \times 10^{-6}$ | $1.3 \times 10^{-5}$ | $1.283 \times 10^{-5}$ | $6.4 \times 10^{-3}$ | $6.396 \times 10^{-3}$ | $2$, $2$ |
| $512$ | $2.8 \times 10^{-7}$ | $2.778 \times 10^{-7}$ | $4.7 \times 10^{-6}$ | $4.535 \times 10^{-6}$ | $4.5 \times 10^{-3}$ | $4.521 \times 10^{-3}$ | $1$, $1$ |
| $1024$ | $7.0 \times 10^{-8}$ | $6.944 \times 10^{-8}$ | $1.7 \times 10^{-6}$ | $1.603 \times 10^{-6}$ | $3.2 \times 10^{-3}$ | $3.204 \times 10^{-3}$ | $1$, $1$ |
| $2048$ | $1.7 \times 10^{-8}$ | $1.736 \times 10^{-8}$ | $5.9 \times 10^{-7}$ | $5.669 \times 10^{-7}$ | $2.3 \times 10^{-3}$ | $2.268 \times 10^{-3}$ | $1$, $1$ |
| $4096$ | $4.3 \times 10^{-9}$ | $4.340 \times 10^{-9}$ | $2.1 \times 10^{-7}$ | $2.004 \times 10^{-7}$ | $1.6 \times 10^{-3}$ | $1.605 \times 10^{-3}$ | $1$, $1$ |
| $8192$ | $1.0 \times 10^{-9}$ | $1.082 \times 10^{-9}$ | $7.4 \times 10^{-8}$ | $7.087 \times 10^{-8}$ | $1.1 \times 10^{-3}$ | $1.135 \times 10^{-3}$ | $1$, $1$ |

Ratios $0.99$ to $1.08$ (T1), $0.94$ to $0.99$ (T3), $0.99$ to $1.03$ (T2); orders: published $2.02$, $1.50$, $0.50$, ours $2.00$, $1.50$, $0.50$.
T1 and T3 take one cycle at every $N$ in both tables, as published.

Max error, lexicographic (Table 2):

| $N$ | T1 published | T1 ours | T3 published | T3 ours | T2 published | T2 ours | T2 Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| $128$ | $5.1 \times 10^{-6}$ | $5.357 \times 10^{-6}$ | $4.3 \times 10^{-5}$ | $4.018 \times 10^{-5}$ | $8.9 \times 10^{-3}$ | $8.961 \times 10^{-3}$ | $3$, $2$ |
| $256$ | $1.3 \times 10^{-6}$ | $1.341 \times 10^{-6}$ | $1.5 \times 10^{-5}$ | $1.422 \times 10^{-5}$ | $6.3 \times 10^{-3}$ | $6.399 \times 10^{-3}$ | $2$, $2$ |
| $512$ | $3.3 \times 10^{-7}$ | $3.355 \times 10^{-7}$ | $5.4 \times 10^{-6}$ | $5.029 \times 10^{-6}$ | $4.4 \times 10^{-3}$ | $4.547 \times 10^{-3}$ | $1$, $2$ |
| $1024$ | $8.2 \times 10^{-8}$ | $8.388 \times 10^{-8}$ | $1.9 \times 10^{-6}$ | $1.778 \times 10^{-6}$ | $3.2 \times 10^{-3}$ | $3.169 \times 10^{-3}$ | $1$, $1$ |
| $2048$ | $2.1 \times 10^{-8}$ | $2.097 \times 10^{-8}$ | $6.8 \times 10^{-7}$ | $6.287 \times 10^{-7}$ | $2.2 \times 10^{-3}$ | $2.243 \times 10^{-3}$ | $1$, $1$ |
| $4096$ | $5.1 \times 10^{-9}$ | $5.242 \times 10^{-9}$ | $2.4 \times 10^{-7}$ | $2.223 \times 10^{-7}$ | $1.6 \times 10^{-3}$ | $1.587 \times 10^{-3}$ | $1$, $1$ |
| $8192$ | $1.3 \times 10^{-9}$ | $1.305 \times 10^{-9}$ | $8.5 \times 10^{-8}$ | $7.858 \times 10^{-8}$ | $1.1 \times 10^{-3}$ | $1.123 \times 10^{-3}$ | $1$, $1$ |

Ratios $1.00$ to $1.05$ (T1), $0.92$ to $0.95$ (T3), $0.99$ to $1.03$ (T2); orders: published $1.99$, $1.50$, $0.50$, ours $2.00$, $1.50$, $0.50$.

Level: order of magnitude for both tables.
$18$ of the $42$ values agree with the published ones to the printed two digits (T2 red-black from $N = 256$ on, T2 lexicographic from $1024$ on, T1 at four $N$ red-black and three lexicographic, T3 red-black at $256$), the others differ in the last digit, so the level is not exact.
The paper's observation holds in ours: red-black is the more accurate ordering on T1 and T3, by $20$ to $21$ percent (T1) and $11$ percent (T3) against the paper's $13$ to $30$ and $12$ to $15$ percent.
After one cycle ours is $6$ percent above the discrete solution's error with red-black sweeps (T1 and T3), $28$ percent (T1) and $17$ percent (T3) above it with lexicographic ones, and within $2$ percent on T2: one FMG cycle reaches the discretisation error, as the paper says, with an algebraic error smaller than it.

## Table 3: T9

| $N$ | Published | Ours | As printed | Ratio | Red-black (ours) | Converged | Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| $8$ | $1.2 \times 10^{-3}$ | $1.507 \times 10^{-3}$ | **$1.5 \times 10^{-3}$** | $1.26$ | $1.396 \times 10^{-3}$ | $1.157 \times 10^{-3}$ | $1$, $1$ |
| $16$ | $2.7 \times 10^{-4}$ | $4.107 \times 10^{-4}$ | **$4.1 \times 10^{-4}$** | $1.52$ | $3.786 \times 10^{-4}$ | $2.977 \times 10^{-4}$ | $1$, $1$ |
| $32$ | $5.7 \times 10^{-5}$ | $1.083 \times 10^{-4}$ | **$1.1 \times 10^{-4}$** | $1.90$ | $9.597 \times 10^{-5}$ | $7.551 \times 10^{-5}$ | $1$, $1$ |
| $64$ | $1.4 \times 10^{-5}$ | $2.786 \times 10^{-5}$ | **$2.8 \times 10^{-5}$** | $1.99$ | $2.384 \times 10^{-5}$ | $1.891 \times 10^{-5}$ | $1$, $1$ |
| $128$ | $6.2 \times 10^{-6}$ | $7.065 \times 10^{-6}$ | **$7.1 \times 10^{-6}$** | $1.14$ | $5.905 \times 10^{-6}$ | $4.730 \times 10^{-6}$ | $1$, $1$ |

Orders: published $1.95$, ours $1.94$.
Level: order of magnitude, with the ratio at $N = 64$ at the edge of $[1/2, 2]$; the ratio of $1.99$ depends on the FMG interpolation, which the paper does not specify, and is $1.35$ with the 4-point cubic one (Finding 1).
The paper's order drops to $1.2$ from $N = 64$ to $128$, which it puts down to rounding (the operator divides by $h^6$); the discrete solution's error falls by a factor $4.0$ there ($1.891 \times 10^{-5}$ to $4.730 \times 10^{-6}$), so in our float64 runs the second order holds at $128$.
The Gauss-Seidel column:

| $N$ | Published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| $8$ | $1.2 \times 10^{-3}$ | $1.158 \times 10^{-3}$ | $72$, $72$ |
| $16$ | $3.0 \times 10^{-4}$ | $2.984 \times 10^{-4}$ | $270$, $270$ |
| $32$ | $8.1 \times 10^{-5}$ | $8.137 \times 10^{-5}$ | $867$, $867$ |

Level: exact, with the same number of sweeps.

## Table 5: T46

| $N$ | Published | Ours | As printed | Ratio | Red-black (ours) | Converged | Iter (pub., ours) |
|---|---|---|---|---|---|---|---|
| $8$ | $1.7 \times 10^{-2}$ | $1.854 \times 10^{-2}$ | **$1.9 \times 10^{-2}$** | $1.09$ | $1.523 \times 10^{-2}$ | $1.712 \times 10^{-2}$ | $2$, $1$ |
| $16$ | $3.3 \times 10^{-3}$ | $4.999 \times 10^{-3}$ | **$5.0 \times 10^{-3}$** | $1.51$ | $4.104 \times 10^{-3}$ | $4.304 \times 10^{-3}$ | $1$, $1$ |
| $32$ | $8.1 \times 10^{-4}$ | $1.296 \times 10^{-3}$ | **$1.3 \times 10^{-3}$** | $1.60$ | $1.042 \times 10^{-3}$ | $1.078 \times 10^{-3}$ | $1$, $1$ |
| $64$ | $2.1 \times 10^{-4}$ | $3.276 \times 10^{-4}$ | **$3.3 \times 10^{-4}$** | $1.56$ | $2.621 \times 10^{-4}$ | $2.695 \times 10^{-4}$ | $1$, $1$ |
| $128$ | $5.2 \times 10^{-5}$ | $8.218 \times 10^{-5}$ | **$8.2 \times 10^{-5}$** | $1.58$ | $6.568 \times 10^{-5}$ | $6.738 \times 10^{-5}$ | $1$, $1$ |

Orders: published $2.07$, ours $1.96$.
Level: order of magnitude (Finding 1).

| $N$ | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| $8$ | $1.7 \times 10^{-2}$ | $1.712 \times 10^{-2}$ | $82$, $82$ |
| $16$ | $4.3 \times 10^{-3}$ | $4.304 \times 10^{-3}$ | $296$, $296$ |
| $32$ | $1.1 \times 10^{-3}$ | $1.073 \times 10^{-3}$ | $954$, $954$ |

Level: exact, with the same number of sweeps.

## Table 6: T28

| $N$ | Published | Ours | As printed | Ratio | Red-black (ours) | Converged |
|---|---|---|---|---|---|---|
| $8$ | $5.3 \times 10^{-4}$ | $4.386 \times 10^{-4}$ | **$4.4 \times 10^{-4}$** | $0.83$ | $4.160 \times 10^{-4}$ | $4.040 \times 10^{-4}$ |
| $16$ | $2.1 \times 10^{-4}$ | $1.673 \times 10^{-4}$ | **$1.7 \times 10^{-4}$** | $0.80$ | $1.555 \times 10^{-4}$ | $1.438 \times 10^{-4}$ |
| $32$ | $7.4 \times 10^{-5}$ | $6.087 \times 10^{-5}$ | **$6.1 \times 10^{-5}$** | $0.82$ | $5.553 \times 10^{-5}$ | $5.090 \times 10^{-5}$ |
| $64$ | $2.6 \times 10^{-5}$ | $2.171 \times 10^{-5}$ | **$2.2 \times 10^{-5}$** | $0.84$ | $1.967 \times 10^{-5}$ | $1.800 \times 10^{-5}$ |
| $128$ | $9.3 \times 10^{-6}$ | $7.691 \times 10^{-6}$ | **$7.7 \times 10^{-6}$** | $0.83$ | $6.957 \times 10^{-6}$ | $6.364 \times 10^{-6}$ |

Orders: published $1.47$, ours $1.46$; one cycle at every $N$ in both.
Level: order of magnitude.
Here the published errors are above ours, by a constant $20$ percent, and $31$ to $46$ percent above the discrete solution's, unlike Tables 3 and 5 (Finding 1).

| $N$ | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| $8$ | $4.0 \times 10^{-4}$ | $4.040 \times 10^{-4}$ | $86$, $86$ |
| $16$ | $1.4 \times 10^{-4}$ | $1.438 \times 10^{-4}$ | $312$, $312$ |
| $32$ | $5.1 \times 10^{-5}$ | $5.091 \times 10^{-5}$ | $925$, $924$ |

Level: exact.

## Table 7: T22

| $N$ | Published | Ours | As printed | Ratio | Red-black (ours) | Converged |
|---|---|---|---|---|---|---|
| $8$ | $1.4 \times 10^{-3}$ | $1.392 \times 10^{-3}$ | $1.4 \times 10^{-3}$ | $0.99$ | $1.388 \times 10^{-3}$ | $1.417 \times 10^{-3}$ |
| $16$ | $1.4 \times 10^{-3}$ | $1.359 \times 10^{-3}$ | $1.4 \times 10^{-3}$ | $0.97$ | $1.358 \times 10^{-3}$ | $1.377 \times 10^{-3}$ |
| $32$ | $4.3 \times 10^{-3}$ | $1.087 \times 10^{-3}$ | **$1.1 \times 10^{-3}$** | $0.25$ | $1.084 \times 10^{-3}$ | $1.102 \times 10^{-3}$ |
| $64$ | $2.2 \times 10^{-3}$ | $2.357 \times 10^{-3}$ | **$2.4 \times 10^{-3}$** | $1.07$ | $2.628 \times 10^{-3}$ | $8.367 \times 10^{-4}$ |
| $128$ | $1.6 \times 10^{-3}$ | $1.987 \times 10^{-3}$ | **$2.0 \times 10^{-3}$** | $1.24$ | $4.059 \times 10^{-3}$ | ($6.372 \times 10^{-4}$) |

Orders: published $-0.10$, ours $-0.18$; one cycle at every $N$ in both.
Level: mismatch (Finding 4), since the ratio at $N = 32$ is $0.25$; at the other four $N$ it lies in $[0.97, 1.24]$.

| $N$ | Gauss-Seidel published | Ours | Sweeps (pub., ours) |
|---|---|---|---|
| $8$ | $1.4 \times 10^{-3}$ | $1.417 \times 10^{-3}$ | $75$, $75$ |
| $16$ | $1.4 \times 10^{-3}$ | $1.377 \times 10^{-3}$ | $266$, $266$ |
| $32$ | $1.1 \times 10^{-3}$ | $1.102 \times 10^{-3}$ | $854$, $851$ |

Level: exact.
At $N = 128$ the cycles do not settle on a discrete solution (Finding 4); the "converged" value in parentheses is the tenth red-black cycle, not a limit.

## Finding 1: on the smooth 3D examples one published FMG cycle lands below the discrete solution's error

On T9 and T46 the published FMG errors lie below the error of the discrete solution, by $9$ to $26$ percent on T9 from $N = 16$ to $64$ ($2.7 \times 10^{-4}$ against $2.98 \times 10^{-4}$, $5.7 \times 10^{-5}$ against $7.55 \times 10^{-5}$, $1.4 \times 10^{-5}$ against $1.89 \times 10^{-5}$) and by $22$ to $25$ percent on T46 at every $N$ from $16$ ($3.3 \times 10^{-3}$ against $4.30 \times 10^{-3}$, $8.1 \times 10^{-4}$ against $1.08 \times 10^{-3}$, $2.1 \times 10^{-4}$ against $2.70 \times 10^{-4}$, $5.2 \times 10^{-5}$ against $6.74 \times 10^{-5}$); at $N = 128$ on T9 the published value is above it ($6.2 \times 10^{-6}$ against $4.73 \times 10^{-6}$), where the paper notes round-off.
The discrete solution is the one the paper solves: its own Gauss-Seidel column, which stops near it, agrees with ours to the printed digits and to the sweep at every $N$, a sweep-by-sweep test of the scheme, the root selection and the lexicographic order.
So the published FMG iterate carries an algebraic error that cancels part of the discretisation error, while ours, after the same one cycle with the lexicographic order, lies $30$ to $49$ percent (T9) and $8$ to $22$ percent (T46) above the discrete solution's error.
Our red-black cycle shows that the sign of the algebraic error depends on the details: on T46 it lands $3$ to $11$ percent below the discrete solution's error, on T9 $21$ to $27$ percent above.
The paper leaves one choice open that moves this: the "cubic interpolation" of the FMG step.
Ours is the tensor-product not-a-knot spline (`cubic_resample`); the 4-point cubic interpolation, which takes $(-1, 9, 9, -1)/16$ at an inner midpoint (Keys' cubic convolution with $a = -1/2$, the "cubic" of MATLAB's `interp2` and `interp3` on a uniform grid, so a natural reading of a MATLAB paper) and the cubic through the $4$ nodes next to an end at the midpoint there, $(5, 15, -5, 1)/16$ (MATLAB's end rule may differ; we did not check it), gives after the same one cycle (`multigrid_checks.csv`, variant `keys`, `KeysFMG` in `run_multigrid.py`):

| $N$ | T9 published | T9 spline (ours) | T9 4-point | T9 converged | T46 published | T46 spline (ours) | T46 4-point | T46 converged |
|---|---|---|---|---|---|---|---|---|
| $8$ | $1.2 \times 10^{-3}$ | $1.507 \times 10^{-3}$ | $1.351 \times 10^{-3}$ | $1.157 \times 10^{-3}$ | $1.7 \times 10^{-2}$ | $1.854 \times 10^{-2}$ | $1.614 \times 10^{-2}$ | $1.712 \times 10^{-2}$ |
| $16$ | $2.7 \times 10^{-4}$ | $4.107 \times 10^{-4}$ | $3.295 \times 10^{-4}$ | $2.977 \times 10^{-4}$ | $3.3 \times 10^{-3}$ | $4.999 \times 10^{-3}$ | $4.045 \times 10^{-3}$ | $4.304 \times 10^{-3}$ |
| $32$ | $5.7 \times 10^{-5}$ | $1.083 \times 10^{-4}$ | $7.856 \times 10^{-5}$ | $7.551 \times 10^{-5}$ | $8.1 \times 10^{-4}$ | $1.296 \times 10^{-3}$ | $1.005 \times 10^{-3}$ | $1.078 \times 10^{-3}$ |
| $64$ | $1.4 \times 10^{-5}$ | $2.786 \times 10^{-5}$ | $1.894 \times 10^{-5}$ | $1.891 \times 10^{-5}$ | $2.1 \times 10^{-4}$ | $3.276 \times 10^{-4}$ | $2.501 \times 10^{-4}$ | $2.695 \times 10^{-4}$ |
| $128$ | $6.2 \times 10^{-6}$ | $7.065 \times 10^{-6}$ | $4.622 \times 10^{-6}$ | $4.730 \times 10^{-6}$ | $5.2 \times 10^{-5}$ | $8.218 \times 10^{-5}$ | $6.250 \times 10^{-5}$ | $6.738 \times 10^{-5}$ |

The ratios to the published values drop from $1.26$, $1.52$, $1.90$, $1.99$ and $1.14$ to $1.13$, $1.22$, $1.38$, $1.35$ and $0.75$ on T9, and from $1.09$, $1.51$, $1.60$, $1.56$ and $1.58$ to $0.95$, $1.23$, $1.24$, $1.19$ and $1.20$ on T46.
From $N = 16$ to $64$ the 4-point interpolation closes $56$ to $66$ percent of the gap between ours and the published value, so the interpolation explains about half of it, and it decides whether Table 3 sits at the edge of the order-of-magnitude level (ratio $1.99$ at $N = 64$) or well inside it ($1.35$).
On T28, T1 and T3 it changes the error by less than $2$ percent (T28 $+0.1$ to $+0.3$, T1 $-1.5$ to $-1.9$, T3 $-0.3$ percent).
The rest of the gap stays open: with the 4-point interpolation one cycle on T9 still lies above the discrete solution's error up to $N = 64$ ($1.894 \times 10^{-5}$ against $1.891 \times 10^{-5}$ there), where the published value lies $26$ percent below it, and on T46 it lies $6$ to $7$ percent below it, against the published $22$ to $25$ percent; nothing else the paper states (the 3D sweep is lexicographic, Section 2) differs from ours.
On T28 the published errors lie $31$ to $46$ percent above the discrete solution's and ours $9$ to $21$ percent above it; in 2D, where the same paper's FMG errors also lie above the discrete solution's (T1: $4.5 \times 10^{-6}$ against $4.20 \times 10^{-6}$ at $N = 128$), ours agree with the published ones to within $8$ percent.
The receipts use the spline; whether `cubic_resample` should become the 4-point interpolation, which would move the Table 3 and 5 columns to the values above, is open (PR #68).

## Finding 2: the reference of the relative residual

The paper stops at $\lVert r_l \rVert_2 / \lVert r_0 \rVert_2 \le 10^{-6}$ and does not say what $r_0$ is for the FMG-FAS method, which has no initial guess on the finest grid.
Ours is the residual of the node values that are $0$ inside, with the Dirichlet data on the ring (the iterate from which the coarsest grid starts), a residual that grows like $N^5$ in 3D at the nodes next to the ring.
With it the iteration counts on T2 match Table 1 in all seven cells and Table 2 in five (ours $2$ against $3$ at $N = 128$, $2$ against $1$ at $512$), and every other example takes one cycle, as published, except T46 at $N = 8$ (ours $1$, published $2$).
The relative residuals after the cycle do not match: in 3D the printed value divided by ours runs from $0.1$ (T46, $N = 8$, where the paper takes a second cycle) to $220$ (T9, $N = 128$); it grows with $N$ on T9 ($3.7$ to $220$) and T46 ($3.0$ to $60$ from $N = 16$), stays at $3$ to $4$ on T28 and between $0.9$ and $11$ on T22, so on the smooth examples the paper's reference grows more slowly with $N$ than ours.
For the stand-alone Gauss-Seidel solver the natural reference, the residual of the start $1.01\,u$, reproduces the published sweep counts exactly, so the norm itself (the plain L2 norm over the unknowns) is the paper's.
The stopping rule decides only whether a second cycle runs; where it does (T2 at $N = 128$ and $256$, and $512$ lexicographic), the error changes by $0.4$ to $1.7$ percent.
Because the reference grows with $N$, the rule says little about the iterate on fine grids, so `FDMultigrid` stops on another one, and the receipt applies the paper's rule only to compare with the paper (Finding 5).

## Finding 3: the smallest real root also where the coarse right-hand side is negative

The theorems of Section 2 select the root with a positive definite Hessian for $f > 0$; the right-hand side of a coarse grid in a FAS cycle, $S_H(I w) + I r$, can be negative next to a singularity, where $r$ is large.
The paper's rule, "the smallest of all the real roots" (eq. (12)), is what `smallest_root` does there; it leaves that coarse node with an indefinite Hessian, which only the coarse correction sees.
The alternative of clamping $b$ at $0$ (the positive semidefinite limit) stalls the cycles on T2 (`multigrid_checks.csv`): the relative residual stops at $1.9 \times 10^{-5}$ ($N = 128$) to $1.7 \times 10^{-6}$ ($N = 1024$) and none of the $8$ clamped runs reaches $10^{-6}$ in $50$ cycles, with errors of $7.1 \times 10^{-3}$ to $1.1 \times 10^{-2}$ against the paper's $3.2 \times 10^{-3}$ to $8.9 \times 10^{-3}$, while the smallest real root converges in $1$ or $2$ cycles to the published errors.

## Finding 4: T22 at $N = 32$, and the cycles next to the corner

The published FMG error on T22 jumps from $1.4 \times 10^{-3}$ at $N = 16$ to $4.3 \times 10^{-3}$ at $32$ and falls again ($2.2 \times 10^{-3}$, $1.6 \times 10^{-3}$); the paper's own Gauss-Seidel column, and ours, give $1.1 \times 10^{-3}$ at $32$, and our FMG cycle gives $1.09 \times 10^{-3}$, within $1.4$ percent of the discrete solution's error, so the published value is $4$ times all three.
The published value is not a lone misprint of the error cell: the paper's Order column prints $-1.7$ at $N = 32$ (and $1.0$ at $64$), which is what $4.3 \times 10^{-3}$ gives against its neighbours ($\log_2(1.4 \times 10^{-3} / 4.3 \times 10^{-3}) = -1.6$ from the rounded values).
So Table 7 is a mismatch; the paper prints nothing that would explain it.

At $N = 64$ and $128$ our FMG cycle stops above the discrete solution's error too ($2.36 \times 10^{-3}$ against $8.37 \times 10^{-4}$ at $64$), with a residual of order $10^3$ to $10^4$ at the nodes next to the corner $(1, 1, 1)$, where $f$ grows like $h^{-5}$, and a few nodes with an indefinite discrete Hessian there (non-convex fraction $7.2 \times 10^{-5}$ at $64$, $1.1 \times 10^{-5}$ at $128$); the relative residual meets $10^{-6}$ because its reference is dominated by the ring.
`multigrid_checks.csv` runs more cycles: at $N = 64$ they converge (the error $8.3671 \times 10^{-4}$ from cycle $10$ on, the residual's max $10^{-9}$ from cycle $20$), at $N = 128$ they do not.
There, at the sampled cycles $5$, $10$, $15$, $20$ and $30$, the error moves between $6.4 \times 10^{-4}$ and $2.0 \times 10^{-3}$ and the residual's max stays between $3 \times 10^{2}$ and $1.2 \times 10^{5}$, with no downward trend in either ordering.
Some samples come back close to an earlier one: with red-black sweeps cycles $15$ and $30$ agree to about $6$ digits (error $1.839861 \times 10^{-3}$ and $1.839860 \times 10^{-3}$, residual max $940.9755$ and $940.9756$), with lexicographic ones cycles $10$ and $30$ ($1.120648 \times 10^{-3}$ and $1.120645 \times 10^{-3}$, $9109.566$ and $9109.572$); red-black cycles $5$ and $20$ do not ($1.80 \times 10^{-3}$ and $1.89 \times 10^{-3}$).
Five samples cannot tell whether the iterates are periodic, nearly periodic or wander.
So at $N = 128$ the multigrid iteration does not converge near the corner within $30$ cycles, and the published $1.6 \times 10^{-3}$ and our $1.99 \times 10^{-3}$ are both points on such a path; the "converged" value of Table 7 at $128$ is the tenth red-black cycle.
`FDMultigrid`'s own stopping rule sees this: at $N = 128$ it runs its $50$ cycles and reports `converged = False` in both orderings, while at $N = 64$ it stops after $5$ (red-black) or $6$ (lexicographic) cycles at the discrete solution's error, $8.367 \times 10^{-4}$ (Finding 5).
Whether a smaller stopping step near the corner, more sweeps there, or another coarse-grid treatment would converge is not tested here.

T2, the 2D version, converges ($10$ cycles bring the relative residual to $10^{-11}$), but at $N \ge 512$ the cycle that meets the tolerance leaves a residual of up to $3 \times 10^{8}$ at the corner node ($f$ there is about $1/(8 h^2){}$) and non-convex fractions of $2 \times 10^{-6}$ to $5 \times 10^{-5}$, while the error is within $2$ percent of the discrete solution's.

## Finding 5: the paper's stopping rule against `FDMultigrid`'s

The paper's rule tests the relative residual, whose reference grows like $N^{2d}$ (Finding 2), so on fine grids it holds after the FMG cycle whatever the iterate: on T22 at $N = 128$ it holds while the residual is $10^3$ to $10^5$ near the corner and the cycles do not converge (Finding 4), and the earlier generation of `FDMultigrid`, which stopped by it, reported `converged = True` there.
`FDMultigrid` now stops when the RMS of the residual over the unknowns is at most `tol` $\cdot \max(1, \max \lvert f \rvert)$ with `tol` $= 10^{-6}$, and reports the relative residual in its history only (`docs/methods/multigrid.md`, Stopping rule).
On the paper's examples (the `default_*` columns; error change against the paper's rule, which takes $1$ or $2$ cycles):

| Table | Problem | Cycles, red-black | Cycles, lexicographic | Error change |
|---|---|---|---|---|
| $1$, $2$ | T1 | $1$ ($2$ at $N = 128$) | $1$ ($2$ at $N = 128$) | $-6$ and $-20$ percent at $N = 128$, none elsewhere |
| $1$, $2$ | T3 | $1$ to $3$ | $2$ or $3$ | $-5$ to $0$ (red-black), $-15$ to $-14$ percent (lexicographic) |
| $1$, $2$ | T2 | $6$ to $10$ | $7$ to $14$ | $+0.1$ to $+1.8$ percent |
| $3$ | T9 | $2$ to $4$ | $2$ to $4$ | $-21$ to $-17$ (red-black), $-30$ to $-23$ percent (lexicographic) |
| $5$ | T46 | $2$ or $3$ | $2$ to $4$ | $+3$ to $+12$ (red-black), $-18$ to $-8$ percent (lexicographic) |
| $6$ | T28 | $2$ or $3$ | $3$ or $4$ | $-9$ to $-3$ (red-black), $-17$ to $-8$ percent (lexicographic) |
| $7$ | T22 | $3$ to $5$; $50$ at $N = 128$, not converged | $4$ to $6$; $50$ at $N = 128$, not converged | $+1$ to $+2$ percent to $N = 32$, $-64$ to $-68$ percent at $64$ |

Every run but T22 at $N = 128$ converges, and its error then lies within $6$ percent of the discrete solution's, or $28$ percent above it where one FMG cycle already meets the bound (T1, lexicographic, from $N = 256$).
The "Iter" columns of the tables above are the paper's rule, not `FDMultigrid`'s cycle counts, and Table 7 at $N = 64$ compares the iterate the paper's rule stops at ($2.36 \times 10^{-3}$), not the discrete solution `FDMultigrid` reaches ($8.37 \times 10^{-4}$).

## Conventions

- Grid: $N$ cells per side, $h = 1/N$ ($\pi / N$ on T46); the paper defines its grid this way (Section 2), so ADR 0001 is the paper's own statement here.
- The paper does not say which cubic interpolation the FMG step uses, nor what $r_0$ is (Findings 1, 2).
- Algorithm 1, step (4), prints two restrictions of the fine iterate: $u_H := I_h^H w_h$, half-weighting, as the start of the coarse cycle, and $w_H := \tilde I_h^H w_h$, injection, for the FAS right-hand side $S_H(w_H) + r_H$ and the correction $u_H - w_H$.
  Ours starts the coarse cycle from the injected $w_H$ too.
  From the half-weighted start the discrete solution is not a fixed point of the cycle: on T1 (lexicographic) one FMG cycle stops at an error of $2.23 \times 10^{-4}$, $2.20 \times 10^{-4}$ and $2.19 \times 10^{-4}$ at $N = 128$, $256$ and $512$, against $5.36 \times 10^{-6}$, $1.34 \times 10^{-6}$ and $3.36 \times 10^{-7}$ from the injected start, and on T9 at $7.4 \times 10^{-4}$ at $N = 64$ against $2.79 \times 10^{-5}$ (`multigrid_checks.csv`, variant `half-weighted-start`).
- Table 3 prints a DOFs column, $216$ to $2{,}000{,}376$, which is $(N - 2)^3$; the grid of Section 2 ($x_i = i h$, $h = 1/N$, the Dirichlet data at $i = 0$ and $N$) has $(N - 1)^3$ unknowns, and Table 4's DOFs column is $(N - 1)^3$.
  The Gauss-Seidel columns settle it: on $h = 1/N$ with $(N - 1)^3$ unknowns ours match them to the printed digits and to the sweep, so we read Table 3's DOFs as a misprint.
- The red-black order updates each colour at once; with the diagonal neighbours of a 9- or 19-point stencil in the same colour this is a Jacobi step within the colour, as in the paper's vectorised MATLAB (Section 4.1).
- In 3D the paper sweeps lexicographically only; our red-black columns are ours.

## Iterations and time

The time is `Result.info["time"]` (the discretisation of every grid, the cycles and the solution fields, the spline through the node values among them) of the run at the paper's stopping rule, on one thread of an Intel Xeon Platinum 8358 of a100server, eight runs at a time on a machine shared with other jobs; the paper's is MATLAB on an Intel Core i3-3120M laptop.
In 2D the red-black time grows by a factor of $4.1$ to $6.0$ per halving of $h$ from $N = 1024$ on, about the factor $4$ of the unknowns, as the paper's does; the lexicographic one by $2.9$ to $4.3$, since the fixed cost of each of its $3 N$ vectorised steps dominates below $N = 8192$.
At $N = 8192$ ours takes $155$ s (red-black) and $117$ s (lexicographic) on T1 against the paper's $67$ s and $246$ s.
In 3D ours grows by $2$ to $4$ per halving up to $N = 128$, less than the factor $8$ of the unknowns, for the same reason; at $N = 128$ it takes $9$ to $14$ s against the paper's $133$ to $137$ s.
`FDMultigrid`'s own rule takes more cycles (Finding 5): at $N = 8192$ on T2, $643$ s (lexicographic, $14$ cycles) and $699$ s (red-black, $10$ cycles).
The times are recorded, not compared (ADR 0002).

## Table 4: T9 against the wide-stencil scheme

Table 4 reprints, for Example 4.4 (T9), the "Wide-stencil FD" errors of Froese-Oberman (J. Comput. Phys. 2011, its Table 4, the 3D hybrid scheme) and the quadratic and cubic interior-penalty elements of Brenner-Neilan (2012); it is not a target of this method.
Its DOFs ($343$ to $29791$) are $(N - 1)^3$, so on its grids $N$ counts cells, $h = 1/N$: the Froese-Oberman grids of $N = 7$ to $31$ read as $N - 1 = 7$ to $31$ interior nodes per side, a third reading of their $N$ besides the two of `monotone19.md` (Finding 1 there).
This reading is tentative: the DOF columns of this paper are not consistent with each other, since Table 3's are $(N - 2)^3$ (Conventions), so they are weak evidence of how Liu et al. read Froese-Oberman's $N$.
`multigrid_table4.csv` (`run_multigrid.py --table4 --workers 3` at commit 3d31695, clean checkout, MLflow `maesolvers-multigrid-T9`, runs `table4-n<N>`) solves T9 on these grids with the 19-point monotone scheme of #27, `FDMonotone(n=N, stencil=19)`, and with `FDMultigrid(n=N)` and its stopping rule ($3$ or $4$ cycles):

| $N$ | DOFs | Wide-stencil FD (reprinted) | `FDMonotone(stencil=19)` | Ratio | `FDMultigrid` |
|---|---|---|---|---|---|
| $8$ | $343$ | $1.51 \times 10^{-2}$ | $1.466 \times 10^{-2}$ | $0.97$ | $1.158 \times 10^{-3}$ |
| $12$ | $1331$ | $1.40 \times 10^{-2}$ | $1.325 \times 10^{-2}$ | $0.95$ | $5.243 \times 10^{-4}$ |
| $16$ | $3375$ | $1.29 \times 10^{-2}$ | $1.283 \times 10^{-2}$ | $0.99$ | $2.986 \times 10^{-4}$ |
| $22$ | $9261$ | $1.21 \times 10^{-2}$ | $1.263 \times 10^{-2}$ | $1.04$ | $1.597 \times 10^{-4}$ |
| $32$ | $29791$ | $1.11 \times 10^{-2}$ | $1.246 \times 10^{-2}$ | $1.12$ | $7.582 \times 10^{-5}$ |

The monotone scheme lies within $12$ percent of the reprinted hybrid values, which is all a cross-check can ask of two different schemes, and stalls near $1.25 \times 10^{-2}$, the limit its angular resolution sets (`monotone19.md`); the standard scheme under multigrid is $13$ to $164$ times more accurate on this smooth solution, as the paper argues.
The interior-penalty columns wait for the 0.3 finite elements.

## Not compared

- The Gauss-Seidel CPU times, and the Gauss-Seidel columns beyond $N = 32$, which the paper does not print.
