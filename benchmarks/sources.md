# Published finite-difference benchmarks for the Monge-Ampere equation

This file holds the published finite-difference tables that maesolvers reproduces, in 2D and 3D, with Dirichlet data and with the transport boundary condition: sections 1 to 9 transcribe them here.
Most of the 2D Dirichlet tables share two problems, $\det D^2 u = f$ on $(0,1)^2$ with Dirichlet data from the exact solution:
solution (a), T1: $u = \exp(\lvert x \rvert^2 / 2)$, $f = (1 + \lvert x \rvert^2) \exp(\lvert x \rvert^2)$;
solution (b), T2: $u = -\sqrt{2 - \lvert x \rvert^2}$, $f = 2 (2 - \lvert x \rvert^2)^{-2}$, gradient unbounded at the corner $(1,1)$.

All numbers below were copied from the journal versions, read from local PDF copies in the Zotero library.
Each value was checked against a rendered image of the printed page.
The arXiv and Numdam links are public equivalents; their tables were not checked against the journal tables.
Values keep the printed scaling (for example $24.45 \times 10^{-5}$, not $2.445 \times 10^{-4}$).
A "--" means the paper printed a dash: the run was not done, or it was too slow to report.
The neural sources are transcribed next to their comparison: Nystrom-Vestberg 2023 in `penalty_mlp.md` (the penalty MLP) and `icnn.md` (the input convex network), and Caboussat-Peruso 2025 in `icnn.md` (T12).

## Grid convention (settled by ADR 0001)

ADR 0001 settles the convention at $h = 1/N$, $N$ cells per side: on T1 at $N = 31$ only $h = 1/N$ matches the published $7.14 \times 10^{-5}$ (`standard_fd.md`, Finding 1).
The quotes below are what the papers say; the last two lines list the tables that count nodes instead.

Benamou-Froese-Oberman 2010, Section 3.3, says verbatim: "The solutions are computed on an $N \times N$ grid ... By convention, $h$ is the spatial discretization parameter $h = L/N$ where $L$ is the length of one side of the square domain."
Froese-Oberman JCP 2011, Section 8, says only: "These computations are performed on an $N \times N$ grid on the square $[0, 1]^2$."
Froese-Oberman SIAM 2011, Section 6, says only: "We perform the computations using a 17 point stencil on an $N \times N$ grid on a square."
SIAM Table 2 adds: "Here $M = N^2$ is the total number of grid points."
None of the 2011 papers defines $h$ in terms of $N$.
The odd $N$ values ($21$, $41$, ...; $31$, $63$, $127$, $255$) suggest $N$ points per side including the boundary, so $h = 1/(N-1)$, but no paper says so.
The check that decided it: the standard-scheme max error for (a) at $N = 31$ is $7.14 \times 10^{-5}$, which tells $h = 1/(N-1)$ apart from $h = 1/N$ or $h = 1/(N+1)$.
One table does not follow $h = L/N$: on $[-1, 1]^2$, BFO 2010 Table 7 matches our solvers at $n = N - 1$ cells (`standard_fd.md`, Finding 3).
Froese-Oberman 2013 (section 5 below) counts nodes too, in every table: its values match $n = N - 1$ cells (`filtered.md`, Finding 1).

## 1. Benamou, Froese, Oberman (2010)

J.-D. Benamou, B. D. Froese, A. M. Oberman, "Two numerical methods for the elliptic Monge-Ampere equation", ESAIM: M2AN 44 (2010) 737-758, DOI 10.1051/m2an/2010017.
Read: local journal PDF (Zotero item RLL6V439).
Open copy: https://www.numdam.org/item/M2AN_2010__44_4_737_0 (Numdam, not checked).
No arXiv version was found.

Methods: both solve the same standard 9-point central-difference discretisation $(D_{xx} u)(D_{yy} u) - (D_{xy} u)^2 = f$, with $D_{xy}$ on the four diagonal neighbours divided by $4h^2$.
(M1) is the explicit Gauss-Seidel iteration: at each point solve the local quadratic for $u_{ij}$ and take the smaller root.
(M2) is the Poisson iteration: $\Delta u^{n+1} = \sqrt{(u^n_{xx})^2 + (u^n_{yy})^2 + 2 (u^n_{xy})^2 + 2f}$, with central differences and Dirichlet data.
The paper states that both methods reach the same discrete solution, so the error depends only on $N$ (Section 3, item 1).
Stopping rule: the max difference between two successive iterates is less than $10^{-14}$.
Initial guess for the timings: the solution of $u_{xx} + u_{yy} = \sqrt{2f}$.

Name map for this paper:
- Solution (a) is Section 4.1, "An exact radial solution".
- Solution (b) is Section 5.4, "Surface of ball", on the domain $[0,1] \times [0,1]$.
- Watch out: Section 4.2 is titled "Blow-up at boundary", but it is a different solution, $u = (2 \sqrt{2}/3)(x^2+y^2)^{3/4}$, $f = 1/\sqrt{x^2+y^2}$, where $f$ blows up at the corner $(0,0)$. Its Table 3 is not a table for (b).

The paper prints no error table for (a) or (b).
For (a), the error is only shown in Figure 2(a), and the text says the accuracy is $O(h^2)$.
For (b), the error is only shown in Figure 8(b), with the label "Slope $\sim -0.45$", and the text says "the convergence rate is about $.5$".
The tables for (a) and (b) give iteration counts and CPU time only.

### Solution (a): Table 2, computation times on an $N \times N$ grid

The paper skips rows between $N = 261$ and $N = 341$ (printed as a vertical ellipsis).

| $N$ | Iterations (M1) | Iterations (M2) | CPU time s (M1) | CPU time s (M2) |
|---|---|---|---|---|
| $21$ | $1014$ | $54$ | $0.9$ | $0.1$ |
| $81$ | $14\thinspace 621$ | $59$ | $39.8$ | $3.4$ |
| $141$ | $42\thinspace 746$ | $60$ | $350.0$ | $11.7$ |
| $201$ | $84\thinspace 611$ | $60$ | $1540.0$ | $27.2$ |
| $261$ | -- | $65$ | -- | $57.8$ |
| ... | ... | ... | ... | ... |
| $341$ | -- | $65$ | -- | $118.0$ |
| $421$ | -- | $78$ | -- | $249.0$ |
| $501$ | -- | $151$ | -- | $804.0$ |
| $581$ | -- | $284$ | -- | $2120.0$ |

Table 1 also gives the CPU time ratio (M1)/(M2) for (a) (Section 4.1 column): $6.3$, $4.1$, $7.7$, $11.7$, $14.1$, $23.5$, $29.9$ at $N = 21$, $41$, $61$, $81$, $101$, $121$, $141$.

### Solution (b): Table 9, computation times on an $N \times N$ grid

| $N$ | Iterations (M1) | Iterations (M2) | CPU time s (M1) | CPU time s (M2) |
|---|---|---|---|---|
| $21$ | $1009$ | $67$ | $0.2$ | $0.2$ |
| $41$ | $3833$ | $79$ | $1.9$ | $1.0$ |
| $61$ | $8343$ | $89$ | $8.5$ | $2.7$ |
| $81$ | $14\thinspace 480$ | $99$ | $26.3$ | $5.2$ |
| $101$ | $22\thinspace 189$ | $106$ | $60.2$ | $10.0$ |
| $121$ | $31\thinspace 445$ | $113$ | $120.1$ | $15.0$ |
| $141$ | $42\thinspace 202$ | $119$ | $218.3$ | $21.3$ |

### T7 (Section 5.3): Table 7, minimum value of $u$ when $f = 1$ with constant boundary values $1$ on an $N \times N$ grid

| $N$ | (M1), (M2) | 9-point stencil | 17-point stencil | 33-point stencil |
|---|---|---|---|---|
| $21$ | $0.2892$ | $0.3115$ | $0.2815$ | $0.2839$ |
| $41$ | $0.2734$ | $0.3090$ | $0.2807$ | $0.2732$ |
| $61$ | $0.2682$ | $0.3082$ | $0.2803$ | $0.2711$ |
| $81$ | $0.2655$ | $0.3078$ | $0.2802$ | $0.2704$ |
| $101$ | $0.2639$ | $0.3076$ | $0.2800$ | $0.2700$ |
| $121$ | $0.2629$ | $0.3075$ | $0.2798$ | $0.2697$ |
| $141$ | $0.2621$ | $0.3074$ | $0.2796$ | $0.2695$ |

### T7 (Section 5.3): Table 8, computation times

| $N$ | Iterations (M1) | Iterations (M2) | CPU time (s) (M1) | CPU time (s) (M2) |
|---|---|---|---|---|
| $21$ | $1615$ | $535$ | $0.1$ | $1.7$ |
| $41$ | $6366$ | $1917$ | $0.6$ | $33.6$ |
| $61$ | $14\thinspace 049$ | $4070$ | $2.8$ | $254.0$ |
| $81$ | $24\thinspace 560$ | $6944$ | $8.1$ | $1170.0$ |
| $101$ | $37\thinspace 824$ | $10\thinspace 503$ | $19.9$ | $3730.0$ |
| $121$ | $53\thinspace 778$ | $14\thinspace 711$ | $41.2$ | $6820.0$ |
| $141$ | $72\thinspace 377$ | $19\thinspace 571$ | $99.7$ | $8560.0$ |

Our comparison, and the node-count convention of Table 7: `standard_fd.md`, section "T7" and Finding 3.

### Other test solutions in this paper

- Section 4.2: $u = (2 \sqrt{2}/3)(x^2+y^2)^{3/4}$ on $[0,1]^2$, $f = 1/\sqrt{x^2+y^2}$; max error in Table 3 ($5.5 \times 10^{-4}$ at $N = 21$ to $1.5 \times 10^{-5}$ at $N = 221$), order $1.5$.
- Section 5.1: solution (a) with high-frequency sinusoidal noise added to $f$ and to the boundary data (Table 5, times only).
- Section 5.2: discontinuous $f$ ($1$ outside a circle of radius $0.16$ about $(0.5,0.5)$, $0$ inside), $u = \max(\lvert x - (0.5,0.5) \rvert^2 / 2, 0.08)$ (Table 6, times only).
- Section 5.3 (T7): $f = 1$ with boundary value $1$ on $[-1,1]^2$, no exact solution; Tables 7 and 8 above; the 9-, 17- and 33-point columns of Table 7 are the monotone wide-stencil method of the paper's reference [21].
- Section 6.1: the cone $u = \sqrt{x^2+y^2}$, $f = 4/h^2$ at the origin grid point; max error in Tables 10 and 13, order about $0.8$.
- Section 6.2: $u = \lvert x \rvert$, $f = 0$; the discrete error goes to zero (Table 12, times only).

### Caveat on the printed solution (maesolvers catalogue, T4)

- Section 5.2 (T4): the printed $u = \max(\lvert x - x_0 \rvert^2 / 2, 0.08)$ does not solve the printed problem.
  Its gradient jumps from $0$ to the radial vector of length $0.4$ on the circle $r = 0.4$, so its Monge-Ampere measure has a singular part of mass $0.16 \pi$ on that circle, and the printed $f$ (integral $1 - 0.16 \pi$ on the unit square) omits it.
  A correct solver of the printed $f$ with the trace of the printed $u$ as Dirichlet data converges to a different function, with no known closed form, so the printed $u$ is no reference for an error.
  The catalogue's T4 keeps the printed $f$ and replaces $u$ with the $C^1$ radial Alexandrov solution of that $f$, $0.08$ on the disk and $0.08 + (r s - 0.16 \ln((r + s)/0.4))/2$ with $s = \sqrt{r^2 - 0.16}$ beyond, and takes its Dirichlet data from it; so T4's boundary data differ from the paper's.
  `tests/test_catalogue.py` checks that the area of $\nabla u(\Omega)$ equals the integral of $f$, which fails for the printed $u$ (area $1$ against $1 - 0.16 \pi$).

## 2. Froese, Oberman, SIAM J. Numer. Anal. (2011)

B. D. Froese, A. M. Oberman, "Convergent finite difference solvers for viscosity solutions of the elliptic Monge-Ampere equation in dimensions two and higher", SIAM J. Numer. Anal. 49 (2011) 1692-1714, DOI 10.1137/100803092.
Read: local journal PDF from the Zotero folder (two copies of the same journal version).
Open copy: https://arxiv.org/abs/1007.0765 (not checked).

Method: a wide-stencil monotone discretisation $MA_M[u] = \min$ over orthogonal grid-direction pairs $(\nu_1, \nu_2)$ of the product of the positive parts of $D_{\nu \nu} u$.
Section 3.5 defines a regularised scheme $MA^\delta$, with $\max^\delta(a, b) = (a + b + \sqrt{(a - b)^2 + \delta^2})/2$ and $\min^\delta$ likewise ($\delta^2$ read in arXiv 1007.0765; the journal text layer garbles it), the positive parts $\max^\delta(D u, 0)$ and the minimum over pairs a succession of $\min^\delta$; Theorem 9 proves that Newton converges for $MA^\delta$.
No value of $\delta$ is printed, and Section 5 states Newton for $MA_M$ itself, with Danskin's theorem for the Jacobian, so the tables need not come from $MA^\delta$.
The equations are solved by damped Newton, with the step reduced until the residual decreases.
The linear systems are solved by MATLAB backslash (sparse LU).
The initial guess is one step of the semi-implicit scheme, $\Delta u^0 = \sqrt{2f}$ in 2D, convexified only for the most singular example.
The Jacobian is regularised by $u_{\nu \nu} \to \max(u_{\nu \nu}, \epsilon)$, with $\epsilon = 10^{-8} / (2 dx^2)$ (Section 5.3, read from the page image; the text layer garbles it).
Section 6 uses a 17-point stencil for Table 1; Table 3 also uses 9- and 33-point stencils.
No Newton stopping tolerance is stated.
The "F.D." columns of Table 3 are the standard 9-point scheme "using the results of [5]" ([5] is Benamou-Froese-Oberman 2010).
That $N$ sequence does not appear in the 2010 paper's tables, so these values were recomputed for this paper, not copied from 2010.
The same applies to the Poisson and Gauss-Seidel CPU columns of Table 1.

Name map for this paper:
- Solution (a) is $C^2$ example, eq. (10).
- Solution (b) is "Example with blowup", eq. (12).

### Solution (a), $C^2$ example (10)

Table 3: max error.

| $N$ | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| $31$ | $17.9 \times 10^{-4}$ | $8.9 \times 10^{-4}$ | $7.0 \times 10^{-4}$ | $7.14 \times 10^{-5}$ |
| $63$ | $16.2 \times 10^{-4}$ | $5.1 \times 10^{-4}$ | $3.1 \times 10^{-4}$ | $1.73 \times 10^{-5}$ |
| $127$ | $15.9 \times 10^{-4}$ | $4.6 \times 10^{-4}$ | $1.8 \times 10^{-4}$ | $4.3 \times 10^{-6}$ |
| $255$ | $15.9 \times 10^{-4}$ | $4.4 \times 10^{-4}$ | $1.5 \times 10^{-4}$ | $1.1 \times 10^{-6}$ |
| $361$ | $15.9 \times 10^{-4}$ | $4.4 \times 10^{-4}$ | $1.5 \times 10^{-4}$ | $0.5 \times 10^{-6}$ |

Table 1: Newton on the 17-point monotone scheme; Poisson and Gauss-Seidel are the 2010 methods (M2) and (M1).

| $N$ | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| $31$ | $3$ | $0.2$ | $0.7$ | $2.2$ |
| $63$ | $6$ | $0.9$ | $1.9$ | $15.0$ |
| $127$ | $7$ | $4.9$ | $9.6$ | $236.7$ |
| $255$ | $7$ | $43.5$ | $52.6$ | -- |
| $361$ | $7$ | $172.8$ | $162.6$ | -- |

### Solution (b), "Example with blowup (12)"

Table 3: max error.

| $N$ | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| $31$ | $1.7 \times 10^{-3}$ | $1.7 \times 10^{-3}$ | $1.7 \times 10^{-3}$ | $17.15 \times 10^{-3}$ |
| $63$ | $0.9 \times 10^{-3}$ | $0.6 \times 10^{-3}$ | $0.6 \times 10^{-3}$ | $12.53 \times 10^{-3}$ |
| $127$ | $0.8 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | $0.2 \times 10^{-3}$ | $9.00 \times 10^{-3}$ |
| $255$ | $0.8 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | $0.2 \times 10^{-3}$ | $6.42 \times 10^{-3}$ |
| $361$ | $0.8 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | $0.2 \times 10^{-3}$ | $5.41 \times 10^{-3}$ |

Table 1: Newton on the 17-point monotone scheme.

| $N$ | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| $31$ | $4$ | $0.3$ | $0.5$ | $0.8$ |
| $63$ | $4$ | $0.5$ | $2.9$ | $19.4$ |
| $127$ | $5$ | $3.7$ | $17.7$ | $293.3$ |
| $255$ | $7$ | $41.4$ | $128.2$ | -- |
| $361$ | $9$ | $184.1$ | $374.5$ | -- |

Table 2 gives the CPU time scaling, with $M = N^2$: Gauss-Seidel about $O(M^{1.8})$ on $C^{2,\alpha}$ and $O(M^{1.9})$ on $C^{1,\alpha}$; Poisson about $O(M^{1.4})$, "blowup" on less regular solutions; Newton about $O(M^{1.3})$ for all regularities.

### $C^1$ example (11), T20 in the catalogue

Table 3: max error; the F.D. column prints a dash at $N = 255$ and $361$.

| $N$ | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | F.D. |
|---|---|---|---|---|
| $31$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-3}$ | $1.5 \times 10^{-3}$ | $2.6 \times 10^{-4}$ |
| $63$ | $2.5 \times 10^{-3}$ | $1.0 \times 10^{-3}$ | $0.6 \times 10^{-3}$ | $1.5 \times 10^{-4}$ |
| $127$ | $2.3 \times 10^{-3}$ | $0.8 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | $0.6 \times 10^{-4}$ |
| $255$ | $2.2 \times 10^{-3}$ | $0.7 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | -- |
| $361$ | $2.2 \times 10^{-3}$ | $0.7 \times 10^{-3}$ | $0.3 \times 10^{-3}$ | -- |

Table 1, Newton iterations on the 17-point scheme: $4$, $7$, $11$, $16$, $20$ at $N = 31$, $63$, $127$, $255$, $361$.

### $C^{0,1}$ (Lipschitz) example (13), the cone T21

Table 3: max error; the Standard F.D. column prints a dash at $N = 255$ and $361$.
$f$ is the Dirac mass $\pi \delta_{x_0}$ averaged over the disk $\lvert x - x_0 \rvert \le h/2$: $4/h^2$ there, $0$ elsewhere (Section 6.1).

| $N$ | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| $31$ | $12 \times 10^{-3}$ | $3 \times 10^{-3}$ | $3 \times 10^{-3}$ | $10 \times 10^{-3}$ |
| $63$ | $11 \times 10^{-3}$ | $3 \times 10^{-3}$ | $2 \times 10^{-3}$ | $6 \times 10^{-3}$ |
| $127$ | $11 \times 10^{-3}$ | $4 \times 10^{-3}$ | $2 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| $255$ | $11 \times 10^{-3}$ | $4 \times 10^{-3}$ | $1 \times 10^{-3}$ | -- |
| $361$ | $11 \times 10^{-3}$ | $4 \times 10^{-3}$ | $1 \times 10^{-3}$ | -- |

Table 1, Newton iterations on the 17-point scheme: $9$, $15$, $32$, $34$, $29$ at $N = 31$, $63$, $127$, $255$, $361$.

### Other test solutions in this paper

- $C^1$ example (11): $u = ((\lvert x - x_0 \rvert - 0.2)^+)^2 / 2$, $f = (1 - 0.2/\lvert x - x_0 \rvert)^+$, $x_0 = (0.5, 0.5)$.
- $C^{0,1}$ (Lipschitz) example (13): the cone $u = \lvert x - x_0 \rvert$, $f = \pi \delta_{x_0}$, discretised as $4/h^2$ on $\lvert x - x_0 \rvert \le h/2$.
- 3D (Table 4, 19-point stencil, $N^3$ grid on $[0,1]^3$): $u = \exp(\lvert x \rvert^2/2)$ with $f = (1+\lvert x \rvert^2) \exp(3\lvert x \rvert^2/2)$; a $C^1$ example; and the ball $u = -\sqrt{3 - \lvert x \rvert^2}$, $f = 3 (3 - \lvert x \rvert^2)^{-5/2}$.

### Table 4, three dimensions (`monotone19.md`)

Section 7: "These computations are done using a 19 point stencil on an $N^3$ grid on the square $[0, 1]^3$", by Newton's method; $x_0 = (.5, .5, .5)$.
$N^3$ grid does not say whether $N$ counts nodes or cells per side.
The examples: (14) $u = \exp(\lvert x \rvert^2/2)$, $f = (1 + \lvert x \rvert^2) \exp(3 \lvert x \rvert^2/2)$, T9 in the catalogue; (15) $u = ((\lvert x - x_0 \rvert - 0.2)^+)^2 / 2$, $f = 1 - 0.4/\lvert x - x_0 \rvert + 0.04/\lvert x - x_0 \rvert^2$ for $\lvert x - x_0 \rvert > 0.2$ and $0$ otherwise, no T id (`C1_3D` in `run_monotone19.py`); (16) $u = -\sqrt{3 - \lvert x \rvert^2}$, $f = 3 (3 - \lvert x \rvert^2)^{-5/2}$, T22.
Max error, Newton iterations and CPU seconds, as printed (checked against the page image, p. 1712):

| $N$ | (14) max error | its | CPU s | (15) max error | its | CPU s | (16) max error | its | CPU s |
|---|---|---|---|---|---|---|---|---|---|
| $7$ | $0.0151$ | $2$ | $0.1$ | $0.0034$ | $1$ | $0.02$ | $9.6 \times 10^{-3}$ | $1$ | $0.02$ |
| $11$ | $0.0140$ | $3$ | $0.1$ | $0.0022$ | $1$ | $0.06$ | $5.3 \times 10^{-3}$ | $3$ | $0.12$ |
| $15$ | $0.0132$ | $5$ | $0.5$ | $0.0019$ | $1$ | $0.17$ | $4.7 \times 10^{-3}$ | $3$ | $0.34$ |
| $21$ | $0.0127$ | $6$ | $3.6$ | $0.0020$ | $2$ | $1.42$ | $4.3 \times 10^{-3}$ | $6$ | $3.65$ |
| $31$ | $0.0125$ | $5$ | $34.7$ | $0.0019$ | $2$ | $16.70$ | $3.9 \times 10^{-3}$ | $8$ | $54.70$ |

## 3. Froese, Oberman, J. Comput. Phys. (2011)

B. D. Froese, A. M. Oberman, "Fast finite difference solvers for singular solutions of the elliptic Monge-Ampere equation", J. Comput. Phys. 230 (2011) 818-834, DOI 10.1016/j.jcp.2010.10.020.
Read: local journal PDF (Zotero item N9IMUP7B).
Open copy: https://arxiv.org/abs/1006.5748 (not checked).

Methods: three discretisations, each solved by damped Newton.
"Standard" is the natural 9-point scheme ($MA_N$).
"Monotone" is the wide-stencil scheme ($MA_M$) with a 17-point stencil.
"Hybrid" is $w(x) MA_N + (1 - w(x)) MA_M$, with the weight $w$ fixed a priori: $w = 0$ near possible singular sets ($f$ near $0$ or infinity, or a flat or non-smooth boundary), and $w$ goes to $1$ elsewhere.
On the square, the monotone scheme is used near the boundary, so the hybrid scheme is less accurate than the standard one for (a).
The linear systems are solved by LU decomposition.
The initial guess is one semi-implicit step followed by a convex envelope on a coarse grid, interpolated to the fine grid.
The Jacobian is regularised by $u_{mm} \to \max(u_{mm}, \epsilon)$, with $\epsilon = 10^{-8} / (2 dx^2)$ as in the SIAM paper (Section 7.2).
Hardware: a laptop with a $2$ GHz Intel processor, MATLAB, Windows Vista.
No Newton stopping tolerance is stated.
Section 8.1 reports that Newton on the standard scheme fails for solution (b): started from the exact solution, after two iterations it oscillates and becomes non-convex (Fig. 2).
The "Standard" column of Table 3 is credited to [29], which is Benamou-Froese-Oberman 2010, so it comes from the (M1)/(M2) iterations, not from Newton.
That $N$ sequence does not appear in the 2010 paper's tables, so these values were recomputed, not copied.
The same applies to the Poisson and Gauss-Seidel CPU columns of Table 1.
Where the two 2011 papers share $N$ values, their standard-scheme errors agree (for (a) at $N = 127$, $0.43 \times 10^{-5}$ here and $4.3 \times 10^{-6}$ in SIAM).

Name map for this paper:
- Solution (a) is $C^2$ Example, eq. (14).
- Solution (b) is "Example with blow-up", eq. (16); it is also the Section 8.1 example.

### Solution (a), $C^2$ Example (14)

Table 3: maximum error.

| $N$ | Standard | Monotone | Hybrid |
|---|---|---|---|
| $31$ | $7.14 \times 10^{-5}$ | $89.09 \times 10^{-5}$ | $24.45 \times 10^{-5}$ |
| $45$ | $3.39 \times 10^{-5}$ | $60.50 \times 10^{-5}$ | $15.29 \times 10^{-5}$ |
| $63$ | $1.73 \times 10^{-5}$ | $50.88 \times 10^{-5}$ | $9.06 \times 10^{-5}$ |
| $89$ | $0.87 \times 10^{-5}$ | $47.51 \times 10^{-5}$ | $5.32 \times 10^{-5}$ |
| $127$ | $0.43 \times 10^{-5}$ | $45.53 \times 10^{-5}$ | $3.02 \times 10^{-5}$ |
| $181$ | $0.21 \times 10^{-5}$ | $44.65 \times 10^{-5}$ | $1.61 \times 10^{-5}$ |
| $255$ | $0.11 \times 10^{-5}$ | $44.22 \times 10^{-5}$ | $0.87 \times 10^{-5}$ |
| $361$ | $0.05 \times 10^{-5}$ | $44.00 \times 10^{-5}$ | $0.46 \times 10^{-5}$ |

Table 1: Newton on the hybrid scheme; Poisson and Gauss-Seidel are the 2010 methods (M2) and (M1).

| $N$ | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| $31$ | $3$ | $0.2$ | $0.7$ | $2.2$ |
| $45$ | $4$ | $0.2$ | $1.1$ | $4.1$ |
| $63$ | $4$ | $0.4$ | $1.9$ | $15.0$ |
| $89$ | $4$ | $1.0$ | $4.8$ | $57.6$ |
| $127$ | $5$ | $2.9$ | $9.6$ | $236.7$ |
| $181$ | $5$ | $9.0$ | $23.2$ | $1004.0$ |
| $255$ | $5$ | $30.5$ | $52.6$ | -- |
| $361$ | $6$ | $131.4$ | $162.6$ | -- |

### Solution (b), "Example with blow-up (16)"

Table 3: maximum error.
The text gives the rates: hybrid $O(h^{1.5})$, standard $O(h^{0.5})$, monotone limited by the stencil width.

| $N$ | Standard | Monotone | Hybrid |
|---|---|---|---|
| $31$ | $17.15 \times 10^{-3}$ | $1.74 \times 10^{-3}$ | $1.74 \times 10^{-3}$ |
| $45$ | $14.59 \times 10^{-3}$ | $0.98 \times 10^{-3}$ | $0.98 \times 10^{-3}$ |
| $63$ | $12.53 \times 10^{-3}$ | $0.59 \times 10^{-3}$ | $0.59 \times 10^{-3}$ |
| $89$ | $10.67 \times 10^{-3}$ | $0.37 \times 10^{-3}$ | $0.35 \times 10^{-3}$ |
| $127$ | $9.00 \times 10^{-3}$ | $0.35 \times 10^{-3}$ | $0.20 \times 10^{-3}$ |
| $181$ | $7.59 \times 10^{-3}$ | $0.34 \times 10^{-3}$ | $0.12 \times 10^{-3}$ |
| $255$ | $6.42 \times 10^{-3}$ | $0.33 \times 10^{-3}$ | $0.07 \times 10^{-3}$ |
| $361$ | $5.41 \times 10^{-3}$ | $0.33 \times 10^{-3}$ | $0.04 \times 10^{-3}$ |

Table 1: Newton on the hybrid scheme.

| $N$ | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| $31$ | $4$ | $0.2$ | $0.5$ | $0.8$ |
| $45$ | $4$ | $0.4$ | $1.4$ | $5.3$ |
| $63$ | $4$ | $0.7$ | $2.9$ | $19.4$ |
| $89$ | $5$ | $1.8$ | $8.1$ | $74.1$ |
| $127$ | $7$ | $5.1$ | $17.7$ | $293.3$ |
| $181$ | $7$ | $12.9$ | $51.4$ | $1637.1$ |
| $255$ | $7$ | $36.1$ | $128.2$ | -- |
| $361$ | $8$ | $152.9$ | $374.5$ | -- |

Table 2 gives the same CPU time scalings as the SIAM paper (Newton about $O(M^{1.3})$, $M = N^2$).

### $C^1$ Example (15), T20 in the catalogue

Table 3: maximum error; the Standard column prints a dash at $N = 255$ and $361$.

| $N$ | Standard | Monotone | Hybrid |
|---|---|---|---|
| $31$ | $2.6 \times 10^{-4}$ | $17.5 \times 10^{-4}$ | $12.2 \times 10^{-4}$ |
| $45$ | $1.8 \times 10^{-4}$ | $11.6 \times 10^{-4}$ | $5.9 \times 10^{-4}$ |
| $63$ | $1.5 \times 10^{-4}$ | $9.8 \times 10^{-4}$ | $4.2 \times 10^{-4}$ |
| $89$ | $0.9 \times 10^{-4}$ | $8.4 \times 10^{-4}$ | $2.6 \times 10^{-4}$ |
| $127$ | $0.6 \times 10^{-4}$ | $7.9 \times 10^{-4}$ | $2.0 \times 10^{-4}$ |
| $181$ | $0.4 \times 10^{-4}$ | $7.4 \times 10^{-4}$ | $1.2 \times 10^{-4}$ |
| $255$ | -- | $7.2 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| $361$ | -- | $7.0 \times 10^{-4}$ | $0.7 \times 10^{-4}$ |

### $C^{0,1}$ (Lipschitz) Example (17), the cone T21

Table 3: maximum error; the Standard column prints a dash at $N = 255$ and $361$.

| $N$ | Standard | Monotone | Hybrid |
|---|---|---|---|
| $31$ | $10 \times 10^{-3}$ | $3 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| $45$ | $8 \times 10^{-3}$ | $3 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| $63$ | $6 \times 10^{-3}$ | $3 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| $89$ | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ |
| $127$ | $3 \times 10^{-3}$ | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ |
| $181$ | $2 \times 10^{-3}$ | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ |
| $255$ | -- | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ |
| $361$ | -- | $4 \times 10^{-3}$ | $4 \times 10^{-3}$ |

### Other test solutions in this paper

- $C^1$ Example (15): $u = ((\lvert x - x_0 \rvert - 0.2)^+)^2 / 2$, $f = (1 - 0.2/\lvert x - x_0 \rvert)^+$, $x_0 = (0.5, 0.5)$; hybrid accuracy $O(h)$.
- $C^{0,1}$ (Lipschitz) Example (17): the cone centred at $x_0$ with $f = \pi \delta_{x_0}$, discretised as $4/h^2$ on $\lvert x - x_0 \rvert \le h/2$; the hybrid scheme equals the monotone scheme there.
- 3D (Table 4, hybrid Newton, 19-point stencil, $N^3$ grid on $[0,1]^3$): $\exp(\lvert x \rvert^2/2)$, a $C^1$ example, and the ball $-\sqrt{3 - \lvert x \rvert^2}$.

### Table 4, three dimensions, hybrid scheme (`monotone19.md`, cross-check only)

Section 9: "These computations are performed on an $N \times N \times N$ grid on the square $[0, 1]^3$. The monotone scheme used a 19 point stencil."
The examples are the SIAM paper's (14), (15) and (16), numbered (18), (19) and (20) here; the paper calls the single Newton step on (19) "simply an artifact".
Max error, Newton iterations and CPU seconds of the hybrid Newton's method, as printed (text layer of the PDF, p. 831):

| $N$ | (18) max error | its | CPU s | (19) max error | its | CPU s | (20) max error | its | CPU s |
|---|---|---|---|---|---|---|---|---|---|
| $7$ | $0.0151$ | $2$ | $0.04$ | $0.0034$ | $1$ | $0.02$ | $9.6 \times 10^{-3}$ | $1$ | $0.03$ |
| $11$ | $0.0140$ | $3$ | $0.10$ | $0.0022$ | $1$ | $0.09$ | $5.2 \times 10^{-3}$ | $3$ | $0.11$ |
| $15$ | $0.0129$ | $5$ | $0.71$ | $0.0016$ | $1$ | $0.22$ | $4.6 \times 10^{-3}$ | $3$ | $0.48$ |
| $21$ | $0.0121$ | $6$ | $6.72$ | $0.0009$ | $1$ | $1.03$ | $4.0 \times 10^{-3}$ | $6$ | $7.42$ |
| $31$ | $0.0111$ | $5$ | $86.63$ | $0.0005$ | $1$ | $17.12$ | $2.9 \times 10^{-3}$ | $8$ | $138.74$ |

## 4. Loeper, Rapetti (2005)

G. Loeper, F. Rapetti, "Numerical solution of the Monge-Ampere equation by a Newton's algorithm", C. R. Acad. Sci. Paris Ser. I 340 (2005) 319-324, DOI 10.1016/j.crma.2004.12.018.
Read: local journal PDF (Zotero item ZH56SNIU).
It is not paywalled for this project because a local copy exists; no arXiv version was found.

This paper does not test (a) or (b), and it has no tables.
It solves the periodic problem $\det(I + D^2 u) = \rho$ on the torus $[0,1]^2$ (or $[0,1]^3$), not the Dirichlet problem.
Method: Newton with step $1/\tau$ on the linearised operator (cofactor matrix of $I + D^2 u$), centred second-order finite differences, $h = 1/N$, $N^d$ points, and unpreconditioned BiCG with residual tolerance $10^{-8}$.
Test density: $\rho = 1 + \beta \sin(2 \pi k x) \sin(2 \pi k y)$, with $\beta = 0.8$, $k = 2$ in most runs and $\beta = 0.99$, $k = 1$ for the near-degenerate run; the start is $u_0 = 0$, $\rho_0 = 1$.
Results are only in figures: about $10$ Newton iterations reach an $L^2$ error on $\rho$ of about $10^{-10}$ for $N = 16$ to $512$ (Fig. 1).
The only printed numbers are in the Fig. 2 caption ($\beta = 0.8$, $k = 2$, $N = 64$): "The highest absolute value is $0.0609$ ($n = 1$, top left), $0.0239$ ($n = 2$, top right), $0.00949$ ($n = 3$, bottom left) and $0.00379$ ($n = 4$, bottom right)".
Fig. 4 shows the $L^2$ and $L^\infty$ error of $u$ for $u_{ex} = \beta \sin(2 \pi k x) \sin(2 \pi k y)$, $\beta = 0.02$, $k = 1$, at $N = 16$, $32$, $64$, $128$, $256$ after $10$ Newton iterations; the text gives order $O(1/N^2)$ but prints no values.

## 5. Froese, Oberman, SIAM J. Numer. Anal. (2013)

B. D. Froese, A. M. Oberman, "Convergent filtered schemes for the Monge-Ampere partial differential equation", SIAM J. Numer. Anal. 51(1) (2013) 423-444, DOI 10.1137/120875065.
Read: local journal PDF (Zotero item Y55FLDKL, attachment 2BQN6YP2); every value below was checked against a rendered image of pp. 438-442 (journal page numbers).

Method (Section 4.1): the filtered scheme $F_H = F_M + \varepsilon S((F_A - F_M) / \varepsilon)$, with $F_M$ the monotone wide-stencil scheme of their Section 1.7 ($\min$ over pairs of $\prod \max\lbrace D_{vv} u, \delta\rbrace + \sum \min\lbrace D_{vv} u, \delta\rbrace$, $\delta$ not printed), $F_A$ the standard 9-point scheme ("see [FO11b] for details"), the filter $S$ of eq. (1.3) (identity on $[-1, 1]$, $\operatorname{sign}(x)(2 - \lvert x \rvert)$ for $1 \le \lvert x \rvert \le 2$, zero beyond) and $\varepsilon(h, d\theta) = \sqrt{h} + d\theta / 10$.
$d\theta$ is not given a value, neither here nor in FO 2011 (SINUM p. 1700 names it the parameter of the directional discretisation); Oberman 2008, eq. (3.3), defines it as $\max_\theta \min_i \lvert \theta - \theta_i \rvert$, the largest angle between a direction and the nearest stencil direction, half the largest gap between stencil directions.
maesolvers takes the full gap ($\pi/4$, $\arctan(1/2)$, $\arctan{(1/3)}$), which the T2 filtered columns of Table 4.1 fit (`filtered.md`, Finding 3).
Solved by Newton, $u^{n+1} = u^n - (\nabla F_H[u^n])^{-1} F_H[u^n]$, with the Jacobian approximated by $(1 - S') \nabla F_M + \max\lbrace S', 0\rbrace \nabla F_A$ "because negative values [of S'] can lead to poorly conditioned or ill-posed linear systems".
"We have regularized the max operator to allow the gradient to be computed, as in [FO11b]."
[FO11b] is the JCP 2011 hybrid paper (section 3 above; the reference list, and p. 424, "In [FO11b] a hybrid solver was built"), not the SINUM paper [FO11a]; its Section 7.2 replaces $u_{vv}$ by $\max\lbrace u_{vv}, \varepsilon\rbrace$ in the Jacobian, $\varepsilon = 10^{-8} / (2 dx^2)$, which leaves the scheme and its roots unchanged.
The Newton iteration is printed with the full step, no damping; no start or stopping rule is stated.
Section 4.2: "All computations are performed on the domain $[0, 1] \times [0, 1]$, which is discretized on an $N \times N$ grid."
No definition of $h$ in terms of $N$; the values match $h = 1/(N - 1)$, $N$ nodes per side (`filtered.md`, Finding 1).

Name map (Section 4.2, $x_0 = {(0.5, 0.5)}$):
- $C^2$ example, eq. (4.1): $u = \exp(\lvert x - x_0 \rvert^2 / 2)$, $f = (1 + \lvert x - x_0 \rvert^2) \exp(\lvert x - x_0 \rvert^2)$. Centred at $x_0$, so not T1 ($\exp{(\lvert x \rvert^2 / 2)}$), although the text says the four examples are those of [FO11a].
- $C^1$ example, eq. (4.2): T20.
- "Example with blow-up", eq. (4.3): T2, $u = -\sqrt{2 - \lvert x \rvert^2}$, uncentred.
- $C^{0,1}$ (Lipschitz) example, eq. (4.4): the cone T21, $f = 4/h^2$ on $\lvert x - x_0 \rvert \le h/2$, $0$ elsewhere.

### Table 4.1, "Accuracy of the monotone and filtered schemes", max error

$C^2$ example (4.1), all values $\times 10^{-5}$:

| $N$ | 9 Monotone | 9 Filtered | 17 Monotone | 17 Filtered | 33 Monotone | 33 Filtered |
|---|---|---|---|---|---|---|
| $31$ | $9.45$ | $4.54$ | $9.12$ | $4.54$ | $9.38$ | $4.54$ |
| $63$ | $4.91$ | $1.04$ | $3.42$ | $1.04$ | $3.40$ | $1.04$ |
| $127$ | $3.79$ | $0.26$ | $1.67$ | $0.26$ | $1.39$ | $0.26$ |
| $255$ | $3.51$ | $0.06$ | $1.17$ | $0.06$ | $0.66$ | $0.06$ |
| $361$ | $3.48$ | $0.03$ | $1.08$ | $0.03$ | $0.51$ | $0.03$ |

$C^1$ example (4.2), all values $\times 10^{-4}$:

| $N$ | 9 Monotone | 9 Filtered | 17 Monotone | 17 Filtered | 33 Monotone | 33 Filtered |
|---|---|---|---|---|---|---|
| $31$ | $21.54$ | $3.73$ | $8.66$ | $3.99$ | $6.39$ | $3.67$ |
| $63$ | $21.33$ | $1.51$ | $6.82$ | $1.40$ | $3.18$ | $1.46$ |
| $127$ | $21.55$ | $0.92$ | $6.63$ | $0.76$ | $2.49$ | $0.78$ |
| $255$ | $21.51$ | $0.38$ | $6.58$ | $0.46$ | $2.36$ | $0.37$ |
| $361$ | $21.53$ | $0.23$ | $6.62$ | $0.31$ | $2.37$ | $0.28$ |

Example with blow-up (4.3), all values $\times 10^{-3}$:

| $N$ | 9 Monotone | 9 Filtered | 17 Monotone | 17 Filtered | 33 Monotone | 33 Filtered |
|---|---|---|---|---|---|---|
| $31$ | $1.74$ | $1.74$ | $1.74$ | $1.74$ | $1.74$ | $1.74$ |
| $63$ | $0.86$ | $0.59$ | $0.59$ | $0.59$ | $0.59$ | $0.59$ |
| $127$ | $0.83$ | $0.20$ | $0.35$ | $0.20$ | $0.20$ | $0.20$ |
| $255$ | $0.83$ | $0.15$ | $0.33$ | $0.13$ | $0.16$ | $0.08$ |
| $361$ | $0.83$ | $0.17$ | $0.33$ | $0.13$ | $0.15$ | $0.08$ |

$C^{0,1}$ (Lipschitz) example (4.4), all values $\times 10^{-3}$:

| $N$ | 9 Monotone | 9 Filtered | 17 Monotone | 17 Filtered | 33 Monotone | 33 Filtered |
|---|---|---|---|---|---|---|
| $31$ | $11.83$ | $10.42$ | $3.56$ | $4.16$ | $1.61$ | $1.23$ |
| $63$ | $11.10$ | $11.56$ | $3.49$ | $2.82$ | $1.65$ | $1.90$ |
| $127$ | $11.80$ | $10.97$ | $3.45$ | $2.83$ | $1.64$ | $1.11$ |
| $255$ | $10.47$ | $11.03$ | $3.46$ | $3.06$ | $1.64$ | $1.15$ |
| $361$ | $10.40$ | $10.37$ | $3.45$ | $3.22$ | $1.64$ | $1.12$ |

### Table 4.2, "Newton iterations, computation time, and accuracy for the hybrid and filtered schemes on a 17-point stencil"

The "Standard" column prints a dash at $N = 255$ and $361$ for the $C^1$ and the Lipschitz examples.
The hybrid columns are the a priori hybrid scheme of FO 2011 JCP.

| Example | $N$ | Iterations Hybrid | Iterations Filtered | CPU s Hybrid | CPU s Filtered | Max error Hybrid | Max error Filtered | Max error Standard |
|---|---|---|---|---|---|---|---|---|
| $C^2$ ($\times 10^{-5}$) | $31$ | $3$ | $2$ | $0.1$ | $0.1$ | $6.76$ | $4.54$ | $4.54$ |
| | $63$ | $3$ | $2$ | $0.6$ | $0.5$ | $1.46$ | $1.06$ | $1.06$ |
| | $127$ | $3$ | $2$ | $2.4$ | $2.0$ | $0.35$ | $0.26$ | $0.26$ |
| | $255$ | $3$ | $2$ | $12.5$ | $9.9$ | $0.09$ | $0.06$ | $0.06$ |
| | $361$ | $3$ | $2$ | $28.2$ | $22.5$ | $0.04$ | $0.03$ | $0.03$ |
| $C^1$ ($\times 10^{-4}$) | $31$ | $2$ | $2$ | $0.1$ | $0.1$ | $6.62$ | $3.99$ | $3.78$ |
| | $63$ | $3$ | $2$ | $0.6$ | $0.5$ | $2.75$ | $1.40$ | $1.34$ |
| | $127$ | $5$ | $3$ | $3.8$ | $2.6$ | $1.68$ | $0.76$ | $0.59$ |
| | $255$ | $5$ | $3$ | $19.1$ | $12.9$ | $0.85$ | $0.46$ | -- |
| | $361$ | $5$ | $6$ | $48.3$ | $50.6$ | $0.60$ | $0.31$ | -- |
| Blow-up ($\times 10^{-3}$) | $31$ | $6$ | $7$ | $0.3$ | $0.3$ | $1.74$ | $1.74$ | $17.38$ |
| | $63$ | $9$ | $9$ | $1.4$ | $1.5$ | $0.59$ | $0.59$ | $12.62$ |
| | $127$ | $11$ | $11$ | $8.6$ | $8.4$ | $0.20$ | $0.20$ | $9.04$ |
| | $255$ | $7$ | $8$ | $30.5$ | $32.4$ | $0.07$ | $0.13$ | $6.43$ |
| | $361$ | $11$ | $12$ | $101.5$ | $108.7$ | $0.04$ | $0.13$ | $5.42$ |
| Lipschitz ($\times 10^{-3}$) | $31$ | $6$ | $7$ | $0.2$ | $0.2$ | $3.57$ | $4.16$ | $5.19$ |
| | $63$ | $6$ | $9$ | $1.0$ | $1.4$ | $3.49$ | $2.82$ | $2.86$ |
| | $127$ | $8$ | $8$ | $6.6$ | $6.4$ | $3.45$ | $2.83$ | $1.54$ |
| | $255$ | $9$ | $10$ | $47.1$ | $38.2$ | $3.46$ | $3.06$ | -- |
| | $361$ | $10$ | $9$ | $155.8$ | $81.7$ | $3.45$ | $3.22$ | -- |

The two tables print different filtered values for the $C^2$ example at $N = 63$ ($1.04$ in Table 4.1, $1.06$ in Table 4.2); every other filtered 17-point value appears in both and agrees.

## 6. Liu, Froese, Oberman, Xiao, Int. J. Comput. Math. (2017)

J. Liu, B. D. Froese, A. M. Oberman, M. Xiao, "A multigrid scheme for 3D Monge-Ampere equations", Int. J. Comput. Math. 94(9) (2017) 1850-1866, DOI 10.1080/00207160.2016.1247443.
Read: local journal PDF (Zotero item RWA942I8, attachment EEEVIK3A); every value below was checked against a rendered image of pp. 1861-1864 (journal page numbers).
The PDF's text layer garbles the first row of Tables 5 to 7 ("86.8e -08") and prints Example 4.5's $\lvert x \rvert^2 / 2$ as $(x + y + z) / 2$; the page image has $(x^2 + y^2 + z^2) / 2$.

Method (Sections 2 and 3): the standard centred differences (9 points in 2D, 19 in 3D) in $\det D^2 u = f$, solved by FMG-FAS multigrid (Algorithms 1 and 2) with the nonlinear Gauss-Seidel smoother that takes the smallest real root of each node's quadratic (2D, from [2], BFO 2010) or cubic (3D, eq. (12)); $2$ pre- and $2$ post-sweeps, one sweep on the coarsest grid $h_0 = 1/2$ from $0$, half-weighting and injection down, (tri)linear interpolation up, "cubic interpolation" in the FMG step.
Stopping rule: $\lVert r_l \rVert_2 / \lVert r_0 \rVert_2 \le 10^{-6}$, $r_0$ not defined for FMG-FAS.
Grid: $x_i = i h$, $h = 1/N$, $i = 0, \ldots, N$ (Section 2), so ADR 0001's convention is printed here; Example 4.5 is on $(0, \pi)^3$.
The stand-alone Gauss-Seidel solver starts from $1.01 u$.

Name map: Example 4.1 is T1, 4.2 T3, 4.3 T2 (2D); 4.4 is T9, 4.5 T46, 4.6 T28, 4.7 T22 (3D).

### Tables 1 and 2, FMG-FAS in 2D: max error, iterations, CPU s

Table 1, red-black ordering:

| $N$ | 4.1 Error | Iter | CPU | 4.2 Error | Iter | CPU | 4.3 Error | Iter | CPU |
|---|---|---|---|---|---|---|---|---|---|
| $128$ | $4.5 \times 10^{-06}$ | $1$ | $0.1$ | $3.8 \times 10^{-05}$ | $1$ | $0.1$ | $8.9 \times 10^{-03}$ | $2$ | $0.1$ |
| $256$ | $1.1 \times 10^{-06}$ | $1$ | $0.1$ | $1.3 \times 10^{-05}$ | $1$ | $0.1$ | $6.4 \times 10^{-03}$ | $2$ | $0.2$ |
| $512$ | $2.8 \times 10^{-07}$ | $1$ | $0.3$ | $4.7 \times 10^{-06}$ | $1$ | $0.3$ | $4.5 \times 10^{-03}$ | $1$ | $0.3$ |
| $1024$ | $7.0 \times 10^{-08}$ | $1$ | $1.0$ | $1.7 \times 10^{-06}$ | $1$ | $1.1$ | $3.2 \times 10^{-03}$ | $1$ | $1.1$ |
| $2048$ | $1.7 \times 10^{-08}$ | $1$ | $3.6$ | $5.9 \times 10^{-07}$ | $1$ | $4.0$ | $2.3 \times 10^{-03}$ | $1$ | $3.8$ |
| $4096$ | $4.3 \times 10^{-09}$ | $1$ | $16.2$ | $2.1 \times 10^{-07}$ | $1$ | $16.5$ | $1.6 \times 10^{-03}$ | $1$ | $18.0$ |
| $8192$ | $1.0 \times 10^{-09}$ | $1$ | $67.1$ | $7.4 \times 10^{-08}$ | $1$ | $69.1$ | $1.1 \times 10^{-03}$ | $1$ | $70.3$ |

Table 2, lexicographic ordering:

| $N$ | 4.1 Error | Iter | CPU | 4.2 Error | Iter | CPU | 4.3 Error | Iter | CPU |
|---|---|---|---|---|---|---|---|---|---|
| $128$ | $5.1 \times 10^{-06}$ | $1$ | $0.1$ | $4.3 \times 10^{-05}$ | $1$ | $0.1$ | $8.9 \times 10^{-03}$ | $3$ | $0.1$ |
| $256$ | $1.3 \times 10^{-06}$ | $1$ | $0.2$ | $1.5 \times 10^{-05}$ | $1$ | $0.3$ | $6.3 \times 10^{-03}$ | $2$ | $0.2$ |
| $512$ | $3.3 \times 10^{-07}$ | $1$ | $0.8$ | $5.4 \times 10^{-06}$ | $1$ | $0.9$ | $4.4 \times 10^{-03}$ | $1$ | $0.8$ |
| $1024$ | $8.2 \times 10^{-08}$ | $1$ | $3.2$ | $1.9 \times 10^{-06}$ | $1$ | $3.27$ | $3.2 \times 10^{-03}$ | $1$ | $3.1$ |
| $2048$ | $2.1 \times 10^{-08}$ | $1$ | $12.7$ | $6.8 \times 10^{-07}$ | $1$ | $12.9$ | $2.2 \times 10^{-03}$ | $1$ | $12.6$ |
| $4096$ | $5.1 \times 10^{-09}$ | $1$ | $53.2$ | $2.4 \times 10^{-07}$ | $1$ | $54.6$ | $1.6 \times 10^{-03}$ | $1$ | $53.5$ |
| $8192$ | $1.3 \times 10^{-09}$ | $1$ | $245.6$ | $8.5 \times 10^{-08}$ | $1$ | $259.4$ | $1.1 \times 10^{-03}$ | $1$ | $250.7$ |

The order columns print $2.0$ (4.1; $2.1$ at $N = 8192$ in Table 1), $1.5$ (4.2) and $0.5$ (4.3).

### Tables 3, 5, 6 and 7, 3D: FMG-FAS and Gauss-Seidel

RelRes, max error, iterations and CPU s for FMG-FAS, then for the stand-alone Gauss-Seidel solver ($N \le 32$ only); Table 3 alone prints a DOFs column.
Its DOFs are $(N - 2)^3$ ($6^3 = 216$ to $126^3 = 2{,}000{,}376$), not the $(N - 1)^3$ interior nodes of the grid of Section 2 ($h = 1/N$, nodes $i = 0, \ldots, N$), which Table 4's DOFs column is (`multigrid.md`, Conventions).

| Example | $N$ | DOFs | RelRes | Error | Iter | CPU | GS RelRes | GS Error | GS Iter | GS CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.4 (T9) | $8$ | $216$ | $3.1 \times 10^{-07}$ | $1.2 \times 10^{-03}$ | $1$ | $0.1$ | $9.3 \times 10^{-07}$ | $1.2 \times 10^{-03}$ | $72$ | $0.4$ |
| | $16$ | $2{,}744$ | $9.5 \times 10^{-09}$ | $2.7 \times 10^{-04}$ | $1$ | $0.2$ | $9.9 \times 10^{-07}$ | $3.0 \times 10^{-04}$ | $270$ | $11.5$ |
| | $32$ | $27{,}000$ | $2.1 \times 10^{-10}$ | $5.7 \times 10^{-05}$ | $1$ | $2.0$ | $1.0 \times 10^{-06}$ | $8.1 \times 10^{-05}$ | $867$ | $324.6$ |
| | $64$ | $238{,}328$ | $4.1 \times 10^{-12}$ | $1.4 \times 10^{-05}$ | $1$ | $16.9$ | | | | |
| | $128$ | $2{,}000{,}376$ | $5.2 \times 10^{-13}$ | $6.2 \times 10^{-06}$ | $1$ | $136.9$ | | | | |
| 4.5 (T46) | $8$ | | $6.8 \times 10^{-08}$ | $1.7 \times 10^{-02}$ | $2$ | $0.1$ | $9.2 \times 10^{-07}$ | $1.7 \times 10^{-02}$ | $82$ | $0.4$ |
| | $16$ | | $2.1 \times 10^{-08}$ | $3.3 \times 10^{-03}$ | $1$ | $0.2$ | $9.8 \times 10^{-07}$ | $4.3 \times 10^{-03}$ | $296$ | $12.9$ |
| | $32$ | | $2.7 \times 10^{-10}$ | $8.1 \times 10^{-04}$ | $1$ | $2.0$ | $1.0 \times 10^{-06}$ | $1.1 \times 10^{-03}$ | $954$ | $356.2$ |
| | $64$ | | $3.7 \times 10^{-12}$ | $2.1 \times 10^{-04}$ | $1$ | $16.4$ | | | | |
| | $128$ | | $4.8 \times 10^{-13}$ | $5.2 \times 10^{-05}$ | $1$ | $134.1$ | | | | |
| 4.6 (T28) | $8$ | | $2.7 \times 10^{-07}$ | $5.3 \times 10^{-04}$ | $1$ | $0.1$ | $9.7 \times 10^{-07}$ | $4.0 \times 10^{-04}$ | $86$ | $0.4$ |
| | $16$ | | $1.1 \times 10^{-08}$ | $2.1 \times 10^{-04}$ | $1$ | $0.2$ | $9.9 \times 10^{-07}$ | $1.4 \times 10^{-04}$ | $312$ | $13.2$ |
| | $32$ | | $4.8 \times 10^{-10}$ | $7.4 \times 10^{-05}$ | $1$ | $2.0$ | $9.9 \times 10^{-07}$ | $5.1 \times 10^{-05}$ | $925$ | $345.8$ |
| | $64$ | | $2.0 \times 10^{-11}$ | $2.6 \times 10^{-05}$ | $1$ | $16.4$ | | | | |
| | $128$ | | $1.2 \times 10^{-12}$ | $9.3 \times 10^{-06}$ | $1$ | $133.9$ | | | | |
| 4.7 (T22) | $8$ | | $1.7 \times 10^{-07}$ | $1.4 \times 10^{-03}$ | $1$ | $0.1$ | $8.9 \times 10^{-07}$ | $1.4 \times 10^{-03}$ | $75$ | $0.4$ |
| | $16$ | | $2.1 \times 10^{-08}$ | $1.4 \times 10^{-03}$ | $1$ | $0.2$ | $9.7 \times 10^{-07}$ | $1.4 \times 10^{-03}$ | $266$ | $12.2$ |
| | $32$ | | $2.7 \times 10^{-08}$ | $4.3 \times 10^{-03}$ | $1$ | $2.0$ | $9.9 \times 10^{-07}$ | $1.1 \times 10^{-03}$ | $854$ | $320.2$ |
| | $64$ | | $4.9 \times 10^{-09}$ | $2.2 \times 10^{-03}$ | $1$ | $16.3$ | | | | |
| | $128$ | | $1.7 \times 10^{-09}$ | $1.6 \times 10^{-03}$ | $1$ | $133.1$ | | | | |

Table 4 reprints, for Example 4.4, the wide-stencil hybrid finite differences of FO 2011 JCP ($N = 8$ to $32$, $1.51 \times 10^{-2}$ to $1.11 \times 10^{-2}$) and the quadratic and cubic interior-penalty elements of Brenner-Neilan 2012; it is not transcribed here.
`multigrid.md` compares the tables with ours.

## 7. Feng, Jensen, SIAM J. Numer. Anal. (2017)

X. Feng, M. Jensen, "Convergent semi-Lagrangian methods for the Monge-Ampere equation on unstructured grids", SIAM J. Numer. Anal. 55(2) (2017) 691-712, DOI 10.1137/16M1061709.
Read: local journal PDF (Zotero item PVLY7QYE, attachment YSSATHUT); every value below was checked against a rendered image of pp. 709-711 (journal page numbers).

Method (Sections 3, 4, 7, 8): the HJB form $\sup_{B \ge 0, \operatorname{tr} B = 1} (-B : D^2 u + f (\det B)^{1/d}) = 0$ of $\det D^2 u = (f/d)^d$, so the paper's $f$ is $2 \sqrt{f}$ in maesolvers' convention $\det D^2 u = f$ (2D).
Each $B = \sigma \lambda \sigma^T$; the second derivatives along the columns of $\sigma$ are the semi-Lagrangian differences $(u(x - k \sigma_j) - 2 u(x) + u(x + k \sigma_j)) / k^2$ (eq. (18)), whose ends take the value of the continuous piecewise linear finite element function.
Section 8: the control set is $SO(2) \times \lbrace \operatorname{diag}(a, 1 - a): a \in [0, 1/2)\rbrace$ together with $(\mathrm{Id}, \mathrm{Id}/2)$; $SO(2)$ is discretised by the rotation angles $i \pi / 64$, $i = 0, \ldots, 63$, and nothing is said about discretising $a$.
The stencil size is $k = m h$ with $h$ the average mesh size, reduced near the boundary to stay inside the triangulated domain (Fig. 5(b)).
Solved by Howard's algorithm (Algorithm 1), started with the control $B_0 = \mathrm{Id} / d$, stopped when a step is below $10^{-6}$ in the max norm.
The domain is the union of the unit disk and the unit square $(0, 1)^2$; the quasi-uniform grid has $91$ nodes at the coarsest level and $1{,}214{,}465$ after $7$ uniform refinements, computed with FEniCS.
Problems: the quartic $u = \lvert x \rvert^4$ (T39) and the nonsmooth $u = \lvert x_1 \rvert$ with $f = 0$ (T40), whose kink "is not aligned with the computational mesh".

The paper prints no numbered table, but its Figure 6 is a table: the smallest relative error over $m \in \lbrace 2, 4, 8, 16, 32, 64\rbrace$ at each level, with the minimising $m$.
The errors are relative to the norm of $u$ ($\lVert u - u_h \rVert / \lVert u \rVert$, the axis label of Figure 7).

### Figure 6, quartic problem (T39)

| DoFs | $L^2$ error | $m$ | $L^\infty$ error | $m$ | $H^1$ error | $m$ |
|---|---|---|---|---|---|---|
| $91$ | $6.92 \times 10^{-2}$ | $2$ | $9.26 \times 10^{-2}$ | $2$ | $1.86 \times 10^{-1}$ | $2$ |
| $329$ | $2.99 \times 10^{-2}$ | $2$ | $3.85 \times 10^{-2}$ | $2$ | $9.08 \times 10^{-2}$ | $2$ |
| $1{,}249$ | $1.69 \times 10^{-2}$ | $4$ | $2.15 \times 10^{-2}$ | $2$ | $4.93 \times 10^{-2}$ | $4$ |
| $4{,}865$ | $7.12 \times 10^{-3}$ | $4$ | $9.59 \times 10^{-3}$ | $4$ | $2.28 \times 10^{-2}$ | $4$ |
| $19{,}201$ | $4.18 \times 10^{-3}$ | $8$ | $5.63 \times 10^{-3}$ | $4$ | $1.28 \times 10^{-2}$ | $8$ |
| $76{,}289$ | $1.78 \times 10^{-3}$ | $8$ | $2.44 \times 10^{-3}$ | $8$ | $5.82 \times 10^{-3}$ | $8$ |
| $304{,}129$ | $1.06 \times 10^{-3}$ | $16$ | $1.51 \times 10^{-3}$ | $8$ | $3.36 \times 10^{-3}$ | $8$ |
| $1{,}214{,}465$ | $4.82 \times 10^{-4}$ | $16$ | $6.59 \times 10^{-4}$ | $16$ | $1.59 \times 10^{-3}$ | $16$ |

### Figure 6, nonsmooth problem (T40)

| DoFs | $L^2$ error | $m$ | $L^\infty$ error | $m$ | $H^1$ error | $m$ |
|---|---|---|---|---|---|---|
| $91$ | $4.50 \times 10^{-2}$ | $4$ | $1.03 \times 10^{-1}$ | $4$ | $2.02 \times 10^{-1}$ | $2$ |
| $329$ | $1.62 \times 10^{-2}$ | $4$ | $5.69 \times 10^{-2}$ | $4$ | $1.51 \times 10^{-1}$ | $4$ |
| $1{,}249$ | $7.11 \times 10^{-3}$ | $8$ | $3.08 \times 10^{-2}$ | $8$ | $1.21 \times 10^{-1}$ | $8$ |
| $4{,}865$ | $3.35 \times 10^{-3}$ | $16$ | $2.03 \times 10^{-2}$ | $16$ | $9.80 \times 10^{-2}$ | $16$ |
| $19{,}201$ | $1.70 \times 10^{-3}$ | $32$ | $1.38 \times 10^{-2}$ | $32$ | $7.91 \times 10^{-2}$ | $32$ |
| $76{,}289$ | $9.63 \times 10^{-4}$ | $32$ | $9.12 \times 10^{-3}$ | $32$ | $6.33 \times 10^{-2}$ | $32$ |
| $304{,}129$ | $5.10 \times 10^{-4}$ | $64$ | $6.04 \times 10^{-3}$ | $64$ | $5.12 \times 10^{-2}$ | $64$ |
| $1{,}214{,}465$ | $3.12 \times 10^{-4}$ | $64$ | $4.51 \times 10^{-3}$ | $64$ | $4.25 \times 10^{-2}$ | $64$ |

### Rates (p. 711), "across the seven levels of refinement"

| | quartic, in $h$ | quartic, in $k$ | nonsmooth, in $h$ | nonsmooth, in $k$ |
|---|---|---|---|---|
| $L^2$ | $h^{1.02}$ | $k^{1.79}$ | $h^{1.02}$ | $k^{2.39}$ |
| $L^\infty$ | $h^{1.02}$ | $k^{1.78}$ | $h^{0.64}$ | $k^{1.50}$ |
| $H^1$ | $h^{0.98}$ | $k^{1.72}$ | $h^{0.33}$ | $k^{1.22}$ |

The $L^\infty$ rates are the end-to-end slopes of the Figure 6 columns: $\log_2(9.26 \times 10^{-2} / 6.59 \times 10^{-4}) / 7 = 1.02$ and $\log_2(1.03 \times 10^{-1} / 4.51 \times 10^{-3}) / 7 = 0.64$ in $h$, which halves at each refinement; in $k = m h$, which the best $m$ makes shrink $16$ and $8$ times less, $7.13 / 4 = 1.78$ and $4.51 / 3 = 1.50$.

### Figure 8, Howard steps to a step below $10^{-6}$

Over all levels and all $m$, $4$ to $16$ steps; at the $m$ that minimises the $L^\infty$ error (the boxed entries), $5$ to $8$ on the quartic problem and $5$ to $9$ on the nonsmooth one.

## 8. Benamou, Froese, Oberman, J. Comput. Phys. (2014)

J.-D. Benamou, B. D. Froese, A. M. Oberman, "Numerical solution of the optimal transportation problem using the Monge-Ampere equation", J. Comput. Phys. 260 (2014) 107-126, DOI 10.1016/j.jcp.2013.12.015.
Read: local journal PDF (Zotero item W5FIFMRA, attachment ZGDPYBIE).

The transport boundary condition as an upwind Hamilton-Jacobi equation on the boundary, the filtered scheme inside, solved by Newton.
Our values next to these are in `bfo2014.md`, which states each table's grid convention (Table 1's max column counts $N$ nodes per side, Finding 1 there).
Every value below was read from the text layer and checked against rendered images of pp. 118-123.

Grid: the paper's $N_X$ is "the number of grid points $N_X$ along each dimension" (Section 5.2.1, p. 117), so it counts nodes; Table 4 says "$N_X$" in the header too.
Norm: the maximum over $x \in X$ of $\lVert \nabla u_{ex}(x) - \nabla u_{comp}(x) \rVert_2$ (Section 6, p. 118); the $L^2$ column of Table 1 does not say over which nodes it is summed.
Each of Tables 1 to 3 gives the number of Newton iterations and the time for the largest $N_Y$ ($512$) only; the paper notes that the time hardly varies with $N_Y$.
The time is MATLAB on a $2.2$ GHz Intel Core i7 laptop with $8$ GB (Section 6.6).

Text-layer problems: the matrix $M_Y = ((0.6, 0.2), (0.2, 0.8))$ prints as "0.60.2 / 0.20.8"; the radii and centres of the half-disks ($0.2$, $0.1$, $0.85$) are dropped from the formulas of Section 6.3 and printed in the image only; Section 6.2 prints $Y = M_y B_2$ in the text layer, while the image has $Y = M_y B_1$; the CPU column of Table 4 loses digits ("$.2$", "$0 .8$").

### Table 1 (Section 6.1, p. 118): the square onto itself (T16)

The square $(-0.5, 0.5)^2$ onto a uniform density in the same square, $f$ built from $q(z)$ of the paper; exact map $(x_1 + 4 q'(x_1) q(x_2), x_2 + 4 q(x_1) q'(x_2))$.
The target is a square, so its boundary is not discretised and $N_Y$ does not apply.

| $N_X$ | Maximum error | $L^2$ error | Iterations | CPU time (s) |
|---|---|---|---|---|
| $32$ | $0.0220$ | $0.0127$ | $2$ | $0.05$ |
| $64$ | $0.0110$ | $0.0064$ | $2$ | $0.18$ |
| $128$ | $0.0055$ | $0.0032$ | $3$ | $1.16$ |
| $256$ | $0.0028$ | $0.0016$ | $3$ | $6.44$ |
| $512$ | $0.0014$ | $0.0008$ | $3$ | $32.86$ |

### Table 2 (Section 6.2, p. 119): an ellipse onto an ellipse

$X = M_X B_1$, $Y = M_Y B_1$ with $M_X = ((0.8, 0), (0, 0.4))$ and $M_Y = ((0.6, 0.2), (0.2, 0.8))$, constant densities; exact map $\nabla u(x) = M_Y R_\theta M_X^{-1} x$.
The size of the square that holds $X$ is not printed (Finding 4 in `bfo2014.md`).

| $N_X$ | $N_Y = 32$ | $64$ | $128$ | $256$ | $512$ | Iterations | Time (s) |
|---|---|---|---|---|---|---|---|
| $32$ | $0.0691$ | $0.0667$ | $0.0662$ | $0.0660$ | $0.0660$ | $4$ | $0.2$ |
| $64$ | $0.0306$ | $0.0284$ | $0.0279$ | $0.0277$ | $0.0277$ | $4$ | $0.5$ |
| $128$ | $0.0203$ | $0.0176$ | $0.0169$ | $0.0167$ | $0.0167$ | $4$ | $1.7$ |
| $256$ | $0.0127$ | $0.0096$ | $0.0088$ | $0.0086$ | $0.0088$ | $5$ | $10.1$ |
| $512$ | $0.0086$ | $0.0056$ | $0.0047$ | $0.0045$ | $0.0047$ | $5$ | $52.2$ |

### Table 3 (Section 6.3, p. 120): two half-disks onto a disk

$X = \lbrace x_1 < -0.2, (x_1 + 0.2)^2 + x_2^2 < 0.85^2 \rbrace \cup \lbrace x_1 > 0.1, (x_1 - 0.1)^2 + x_2^2 < 0.85^2 \rbrace$ onto $Y = \lbrace y_1^2 + y_2^2 < 0.85^2 \rbrace$; the paper calls the problem degenerate, and notes that with a fixed stencil width the error need not tend to zero.

| $N_X$ | $N_Y = 32$ | $64$ | $128$ | $256$ | $512$ | Iterations | Time (s) |
|---|---|---|---|---|---|---|---|
| $32$ | $0.0280$ | $0.0284$ | $0.0286$ | $0.0286$ | $0.0286$ | $3$ | $0.2$ |
| $64$ | $0.0158$ | $0.0164$ | $0.0165$ | $0.0165$ | $0.0165$ | $3$ | $0.4$ |
| $128$ | $0.0092$ | $0.0093$ | $0.0092$ | $0.0092$ | $0.0092$ | $3$ | $1.3$ |
| $256$ | $0.0047$ | $0.0036$ | $0.0036$ | $0.0036$ | $0.0036$ | $4$ | $8.3$ |
| $512$ | $0.0049$ | $0.0040$ | $0.0034$ | $0.0033$ | $0.0033$ | $5$ | $51.7$ |

### Table 4 (Section 6.4, p. 121): forward map against the inverse of the inverse map

Gaussian densities, four Gaussians at the corners as the source and one at the centre as the target; the paper says "the unit square" but writes the quadrant $[-1, 0] \times [-1, 0]$, so the domain is $[-1, 1]^2$.
Distance: $\max_{x \in X} \lVert \nabla u(\nabla u^*(x)) - x \rVert$; the target is a square, so the results do not depend on $N_Y$.
Not compared: the target densities are not uniform.

| $N_X$ | Max distance | Iterations, forward | Iterations, inverse | CPU (s), forward | CPU (s), inverse |
|---|---|---|---|---|---|
| $32$ | $0.2337$ | $4$ | $5$ | $0.2$ | $0.2$ |
| $64$ | $0.1252$ | $5$ | $6$ | $0.4$ | $0.8$ |
| $128$ | $0.0649$ | $5$ | $6$ | $1.9$ | $3.4$ |
| $256$ | $0.0329$ | $8$ | $8$ | $15.7$ | $20.8$ |
| $512$ | $0.0165$ | $9$ | $10$ | $91.3$ | $128.4$ |

### Table 5 (Section 6.6, p. 123): Pogorelov solutions, normalised errors in percent, $N_X = 256$

A Lebesgue source onto $N_d$ weighted Dirac masses; the errors are those of the cell areas against the optimality condition (9).
Not compared: the target is a sum of Dirac masses.

| $N_d$ | $L^\infty$ error | $L^2$ error | CPU time (s) |
|---|---|---|---|
| $3$ | $0.05$ | $0.02$ | $10.48$ |
| $30$ | $0.48$ | $0.21$ | $10.56$ |
| $300$ | $0.56$ | $0.18$ | $11.8$ |

### Table 6 (Section 6.6, p. 123): $300$ Dirac masses, increasing resolution, errors in percent

| $N_X$ | $L^\infty$ error | $L^2$ error | CPU time (s) |
|---|---|---|---|
| $64$ | $0.93$ | $0.28$ | $1.48$ |
| $128$ | $0.96$ | $0.24$ | $2.6$ |
| $256$ | $0.83$ | $0.21$ | $11.7$ |
| $512$ | $0.66$ | $0.20$ | $46.76$ |
| $1024$ | $0.64$ | $0.20$ | $281.53$ |

The paper's text says the cost is linear until $N_X = 1024$, where out-of-core memory overheads set in.

## 9. Chen, Wan, Lin, J. Sci. Comput. (2018)

Y. Chen, J. W. L. Wan, J. Lin, "Monotone mixed finite difference scheme for Monge-Ampere equation", J. Sci. Comput. 76 (2018) 1839-1867, DOI 10.1007/s10915-018-0685-y.
Read: local journal PDF (Zotero attachments 3XWEWWT6, 9WII8MMM; the survey calls the paper "Chen-Wan-Wang 2018").

The HJB form, each control on the standard 7-point stencil where it is monotone and on the semi-Lagrangian wide stencil elsewhere, solved by policy iteration; grid: $N$ cells per side (ADR 0001), so $h = 1/N$ except on Table 1's $[-1, 1]^2$, where $h = 2/N$ (Finding 1 there).
Our values next to these are in `chen_wan_lin2018.md`.
Every value below was read from the text layer and checked against rendered images of pp. 1861-1866 (PDF pages 23 to 28).

Grid: Section 4 sets up "an $N \times N$ square grid" with the unknowns at $i, j = 1, \ldots, N$ and the boundary at $0$ and $N + 1$, and Section 7 takes "the grid size $N^2 = 32^2, 64^2, \ldots, 512^2$", so the paper's words give $N + 1$ cells per side; its values match $N$ cells (Finding 1 there).
Norm: the tables print $\lVert u - u_h \rVert_2$ and $\lVert u - u_h \rVert_\infty$ without a definition; the order is $\log_2$ of the ratio of the errors at $N$ and $2N$, printed against the finer $N$.
Every example uses the initial guess $\Delta u = 2 \sqrt{f}$ with the Dirichlet data (eq. (50)) and the tolerance $10^{-6}$ on the residual of the policy iteration.
The control set is $M \times M$; Section 7 does not state $M$, Section 5.2 says that a typical choice is $M = N$, and Section 6.1 chooses it for consistency.

Text-layer problems: the square roots of the exact solutions and of $f$ drop out (Example 2's $u = -\sqrt{2 - x^2 - y^2}$, Example 3's $f = \max(1 - 0.1/\sqrt{x^2 + y^2}, 0)$, $g$ and $u$, Example 4's $u = \sqrt{x^2 + y^2}$, and the exponent of Example 1's $u = e^{(x^2 + y^2)/2}$ prints as "e 2 (x +y )" with the $1/2$ detached); the minus signs of the exponents ($\times 10^{-3}$) print as the Unicode minus and survive, but Tables 2 and 4 print some rows as "$4.084 \times 10^{-6}$" without the spaces around the multiplication sign; the rate formula of Section 7 loses its fraction bar.

Problems:
Example 1: $f = (1 + x^2 + y^2) e^{x^2 + y^2}$, $g = u = e^{(x^2 + y^2)/2}$ on $[-1, 1]^2$ (T1 moved from $(0, 1)^2$).
Example 2: $f = 2 / (2 - x^2 - y^2)^2$, $g = u = -\sqrt{2 - x^2 - y^2}$ on $[0, 1]^2$ (T2); $f$ is singular at $(1, 1)$.
Example 3: $f = \max(1 - 0.1 / \lvert x \rvert, 0)$, $g = u = \tfrac12 (\max(\lvert x \rvert - 0.1, 0))^2$ on $[-0.5, 0.5]^2$ (T20c).
Example 4: $f = \pi \delta(0, 0)$, $g = u = \lvert x \rvert$ on $[-0.5, 0.5]^2$ (the cone T21); the paper does not say how the Dirac mass is discretised (Finding 3 in `chen_wan_lin2018.md`).
Example 5: $f = 1$, $g = 0$ on $[-0.5, 0.5]^2$; no closed form, so $u_h(0, 0)$ is tabulated.

In Tables 1 to 3, rows (1) are the proposed mixed stencil scheme and rows (2) the pure semi-Lagrangian wide stencil scheme of Feng and Jensen; Table 4 has the mixed scheme only.
Columns: $\lVert u - u_h \rVert_2$, its order, $\lVert u - u_h \rVert_\infty$, its order, the number of policy iterations; a blank order is blank in the paper.

### Table 1, Example 1

(1) Proposed mixed stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.201 \times 10^{-3}$ |  | $9.598 \times 10^{-4}$ |  | $4$ |
| $64$ | $3.009 \times 10^{-4}$ | $2.00$ | $2.404 \times 10^{-4}$ | $2.00$ | $4$ |
| $128$ | $7.526 \times 10^{-5}$ | $2.00$ | $6.013 \times 10^{-5}$ | $2.00$ | $4$ |
| $256$ | $1.882 \times 10^{-5}$ | $2.00$ | $1.504 \times 10^{-5}$ | $2.00$ | $4$ |
| $512$ | $4.705 \times 10^{-6}$ | $2.00$ | $3.759 \times 10^{-6}$ | $2.00$ | $4$ |

(2) Pure semi-Lagrangian wide stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.868 \times 10^{-2}$ |  | $1.557 \times 10^{-2}$ |  | $5$ |
| $64$ | $1.020 \times 10^{-2}$ | $0.87$ | $8.364 \times 10^{-3}$ | $0.90$ | $5$ |
| $128$ | $5.263 \times 10^{-3}$ | $0.95$ | $4.240 \times 10^{-3}$ | $0.98$ | $6$ |
| $256$ | $2.801 \times 10^{-3}$ | $0.91$ | $2.259 \times 10^{-3}$ | $0.91$ | $5$ |
| $512$ | $1.600 \times 10^{-3}$ | $0.81$ | $1.268 \times 10^{-3}$ | $0.83$ | $5$ |

### Table 2, Example 2

(1) Proposed mixed stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $6.450 \times 10^{-5}$ |  | $2.359 \times 10^{-4}$ |  | $4$ |
| $64$ | $1.628 \times 10^{-5}$ | $1.99$ | $8.211 \times 10^{-5}$ | $1.52$ | $5$ |
| $128$ | $4.084 \times 10^{-6}$ | $2.00$ | $2.882 \times 10^{-5}$ | $1.51$ | $5$ |
| $256$ | $1.022 \times 10^{-6}$ | $2.00$ | $1.015 \times 10^{-5}$ | $1.51$ | $5$ |
| $512$ | $2.557 \times 10^{-7}$ | $2.00$ | $3.583 \times 10^{-6}$ | $1.50$ | $5$ |

(2) Pure semi-Lagrangian wide stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.493 \times 10^{-3}$ |  | $5.799 \times 10^{-3}$ |  | $5$ |
| $64$ | $9.634 \times 10^{-4}$ | $0.63$ | $4.394 \times 10^{-3}$ | $0.40$ | $4$ |
| $128$ | $5.166 \times 10^{-4}$ | $0.90$ | $2.697 \times 10^{-3}$ | $0.70$ | $5$ |
| $256$ | $3.153 \times 10^{-4}$ | $0.71$ | $1.824 \times 10^{-3}$ | $0.56$ | $5$ |
| $512$ | $1.583 \times 10^{-4}$ | $0.99$ | $1.120 \times 10^{-3}$ | $0.70$ | $5$ |

### Table 3, Example 3

(1) Proposed mixed stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.270 \times 10^{-4}$ |  | $4.298 \times 10^{-4}$ |  | $4$ |
| $64$ | $4.273 \times 10^{-5}$ | $1.57$ | $1.520 \times 10^{-4}$ | $1.50$ | $6$ |
| $128$ | $1.835 \times 10^{-5}$ | $1.22$ | $6.907 \times 10^{-5}$ | $1.14$ | $7$ |
| $256$ | $1.544 \times 10^{-5}$ | $0.25$ | $5.959 \times 10^{-5}$ | $0.21$ | $9$ |
| $512$ | $3.396 \times 10^{-6}$ | $2.18$ | $1.513 \times 10^{-5}$ | $1.98$ | $20$ |

(2) Pure semi-Lagrangian wide stencil scheme:

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.337 \times 10^{-3}$ |  | $6.604 \times 10^{-3}$ |  | $5$ |
| $64$ | $9.084 \times 10^{-4}$ | $0.56$ | $3.304 \times 10^{-3}$ | $1.00$ | $6$ |
| $128$ | $6.940 \times 10^{-4}$ | $0.39$ | $1.901 \times 10^{-3}$ | $0.80$ | $7$ |
| $256$ | $3.815 \times 10^{-4}$ | $0.86$ | $9.335 \times 10^{-4}$ | $1.03$ | $7$ |
| $512$ | $1.998 \times 10^{-4}$ | $0.93$ | $4.563 \times 10^{-4}$ | $1.03$ | $9$ |

### Table 4, Example 4 (the cone), mixed scheme

| $N$ | $\lVert u - u_h \rVert_2$ | order | $\lVert u - u_h \rVert_\infty$ | order | policy iterations |
|---|---|---|---|---|---|
| $32$ | $1.156 \times 10^{-3}$ |  | $3.868 \times 10^{-3}$ |  | $9$ |
| $64$ | $6.484 \times 10^{-4}$ | $0.83$ | $2.583 \times 10^{-3}$ | $0.58$ | $15$ |
| $128$ | $3.803 \times 10^{-4}$ | $0.77$ | $1.848 \times 10^{-3}$ | $0.48$ | $17$ |
| $256$ | $2.159 \times 10^{-4}$ | $0.82$ | $1.305 \times 10^{-3}$ | $0.50$ | $23$ |
| $512$ | $1.148 \times 10^{-4}$ | $0.91$ | $9.203 \times 10^{-4}$ | $0.50$ | $27$ |

### Table 5, Example 5: $u_h(0, 0)$

| $N$ | monotone scheme (mixed) | non-monotone scheme |
|---|---|---|
| $32$ | $-0.18380$ | $0.18063$ |
| $64$ | $-0.18444$ | $0.18312$ |
| $128$ | $-0.18461$ | $0.18436$ |
| $256$ | $-0.18485$ | $0.18499$ |
| $512$ | $-0.18507$ | $0.18530$ |

The caption calls the first column the minimum values $u_{\min}$ of the numerical solutions (convex, the viscosity solution) and the second the maximum values $u_{\max}$ (concave, not a viscosity solution), while the header says $u_h(0, 0)$.
The non-monotone scheme is the standard central $9$-point discretisation of $u_{xx} u_{yy} - u_{xy}^2 = f$.

## What a reproduction must match

For the standard 9-point, 17-point monotone and hybrid schemes, the published error tables for (a) and (b) in the Dirichlet setting are Table 3 of the JCP paper and Table 3 of the SIAM paper; sections 5, 6 and 9 add the tables of other schemes on T1 and T2.
They report the max norm of $u$ minus the exact solution at the grid points, on an $N \times N$ grid with $N \in \lbrace 31, 45, 63, 89, 127, 181, 255, 361\rbrace$ (SIAM uses the subset $31$, $63$, $127$, $255$, $361$).
For the standard 9-point scheme, solved to its fixed point by Gauss-Seidel, Poisson or any other convergent iteration, the targets are:
for (a), from $7.14 \times 10^{-5}$ at $N = 31$ down to $0.05 \times 10^{-5}$ at $N = 361$, order $O(h^2)$;
for (b), from $17.15 \times 10^{-3}$ at $N = 31$ down to $5.41 \times 10^{-3}$ at $N = 361$, order about $O(h^{0.5})$.
For (b), the 17-point monotone scheme should reach about $0.3 \times 10^{-3}$, and the JCP hybrid scheme should reach $0.04 \times 10^{-3}$ at $N = 361$, order about $O(h^{1.5})$.
No paper tabulates Newton on the standard scheme, and the JCP paper reports that it fails on (b), so Newton iteration counts can only be compared for the monotone (SIAM Table 1) and hybrid (JCP Table 1) schemes, where they stay between $3$ and $9$ for (a) and (b).
Iteration counts for Gauss-Seidel and Poisson come only from the 2010 paper at $N = 21$ to $141$ (to $581$ for Poisson on (a)), with a $10^{-14}$ stopping rule on successive iterates.
The CPU times are bound to 2010 MATLAB hardware, so only the scaling exponents (Newton about $M^{1.3}$, Poisson about $M^{1.4}$, Gauss-Seidel about $M^{1.8}$ to $M^2$) are reproducible.
Before matching errors to two digits, fix the grid convention, because none of the papers defines $h$ for its $N$ clearly.
