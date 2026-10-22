# Published finite-difference benchmarks for the 2D Dirichlet Monge-Ampere problem

Target problem: det D^2 u = f on (0,1)^2 with Dirichlet data from the exact solution.
Target solution (a): u = exp(|x|^2 / 2), f = (1 + |x|^2) exp(|x|^2).
Target solution (b): u = -sqrt(2 - |x|^2), f = 2 (2 - |x|^2)^(-2), gradient unbounded at the corner (1,1).

All numbers below were copied from the journal versions, read from local PDF copies in the Zotero library.
Each value was checked against a rendered image of the printed page.
The arXiv and Numdam links are public equivalents; their tables were not checked against the journal tables.
Values keep the printed scaling (for example "24.45 x 10^-5", not 2.445e-4).
A "--" means the paper printed a dash: the run was not done, or it was too slow to report.
The neural sources are transcribed next to their comparison: Nystrom-Vestberg 2023 in `penalty_mlp.md`.

## Grid convention (unresolved)

Benamou-Froese-Oberman 2010, Section 3.3, says verbatim: "The solutions are computed on an N x N grid ... By convention, h is the spatial discretization parameter h = L/N where L is the length of one side of the square domain."
Froese-Oberman JCP 2011, Section 8, says only: "These computations are performed on an N x N grid on the square [0, 1]^2."
Froese-Oberman SIAM 2011, Section 6, says only: "We perform the computations using a 17 point stencil on an N x N grid on a square."
SIAM Table 2 adds: "Here M = N^2 is the total number of grid points."
None of the 2011 papers defines h in terms of N.
The odd N values (21, 41, ...; 31, 63, 127, 255) suggest N points per side including the boundary, so h = 1/(N-1), but no paper says so.
A useful check: the standard-scheme max error for (a) at N = 31 is 7.14 x 10^-5, and it should tell h = 1/(N-1) apart from h = 1/N or h = 1/(N+1).
One table does not follow h = L/N: on [-1, 1]^2, BFO 2010 Table 7 matches our solvers at n = N - 1 cells (`standard_fd.md`, Finding 3).

## 1. Benamou, Froese, Oberman (2010)

J.-D. Benamou, B. D. Froese, A. M. Oberman, "Two numerical methods for the elliptic Monge-Ampere equation", ESAIM: M2AN 44 (2010) 737-758, DOI 10.1051/m2an/2010017.
Read: local journal PDF (Zotero item RLL6V439).
Open copy: https://www.numdam.org/item/M2AN_2010__44_4_737_0 (Numdam, not checked).
No arXiv version was found.

Methods: both solve the same standard 9-point central-difference discretisation (D_xx u)(D_yy u) - (D_xy u)^2 = f, with D_xy on the four diagonal neighbours divided by 4h^2.
(M1) is the explicit Gauss-Seidel iteration: at each point solve the local quadratic for u_ij and take the smaller root.
(M2) is the Poisson iteration: Laplacian u^{n+1} = sqrt((u^n_xx)^2 + (u^n_yy)^2 + 2 (u^n_xy)^2 + 2f), with central differences and Dirichlet data.
The paper states that both methods reach the same discrete solution, so the error depends only on N (Section 3, item 1).
Stopping rule: the max difference between two successive iterates is less than 10^-14.
Initial guess for the timings: the solution of u_xx + u_yy = sqrt(2f).

Name map for this paper:
- Solution (a) is Section 4.1, "An exact radial solution".
- Solution (b) is Section 5.4, "Surface of ball", on the domain [0,1] x [0,1].
- Watch out: Section 4.2 is titled "Blow-up at boundary", but it is a different solution, u = (2 sqrt(2)/3)(x^2+y^2)^(3/4), f = 1/sqrt(x^2+y^2), where f blows up at the corner (0,0). Its Table 3 is not a table for (b).

The paper prints no error table for (a) or (b).
For (a), the error is only shown in Figure 2(a), and the text says the accuracy is O(h^2).
For (b), the error is only shown in Figure 8(b), with the label "Slope ~ -0.45", and the text says "the convergence rate is about .5".
The tables for (a) and (b) give iteration counts and CPU time only.

### Solution (a): Table 2, computation times on an N x N grid

The paper skips rows between N = 261 and N = 341 (printed as a vertical ellipsis).

| N | Iterations (M1) | Iterations (M2) | CPU time s (M1) | CPU time s (M2) |
|---|---|---|---|---|
| 21 | 1014 | 54 | 0.9 | 0.1 |
| 81 | 14 621 | 59 | 39.8 | 3.4 |
| 141 | 42 746 | 60 | 350.0 | 11.7 |
| 201 | 84 611 | 60 | 1540.0 | 27.2 |
| 261 | -- | 65 | -- | 57.8 |
| ... | ... | ... | ... | ... |
| 341 | -- | 65 | -- | 118.0 |
| 421 | -- | 78 | -- | 249.0 |
| 501 | -- | 151 | -- | 804.0 |
| 581 | -- | 284 | -- | 2120.0 |

Table 1 also gives the CPU time ratio (M1)/(M2) for (a) (Section 4.1 column): 6.3, 4.1, 7.7, 11.7, 14.1, 23.5, 29.9 at N = 21, 41, 61, 81, 101, 121, 141.

### Solution (b): Table 9, computation times on an N x N grid

| N | Iterations (M1) | Iterations (M2) | CPU time s (M1) | CPU time s (M2) |
|---|---|---|---|---|
| 21 | 1009 | 67 | 0.2 | 0.2 |
| 41 | 3833 | 79 | 1.9 | 1.0 |
| 61 | 8343 | 89 | 8.5 | 2.7 |
| 81 | 14 480 | 99 | 26.3 | 5.2 |
| 101 | 22 189 | 106 | 60.2 | 10.0 |
| 121 | 31 445 | 113 | 120.1 | 15.0 |
| 141 | 42 202 | 119 | 218.3 | 21.3 |

### T7 (Section 5.3): Table 7, minimum value of u when f = 1 with constant boundary values 1 on an N x N grid

| N | (M1), (M2) | 9-point stencil | 17-point stencil | 33-point stencil |
|---|---|---|---|---|
| 21 | 0.2892 | 0.3115 | 0.2815 | 0.2839 |
| 41 | 0.2734 | 0.3090 | 0.2807 | 0.2732 |
| 61 | 0.2682 | 0.3082 | 0.2803 | 0.2711 |
| 81 | 0.2655 | 0.3078 | 0.2802 | 0.2704 |
| 101 | 0.2639 | 0.3076 | 0.2800 | 0.2700 |
| 121 | 0.2629 | 0.3075 | 0.2798 | 0.2697 |
| 141 | 0.2621 | 0.3074 | 0.2796 | 0.2695 |

### T7 (Section 5.3): Table 8, computation times

| N | Iterations (M1) | Iterations (M2) | CPU time (s) (M1) | CPU time (s) (M2) |
|---|---|---|---|---|
| 21 | 1615 | 535 | 0.1 | 1.7 |
| 41 | 6366 | 1917 | 0.6 | 33.6 |
| 61 | 14 049 | 4070 | 2.8 | 254.0 |
| 81 | 24 560 | 6944 | 8.1 | 1170.0 |
| 101 | 37 824 | 10 503 | 19.9 | 3730.0 |
| 121 | 53 778 | 14 711 | 41.2 | 6820.0 |
| 141 | 72 377 | 19 571 | 99.7 | 8560.0 |

Our comparison, and the node-count convention of Table 7: `standard_fd.md`, section "T7" and Finding 3.

### Other test solutions in this paper

- Section 4.2: u = (2 sqrt(2)/3)(x^2+y^2)^(3/4) on [0,1]^2, f = 1/sqrt(x^2+y^2); max error in Table 3 (5.5 x 10^-4 at N = 21 to 1.5 x 10^-5 at N = 221), order 1.5.
- Section 5.1: solution (a) with high-frequency sinusoidal noise added to f and to the boundary data (Table 5, times only).
- Section 5.2: discontinuous f (1 outside a circle of radius 0.16 about (0.5,0.5), 0 inside), u = max(|x - (0.5,0.5)|^2 / 2, 0.08) (Table 6, times only).
- Section 5.3 (T7): f = 1 with boundary value 1 on [-1,1]^2, no exact solution; Tables 7 and 8 above; the 9-, 17- and 33-point columns of Table 7 are the monotone wide-stencil method of the paper's reference [21].
- Section 6.1: the cone u = sqrt(x^2+y^2), f = 4/h^2 at the origin grid point; max error in Tables 10 and 13, order about 0.8.
- Section 6.2: u = |x|, f = 0; the discrete error goes to zero (Table 12, times only).

### Caveat on the printed solution (maesolvers catalogue, T4)

- Section 5.2 (T4): the printed u = max(|x - x0|^2 / 2, 0.08) does not solve the printed problem.
  Its gradient jumps from 0 to the radial vector of length 0.4 on the circle r = 0.4, so its Monge-Ampere measure has a singular part of mass 0.16 pi on that circle, and the printed f (integral 1 - 0.16 pi on the unit square) omits it.
  A correct solver of the printed f with the trace of the printed u as Dirichlet data converges to a different function, with no known closed form, so the printed u is no reference for an error.
  The catalogue's T4 keeps the printed f and replaces u with the C^1 radial Alexandrov solution of that f, 0.08 on the disk and 0.08 + (r s - 0.16 ln((r + s)/0.4))/2 with s = sqrt(r^2 - 0.16) beyond, and takes its Dirichlet data from it; so T4's boundary data differ from the paper's.
  `tests/test_catalogue.py` checks that the area of grad u(Omega) equals the integral of f, which fails for the printed u (area 1 against 1 - 0.16 pi).

## 2. Froese, Oberman, SIAM J. Numer. Anal. (2011)

B. D. Froese, A. M. Oberman, "Convergent finite difference solvers for viscosity solutions of the elliptic Monge-Ampere equation in dimensions two and higher", SIAM J. Numer. Anal. 49 (2011) 1692-1714, DOI 10.1137/100803092.
Read: local journal PDF from the Zotero folder (two copies of the same journal version).
Open copy: https://arxiv.org/abs/1007.0765 (not checked).

Method: a wide-stencil monotone discretisation MA_M[u] = min over orthogonal grid-direction pairs (nu_1, nu_2) of the product of the positive parts of D_{nu nu} u.
Section 3.5 defines a regularised scheme MA^delta, with max^delta(a, b) = (a + b + sqrt((a - b)^2 + delta^2))/2 and min^delta likewise (delta^2 read in arXiv 1007.0765; the journal text layer garbles it), the positive parts max^delta(D u, 0) and the minimum over pairs a succession of min^delta; Theorem 9 proves that Newton converges for MA^delta.
No value of delta is printed, and Section 5 states Newton for MA_M itself, with Danskin's theorem for the Jacobian, so the tables need not come from MA^delta.
The equations are solved by damped Newton, with the step reduced until the residual decreases.
The linear systems are solved by MATLAB backslash (sparse LU).
The initial guess is one step of the semi-implicit scheme, Laplacian u^0 = sqrt(2f) in 2D, convexified only for the most singular example.
The Jacobian is regularised by u_{nu nu} -> max(u_{nu nu}, epsilon), with epsilon = 10^-8 / (2 dx^2) (Section 5.3, read from the page image; the text layer garbles it).
Section 6 uses a 17-point stencil for Table 1; Table 3 also uses 9- and 33-point stencils.
No Newton stopping tolerance is stated.
The "F.D." columns of Table 3 are the standard 9-point scheme "using the results of [5]" ([5] is Benamou-Froese-Oberman 2010).
That N sequence does not appear in the 2010 paper's tables, so these values were recomputed for this paper, not copied from 2010.
The same applies to the Poisson and Gauss-Seidel CPU columns of Table 1.

Name map for this paper:
- Solution (a) is "C^2 example", eq. (10).
- Solution (b) is "Example with blowup", eq. (12).

### Solution (a), "C^2 example (10)"

Table 3: max error.

| N | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| 31 | 17.9 x 10^-4 | 8.9 x 10^-4 | 7.0 x 10^-4 | 7.14 x 10^-5 |
| 63 | 16.2 x 10^-4 | 5.1 x 10^-4 | 3.1 x 10^-4 | 1.73 x 10^-5 |
| 127 | 15.9 x 10^-4 | 4.6 x 10^-4 | 1.8 x 10^-4 | 4.3 x 10^-6 |
| 255 | 15.9 x 10^-4 | 4.4 x 10^-4 | 1.5 x 10^-4 | 1.1 x 10^-6 |
| 361 | 15.9 x 10^-4 | 4.4 x 10^-4 | 1.5 x 10^-4 | 0.5 x 10^-6 |

Table 1: Newton on the 17-point monotone scheme; Poisson and Gauss-Seidel are the 2010 methods (M2) and (M1).

| N | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| 31 | 3 | 0.2 | 0.7 | 2.2 |
| 63 | 6 | 0.9 | 1.9 | 15.0 |
| 127 | 7 | 4.9 | 9.6 | 236.7 |
| 255 | 7 | 43.5 | 52.6 | -- |
| 361 | 7 | 172.8 | 162.6 | -- |

### Solution (b), "Example with blowup (12)"

Table 3: max error.

| N | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| 31 | 1.7 x 10^-3 | 1.7 x 10^-3 | 1.7 x 10^-3 | 17.15 x 10^-3 |
| 63 | 0.9 x 10^-3 | 0.6 x 10^-3 | 0.6 x 10^-3 | 12.53 x 10^-3 |
| 127 | 0.8 x 10^-3 | 0.3 x 10^-3 | 0.2 x 10^-3 | 9.00 x 10^-3 |
| 255 | 0.8 x 10^-3 | 0.3 x 10^-3 | 0.2 x 10^-3 | 6.42 x 10^-3 |
| 361 | 0.8 x 10^-3 | 0.3 x 10^-3 | 0.2 x 10^-3 | 5.41 x 10^-3 |

Table 1: Newton on the 17-point monotone scheme.

| N | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| 31 | 4 | 0.3 | 0.5 | 0.8 |
| 63 | 4 | 0.5 | 2.9 | 19.4 |
| 127 | 5 | 3.7 | 17.7 | 293.3 |
| 255 | 7 | 41.4 | 128.2 | -- |
| 361 | 9 | 184.1 | 374.5 | -- |

Table 2 gives the CPU time scaling, with M = N^2: Gauss-Seidel about O(M^1.8) on C^{2,alpha} and O(M^1.9) on C^{1,alpha}; Poisson about O(M^1.4), "blowup" on less regular solutions; Newton about O(M^1.3) for all regularities.

### C^1 example (11), T20 in the catalogue

Table 3: max error; the F.D. column prints a dash at N = 255 and 361.

| N | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | F.D. |
|---|---|---|---|---|
| 31 | 3.0 x 10^-3 | 1.7 x 10^-3 | 1.5 x 10^-3 | 2.6 x 10^-4 |
| 63 | 2.5 x 10^-3 | 1.0 x 10^-3 | 0.6 x 10^-3 | 1.5 x 10^-4 |
| 127 | 2.3 x 10^-3 | 0.8 x 10^-3 | 0.3 x 10^-3 | 0.6 x 10^-4 |
| 255 | 2.2 x 10^-3 | 0.7 x 10^-3 | 0.3 x 10^-3 | -- |
| 361 | 2.2 x 10^-3 | 0.7 x 10^-3 | 0.3 x 10^-3 | -- |

Table 1, Newton iterations on the 17-point scheme: 4, 7, 11, 16, 20 at N = 31, 63, 127, 255, 361.

### C^{0,1} (Lipschitz) example (13), the cone T21

Table 3: max error; the Standard F.D. column prints a dash at N = 255 and 361.
f is the Dirac mass pi delta_{x0} averaged over the disk |x - x0| <= h/2: 4/h^2 there, 0 elsewhere (Section 6.1).

| N | Monotone 9 Point | Monotone 17 Point | Monotone 33 Point | Standard F.D. |
|---|---|---|---|---|
| 31 | 12 x 10^-3 | 3 x 10^-3 | 3 x 10^-3 | 10 x 10^-3 |
| 63 | 11 x 10^-3 | 3 x 10^-3 | 2 x 10^-3 | 6 x 10^-3 |
| 127 | 11 x 10^-3 | 4 x 10^-3 | 2 x 10^-3 | 3 x 10^-3 |
| 255 | 11 x 10^-3 | 4 x 10^-3 | 1 x 10^-3 | -- |
| 361 | 11 x 10^-3 | 4 x 10^-3 | 1 x 10^-3 | -- |

Table 1, Newton iterations on the 17-point scheme: 9, 15, 32, 34, 29 at N = 31, 63, 127, 255, 361.

### Other test solutions in this paper

- "C^1 example" (11): u = ((|x - x0| - 0.2)^+)^2 / 2, f = (1 - 0.2/|x - x0|)^+, x0 = (0.5, 0.5).
- "C^{0,1} (Lipschitz) example" (13): the cone u = |x - x0|, f = pi delta_{x0}, discretised as 4/h^2 on |x - x0| <= h/2.
- 3D (Table 4, 19-point stencil, N^3 grid on [0,1]^3): u = exp(|x|^2/2) with f = (1+|x|^2) exp(3|x|^2/2); a C^1 example; and the ball u = -sqrt(3 - |x|^2), f = 3 (3 - |x|^2)^(-5/2).

## 3. Froese, Oberman, J. Comput. Phys. (2011)

B. D. Froese, A. M. Oberman, "Fast finite difference solvers for singular solutions of the elliptic Monge-Ampere equation", J. Comput. Phys. 230 (2011) 818-834, DOI 10.1016/j.jcp.2010.10.020.
Read: local journal PDF (Zotero item N9IMUP7B).
Open copy: https://arxiv.org/abs/1006.5748 (not checked).

Methods: three discretisations, each solved by damped Newton.
"Standard" is the natural 9-point scheme (MA_N).
"Monotone" is the wide-stencil scheme (MA_M) with a 17-point stencil.
"Hybrid" is w(x) MA_N + (1 - w(x)) MA_M, with the weight w fixed a priori: w = 0 near possible singular sets (f near 0 or infinity, or a flat or non-smooth boundary), and w goes to 1 elsewhere.
On the square, the monotone scheme is used near the boundary, so the hybrid scheme is less accurate than the standard one for (a).
The linear systems are solved by LU decomposition.
The initial guess is one semi-implicit step followed by a convex envelope on a coarse grid, interpolated to the fine grid.
The Jacobian is regularised by u_{mm} -> max(u_{mm}, epsilon), with epsilon = 10^-8 / (2 dx^2) as in the SIAM paper (Section 7.2).
Hardware: a laptop with a 2 GHz Intel processor, MATLAB, Windows Vista.
No Newton stopping tolerance is stated.
Section 8.1 reports that Newton on the standard scheme fails for solution (b): started from the exact solution, after two iterations it oscillates and becomes non-convex (Fig. 2).
The "Standard" column of Table 3 is credited to [29], which is Benamou-Froese-Oberman 2010, so it comes from the (M1)/(M2) iterations, not from Newton.
That N sequence does not appear in the 2010 paper's tables, so these values were recomputed, not copied.
The same applies to the Poisson and Gauss-Seidel CPU columns of Table 1.
Where the two 2011 papers share N values, their standard-scheme errors agree (for (a) at N = 127, 0.43 x 10^-5 here and 4.3 x 10^-6 in SIAM).

Name map for this paper:
- Solution (a) is "C^2 Example", eq. (14).
- Solution (b) is "Example with blow-up", eq. (16); it is also the Section 8.1 example.

### Solution (a), "C^2 Example (14)"

Table 3: maximum error.

| N | Standard | Monotone | Hybrid |
|---|---|---|---|
| 31 | 7.14 x 10^-5 | 89.09 x 10^-5 | 24.45 x 10^-5 |
| 45 | 3.39 x 10^-5 | 60.50 x 10^-5 | 15.29 x 10^-5 |
| 63 | 1.73 x 10^-5 | 50.88 x 10^-5 | 9.06 x 10^-5 |
| 89 | 0.87 x 10^-5 | 47.51 x 10^-5 | 5.32 x 10^-5 |
| 127 | 0.43 x 10^-5 | 45.53 x 10^-5 | 3.02 x 10^-5 |
| 181 | 0.21 x 10^-5 | 44.65 x 10^-5 | 1.61 x 10^-5 |
| 255 | 0.11 x 10^-5 | 44.22 x 10^-5 | 0.87 x 10^-5 |
| 361 | 0.05 x 10^-5 | 44.00 x 10^-5 | 0.46 x 10^-5 |

Table 1: Newton on the hybrid scheme; Poisson and Gauss-Seidel are the 2010 methods (M2) and (M1).

| N | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| 31 | 3 | 0.2 | 0.7 | 2.2 |
| 45 | 4 | 0.2 | 1.1 | 4.1 |
| 63 | 4 | 0.4 | 1.9 | 15.0 |
| 89 | 4 | 1.0 | 4.8 | 57.6 |
| 127 | 5 | 2.9 | 9.6 | 236.7 |
| 181 | 5 | 9.0 | 23.2 | 1004.0 |
| 255 | 5 | 30.5 | 52.6 | -- |
| 361 | 6 | 131.4 | 162.6 | -- |

### Solution (b), "Example with blow-up (16)"

Table 3: maximum error.
The text gives the rates: hybrid O(h^1.5), standard O(h^0.5), monotone limited by the stencil width.

| N | Standard | Monotone | Hybrid |
|---|---|---|---|
| 31 | 17.15 x 10^-3 | 1.74 x 10^-3 | 1.74 x 10^-3 |
| 45 | 14.59 x 10^-3 | 0.98 x 10^-3 | 0.98 x 10^-3 |
| 63 | 12.53 x 10^-3 | 0.59 x 10^-3 | 0.59 x 10^-3 |
| 89 | 10.67 x 10^-3 | 0.37 x 10^-3 | 0.35 x 10^-3 |
| 127 | 9.00 x 10^-3 | 0.35 x 10^-3 | 0.20 x 10^-3 |
| 181 | 7.59 x 10^-3 | 0.34 x 10^-3 | 0.12 x 10^-3 |
| 255 | 6.42 x 10^-3 | 0.33 x 10^-3 | 0.07 x 10^-3 |
| 361 | 5.41 x 10^-3 | 0.33 x 10^-3 | 0.04 x 10^-3 |

Table 1: Newton on the hybrid scheme.

| N | Newton iterations | CPU s Newton | CPU s Poisson | CPU s Gauss-Seidel |
|---|---|---|---|---|
| 31 | 4 | 0.2 | 0.5 | 0.8 |
| 45 | 4 | 0.4 | 1.4 | 5.3 |
| 63 | 4 | 0.7 | 2.9 | 19.4 |
| 89 | 5 | 1.8 | 8.1 | 74.1 |
| 127 | 7 | 5.1 | 17.7 | 293.3 |
| 181 | 7 | 12.9 | 51.4 | 1637.1 |
| 255 | 7 | 36.1 | 128.2 | -- |
| 361 | 8 | 152.9 | 374.5 | -- |

Table 2 gives the same CPU time scalings as the SIAM paper (Newton about O(M^1.3), M = N^2).

### C^1 Example (15), T20 in the catalogue

Table 3: maximum error; the Standard column prints a dash at N = 255 and 361.

| N | Standard | Monotone | Hybrid |
|---|---|---|---|
| 31 | 2.6 x 10^-4 | 17.5 x 10^-4 | 12.2 x 10^-4 |
| 45 | 1.8 x 10^-4 | 11.6 x 10^-4 | 5.9 x 10^-4 |
| 63 | 1.5 x 10^-4 | 9.8 x 10^-4 | 4.2 x 10^-4 |
| 89 | 0.9 x 10^-4 | 8.4 x 10^-4 | 2.6 x 10^-4 |
| 127 | 0.6 x 10^-4 | 7.9 x 10^-4 | 2.0 x 10^-4 |
| 181 | 0.4 x 10^-4 | 7.4 x 10^-4 | 1.2 x 10^-4 |
| 255 | -- | 7.2 x 10^-4 | 1.0 x 10^-4 |
| 361 | -- | 7.0 x 10^-4 | 0.7 x 10^-4 |

### C^{0,1} (Lipschitz) Example (17), the cone T21

Table 3: maximum error; the Standard column prints a dash at N = 255 and 361.

| N | Standard | Monotone | Hybrid |
|---|---|---|---|
| 31 | 10 x 10^-3 | 3 x 10^-3 | 3 x 10^-3 |
| 45 | 8 x 10^-3 | 3 x 10^-3 | 3 x 10^-3 |
| 63 | 6 x 10^-3 | 3 x 10^-3 | 3 x 10^-3 |
| 89 | 4 x 10^-3 | 4 x 10^-3 | 4 x 10^-3 |
| 127 | 3 x 10^-3 | 4 x 10^-3 | 4 x 10^-3 |
| 181 | 2 x 10^-3 | 4 x 10^-3 | 4 x 10^-3 |
| 255 | -- | 4 x 10^-3 | 4 x 10^-3 |
| 361 | -- | 4 x 10^-3 | 4 x 10^-3 |

### Other test solutions in this paper

- "C^1 Example" (15): u = ((|x - x0| - 0.2)^+)^2 / 2, f = (1 - 0.2/|x - x0|)^+, x0 = (0.5, 0.5); hybrid accuracy O(h).
- "C^{0,1} (Lipschitz) Example" (17): the cone centred at x0 with f = pi delta_{x0}, discretised as 4/h^2 on |x - x0| <= h/2; the hybrid scheme equals the monotone scheme there.
- 3D (Table 4, hybrid Newton, 19-point stencil, N^3 grid on [0,1]^3): exp(|x|^2/2), a C^1 example, and the ball -sqrt(3 - |x|^2).

## 4. Loeper, Rapetti (2005)

G. Loeper, F. Rapetti, "Numerical solution of the Monge-Ampere equation by a Newton's algorithm", C. R. Acad. Sci. Paris Ser. I 340 (2005) 319-324, DOI 10.1016/j.crma.2004.12.018.
Read: local journal PDF (Zotero item ZH56SNIU).
It is not paywalled for this project because a local copy exists; no arXiv version was found.

This paper does not test (a) or (b), and it has no tables.
It solves the periodic problem det(I + D^2 u) = rho on the torus [0,1]^2 (or [0,1]^3), not the Dirichlet problem.
Method: Newton with step 1/tau on the linearised operator (cofactor matrix of I + D^2 u), centred second-order finite differences, h = 1/N, N^d points, and unpreconditioned BiCG with residual tolerance 10^-8.
Test density: rho = 1 + beta sin(2 pi k x) sin(2 pi k y), with beta = 0.8, k = 2 in most runs and beta = 0.99, k = 1 for the near-degenerate run; the start is u0 = 0, rho0 = 1.
Results are only in figures: about 10 Newton iterations reach an L2 error on rho of about 10^-10 for N = 16 to 512 (Fig. 1).
The only printed numbers are in the Fig. 2 caption (beta = 0.8, k = 2, N = 64): "The highest absolute value is 0.0609 (n = 1, top left), 0.0239 (n = 2, top right), 0.00949 (n = 3, bottom left) and 0.00379 (n = 4, bottom right)".
Fig. 4 shows the L2 and L-infinity error of u for u_ex = beta sin(2 pi k x) sin(2 pi k y), beta = 0.02, k = 1, at N = 16, 32, 64, 128, 256 after 10 Newton iterations; the text gives order O(1/N^2) but prints no values.

## What a reproduction must match

The only published error tables for (a) and (b) in the Dirichlet setting are Table 3 of the JCP paper and Table 3 of the SIAM paper.
They report the max norm of u minus the exact solution at the grid points, on an N x N grid with N in {31, 45, 63, 89, 127, 181, 255, 361} (SIAM uses the subset 31, 63, 127, 255, 361).
For the standard 9-point scheme, solved to its fixed point by Gauss-Seidel, Poisson or any other convergent iteration, the targets are:
for (a), from 7.14 x 10^-5 at N = 31 down to 0.05 x 10^-5 at N = 361, order O(h^2);
for (b), from 17.15 x 10^-3 at N = 31 down to 5.41 x 10^-3 at N = 361, order about O(h^0.5).
For (b), the 17-point monotone scheme should reach about 0.3 x 10^-3, and the JCP hybrid scheme should reach 0.04 x 10^-3 at N = 361, order about O(h^1.5).
No paper tabulates Newton on the standard scheme, and the JCP paper reports that it fails on (b), so Newton iteration counts can only be compared for the monotone (SIAM Table 1) and hybrid (JCP Table 1) schemes, where they stay between 3 and 9 for (a) and (b).
Iteration counts for Gauss-Seidel and Poisson come only from the 2010 paper at N = 21 to 141 (to 581 for Poisson on (a)), with a 10^-14 stopping rule on successive iterates.
The CPU times are bound to 2010 MATLAB hardware, so only the scaling exponents (Newton about M^1.3, Poisson about M^1.4, Gauss-Seidel about M^1.8 to M^2) are reproducible.
Before matching errors to two digits, fix the grid convention, because none of the papers defines h for its N clearly.
