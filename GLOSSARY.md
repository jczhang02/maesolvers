# Glossary

Terms maesolvers uses in one fixed way, in Chinese writing and in English writing.
Rows marked (shared) are copies of entries in the glossary of the research repository (jczhang02/monge-ampere-equations, private), which owns them: change them there first and copy the change here in the same change.
The other rows are maesolvers' own, and the research repository uses them when it writes about the library.
Agents and skills use these forms and avoid the ones listed under "Not".

## Methods

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 传统方法 (shared) | classical method | A numerical method for `det D^2 u = f` that uses no neural network: finite-difference schemes, finite elements, least-squares and augmented-Lagrangian methods. | 经典方法, 经典数值方法, 传统数值方法, 传统格式, 传统求解器 |
| 深度学习方法 (shared) | neural method | A method that represents the solution, or the map to it, with a neural network; the opposite class to 传统方法. A 求解器 (solver) is one concrete method, so "a neural solver" names one solver of this class. | 神经方法, 神经网络方法, AI 方法, 神经求解器 (for the whole class) |
| 方法族 | method family | A group of methods that share their parts and dependencies: the finite-difference family, the neural family, later the finite-element family. | |
| 差分格式 | scheme | One finite-difference discretisation of the equation, named by its stencil or its construction (standard 9-point, monotone 17-point, filtered, hybrid, mixed, semi-Lagrangian). Not used for finite elements. | 离散格式 (alone), format |
| 模板 | stencil | The grid points one equation of a scheme reads around a node: 9 points for the standard scheme, 9, 17 or 33 for the monotone scheme in 2D and 19 in 3D, whose stencil size sets the directions it resolves and, for the monotone scheme, the dimension; for the semi-Lagrangian scheme, the two ends `x +/- k s` along each stencil direction `s`, which fall between nodes. | 网格模式 |
| 标架 | frame | `d` mutually orthogonal grid directions of a stencil, one per factor of the monotone scheme's product: a pair in 2D, a triple in 3D (`stencil_frames`). | 基 (alone) |
| 模板因子 | stencil factor | `m` in the stencil size `k = m h` of the semi-Lagrangian scheme (Feng-Jensen 2017): how many grid steps a stencil end lies from its node, away from the boundary. Its best value grows as the grid is refined. | 模板宽度 (that is the monotone scheme's, in grid directions) |
| 切换规则 | switching rule | How the hybrid scheme weights its two schemes at each node, fixed before the solve from the data: by default (Froese-Oberman 2011) the monotone scheme near the boundary and where `f` is near 0 or infinite, the standard scheme elsewhere. | 混合权重 (for the rule itself) |
| 滤波 | filter | How the filtered scheme combines its two schemes at each node from their difference, during the solve: the accurate scheme where they differ by at most the filter width `eps`, the monotone scheme where they differ by more than `2 eps`, linear in between (Froese-Oberman 2013). | 切换规则 (that is the hybrid scheme's, fixed before the solve) |
| 滤波宽度 | filter width | The difference `eps` between the two schemes up to which the filter keeps the accurate one; by default `sqrt(h) + dtheta / 10` in `Filtered`, with `dtheta` the largest gap between neighbouring stencil directions, twice `angular_resolution` (Froese-Oberman 2013; `benchmarks/filtered.md`, Finding 3), and `sqrt(h) + dtheta` in `SecondBoundary` (Benamou-Froese-Oberman 2014, Section 5.2.1). | 阈值 (alone) |
| HJB 形式 | HJB form | The equation of a convex `u` written as a maximum over controls of a linear operator, `max over A of (-tr(A D^2 u) + 2 sqrt(det(A) f)) = 0`, `A` positive semidefinite with trace 1 (Krylov 1972; Chen-Wan-Lin 2018, Theorem 1); convexity needs no constraint there. | Bellman 形式 |
| 控制 | control | In the HJB form, the matrix `A` at a node, parametrised by `(a, theta)`: eigenvalues `a` and `1 - a` along the axes turned by `-theta`. | 策略 (for one node's control; a policy is the controls at all nodes) |
| 混合模板格式 | mixed scheme | The scheme of Chen-Wan-Lin 2018 on the HJB form: each control takes the standard 7-point stencil where that stencil is monotone, the semi-Lagrangian wide stencil elsewhere. | 混合格式 (that is the hybrid scheme) |
| 半 Lagrange 宽模板 | semi-Lagrangian wide stencil | Second differences along the turned axes of a control with arms of length `sqrt(h)`, the values at the arms' ends interpolated bilinearly; monotone for every control. | 宽模板 (alone: that is the monotone scheme's) |
| 策略迭代 | policy iteration | Howard's algorithm for a scheme in HJB form: find the optimal control at every node (the policy), solve its linear system, repeat until the residual is small (`PolicyIteration`) or a step is (`Howard`, whose step is a full Newton step with the scheme's generalised Jacobian, the linear part of the optimal control's equation). It converges from any start, monotonically, where every control's matrix is an M-matrix (Bokanowski-Maroso-Zidani 2009; Feng-Jensen 2017, Algorithm 1). | Howard 迭代 |
| 迭代方法 | iteration | The algorithm that solves a scheme's discrete equations (Newton, the Poisson iteration, semi-smooth Newton, multigrid, policy iteration). One scheme under two iterations can reach different discrete solutions. | solver (for this part alone) |
| 初值 | start | The iterate an iteration steps from: for Newton and the Poisson iteration by default the solution of a Poisson equation with a right-hand side from `f` (`Laplace u = d f^(1/d)` in dimension `d` for Newton, `2 sqrt(f)` in 2D), for multigrid the FMG cycle below the finest grid; or the final iterate of another solver on the same unknowns, such as the monotone scheme's root under the hybrid scheme. The start can decide which discrete root Newton reaches. For a neural solver, the network its training steps from, such as one fitted to `grad u = x` (`IdentityStart`). | 初始猜测 (for a whole solver's root) |
| 扫描 | sweep | One pass of the nonlinear Gauss-Seidel iteration over the interior nodes: each node's equation solved for its own value (`relax`) with its neighbours held, in the order the ordering (排序) sets, lexicographic (node by node in flat order, each reading its neighbours' newest values) or red-black (the nodes with an even index sum, then the odd ones). The smoother of multigrid. | 迭代 (for one sweep), 遍历 |
| V 循环 | V-cycle | One cycle of the FAS multigrid method from a grid down to the coarsest and back: sweeps, the coarse problem with the restricted residual solved by the same cycle, the interpolated correction, sweeps (Liu et al. 2017, Algorithm 1). Every step of multigrid after the first is one V-cycle on the finest grid. | V 型迭代 |
| FMG 循环 | FMG cycle | The full multigrid cycle (Algorithm 2): the coarsest grid solved by sweeps, then on each finer grid the coarser solution interpolated and one V-cycle. The first step of multigrid; the paper's "Iter = 1". | 完全多重网格迭代 |
| 求解器 | solver | A complete method that takes a problem and returns a result: a scheme with an iteration, or a network with a loss and a training loop. | 模型 (for a classical solver) |
| 预设 | preset | A named solver for a common combination of parts, such as the standard 9-point scheme under the Poisson iteration. The comparison table lists presets. | |
| 部件 | part | One exchangeable piece of a composed solver, itself configured by its parameters: a scheme or an iteration for a classical solver; a network, a loss, a sampler or an optimizer for a neural solver. A part of your own replaces a shipped one without a new solver. | 组件, 模块 (for a part) |
| 网络 | network | The part of a neural solver that represents the solution as a function of the point, including any fixed form around the trained network, such as an ansatz. | 模型 (for the network alone) |
| 拟设 | ansatz | A fixed form of the solution around the trained network, such as `u = G + D N` (Nystrom-Vestberg 2023): `G` fitted to the boundary data and `D` vanishing on the boundary, so `u` meets the Dirichlet condition up to the fits. | 试探函数 |
| 恒等初值 | identity start | The start of a neural transport solver: the network fitted, before the training, so that `grad u` is near the identity map `x`, `u` near the convex `|x|^2 / 2` (Caboussat-Peruso 2025, Section 4.1, after Singh-Bauer-Joshi 2022), since an input convex network trains badly from a random start (`IdentityStart`). | 恒等初始化 (初始化 is the random draw of the parameters), 预训练 (the fit solves no equation) |
| QS 形式 | QS form | The network of QS-PINN: `u = (x - x0)^T A(x)^T A(x) (x - x0) + b . x + c`, with `A(x)` from a plain network and `b`, `c` constant (`QuadraticForm`); it lies above its tangent plane at the reference point, where its Hessian is `2 A^T A`. | 二次型网络, quadratic network |
| 参考点 | reference point | The point `x0` of the QS form, inside the domain with `f(x0) > 0`, where the form's Hessian `2 A^T A` is positive definite for every value of the parameters, since `A = alpha (I + B)` with `alpha > 0` and `|B| < 1`; by default the centre of the domain's bounding box. | 原点 (for `x0` in general), 中心点 |
| 损失 | loss | The part of a neural solver that turns the network's values at the collocation points into the number training minimises: the equation's residual plus penalty terms, each reported by name. | 目标函数 (for the loss part) |
| Hausdorff 型损失 | Hausdorff-type loss | The loss of a neural transport solver (Caboussat-Peruso 2025): the equation's mean residual plus a discrete Hausdorff distance between `grad u` of the boundary points and the target set's boundary, in two halves, `into` (each image near the target's boundary) and `onto` (no point of the target's boundary far from the images) (`HausdorffLoss`). | Hausdorff 距离 (that is the distance between two sets, one term of the loss), 边界损失 |
| 采样器 | sampler | The part of a neural solver that draws the collocation points of a problem from the seed. | |
| 优化器 | optimizer | The part of a neural solver that updates the network's parameters once per training step, such as L-BFGS or SciPy's dense BFGS. | |
| 训练循环 | training loop | The one loop every neural solver trains on (ADR 0008): float64, all collocation points in every step, one optimizer step per training step, stopping on the gradient tolerance, the step budget, a step that changes nothing, or a non-finite loss. | 训练流程 |
| 配点 | collocation points | The points where a neural method evaluates its loss: inside the domain for the equation, on the boundary for the boundary condition, and for a transport problem on the target's boundary too, where the loss compares them with `grad u` of the boundary points. Drawn from the seed by the sampler; whether ADR 0008's shared set is per problem or per sampler, and whether it is stored, is open (#52). The primary error is never measured at them. | 训练点 (for the set itself), sample points |
| 分支选择 (shared) | branch selection | Whether a solver reaches the convex solution (凸解) or a non-convex one that also satisfies `det D^2 u = f`, such as the concave solution (凹解) of the unit-disk example with `f = 1`. | |

## Problems

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 问题 | problem | A domain, a right-hand side `f`, a boundary condition, and the exact solution when it is known. | 算例 (for the definition itself), case |
| 目录 | catalogue | Every problem maesolvers defines, named by T ids. Wider than the shared test set. | benchmark suite |
| T 编号 | T id | The id of a problem in the research programme's literature survey (T1, T2, ...). An id once used is never reused. The P1 to P5 ids of 0.0.x are retired. | |
| 共用集 | shared test set | The problems every method reports on and the rows of the comparison table: T1, T2, T20, T8, T12, T9, T10. | 基准集 |
| 第二边值问题, 传输边界条件 (shared) | second boundary value problem, transport boundary condition | The Monge-Ampere equation with the transport boundary condition `grad u(Omega) = Omega*` in place of boundary values; its boundary condition is the 传输边界条件 (transport boundary condition). | transport 问题, transport 边界, 输运边界条件, 运输边界 |
| 目标集 | target set | The convex set onto which `grad u` maps the domain in a transport problem (`Transport(target)`), with a uniform density 1; maesolvers knows the geometry of `Box`, `Disk` and `Ellipse` targets (signed distance, support function). | 目标区域 (for the set itself) |
| 支撑函数 | support function | `H*(n) = sup_{y in Y} y . n` of a convex target set: the transport condition on the boundary is the largest `grad u . n - H*(n)` over the directions `n` (Benamou-Froese-Oberman 2014); `support_function` computes it for `Box`, `Disk`, `Ellipse` and any target with the method of `SupportFunction`, the targets for which `knows_support_function` is True. | 支持函数 |
| 质量平衡 | mass balance | For a transport problem: the integral of `f` over the domain equals the volume of the target set, without which no solution exists; `check_mass_balance` checks it, called from the `check` of a transport solver written as a subclass (where `can_check_mass_balance` is True), or by the caller of a composed one. | 质量守恒 |
| 观测量 | observable | A named number computed from a solution, reported for problems without an exact solution, such as `u(0, 0)`. | |

## Measurement

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 节点 | nodes | The points where a classical method itself defines its solution: its grid points, and for a grid cut by a curved boundary the points where grid lines meet the boundary, which carry the Dirichlet data. | |
| 边界环 | boundary ring | The nodes of a box grid that lie on the faces of the box, those with some index 0 or `n`; a finite-difference scheme imposes the Dirichlet data there, and the other nodes carry the unknowns; for the transport condition (`SecondBoundary`) the ring nodes are unknowns too and carry the boundary equation. A ring in 2D, a shell in 3D; the name is the same in every dimension. | 边界层 (that is a layer of the solution) |
| 共用评估网格 | shared evaluation grid | The uniform grid over the bounding box of a problem's domain, points in the domain only. Never the training points of a neural method. | test grid |
| 主误差 | primary error | The error at the method's nodes for a classical method, on the shared evaluation grid for a neural method. The only error used to rank methods. | |
| 次要值 | secondary value | The error of a classical method on the shared evaluation grid, through its interpolant. Reported, never used for ranking. | |
| 梯度映射误差 | gradient-map error | The max and RMS over some points of `\|grad u - grad u_exact\|`, the error of the transport map that the transport papers report; computed by `gradient_map_error`, never a field of `evaluate`. | 映射误差 (alone) |
| 非正定比例 | non-convex fraction | The share of points where the solution's smallest Hessian eigenvalue is negative. | 凸性指标 (alone) |
| 种子 | seed | One independent run of a neural method. Results over seeds are reported as median and 5% to 95% quantiles, with failed seeds counted. | trial |
| 成功率 | success rate | The share of seeds whose error is below a stated threshold. | |

## Reproduction and tables

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 复现 | reproduction | Running a method on its source paper's own test problem and comparing with the paper's own table. Its verdict is one of five levels: exact, order of magnitude, statistical, figure only, mismatch. | 验证, 对齐 |
| 发现 | finding | A recorded difference between our run and the published one, kept with its explanation instead of tuned away. | |
| 复现表 | reproduction table | Published numbers next to ours, per method. The only table with published numbers. | |
| 对比表, 横向比较表 | comparison table | Our runs of every method on the shared test set. It carries no published numbers. | leaderboard |
