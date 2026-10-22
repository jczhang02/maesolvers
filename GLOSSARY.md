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
| 差分格式 | scheme | One finite-difference discretisation of the equation, named by its stencil (standard 9-point, monotone 17-point, filtered, hybrid). Not used for finite elements. | 离散格式 (alone), format |
| 迭代方法 | iteration | The algorithm that solves a scheme's discrete equations (Newton, the Poisson iteration, semi-smooth Newton). One scheme under two iterations can reach different discrete solutions. | solver (for this part alone) |
| 求解器 | solver | A complete method that takes a problem and returns a result: a scheme with an iteration, or a network with a loss and a training loop. | 模型 (for a classical solver) |
| 预设 | preset | A named solver for a common combination of parts, such as the standard 9-point scheme under the Poisson iteration. The comparison table lists presets. | |
| 配点 | collocation points | The points where a neural method evaluates its loss: inside the domain for the equation, on the boundary for the boundary condition. Drawn from the seed by the sampler; whether ADR 0008's shared set is per problem or per sampler, and whether it is stored, is open (#52). The primary error is never measured at them. | 训练点 (for the set itself), sample points |
| 分支选择 (shared) | branch selection | Whether a solver reaches the convex solution (凸解) or a non-convex one that also satisfies `det D^2 u = f`, such as the concave solution (凹解) of the unit-disk example with `f = 1`. | |

## Problems

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 问题 | problem | A domain, a right-hand side `f`, a boundary condition, and the exact solution when it is known. | 算例 (for the definition itself), case |
| 目录 | catalogue | Every problem maesolvers defines, named by T ids. Wider than the shared test set. | benchmark suite |
| T 编号 | T id | The id of a problem in the research programme's literature survey (T1, T2, ...). An id once used is never reused. The P1 to P5 ids of 0.0.x are retired. | |
| 共用集 | shared test set | The problems every method reports on and the rows of the comparison table: T1, T2, T20, T8, T12, T9, T10. | 基准集 |
| 观测量 | observable | A named number computed from a solution, reported for problems without an exact solution, such as `u(0, 0)`. | |

## Measurement

| Term (zh) | Term (en) | Meaning | Not |
|---|---|---|---|
| 节点 | nodes | The points where a classical method itself defines its solution: its grid points, and for a grid cut by a curved boundary the points where grid lines meet the boundary, which carry the Dirichlet data. | |
| 共用评估网格 | shared evaluation grid | The uniform grid over the bounding box of a problem's domain, points in the domain only. Never the training points of a neural method. | test grid |
| 主误差 | primary error | The error at the method's nodes for a classical method, on the shared evaluation grid for a neural method. The only error used to rank methods. | |
| 次要值 | secondary value | The error of a classical method on the shared evaluation grid, through its interpolant. Reported, never used for ranking. | |
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
