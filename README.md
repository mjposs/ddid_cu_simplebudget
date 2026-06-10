# Robust Minimum Spanning Tree with Decision-Dependent Information Discovery

A small, self-contained Julia implementation of two solvers for the **robust budgeted minimum spanning tree (MST) problem with Decision-Dependent Information Discovery (DDID)**, together with a benchmarking harness that compares them and emits a LaTeX results table. This code was refactored, documented, and prepared for publication with the assistance of Claude, an AI assistant developed by Anthropic.

DDID is a two-stage robust model in which the first stage *queries* (observes) a subset of the uncertain edge costs, and the second stage picks a spanning tree that is robust against the still-uncertain remaining costs. The goal is to choose the query set so that the worst-case tree cost is minimized.

## The problem

For a graph `G` with `n` edges and cost vector `c`:

```
𝒴       = { spanning trees of G }
Ξ       = { ξ ∈ [0,1]ⁿ : eᵀξ ≤ Γ }            uncertainty set (budget Γ)
Ξ(Q,μ)  = { ξ ∈ Ξ : ξᵢ = μᵢ  ∀ i ∈ Q }        realizations consistent with the query Q

φ(Q,μ)  = min_{y ∈ 𝒴} { cᵀy : ξᵀy ≤ b  ∀ ξ ∈ Ξ(Q,μ) }
Φ(Q)    = max_{μ ∈ Ξ} φ(Q,μ)
DDID    = min_{Q : |Q| = q} Φ(Q)
```

`q` is the query budget (number of edges observed), `Γ` is the uncertainty
budget, and `b` is the robust constraint bound.

## Methods

The repository provides two solvers for the same instance:

1. **`ddid_mst` — exact (Chen & Poss).** Enumerates the query sets `Q`, and for
   each one evaluates `Φ(Q)` via the reduced candidate set `Ỹ(Q)` (one
   constrained MST per queried pattern), a boundedness test, and a membership
   test. Equation, proposition, and lemma numbers from the source paper are
   quoted in the code comments.

2. **`kadapt_vayanos_mst` — K-adaptability MILP (Vayanos, Georghiou & Yu).** The
   K-adaptability counterpart, reformulated as the mixed-integer linear program
   of the paper's main theorem (eq. (11)). `K` here-and-now spanning-tree
   policies are chosen, and the best is selected once the queried costs are
   revealed. Increasing `K` gives a non-increasing upper bound that converges to
   the exact value.

The problem-independent dual blocks of the MILP are factored into
`_kadapt_dual_blocks!`; only the feasible set `𝒴` (spanning-tree constraints,
via `add_tree_constraints!`) is problem-specific.

## Requirements

- [Julia](https://julialang.org/) (1.6 or later).
- A working **Gurobi** installation and license. Academic licenses are free.
- Julia packages: `JuMP`, `Gurobi`, `Combinatorics` (the rest — `LinearAlgebra`,
  `Random`, `Printf` — ship with Julia).

```julia
using Pkg
Pkg.add(["JuMP", "Gurobi", "Combinatorics"])
```

`Gurobi.jl` needs Gurobi installed and discoverable (e.g. via the `GUROBI_HOME`
environment variable). See the [Gurobi.jl](https://github.com/jump-dev/Gurobi.jl)
setup instructions.

## Usage

### Run the benchmark

```bash
julia ddid_mst.jl
```

This solves five random instances (seeds 1–5) on the complete graph `K₅` with
`q = 4`, `Γ = 2.5`, `b = 2`, sweeping `K = 1, 2, …`, prints progress as it goes,
and writes the LaTeX table to `MST.txt`. For each seed the `K` sweep stops as
soon as the optimum is reached, or as soon as a solve hits the one-hour time
limit.

### Call the solvers directly

```julia
include("ddid_mst.jl")

inst = gen_mst_instance(; V = 5, seed = 1)          # random K₅ instance
c, edges, V = inst.c, inst.edges, inst.V

# exact value
Φstar, Qstar, ystar = ddid_mst(c, edges, V, 4, 2.5, 2)

# K-adaptability upper bound for K = 3, with a 1-hour cap
val, w, policies, status =
    kadapt_vayanos_mst(c, edges, V, 4, 2.5, 2, 3; time_limit = 3600.0)
```

To reproduce the table over your own seeds:

```julia
Kmax = 10
rows = [run_mst_instance(; seed = s, q = 4, Γ = 2.5, b = 2,
                           Kmax = Kmax, time_limit = 3600.0) for s in 1:5]
write_mst_table(rows, Kmax; path = "MST.txt")
```

## Output

While running, the script prints per-instance progress, e.g.

```
##### MST on K5: n=10 edges, q=4, Γ=2.5, b=2 (seed 1) #####
  edges = [(1,2), (1,3), ...]
  costs = [7, 3, ...]
  Chen–Poss  Φ* = 14.0000   (Q* = [1, 2, 4, 7])   [0.04s]
    K=1 : ∞            inf   [0.10s]
    K=2 : 14.0000      Opt   [0.31s]
  => optimal K = 2
```

The `Opt / Feas / inf / Error` tag compares each K-adaptability value to the
exact `Φ*`: `Opt` means it matches, `Feas` means it is still an upper bound,
`inf` means infeasible, and `Error` flags an impossible value below `Φ*`.

### LaTeX table (`MST.txt`)

The table has one `(time, value)` column pair per seed. The first data row is
the exact Chen–Poss result (its runtime and `Φ*`); each subsequent row is one
`K`, for every `K` attempted by some seed.

```latex
\begin{tabular}{lrrrrrrrr}
\hline
 & \multicolumn{2}{c}{seed 1} & \multicolumn{2}{c}{seed 2} & \multicolumn{2}{c}{seed 3} & \multicolumn{2}{c}{seed 4} \\
 & time & value & time & value & time & value & time & value \\
\hline
Chen--Poss & 0.04 & 14 & 0.05 & 16 & 0.03 & 12 & 0.06 & 15 \\
$K=1$ & 0.10 & $\infty$ & 0.22 & 16 & 0.15 & 18 & 0.20 & 20 \\
$K=2$ & 0.31 & 14 &  &  & 0.40 & 14 & T & T \\
$K=3$ &  &  &  &  & 0.77 & 12 &  &  \\
\hline
\end{tabular}
```

Value-cell legend:

- an integer — the K-adaptability objective;
- `$\infty$` — that K-adaptability model is infeasible;
- `T` (in both the time and value cells) — the solve hit the time limit;
- blank — the seed never attempted that `K` (its optimum was already reached, or
  the sweep stopped after a timeout).

## API reference

| Function | Returns |
| --- | --- |
| `gen_mst_instance(; V, seed, cmin, cmax)` | named tuple `(edges, V, c, n)` for a random complete graph |
| `ddid_mst(c, edges, V, q, Γ, b; opt)` | `(Φ*, Q*, y*)` — exact value, optimal query set, optimal tree |
| `kadapt_vayanos_mst(c, edges, V, q, Γ, b, K; ε, M, optimizer, time_limit, silent)` | `(value, w, [y¹…yᴷ], status)` |
| `run_mst_instance(; seed, q, Γ, b, Kmax, time_limit)` | per-seed results (times, values, time-limit flags) for the table |
| `write_mst_table(rows, Kmax; path)` | writes the LaTeX table, returns the path |

Common arguments: `c` edge costs (length `n`), `edges` list of `(u, v)` tuples,
`V` number of vertices, `q` query budget, `Γ` uncertainty budget, `b` robust
bound, `K` adaptability level, `time_limit` in seconds.

## Repository layout

```
ddid_mst.jl    all code: solvers, harness, and the LaTeX table writer
MST.txt        generated results table (created when the script is run)
README.md
```

## References

The method names and the equation numbers in the code comments refer to:

- X. Chen, M. Goerigk, M. Poss. *The robust selection problem with information
  discovery.* Discrete Applied Mathematics 382 (2026) 277–292. — the exact
  algorithm (`ddid_mst`).
- P. Vayanos, A. Georghiou, H. Yu. *Robust Optimization with Decision-Dependent
  Information Discovery.* Management Science 72(2) (2025) 1509–1528.
  doi:10.1287/mnsc.2021.00160 — the K-adaptability MILP (`kadapt_vayanos_mst`).

Related, on the DDID combinatorial framework (including a spanning-tree
application): J. Omer, M. Poss, M. Rougier. *Combinatorial Robust Optimization
with Decision-Dependent Information Discovery and Polyhedral Uncertainty.* Open
Journal of Mathematical Optimization 5 (2024), article 5. doi:10.5802/ojmo.33.

## License

Add your license of choice here (e.g. MIT) before publishing.
