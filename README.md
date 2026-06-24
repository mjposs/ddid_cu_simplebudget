# Robust Minimum Spanning Tree with Decision-Dependent Information Discovery

A small, self-contained Julia implementation of three solvers for the **robust
budgeted minimum spanning tree (MST) problem with Decision-Dependent Information
Discovery (DDID)**, together with a benchmarking harness that compares them and
emits a LaTeX results table. This code was refactored, documented, and prepared
for publication with the assistance of Claude, an AI assistant developed by
Anthropic.

DDID is a two-stage robust model in which the first stage *queries* (observes) a
subset of the uncertain edge costs, and the second stage picks a spanning tree
that is robust against the still-uncertain remaining costs. The goal is to choose
the query set so that the worst-case tree cost is minimized.

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

The repository provides three solvers for the same instance:

1. **`ddid_mst` — exact (Chen & Poss).** Enumerates the query sets `Q`, and for
   each one evaluates `Φ(Q)` via the reduced candidate set `Ỹ(Q)` (one
   constrained MST per queried pattern), a boundedness test, and a membership
   test. Returns the exact value `Φ*`, an optimal query set `Q*`, and the
   corresponding tree. The enumeration runs in parallel over Julia's threads.
   Equation, proposition, and lemma numbers from the source paper are quoted in
   the code comments.

2. **`ddid_mst_LS` — local-search heuristic.** Starts from the `q` highest-cost
   edges and repeatedly moves to the best improving 1-swap neighbour (replace one
   edge in `Q` by one outside it) until no swap improves `Φ`. Each round of
   `q·(n−q)` neighbours is evaluated in parallel. The result is a local optimum
   of `Φ` — an upper bound on `Φ*`, returned much faster than the exact
   enumeration on larger instances. If a search ends without a feasible query set
   (`Φ = ∞`), it restarts from random query sets — each kept at least two swaps
   away from every previously tried start, so it differs from all visited starts
   *and* their neighbours — until a feasible one is found or a restart budget is
   reached.

3. **`kadapt_vayanos_mst` — K-adaptability MILP (Vayanos, Georghiou & Yu).** The
   K-adaptability counterpart, reformulated as the mixed-integer linear program
   of the paper's main theorem (eq. (11)). `K` here-and-now spanning-tree
   policies are chosen, and the best is selected once the queried costs are
   revealed. Increasing `K` gives a non-increasing upper bound that converges to
   the exact value.

The problem-independent dual blocks of the MILP are factored into
`_kadapt_dual_blocks!`; only the feasible set `𝒴` (spanning-tree constraints,
via `add_tree_constraints!`) is problem-specific.

All three solvers are solver-agnostic: pass `solver = :gurobi` (default) or
`solver = :highs` to switch between the commercial Gurobi and the open-source
HiGHS backend.

## Requirements

- [Julia](https://julialang.org/) — 1.10 (LTS) or newer is recommended, since the
  solvers parallelize over threads.
- A working **Gurobi** installation and license. Academic licenses are free. The
  file initializes a shared Gurobi environment when it is loaded, so Gurobi must
  be available even if you intend to solve with HiGHS.
- Julia packages: `JuMP`, `Gurobi`, `HiGHS`, `Combinatorics` (the rest —
  `LinearAlgebra`, `Random`, `Printf` — ship with Julia).

```julia
using Pkg
Pkg.add(["JuMP", "Gurobi", "HiGHS", "Combinatorics"])
```

`Gurobi.jl` needs Gurobi installed and discoverable (e.g. via the `GUROBI_HOME`
environment variable). See the [Gurobi.jl](https://github.com/jump-dev/Gurobi.jl)
setup instructions.

To render the generated LaTeX table you will also need the `booktabs` package in
your LaTeX document (`\usepackage{booktabs}`).

## Usage

### Run the benchmark

The solvers evaluate many small subproblems in parallel, so start Julia with
several threads:

```bash
julia -t auto ddid_mst.jl
```

(`-t auto` uses one thread per core; `-t N` requests `N`. Without `-t` the code
still runs, single-threaded.)

As shipped, the script warms up on `K₅`, then solves five random instances
(seeds 1–5) on the complete graph `K₇` with `q = 5`, `Γ = 2.5`, `b = 2`, sweeping
`K = 1, 2, …, 10`, prints progress as it goes, and writes the LaTeX table to
`MST7.txt`. For each seed the `K` sweep stops as soon as the optimum is reached,
or as soon as a solve hits the time limit. The top of the script exposes a few
knobs:

```julia
Kmax   = 10
solver = :gurobi   # or :highs
exact  = true      # false ⇒ skip the exact solver (for large instances)
```

Set `exact = false` to skip the exact enumeration and run only the heuristic and
the K-adaptability MILP — useful for instances too large to solve exactly.

### Call the solvers directly

```julia
include("ddid_mst.jl")

inst = gen_mst_instance(; V = 5, seed = 1)          # random K₅ instance
c, edges, V = inst.c, inst.edges, inst.V

# exact value
Φstar, Qstar, ystar = ddid_mst(c, edges, V, 4, 2.5, 2)

# local-search heuristic (upper bound on Φ*)
Φls, Qls, yls = ddid_mst_LS(c, edges, V, 4, 2.5, 2)

# solve with the open-source HiGHS backend instead of Gurobi
Φls, Qls, yls = ddid_mst_LS(c, edges, V, 4, 2.5, 2; solver = :highs)

# K-adaptability upper bound for K = 3, with a 1-hour cap
val, w, policies, status =
    kadapt_vayanos_mst(c, edges, V, 4, 2.5, 2, 3; time_limit = 3600.0)
```

To reproduce the table over your own seeds:

```julia
Kmax = 10
rows = [run_mst_instance(; seed = s, q = 4, Γ = 2.5, b = 2, Kmax = Kmax,
                           time_limit = 3600.0, solver = :gurobi, exact = true)
        for s in 1:5]
write_mst_table(rows, Kmax; path = "MST.txt")
```

## Output

While running, the script prints per-instance progress, e.g.

```
##### MST on K5: n=10 edges, q=4, Γ=2.5, b=2 (seed 1) #####
  edges = [(1,2), (1,3), ...]
  costs = [7, 3, ...]
  Chen–Poss  Φ* = 14.0000   (Q* = [1, 2, 4, 7])   [0.04s]
  LS         Φ  = 14.0000   (Q  = [1, 2, 4, 7])   [0.02s]   Opt
    K=1 : ∞            inf   [0.10s]
    K=2 : 14.0000      Opt   [0.31s]
  => optimal K = 2
```

When `exact = true`, the tag after each value compares it to the exact `Φ*`:
`Opt` means it matches, `Feas` means it is still an upper bound, `inf` means
infeasible, and `Error` flags an impossible value below `Φ*`. The local-search
line carries the same tag.

When `exact = false`, the `Chen–Poss` line is omitted and there is no exact `Φ*`
to compare against, so the K-adaptability values are tagged relative to the
heuristic instead — `<LS`, `=LS`, `>LS` — and the sweep stops at the first `K`
that reaches the LS bound (reported as `K reaching LS`).

### LaTeX table (`MST*.txt`)

The table has one `(time, value)` column pair per seed. The first data row is the
exact Chen–Poss result (its runtime and `Φ*`), the second is the local-search
heuristic, and each subsequent row is one `K`, for every `K` attempted by some
seed. When the instances were run with `exact = false`, the `exact` row is
omitted.

```latex
\begin{tabular}{l|rr|rr|}
\toprule
 & \multicolumn{2}{c}{seed 1} & \multicolumn{2}{c}{seed 2} \\
 & time & value & time & value \\
\midrule
exact & 0.0400 & 14 & 0.0500 & 16 \\
LS & 0.0200 & 14 & 0.0300 & 16 \\
\midrule
$K=1$ & 0.100 & $\infty$ & 0.220 & 16 \\
$K=2$ & 0.310 & 14 &  &  \\
\bottomrule
\end{tabular}
```

Value-cell legend:

- an integer — the objective (`Φ*`, the LS value, or the K-adaptability value);
- `$\infty$` — that model is infeasible / unbounded;
- `T` (in both the time and value cells) — the solve hit the time limit with no
  feasible incumbent (a `T` in the time cell with a value keeps the best
  incumbent found so far);
- blank — the seed never attempted that `K` (its optimum was already reached, or
  the sweep stopped after a timeout).

## API reference

| Function | Returns |
| --- | --- |
| `gen_mst_instance(; V, seed, cmin, cmax)` | named tuple `(edges, V, c, n)` for a random complete graph |
| `ddid_mst(c, edges, V, q, Γ, b; solver)` | `(Φ*, Q*, y*)` — exact value, optimal query set, optimal tree |
| `ddid_mst_LS(c, edges, V, q, Γ, b; solver, rng, max_restarts)` | `(Φ, Q, y)` — local-optimum value (an upper bound on `Φ*`), its query set and tree |
| `kadapt_vayanos_mst(c, edges, V, q, Γ, b, K; ε, M, optimizer, time_limit, silent)` | `(value, w, [y¹…yᴷ], status)` |
| `run_mst_instance(; seed, q, V, Γ, b, Kmax, time_limit, solver, exact)` | per-seed results (times, values, time-limit flags) for the table |
| `write_mst_table(rows, Kmax; path)` | writes the LaTeX table, returns the path |

Common arguments: `c` edge costs (length `n`), `edges` list of `(u, v)` tuples,
`V` number of vertices, `q` query budget, `Γ` uncertainty budget, `b` robust
bound, `K` adaptability level, `time_limit` in seconds, `solver` either `:gurobi`
or `:highs`, `exact` whether to run the exact enumeration.

## Notes on parallelism

`ddid_mst` and `ddid_mst_LS` distribute their subproblem solves across Julia's
threads (up to `Threads.nthreads()` workers), each with its own optimizer — for
Gurobi, a private `Gurobi.Env`, since an environment must not be shared across
threads. Start Julia with `-t` to use more than one thread. Results are
deterministic: the exact enumeration breaks ties by the lowest query-set index
(matching a sequential scan), and the heuristic's random restarts use a seeded
RNG.

## Repository layout

```
ddid_mst.jl    all code: solvers, harness, and the LaTeX table writer
MST*.txt       generated results table (created when the script is run)
README.md
```

## References

The method names and the equation numbers in the code comments refer to:

- X. Chen, M. Poss. *Decision-dependent information discovery for robust cardinality-constrained combinatorial optimization problems.* (https://hal.science/hal-05667307) (`ddid_mst`).
- P. Vayanos, A. Georghiou, H. Yu. *Robust Optimization with Decision-Dependent
  Information Discovery.* Management Science 72(2) (2025) 1509–1528.
  doi:10.1287/mnsc.2021.00160 — the K-adaptability MILP (`kadapt_vayanos_mst`).

Related, on the DDID combinatorial framework (including a spanning-tree
application): J. Omer, M. Poss, M. Rougier. *Combinatorial Robust Optimization
with Decision-Dependent Information Discovery and Polyhedral Uncertainty.* Open
Journal of Mathematical Optimization 5 (2024), article 5. doi:10.5802/ojmo.33.
