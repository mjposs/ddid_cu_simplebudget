# ddid_mst.jl
#
# Robust budgeted MINIMUM SPANNING TREE with Decision-Dependent Information
# Discovery (DDID).
#
#   𝒴 = { spanning trees of G },   cost c,   robust constraint  ξᵀy ≤ b,
#   Ξ = { ξ∈[0,1]ⁿ : eᵀξ ≤ Γ }                       (n = number of edges).
#
#   DDID    = min_{Q:|Q|=q} Φ(Q),     Φ(Q)   = max_{μ∈Ξ} φ(Q,μ),
#   φ(Q,μ)  = min_{y∈𝒴} { cᵀy : ξᵀy ≤ b  ∀ξ∈Ξ(Q,μ) },
#   Ξ(Q,μ)  = { ξ∈Ξ : ξᵢ = μᵢ ∀i∈Q }.
#
# Two solvers are provided:
#   (1) ddid_mst            – exact, Chen & Poss (equation numbers in comments).
#   (2) kadapt_vayanos_mst  – K-adaptability MILP, Vayanos, Georghiou & Yu
#                             (Theorem 4, eq (11)); the matching line of (11) is
#                             quoted above each constraint block.
#
# deps:  JuMP, Gurobi, HiGHS, Combinatorics, LinearAlgebra, Random, Printf

using JuMP, Gurobi, HiGHS, Combinatorics, LinearAlgebra, Random, Printf

# One shared Gurobi environment, created once inside a stdout redirect so the
# "Set parameter LicenseID to value …" banner is never printed.  Reusing this
# environment for every model means the banner is not reprinted per model.
const GRB_ENV = redirect_stdout(() -> Gurobi.Env(), devnull)
grb_opt() = Gurobi.Optimizer(GRB_ENV)   # optimizer factory passed to Model(...)

cᵀ(c, y) = sum(c[i] * y[i] for i in eachindex(c))     # cᵀy

# =====================================================================
#  (1)  CHEN & POSS  – exact algorithm
# =====================================================================

# Min-cost spanning tree that INCLUDES every edge in Tset and EXCLUDES every
# queried edge not in Tset (Qset∖Tset), completing only with non-queried edges
# (Kruskal with forced/forbidden edges).  Returns a 0/1 length-n vector, or
# `nothing` if no such tree exists.  This is the GenCard-Π oracle for MST.
function constrained_mst(c, edges, V, Tset, Qset)
    parent = collect(1:V)
    function root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]            # path compression
            x = parent[x]
        end
        return x
    end
    addedge!(e) = begin
        u, v = edges[e]; ru, rv = root(u), root(v)
        ru == rv ? false : (parent[ru] = rv; true)   # false ⇔ would close a cycle
    end
    tree = Int[]
    for e in Tset                                    # forced queried edges first
        addedge!(e) || return nothing                #   T contains a cycle ⇒ infeasible
        push!(tree, e)
    end
    comp = sort([e for e in eachindex(edges) if !(e in Qset)], by = e -> c[e])
    for e in comp                                    # Kruskal over non-queried edges
        length(tree) == V - 1 && break
        addedge!(e) && push!(tree, e)
    end
    length(tree) == V - 1 || return nothing          # could not span ⇒ infeasible
    y = zeros(Int, length(edges)); for e in tree; y[e] = 1; end
    return y
end

# Reduced candidate set Ỹ(Q): one constrained MST per queried pattern T⊆Q, the
# cheapest tree that INCLUDES T and EXCLUDES Q∖T, completed from non-queried
# edges.  Deduplicated.  |Ỹ(Q)| ≤ 2^q.                            [eqs (5),(26)]
function reduced_mst(c, edges, V, Q)
    Qset = Set(Q); q = length(Q)
    cands = Vector{Vector{Int}}()
    for bits in 0:(2^q - 1)
        T = [Q[j] for j in 1:q if (bits >> (j - 1)) & 1 == 1]
        y = constrained_mst(c, edges, V, T, Qset)
        y === nothing || push!(cands, y)
    end
    return unique!(cands)
end

# Boundedness test:  Φ(Q)=∞  ⟺  the LP below has optimum ε* > 0.    [eqs (23)–(25)]
#   max_{μ,ε} ε
#   s.t.  Σ_{i∈Q} μᵢyᵢ + Σ_{i∈Q̄} yᵢ        ≥ b+ε   ∀y∈Ỹ(Q)              (24)
#         Σ_{i∈Q} μᵢyᵢ + Γ − Σ_{i∈Q} μᵢ    ≥ b+ε   ∀y∈Ỹ(Q)              (25)
#         Σ_{i∈Q} μᵢ ≤ Γ,   0 ≤ μᵢ ≤ 1                                 (21),(22)
function Φ_unbounded(Γ, b, Q, cands, Q̄idx; opt = grb_opt)
    m = Model(opt); set_silent(m)
    @variable(m, μ[Q], lower_bound = 0.0, upper_bound = 1.0)
    @variable(m, ε, upper_bound = Γ + 1.0)
    @objective(m, Max, ε)
    Σμ = sum(μ[i] for i in Q)
    @constraint(m, Σμ ≤ Γ)                                              # (21)
    for y in cands
        Py = sum(μ[i] * y[i] for i in Q)
        rY = sum(y[i] for i in Q̄idx)                       # Σ_{i∈Q̄} yᵢ
        @constraint(m, Py + rY      ≥ b + ε)                            # (24)
        @constraint(m, Py + Γ - Σμ  ≥ b + ε)                            # (25)
    end
    optimize!(m)
    return termination_status(m) == MOI.OPTIMAL && value(ε) > 1e-7
end

# Membership test  ŷ ∈ Ŷ(Q):  ŷ is the cheapest feasible solution for some μ
# ⟺ max_{ℓ∈[Γ−1]₀} ε(ŷ,ℓ) > 0, where ε(ŷ,ℓ) solves                  [Lemma 3, (15)–(22)]
#   max_{μ,ε} ε
#   s.t.  Σ_{i∈Q} μᵢŷᵢ + Σ_{i∈Q̄} ŷᵢ  ≤ b       if Σ_{i∈Q̄} ŷᵢ ≤ ℓ        (16)
#         Σ_{i∈Q} μᵢŷᵢ + Γ − Σ_{i∈Q} μᵢ ≤ b     if Σ_{i∈Q̄} ŷᵢ ≥ ℓ+1      (17)
#         Σ_{i∈Q} μᵢy'ᵢ + Σ_{i∈Q̄} y'ᵢ   ≥ b+ε   ∀y'∈Ỹ(Q): cᵀy' < cᵀŷ     (18)
#         Σ_{i∈Q} μᵢy'ᵢ + Γ − Σ_{i∈Q} μᵢ ≥ b+ε  ∀y'∈Ỹ(Q): cᵀy' < cᵀŷ     (19)
#         ℓ ≤ Γ − Σ_{i∈Q} μᵢ ≤ ℓ+1                                      (20)
#         Σ_{i∈Q} μᵢ ≤ Γ,   0 ≤ μᵢ ≤ 1                                  (21),(22)
function in_Ŷ(Γ, b, Q, ŷ, cheaper, Q̄idx; opt = grb_opt)
    rhat   = sum(ŷ[i] for i in Q̄idx)                       # Σ_{i∈Q̄} ŷᵢ
    chinfo = [(yp, sum(yp[i] for i in Q̄idx)) for yp in cheaper]   # (y', Σ_{Q̄} y'ᵢ)
    for ℓ in 0:(round(Int, Γ) - 1)                         # ℓ ∈ [Γ−1]₀
        m = Model(opt); set_silent(m)
        @variable(m, μ[Q], lower_bound = 0.0, upper_bound = 1.0)
        @variable(m, ε, upper_bound = 1.0)                 # only the sign of ε* matters
        @objective(m, Max, ε)
        Σμ = sum(μ[i] for i in Q)
        @constraint(m, Σμ ≤ Γ)                                          # (21)
        @constraint(m, ℓ ≤ Γ - Σμ)                                      # (20)
        @constraint(m, Γ - Σμ ≤ ℓ + 1)                                  # (20)
        Pŷ = sum(μ[i] * ŷ[i] for i in Q)
        if rhat ≤ ℓ
            @constraint(m, Pŷ + rhat     ≤ b)                           # (16)
        else
            @constraint(m, Pŷ + Γ - Σμ   ≤ b)                           # (17)
        end
        for (yp, ryp) in chinfo
            Pyp = sum(μ[i] * yp[i] for i in Q)
            @constraint(m, Pyp + ryp     ≥ b + ε)                       # (18)
            @constraint(m, Pyp + Γ - Σμ  ≥ b + ε)                       # (19)
        end
        optimize!(m)
        if termination_status(m) == MOI.OPTIMAL && value(ε) > 1e-7
            return true
        end
    end
    return false
end

# Φ(Q) via Algorithm 1:  Φ(Q) = max_{ŷ∈Ŷ(Q)} cᵀŷ.  Scan Ỹ(Q) by decreasing cost
# and return the first ŷ∈Ŷ(Q).                                        [Prop. 1, Alg. 1]
function Φ_mst(c, edges, V, Γ, b, Q; opt = grb_opt)
    n = length(edges)
    cands = reduced_mst(c, edges, V, Q)
    isempty(cands) && return Inf, nothing
    Q̄idx = [e for e in 1:n if e ∉ Q]
    Φ_unbounded(Γ, b, Q, cands, Q̄idx; opt = opt) && return Inf, nothing
    costs = [cᵀ(c, y) for y in cands]
    for j in sortperm(costs; rev = true)
        ŷ = cands[j]
        cheaper = [cands[t] for t in eachindex(cands) if costs[t] < costs[j]]
        in_Ŷ(Γ, b, Q, ŷ, cheaper, Q̄idx; opt = opt) && return costs[j], ŷ
    end
    return Inf, nothing
end

# Build one optimizer factory per worker.  Gurobi needs a private Gurobi.Env per
# worker (an env must not be shared across threads); HiGHS optimizers are
# independent, so none is needed.  Capping the solver at one thread per LP keeps
# the parallelism at the worker level, not inside each tiny solve.  MOI.Silent()
# is listed first so it is applied before "Threads", suppressing the Gurobi
# "Set parameter Threads to value 1" banner.
function worker_factories(solver::Symbol, nworkers::Integer)
    if solver === :gurobi
        envs = [redirect_stdout(() -> Gurobi.Env(), devnull) for _ in 1:nworkers]
        return [optimizer_with_attributes(() -> Gurobi.Optimizer(envs[w]), MOI.Silent() => true, "Threads" => 1) for w in 1:nworkers]
    elseif solver === :highs
        return [optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true, "threads" => 1) for _ in 1:nworkers]
    else
        error("solver must be :gurobi or :highs (got :$solver)")
    end
end

"""
    ddid_mst(c, edges, V, q, Γ, b; solver = :gurobi) -> (Φ*, Q*, y*)

Exact DDID value for the robust MST (Chen & Poss): enumerate every query set
Q ∈ combinations(1:n, q) and return the smallest Φ(Q).

`solver` selects the LP solver, `:gurobi` or `:highs`.

The enumeration runs in parallel over Julia's threads (start Julia with
`julia -t N`); it uses up to `Threads.nthreads()` workers.  Each worker gets its
own optimizer factory (for Gurobi, a private Gurobi.Env — an env must not be
shared across threads), and pulls query sets from a shared iterator for dynamic
load balancing.  Ties are broken by the lowest combinations-index, so the result
matches a sequential scan.
"""
function ddid_mst(c, edges, V, q, Γ, b; solver::Symbol = :gurobi)
    n = length(edges)
    nworkers = max(1, min(Threads.nthreads(), binomial(n, q)))

    optf = worker_factories(solver, nworkers)

    # Compile the JuMP→Gurobi solve path on one thread first, so the workers do
    # not all hit first-call compilation at once and serialize on the compiler
    # lock (which would stall the first run for minutes).
    nworkers > 1 && Φ_mst(c, edges, V, Γ, b, first(combinations(1:n, q)); opt = optf[1])

    # Per-worker incumbents, combined at the end.  `bi` is the position of the
    # incumbent Q in combinations order, used to break ties as a scan would.
    bφ = fill(Inf, nworkers)
    bi = fill(typemax(Int), nworkers)
    bQ = Vector{Union{Nothing,Vector{Int}}}(nothing, nworkers)
    by = Vector{Union{Nothing,Vector{Int}}}(nothing, nworkers)

    # Shared, lock-protected enumeration of combinations(1:n, q): workers pull
    # the next Q on demand (Φ_mst cost varies a lot with Q).
    lk  = ReentrantLock()
    it  = combinations(1:n, q)
    st  = iterate(it)
    cnt = 0
    function next_Q()
        lock(lk)
        try
            st === nothing && return nothing
            Q, s = st
            cnt += 1; j = cnt
            st = iterate(it, s)
            return (j, Q)
        finally
            unlock(lk)
        end
    end

    @sync for w in 1:nworkers
        Threads.@spawn begin
            f = optf[w]
            while true
                item = next_Q()
                item === nothing && break
                j, Q = item
                φ, ŷ = Φ_mst(c, edges, V, Γ, b, Q; opt = f)
                if φ < bφ[w]
                    bφ[w], bi[w], bQ[w], by[w] = φ, j, collect(Q), ŷ
                end
            end
        end
    end

    best, besti, Qstar, ystar = Inf, typemax(Int), nothing, nothing
    for w in 1:nworkers
        if bφ[w] < best || (bφ[w] == best && bi[w] < besti)
            best, besti, Qstar, ystar = bφ[w], bi[w], bQ[w], by[w]
        end
    end
    return best, Qstar, ystar
end

"""
    ddid_mst_LS(c, edges, V, q, Γ, b; solver = :gurobi, rng = MersenneTwister(1),
                max_restarts = 1000) -> (Φ, Q, y)

Local-search heuristic for the DDID robust MST (same Φ objective as `ddid_mst`).
Instead of enumerating all C(n,q) query sets it:

  1. starts from the q highest-cost edges (sorted by decreasing cost), then
  2. repeatedly moves to the best 1-swap neighbour — replace one edge in Q by one
     edge outside Q — until no swap improves Φ.

Each round evaluates the q·(n−q) neighbours in parallel over Julia's threads,
exactly like `ddid_mst`.  The result is a local optimum of Φ, hence an upper
bound on the exact Φ*, not necessarily equal to it.

A search can end without a feasible query set (Φ = ∞: the start and all its
neighbours give an unbounded robust value).  When that happens the heuristic
restarts from random q-subsets until it finds a feasible one, then local-searches
from there; it stops after `max_restarts` restarts or when the search space is
exhausted.  Each restart is drawn at least two swaps away from every previously
tried start, so it differs from each visited start *and* its 1-swap neighbours.
Because only a feasible *start* is needed, restart candidates are screened by a
single Φ evaluation (in parallel batches) rather than a full local search apiece,
which keeps the extra work — and the solver churn — small.  Returns
(∞, nothing, nothing) if no feasible query set is found.  `solver` is `:gurobi`
or `:highs`.
"""
function ddid_mst_LS(c, edges, V, q, Γ, b; solver::Symbol = :gurobi,
                     rng::AbstractRNG = MersenneTwister(1), max_restarts::Integer = 1000)
    n = length(edges)
    nworkers = max(1, min(Threads.nthreads(), q * (n - q)))
    optf = worker_factories(solver, nworkers)

    # Evaluate Φ for many query sets in parallel; out[k] ⇔ Qs[k].  Each worker
    # writes distinct indices, so there is no race.
    function eval_phis(Qs)
        N = length(Qs)
        out = Vector{Tuple{Float64,Union{Nothing,Vector{Int}}}}(undef, N)
        nxt = Threads.Atomic{Int}(1)
        @sync for w in 1:nworkers
            Threads.@spawn begin
                f = optf[w]
                while true
                    k = Threads.atomic_add!(nxt, 1)
                    k > N && break
                    out[k] = Φ_mst(c, edges, V, Γ, b, Qs[k]; opt = f)
                end
            end
        end
        return out
    end

    # Best-improvement 1-swap local search from a starting query set Q.  The
    # first Φ_mst is single-threaded, which also compiles the path before the
    # parallel rounds (avoids the all-workers-compile-at-once stall).
    function local_search(Q)
        φ, ŷ = Φ_mst(c, edges, V, Γ, b, Q; opt = optf[1])
        while true
            Qset = Set(Q)
            outside = [j for j in 1:n if j ∉ Qset]
            neighbours = [sort([e == i ? j : e for e in Q]) for i in Q for j in outside]  # Q with i↦j
            isempty(neighbours) && break
            res = eval_phis(neighbours)
            GC.gc(false)                        # drain Gurobi finalizers at a single-threaded
            bestφ, k = findmin(first.(res))
            bestφ < φ - 1e-9 || break          # no improving swap ⇒ local optimum
            φ, ŷ, Q = bestφ, res[k][2], neighbours[k]
        end
        return φ, Q, ŷ
    end

    # A random q-subset at least two swaps from every tried start (|Q ∩ T| ≤ q−2),
    # i.e. different from each visited start and its 1-swap neighbours.  Returns
    # `nothing` if rejection sampling cannot find one (the space is exhausted).
    function fresh_start(tried)
        for _ in 1:5000
            Q = sort!(randperm(rng, n)[1:q])
            all(T -> count(in(T), Q) ≤ q - 2, tried) && return Q
        end
        return nothing
    end

    # Informed start: the q highest-cost edges.  Its full local search also
    # explores the neighbourhood, so a feasible neighbour is found if one is there.
    Q0 = sort(sortperm(c; rev = true)[1:q])
    φ, Qbest, ŷ = local_search(Q0)
    isfinite(φ) && return φ, Qbest, ŷ

    # Start (and whole neighbourhood) infeasible: probe random starts for a feasible
    # one — one Φ each, in parallel batches — then local-search from the first hit.
    tried = [Set(Q0)]
    left = max_restarts
    while left > 0
        batch = Vector{Int}[]
        while length(batch) < min(nworkers, left)
            Q = fresh_start(tried)
            Q === nothing && break
            push!(tried, Set(Q)); push!(batch, Q)
        end
        isempty(batch) && break                # no admissible new start remains
        left -= length(batch)
        res = eval_phis(batch)
        GC.gc(false)
        k = findfirst(r -> isfinite(first(r)), res)
        k === nothing || return local_search(batch[k])   # feasible start ⇒ optimise from it
    end
    @warn "ddid_mst_LS: no feasible query set found after $(max_restarts - left) random restart(s)"
    return Inf, nothing, nothing
end

# =====================================================================
#  (2)  VAYANOS, GEORGHIOU & YU – K-adaptability MILP  (§4.2, Theorem 4)
# =====================================================================

# Exact linearization of  z = δ·x  with δ∈{0,1}, x∈[lo,hi]  (McCormick).
# Used for every binary×continuous product in (11) (Cor. 1's linearization).
function mccormick(model, δ, x, lo, hi)
    z = @variable(model)
    @constraint(model, z ≤ hi * δ)
    @constraint(model, z ≥ lo * δ)
    @constraint(model, z ≤ x - lo * (1 - δ))
    @constraint(model, z ≥ x - hi * (1 - δ))
    return z
end

# The ℓ-pattern dual blocks of (11).  w, y (K×n) and τ are existing model
# variables.  Specialization of (11) to the robust MST:
#   • decisions:  w∈𝒲={w∈{0,1}ⁿ:eᵀw=q},  yᵏ∈𝒴={spanning trees}, k∈𝒦={1..K}, no x.
#   • Ξ = {ξ : Aξ ≤ d},  A=[I;−I;eᵀ], d=[1;0;Γ]  (R=2n+1).  d is the paper's b in
#     Aξ≤b, renamed so it does not clash with the cardinality bound b.
#   • single constraint (L=1)  ξᵀyᵏ ≤ b  ⇒  [H]_{ℓₖ} ≡ −yᵏ and T=V=W=0 (Rmk 4).
#   • deterministic cost cᵀyᵏ ⇒ C=D=0, ξ-coefficient [Cx+Dw+Qyᵏ]=0; the cost is
#     carried explicitly through λₖcᵀyᵏ and the constant b through βᵏ,γₖ.
#   • patterns  ℓ∈𝓛={0,1}ᴷ;  ∂𝓛=𝓛∖{𝟙},  𝓛₊={𝟙};
#     Λᴷ(ℓ)={λ∈ℝᴷ₊:eᵀλ=1,λₖ=0 ∀k:ℓₖ=1}.
function _kadapt_dual_blocks!(model, w, y, τ, c, A, d, n, K, b, ε, M)
    R = size(A, 1)
    Ls = vec([collect(t) for t in Iterators.product(ntuple(_ -> 0:1, K)...)])  # 𝓛={0,1}ᴷ
    for ℓ in Ls
        allone = all(==(1), ℓ)                       # ℓ=𝟙  ⇔  ℓ∈𝓛₊

        α  = @variable(model, [1:R], lower_bound = 0.0)                 # α(ℓ)
        αk = @variable(model, [1:K, 1:R], lower_bound = 0.0)            # αᵏ(ℓ)
        η  = @variable(model, [1:K, 1:n], lower_bound = -M, upper_bound = M)  # ηᵏ(ℓ)
        β  = Dict(k => @variable(model, lower_bound = 0.0, upper_bound = M) for k in 1:K if ℓ[k] == 0)
        γ  = Dict(k => @variable(model, lower_bound = 0.0, upper_bound = M) for k in 1:K if ℓ[k] == 1)
        wη = [[mccormick(model, w[i], η[k, i], -M, M) for i in 1:n] for k in 1:K]  # w∘ηᵏ

        Aα = A' * α                                                    # (11): Aᵀα = Σₖ w∘ηᵏ
        @constraint(model, [i = 1:n], Aα[i] == sum(wη[k][i] for k in 1:K))

        for k in 1:K
            Aαk = A' * αk[k, :]
            if ℓ[k] == 0
                # (11), ℓₖ=0:  Aᵀαᵏ − Hᵀβᵏ + w∘ηᵏ = λₖ[Cx+Dw+Qyᵏ];  −Hᵀβᵏ=+yᵏβᵏ, RHS 0
                yβ = [mccormick(model, y[k, i], β[k], 0.0, M) for i in 1:n]
                @constraint(model, [i = 1:n], Aαk[i] + wη[k][i] + yβ[i] == 0)
            else
                # (11), ℓₖ≠0:  Aᵀαᵏ + [H]_{ℓₖ}γₖ + w∘ηᵏ = λₖ[…];  [H]_{ℓₖ}=−yᵏ, RHS 0
                yγ = [mccormick(model, y[k, i], γ[k], 0.0, M) for i in 1:n]
                @constraint(model, [i = 1:n], Aαk[i] + wη[k][i] - yγ[i] == 0)
            end
        end

        dualobj = sum(d[r] * α[r] for r in 1:R) + sum(d[r] * αk[k, r] for k in 1:K, r in 1:R)
        if !allone
            λ = @variable(model, [1:K], lower_bound = 0.0, upper_bound = 1.0)   # λ(ℓ)∈Λᴷ(ℓ)
            for k in 1:K; ℓ[k] == 1 && @constraint(model, λ[k] == 0); end
            @constraint(model, sum(λ[k] for k in 1:K) == 1)
            λy = [mccormick(model, y[k, i], λ[k], 0.0, 1.0) for k in 1:K, i in 1:n]
            cost = sum(c[i] * λy[k, i] for k in 1:K, i in 1:n)         # Σₖ λₖ cᵀyᵏ
            # (11):  τ ≥ dᵀ(α+Σαᵏ) + Σ_{ℓₖ=0} b βᵏ − Σ_{ℓₖ≠0}(b+ε)γₖ + cost  (T=V=W=0)
            @constraint(model, τ ≥ cost + dualobj
                                   + sum(b * β[k]       for k in 1:K if ℓ[k] == 0; init = 0.0)
                                   - sum((b + ε) * γ[k] for k in 1:K if ℓ[k] == 1; init = 0.0))
        else
            # (11), 𝓛₊:  dᵀ(α+Σαᵏ) − Σₖ(b+ε)γₖ ≤ −1   (forbids S(𝟙)=∅)
            @constraint(model, dualobj - sum((b + ε) * γ[k] for k in 1:K) ≤ -1.0)
        end
    end
    return model
end

# Spanning-tree constraints on a binary edge vector yk:  eᵀyk = |V|−1  plus
# subtour elimination Σ_{e⊆S} yk_e ≤ |S|−1 for every 2≤|S|≤|V|−1.  A forest with
# |V|−1 edges on |V| nodes is a spanning tree, so this is exactly 𝒴.
function add_tree_constraints!(model, yk, edges, V)
    @constraint(model, sum(yk) == V - 1)
    for s in 2:(V - 1)
        for S in combinations(1:V, s)
            Sset = Set(S)
            inS = [e for e in eachindex(edges) if edges[e][1] ∈ Sset && edges[e][2] ∈ Sset]
            isempty(inS) || @constraint(model, sum(yk[e] for e in inS) ≤ s - 1)
        end
    end
    return model
end

# Lexicographic symmetry-breaking constraints (Vayanos, Georghiou & Yu, §EC.3.1,
# eqs (EC.3)–(EC.4)).  The K candidate policies are forced to be lexicographically
# decreasing, removing the K! permutation symmetry of (𝒫ₖ) that otherwise slows
# branch-and-bound.  z[k,i] = 1 ⇔ yᵏ and yᵏ⁺¹ differ in component i; with that,
# the first differing component must have yᵏ ≥ yᵏ⁺¹.  These are deterministic,
# so they leave the optimum unchanged.
function add_symmetry_breaking!(model, y, n, K)
    K < 2 && return model
    @variable(model, z[1:K-1, 1:n], Bin)               # zᵏ,ᵏ⁺¹ ∈ {0,1}ⁿ
    for k in 1:K-1, i in 1:n
        @constraint(model, z[k, i] ≤ y[k, i] + y[k+1, i])           # (EC.3): z=0 if equal at 0
        @constraint(model, z[k, i] ≤ 2 - y[k, i] - y[k+1, i])       #         z=0 if equal at 1
        @constraint(model, z[k, i] ≥ y[k, i] - y[k+1, i])           #         z=1 if they differ
        @constraint(model, z[k, i] ≥ y[k+1, i] - y[k, i])
        # (EC.4): equal on all i'<i  ⇒  yᵏᵢ ≥ yᵏ⁺¹ᵢ  (lexicographically decreasing)
        @constraint(model, y[k, i] ≥ y[k+1, i] - sum(z[k, ip] for ip in 1:i-1; init = 0))
    end
    return model
end

"""
    kadapt_vayanos_mst(c, edges, V, q, Γ, b, K; optimizer = grb_opt, ...) -> (value, w, [y¹..yᴷ], status)

K-adaptability MILP for the robust MST, the reformulation of Theorem 4, eq (11).
`optimizer` selects the MILP solver (e.g. `grb_opt` for Gurobi or
`HiGHS.Optimizer` for HiGHS).  The dual blocks (`_kadapt_dual_blocks!`) are
problem-independent; the feasible set 𝒴 enters only through
`add_tree_constraints!`; the K! policy symmetry is removed by the lexicographic
constraints of `add_symmetry_breaking!`.
"""
function kadapt_vayanos_mst(c, edges, V, q, Γ, b, K;
                            ε = 1e-3, M = 1e3, optimizer = grb_opt,
                            time_limit = 600.0, silent = true)
    n = length(edges)
    A = vcat(Matrix(1.0I, n, n), Matrix(-1.0I, n, n), ones(1, n))   # Ξ = {ξ : Aξ ≤ d}
    d = vcat(ones(n), zeros(n), float(Γ))
    model = Model(optimizer); silent && set_silent(model)
    time_limit === nothing || set_time_limit_sec(model, time_limit)

    @variable(model, w[1:n], Bin);  @constraint(model, sum(w) == q)         # query q edges
    @variable(model, y[1:K, 1:n], Bin)
    for k in 1:K; add_tree_constraints!(model, y[k, :], edges, V); end       # yᵏ ∈ trees
    @variable(model, τ);  @objective(model, Min, τ)
    add_symmetry_breaking!(model, y, n, K)             # lexicographic (EC.3)–(EC.4)
    _kadapt_dual_blocks!(model, w, y, τ, c, A, d, n, K, b, ε, M)

    optimize!(model); st = termination_status(model)
    # Return whatever the solver found: on OPTIMAL the optimum; on TIME_LIMIT the
    # best incumbent found in the branch-and-bound tree.  `has_values` is false
    # only when no feasible solution was found at all, in which case we report Inf.
    return has_values(model) ?
        (objective_value(model), round.(Int, value.(w)),
         [round.(Int, value.(y[k, :])) for k in 1:K], st) :
        (Inf, nothing, nothing, st)
end

# =====================================================================
#  Experiment harness
# =====================================================================

# Status of a K-adaptability solve vs the Chen–Poss exact value Φ*:
#   inf  infeasible | Opt  = Φ* | Error  < Φ* (impossible bound!) | Feas  > Φ*
function kstatus(v, Φstar; tol = 1e-4)
    v == Inf && return "inf"
    Φstar == Inf && return "inf"
    v < Φstar - tol  && return "Error"
    v ≤ Φstar + tol  && return "Opt"
    return "Feas"
end

# Complete graph on V nodes (V=5 ⇒ 10 edges), random integer edge costs.
function gen_mst_instance(; V = 5, seed = 1, cmin = 1, cmax = 100)
    rng = MersenneTwister(seed)
    edges = [(u, v) for u in 1:V for v in (u + 1):V]            # K_V : C(V,2) edges
    c = Float64[rand(rng, cmin:cmax) for _ in eachindex(edges)]
    return (; edges, V, c, n = length(edges))
end

"""
    run_mst_instance(; seed, q, V, Γ, b, Kmax, time_limit, solver) -> NamedTuple

Compute the exact Φ* (Chen–Poss) for one random robust-MST instance, run the
`ddid_mst_LS` local-search heuristic on the same instance, then solve the
K-adaptability MILP for K=1,2,… and stop at the first K whose value matches
Φ* (the optimal K), or at the first K that hits the time limit (larger K only
grow harder, so they are skipped).

`solver` chooses the solver (`:gurobi` or `:highs`) for both the Chen–Poss LPs
and the K-adaptability MILP.

`exact` toggles the exact Chen–Poss enumeration.  With `exact = false` it is
skipped (so the method does not enumerate all C(n,q) query sets) and only LS and
the K-adaptability MILP are run — useful for large instances.  In that case the
K-adaptability loop compares against, and stops once it reaches, the LS value
instead of the (unavailable) exact Φ*.

`ktimes[K]` / `kvals[K]` hold the wall-clock time and objective of each attempted
K (`NaN` for a K skipped once the optimum was reached or the time limit was hit);
`ktimedout[K]` flags a K that hit the time limit, and `kvals[K]` is `Inf` for an
infeasible model.
"""
function run_mst_instance(; seed = 1, q = 2, V = 5, Γ = 3, b = 2, Kmax = 4,
                          time_limit = 3600.0, solver::Symbol = :gurobi, exact::Bool = true)
    kadapt_opt = solver === :gurobi ? grb_opt :
                 solver === :highs  ? HiGHS.Optimizer :
                 error("solver must be :gurobi or :highs (got :$solver)")

    inst = gen_mst_instance(; seed = seed, V = V)
    c, edges, V, n = inst.c, inst.edges, inst.V, inst.n
    @printf("\n##### MST on K%d: n=%d edges, q=%d, Γ=%.1f, b=%d (seed %d) #####\n",
            V, n, q, Γ, b, seed)
    println("  edges = ", edges)
    println("  costs = ", Int.(c))
    flush(stdout)

    # Exact Chen–Poss value.  Skipped when `exact = false` (it enumerates all
    # C(n,q) query sets, so it is the part that does not scale to large n).
    if exact
        t = time()
        Φstar, Qstar, _ = ddid_mst(c, edges, V, q, Γ, b; solver = solver)
        tCP = time() - t
        @printf("  Chen–Poss  Φ* = %.4f   (Q* = %s)   [%.2fs]\n",
                Φstar, Qstar === nothing ? "—" : string(Qstar), tCP)
    else
        Φstar, Qstar, tCP = NaN, nothing, NaN
    end
    flush(stdout)

    t = time()
    Φls, Qls, _ = ddid_mst_LS(c, edges, V, q, Γ, b; solver = solver)
    tLS = time() - t
    @printf("  LS         Φ  = %.4f   (Q  = %s)   [%.2fs]%s\n",
            Φls, Qls === nothing ? "—" : string(Qls), tLS,
            exact ? "   " * kstatus(Φls, Φstar) : "")
    flush(stdout)

    # The K-adaptability loop tags solutions against, and stops once it reaches, a
    # reference value: the exact Φ* when available, otherwise the LS upper bound.
    ref = exact ? Φstar : Φls

    ktimes = fill(NaN, Kmax); kvals = fill(NaN, Kmax)   # NaN ⇒ K not attempted
    ktimedout = falses(Kmax)
    Kstar = nothing
    for K in 1:Kmax
        t = time()
        v, _, _, st = kadapt_vayanos_mst(c, edges, V, q, Γ, b, K;
                                         optimizer = kadapt_opt, time_limit = time_limit)
        dt = time() - t
        timedout = st == MOI.TIME_LIMIT
        ktimes[K] = dt; kvals[K] = v; ktimedout[K] = timedout
        reached = isfinite(v) && isfinite(ref) && v ≤ ref + 1e-4   # hit Φ* (exact) or LS bound
        note = timedout ? " (time limit)" : ""
        tag = exact ? kstatus(v, Φstar) :
              v == Inf       ? "inf" :
              v < ref - 1e-4 ? "<LS" :
              reached        ? "=LS" : ">LS"
        if v == Inf
            @printf("    K=%d : %-12s %-5s [%.2fs]%s\n", K, "∞", tag, dt, note)
        else
            @printf("    K=%d : %-12.4f %-5s [%.2fs]%s\n", K, v, tag, dt, note)
        end
        flush(stdout)
        reached && (Kstar = K; break)   # reached the reference ⇒ larger K only harder, stop
        timedout && break               # time limit hit ⇒ stop
    end
    @printf("  => %s = %s\n", exact ? "optimal K" : "K reaching LS",
            Kstar === nothing ? ">$Kmax" : string(Kstar))
    flush(stdout)
    return (; seed, exact, Φstar, Qstar, Kstar, tCP, Φls, Qls, tLS, ktimes, kvals, ktimedout)
end

# Format a value column: integer (no decimals), or ∞ for an infeasible model.
_fmtval(v) = v == Inf ? "\$\\infty\$" : string(round(Int, v))

# Format a time to 3 significant figures, fixed-point (e.g. 453.14→453, 18.36→18.4,
# 1315.69→1320, 0.05→0.0500); the decimal count is chosen from the magnitude.
function _fmttime(x)
    isfinite(x) || return string(x)
    m = x == 0 ? 0 : floor(Int, log10(abs(x)))
    return Printf.format(Printf.Format("%.$(max(0, 2 - m))f"), round(x; sigdigits = 3))
end

"""
    write_mst_table(rows, Kmax; path="MST.txt")

Write the results of `run_mst_instance` as a LaTeX `tabular` with a (time, value)
pair of columns per seed.  The first row is the exact Chen–Poss result (its time
and the exact Φ*), the second is the `LS` local-search heuristic, and each
following row is one K, for every K attempted by some seed.
The search loop stops at the optimal K or at a time-limit hit, so these are
exactly the informative K.  In a value cell: an integer is the K-adaptability
value and `\$\\infty\$` an infeasible model; a solve that hit the time limit shows
`T` in its time cell and the best incumbent found (an upper bound) in its value
cell — or `T` there too if no feasible point was found in time.  A cell pair is blank for a K a seed never
attempted (optimum already reached, or skipped after a timeout).
"""
function write_mst_table(rows, Kmax; path = "MST.txt")
    attempted(r, K) = !isnan(r.kvals[K])
    Kshow = maximum((K for r in rows for K in 1:Kmax if attempted(r, K)); init = 0)
    S = length(rows)
    open(path, "w") do io
        println(io, "\\begin{tabular}{l|", repeat("rr|", S), "}")
        println(io, "\\toprule")
        println(io, " & ", join(("\\multicolumn{2}{c}{seed $(r.seed)}" for r in rows), " & "), " \\\\")
        println(io, " & ", join(("time & value" for _ in rows), " & "), " \\\\")
        println(io, "\\midrule")
        cp = String[]
        for r in rows
            r.exact ? push!(cp, _fmttime(r.tCP), _fmtval(r.Φstar)) : push!(cp, "—", "—")
        end
        any(r -> r.exact, rows) && println(io, "exact & ", join(cp, " & "), " \\\\")
        ls = String[]
        for r in rows; push!(ls, _fmttime(r.tLS), _fmtval(r.Φls)); end
        println(io, "LS & ", join(ls, " & "), " \\\\")
        println(io, "\\midrule")
        for K in 1:Kshow
            cells = String[]
            for r in rows
                if !attempted(r, K)
                    push!(cells, "", "")                                   # K never attempted
                elseif r.ktimedout[K]
                    # time limit hit: keep the best incumbent found (an upper
                    # bound) with a T-flagged time; T/T if no feasible point.
                    push!(cells, "T", isfinite(r.kvals[K]) ? _fmtval(r.kvals[K]) : "T")
                else
                    push!(cells, _fmttime(r.ktimes[K]), _fmtval(r.kvals[K]))
                end
            end
            println(io, "\$K=$K\$ & ", join(cells, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    @printf("\n[latex] wrote table (%d seeds, K up to %d) -> %s\n", S, Kshow, path)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    Kmax   = 10
    solver = :gurobi                          # :gurobi or :highs for the Chen–Poss LPs and the MILP
    exact = Dict(5 => true, 10 => false)      # false ⇒ skip exact Chen–Poss (for large instances): LS + K-adaptability only
    Q = Dict(5 => 4, 10 => 8)

    @info "warm up"
    run_mst_instance(; seed = 1, q = Q[5], Γ = 2.5, b = 2, V = 5,
                     Kmax = Kmax, time_limit = 3.0, solver = solver, exact = exact[5])

    for V in (5,10)
        rows = [run_mst_instance(; seed = seed, q = Q[V], Γ = 2.5, b = 2, V = V,
                                 Kmax = Kmax, time_limit = 1800, solver = solver, exact = exact[V])
                for seed in 1:5]
        write_mst_table(rows, Kmax; path = "MST$V.txt")
    end
end
