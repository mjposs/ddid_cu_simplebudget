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
# deps:  JuMP, Gurobi, Combinatorics, LinearAlgebra, Random, Printf

using JuMP, Gurobi, Combinatorics, LinearAlgebra, Random, Printf

# One shared Gurobi environment, created once inside a stdout redirect so the
# "Set parameter LicenseID to value …" banner is never printed.  Reusing this
# environment for every model means the banner is not reprinted per model.
const GRB_ENV = redirect_stdout(devnull) do
    Gurobi.Env()
end
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

"""
    ddid_mst(c, edges, V, q, Γ, b) -> (Φ*, Q*, y*)

Exact DDID value for the robust MST, enumerating the query sets Q (Chen & Poss).
"""
function ddid_mst(c, edges, V, q, Γ, b; opt = grb_opt)
    n = length(edges)
    best = Inf; Qstar = nothing; ystar = nothing
    for Q in combinations(1:n, q)
        φ, ŷ = Φ_mst(c, edges, V, Γ, b, Q; opt = opt)
        if φ < best
            best, Qstar, ystar = φ, collect(Q), ŷ
        end
    end
    return best, Qstar, ystar
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

"""
    kadapt_vayanos_mst(c, edges, V, q, Γ, b, K; ...) -> (value, w, [y¹..yᴷ], status)

K-adaptability MILP for the robust MST, the reformulation of Theorem 4, eq (11).
The dual blocks (`_kadapt_dual_blocks!`) are problem-independent; the feasible
set 𝒴 enters only through `add_tree_constraints!`.
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
    for k in 1:K-1
        @constraint(model, cᵀ(c, y[k, :]) ≤ cᵀ(c, y[k+1, :]))              # symmetry break
    end
    _kadapt_dual_blocks!(model, w, y, τ, c, A, d, n, K, b, ε, M)

    optimize!(model); st = termination_status(model)
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
function gen_mst_instance(; V = 5, seed = 1, cmin = 1, cmax = 10)
    rng = MersenneTwister(seed)
    edges = [(u, v) for u in 1:V for v in (u + 1):V]            # K_V : C(V,2) edges
    c = Float64[rand(rng, cmin:cmax) for _ in eachindex(edges)]
    return (; edges, V, c, n = length(edges))
end

"""
    run_mst_instance(; seed, q, Γ, b, Kmax, time_limit) -> NamedTuple

Compute the exact Φ* (Chen–Poss) for one random robust-MST instance, then solve
the K-adaptability MILP for K=1,2,… and stop at the first K whose value matches
Φ* (the optimal K), or at the first K that hits the time limit (larger K only
grow harder, so they are skipped).  Output is printed as the experiment
proceeds; the per-K solve times/values needed for the LaTeX table are returned.

`ktimes[K]` / `kvals[K]` hold the wall-clock time and objective of each attempted
K (`NaN` for a K skipped once the optimum was reached or the time limit was hit);
`ktimedout[K]` flags a K that hit the time limit, and `kvals[K]` is `Inf` for an
infeasible model.
"""
function run_mst_instance(; seed = 1, q = 2, Γ = 3, b = 2, Kmax = 4, time_limit = 3600.0)
    inst = gen_mst_instance(; seed = seed)
    c, edges, V, n = inst.c, inst.edges, inst.V, inst.n
    @printf("\n##### MST on K%d: n=%d edges, q=%d, Γ=%.1f, b=%d (seed %d) #####\n",
            V, n, q, Γ, b, seed)
    println("  edges = ", edges)
    println("  costs = ", Int.(c))
    flush(stdout)

    t = time()
    Φstar, Qstar, _ = ddid_mst(c, edges, V, q, Γ, b)
    tCP = time() - t
    @printf("  Chen–Poss  Φ* = %.4f   (Q* = %s)   [%.2fs]\n",
            Φstar, Qstar === nothing ? "—" : string(Qstar), tCP)
    flush(stdout)

    ktimes = fill(NaN, Kmax); kvals = fill(NaN, Kmax)   # NaN ⇒ K not attempted
    ktimedout = falses(Kmax)
    Kstar = nothing
    for K in 1:Kmax
        t = time()
        v, _, _, st = kadapt_vayanos_mst(c, edges, V, q, Γ, b, K; time_limit = time_limit)
        dt = time() - t
        timedout = st == MOI.TIME_LIMIT
        ktimes[K] = dt; kvals[K] = v; ktimedout[K] = timedout
        tag = kstatus(v, Φstar)
        note = timedout ? " (time limit)" : ""
        if v == Inf
            @printf("    K=%d : %-12s %-5s [%.2fs]%s\n", K, "∞", tag, dt, note)
        else
            @printf("    K=%d : %-12.4f %-5s [%.2fs]%s\n", K, v, tag, dt, note)
        end
        flush(stdout)
        if tag == "Opt"            # optimal K found ⇒ stop trying larger K
            Kstar = K
            break
        end
        timedout && break          # time limit hit ⇒ larger K only harder, skip them
    end
    @printf("  => optimal K = %s\n", Kstar === nothing ? ">$Kmax" : string(Kstar))
    flush(stdout)
    return (; seed, Φstar, Qstar, Kstar, tCP, ktimes, kvals, ktimedout)
end

# Format a value column: integer (no decimals), or ∞ for an infeasible model.
_fmtval(v) = v == Inf ? "\$\\infty\$" : string(round(Int, v))

"""
    write_mst_table(rows, Kmax; path="MST.txt")

Write the results of `run_mst_instance` as a LaTeX `tabular` with a (time, value)
pair of columns per seed.  The first row is the Chen–Poss result (its time and
the exact Φ*); each following row is one K, for every K attempted by some seed.
The search loop stops at the optimal K or at a time-limit hit, so these are
exactly the informative K.  In a value cell: an integer is the K-adaptability
value and `\$\\infty\$` an infeasible model; a solve that hit the time limit shows
`T` in both its time and value cells.  A cell pair is blank for a K a seed never
attempted (optimum already reached, or skipped after a timeout).
"""
function write_mst_table(rows, Kmax; path = "MST.txt")
    attempted(r, K) = !isnan(r.kvals[K])
    Kshow = maximum((K for r in rows for K in 1:Kmax if attempted(r, K)); init = 0)
    S = length(rows)
    open(path, "w") do io
        println(io, "\\begin{tabular}{l", repeat("rr", S), "}")
        println(io, "\\hline")
        println(io, " & ", join(("\\multicolumn{2}{c}{seed $(r.seed)}" for r in rows), " & "), " \\\\")
        println(io, " & ", join(("time & value" for _ in rows), " & "), " \\\\")
        println(io, "\\hline")
        cp = String[]
        for r in rows; push!(cp, @sprintf("%.2f", r.tCP), _fmtval(r.Φstar)); end
        println(io, "Chen--Poss & ", join(cp, " & "), " \\\\")
        for K in 1:Kshow
            cells = String[]
            for r in rows
                if !attempted(r, K)
                    push!(cells, "", "")                                   # K never attempted
                elseif r.ktimedout[K]
                    push!(cells, "T", "T")                                 # time limit hit
                else
                    push!(cells, @sprintf("%.2f", r.ktimes[K]), _fmtval(r.kvals[K]))
                end
            end
            println(io, "\$K=$K\$ & ", join(cells, " & "), " \\\\")
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
    end
    @printf("\n[latex] wrote table (%d seeds, K up to %d) -> %s\n", S, Kshow, path)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    Kmax = 10
    rows = [run_mst_instance(; seed = seed, q = 4, Γ = 2.5, b = 2,
                               Kmax = Kmax, time_limit = 1800.0) for seed in 1:5]
    write_mst_table(rows, Kmax; path = "MST.txt")
end
