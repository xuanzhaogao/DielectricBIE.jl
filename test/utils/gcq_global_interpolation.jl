# The global interpolant of a weighted GCQ rule (plan 2026-11-05-global-gcq-local-duffy, Tasks 2-3): the retained SVD
# data of edge_gcq_rule, the in-place evaluator global_interp! and its accuracy against a BigFloat evaluation of the same
# retained subspace and nodes.
using Test, LinearAlgebra, Random
using DielectricBIE
import DielectricBIE as DB

# The construction of edge_gcq_rule as it was before the global data were retained (src/utils/edge_gcq.jl at 3a9d634),
# returning its nodes, weights, P and the retained factors: the retained data must come from the same construction.
function gg_reference_construction(dpows; tol, nsmooth = 20, qmax = nothing, grid = :graded, floor = 0.0)
    p1 = minimum(dpows)
    qmax === nothing || (nsmooth = ceil(Int, qmax - p1))
    q = sort(unique(round.(vcat([(p - p1) .+ (0:nsmooth) for p in dpows]..., collect((0:nsmooth) .- p1)); digits = 12)))
    q = q[q .>= -1e-12]
    qmax === nothing || (q = q[q .+ p1 .<= qmax + 1e-12])
    fs = s -> [s^qi for qi in q]
    z0, w0 = DB.gausslegendre(14)
    if grid == :adaptive
        _, _, segbuf, _ = DB.quadgk_segbuf_count(s -> s^p1 .* fs(s), 0.0, 1.0; atol = tol / 10, rtol = tol / 10)
        segs = sort([(sg.a, sg.b) for sg in segbuf])
        xq = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs])
        ωq = reduce(vcat, [(b - a) / 2 .* w0 for (a, b) in segs]) .* xq .^ p1
        xi = xq; ngj = 0
    else
        cs = [c for c in q if abs(c - round(c)) > 1e-9]
        cmin = isempty(cs) ? 1.0 : minimum(cs)
        K = max(20, ceil(Int, log2(10 / tol) / (1 + p1 + cmin)))
        δ = 2.0^(-K)
        segs = vcat([(0.0, δ)], [(2.0^(-k), 2.0^(-k + 1)) for k in K:-1:1])
        ngj = max(24, ceil(Int, maximum(q) / 2) + 4)
        tj, wj = DB.gaussjacobi(ngj, 0.0, p1)
        xq = vcat(δ .* (1 .+ tj) ./ 2, reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs[2:end]]))
        ωq = vcat((δ / 2)^(1 + p1) .* wj, reduce(vcat, [(b - a) / 2 .* w0 for (a, b) in segs[2:end]]) .* xq[ngj+1:end] .^ p1)
        xi = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs])
    end
    A = reduce(vcat, [sqrt(ωj) .* fs(xj)' for (xj, ωj) in zip(xq, ωq)])
    S = svd(A)
    r = sum(S.S .> tol * S.S[1])
    U = S.U[:, 1:r]
    cand = grid == :graded ? collect(ngj+1:length(xq)) : collect(eachindex(xq))
    cand = filter(i -> xq[i] >= floor, cand)
    nodeinds = sort(cand[qr(U[cand, :]', ColumnNorm()).p[1:r]])
    x = xq[nodeinds]
    Is = vec(sum(sqrt.(ωq) .* U, dims = 1))
    W = sqrt.(ωq[nodeinds]) .* (U[nodeinds, :]' \ Is)
    B = S.V[:, 1:r] ./ S.S[1:r]'
    Ui = reduce(vcat, [fs(xj)' for xj in xi]) * B
    Un = (U ./ sqrt.(ωq))[nodeinds, :]
    P = (Ui / Un) * Diagonal(x .^ (-p1))
    return (; x, w = W ./ x .^ p1, P, q, p1, V = S.V[:, 1:r], S = S.S[1:r])
end

const GG_UNIVERSAL = (-0.3, 6.0)          # universal_gcq_rule(pmin; qmax) as used below
gg_udpows(pmin; h = 0.01) = pmin .+ unique(vcat(collect(0:h:1-h/2), collect(0:h/10:10h)))   # as universal_gcq_rule

@testset "GCQ global interpolation: retained data, construction unchanged" begin
    cases = (("dedicated -0.9", [-0.9], (; tol = 1e-8)),
             ("dedicated -0.268308", [-0.268308], (; tol = 1e-12)),
             ("two powers (10, 1)", DB.edge_density_powers(10.0, 1.0), (; tol = 1e-8)),
             ("qmax", DB.edge_density_powers(Inf, 1.0; gmax = 2.0), (; tol = 1e-6, qmax = 12.0)),
             ("adaptive", DB.edge_density_powers(4.0, 1.0; angle = 2π / 3), (; tol = 1e-6, grid = :adaptive)),
             ("universal", gg_udpows(GG_UNIVERSAL[1]), (; tol = 1e-8, qmax = GG_UNIVERSAL[2], floor = 1e-10)))
    for (name, dp, kw) in cases
        er = name == "universal" ? universal_gcq_rule(GG_UNIVERSAL[1]; qmax = GG_UNIVERSAL[2], tol = 1e-8) :
                                   edge_gcq_rule(dp; kw...)
        ref = gg_reference_construction(dp; kw...)
        # the rule itself is unchanged, bit for bit
        @test er.x == ref.x && er.w == ref.w && er.P == ref.P && er.pw == ref.p1
        g = DB.gcq_global_interp(er)
        @test g isa DB.GCQGlobalInterp
        @test g.q == ref.q && g.q .+ g.pw == er.pows && g.pw == er.pw && g.x == er.x
        @test g.V == ref.V && g.S == ref.S                     # the retained factors of the same SVD, not a new one
        @test size(g.V) == (length(ref.q), length(er.x))      # retained rank = number of nodes
        @test g.Hn ≈ [t^qi for t in er.x, qi in g.q] * g.V rtol = 1e-14
        # the tensor rule carries the same data (built from the same P)
        tr = gcq_tensor_rule(er)
        @test DB.gcq_global_interp(tr) === g
    end
    # rules lacking the data are refused explicitly: a GL rule, a copied P (e.g. deserialized), changed nodes or weight
    er = edge_gcq_rule([-0.6]; tol = 1e-8); tr = gcq_tensor_rule(er)
    @test_throws ArgumentError DB.gcq_global_interp(DB.gl_tensor_rule(8))
    @test_throws ArgumentError DB.gcq_global_interp(DB.TensorRule1D(tr.x, tr.w, tr.segs, tr.z0, tr.wz, copy(tr.P), tr.pw))
    @test_throws ArgumentError DB.gcq_global_interp(DB.TensorRule1D(nextfloat.(tr.x), tr.w, tr.segs, tr.z0, tr.wz, tr.P, tr.pw))
    @test_throws ArgumentError DB.gcq_global_interp(DB.TensorRule1D(tr.x, tr.w, tr.segs, tr.z0, tr.wz, tr.P, 0.0))
    # a rule built twice from the same arguments gets its own (identical) data
    er2 = edge_gcq_rule([-0.6]; tol = 1e-8)
    @test DB.gcq_global_interp(er2) !== DB.gcq_global_interp(er) && DB.gcq_global_interp(er2).V == DB.gcq_global_interp(er).V
    # the registry does not keep rules alive: the data of collected rules go at the next registration (meshes build
    # their rules per call, and most never use the global backend)
    gg_mk() = (edge_gcq_rule([-0.37]; tol = 1e-6); nothing)
    gg_mk(); GC.gc(true)
    n0 = length(DB.GCQ_GLOBAL[])
    for _ in 1:20; gg_mk(); end
    GC.gc(true); gg_mk()
    @test length(DB.GCQ_GLOBAL[]) <= n0 + 2
    @test DB.gcq_global_interp(er) isa DB.GCQGlobalInterp          # live rules keep theirs
end

# BigFloat evaluation of the same interpolant (the same q, V and nodes; powers, nodal matrix and solve at `prec` bits)
function gg_bigref(g, ts; prec = 256)
    setprecision(BigFloat, prec) do
        q = big.(g.q); V = big.(g.V); x = big.(g.x); t = big.(ts); p = big(g.pw)
        H = [ti^qi for ti in t, qi in q] * V; Hn = [xi^qi for xi in x, qi in q] * V
        return Float64.(Diagonal(t .^ p) * (H / Hn) * Diagonal(x .^ (-p)))
    end
end
gg_eval(r, ts) = DB.global_interp!(zeros(length(ts), length(r.x)), r, ts, DB.GlobalInterpWorkspace(r))
# physical densities (node values, values at ts): s^p smooth, a constant (in the family as s^p s^-p), s^(p+1) cos s, s^2 e^s
gg_phys(p, s) = (s .^ p .* (1 .+ 0.3 .* s .- 0.2 .* s .^ 2), s .^ 0, s .^ (p + 1) .* cos.(s), s .^ 2 .* exp.(s))

@testset "GCQ global interpolation: evaluator against a BigFloat reference" begin
    rng = MersenneTwister(7)
    rules = [("p=$p tol=$tol", gcq_tensor_rule(edge_gcq_rule([p]; tol)), tol)
             for tol in (1e-4, 1e-8, 1e-12) for p in (-0.9, -0.6, -0.268308, -0.04)]
    push!(rules, ("universal -0.3", gcq_tensor_rule(universal_gcq_rule(GG_UNIVERSAL[1]; qmax = GG_UNIVERSAL[2], tol = 1e-12)), 1e-12))
    for (name, r, tol) in rules
        g = DB.gcq_global_interp(r)
        lo = minimum(r.x)                                   # (a universal rule's nodes stop at its floor)
        bnd = [sb for (_, sb) in r.segs[1:end-1] if sb > lo]
        hs = (1e-4, 1e-6, 1e-8)
        ts = sort(unique(vcat(r.x, filter(t -> t >= lo, vec(r.fx)), exp.(range(log(lo), 0, length = 120)) .* (1 .- 0.01 .* rand(rng, 120)),
                              lo .+ (1 - lo) .* rand(rng, 40), bnd, [b * (1 + s * h) for b in bnd for h in hs for s in (-1, 1)])))
        ts = ts[ts .<= 1]
        E = gg_eval(r, ts)
        E0 = gg_bigref(g, ts); E1 = gg_bigref(g, ts; prec = 128)
        sc = ts .^ r.pw                                     # values relative to the singular factor
        @test maximum(abs.(E1 - E0) ./ sc) < 1e-20          # the reference is converged in its precision
        car = maximum(abs.(E - E0) ./ sc)
        phys = maximum(maximum(abs.(E * σ - E0 * σ) ./ sc) for σ in gg_phys(r.pw, r.x))
        model = maximum(maximum(abs.(E0 * σ - σt) ./ sc) for (σ, σt) in zip(gg_phys(r.pw, r.x), gg_phys(r.pw, ts)))
        nodal = maximum(abs.(gg_eval(r, r.x) - I))
        @info "global interpolant $name" cardinal = car physical = phys model = model nodal
        # the gate: physical densities to rounding (tol 1e-12 rules: a 1e-12 potential needs < 1e-13 here). The universal
        # rule is a diagnostic: its nodal matrix is worse conditioned, and its family stops at qmax = 6 (s^(p+1) cos s and
        # s^2 e^s are outside it), so its model error is the family's, not the rule tolerance's
        uni = occursin("universal", name)
        @test phys < (uni ? 1e-11 : 1e-13)
        @test car < (uni ? 10 : 1) * eps() / tol && nodal < (uni ? 100 : 1) * eps() / tol
        uni || @test model < 10 * tol                       # the interpolant itself (a diagnostic of the rule)
        # continuous at the old fine boundaries: the reference converges to its value at the boundary linearly in h,
        # and the Float64 values of physical densities do not jump
        ib = [findfirst(==(b), ts) for b in bnd]
        for (k, b) in enumerate(bnd), h in hs, s in (-1, 1)
            i = findfirst(==(b * (1 + s * h)), ts)
            i === nothing && continue
            @test maximum(abs.(E0[i, :] - E0[ib[k], :])) <= 1e4 * h * maximum(abs.(E0[ib[k], :])) + 1e-30
        end
        # no jump: across a boundary, the Float64 difference of a physical density is the reference's (a difference over
        # 2e-8 b is the density's own variation, not a jump measurement); the piecewise interpolant, for comparison
        jp = 0.0; jpw = 0.0
        Ep = DB.tensor_rule_interp(r, ts)
        for b in bnd, σ in gg_phys(r.pw, r.x)
            ip = findfirst(==(b * (1 + 1e-8)), ts); im = findfirst(==(b * (1 - 1e-8)), ts)
            (ip === nothing || im === nothing) && continue
            dref = dot(E0[ip, :] - E0[im, :], σ)
            jp = max(jp, abs(dot(E[ip, :] - E[im, :], σ) - dref) / b^r.pw)
            jpw = max(jpw, abs(dot(Ep[ip, :] - Ep[im, :], σ) - dref) / b^r.pw)
        end
        @info "jump at the fine boundaries (physical) $name" global_ = jp piecewise = jpw
        @test jp < (uni ? 1e-11 : 1e-13)
        # at the fine nodes the piecewise interpolant is P, the same interpolant through the SVD's nodal factor: the two
        # agree to the cardinal rounding there; off the fine nodes they are different interpolants (measured, not asserted)
        tf = filter(t -> t >= lo, vec(r.fx))
        # (the universal rule, worse conditioned, 10x looser, as above: on GitHub's runners 4.5e-3 x eps / tol and
        # 1.08e-11, under 2e-3 and 1e-11 on genoa)
        @test maximum(abs.(gg_eval(r, tf) - DB.tensor_rule_interp(r, tf)) ./ (tf .^ r.pw)) < (uni ? 100 : 10) * eps() / tol
        @test maximum(maximum(abs.((gg_eval(r, tf) - DB.tensor_rule_interp(r, tf)) * σ) ./ (tf .^ r.pw)) for σ in gg_phys(r.pw, r.x)) < (uni ? 1e-10 : 1e-12)
        @info "piecewise vs global off the fine nodes $name" cardinal = maximum(abs.(DB.tensor_rule_interp(r, ts) - E) ./ sc)
    end
end

@testset "GCQ global interpolation: interface" begin
    r = gcq_tensor_rule(edge_gcq_rule([-0.6]; tol = 1e-8)); n = length(r.x)
    ws = DB.GlobalInterpWorkspace(r, 16)
    # chunks of the workspace's size give the values of one pass
    ts = sort(rand(MersenneTwister(3), 50)) .* 0.999 .+ 1e-6
    @test DB.global_interp!(zeros(50, n), r, ts, ws) ≈ DB.global_interp!(zeros(50, n), r, ts, DB.GlobalInterpWorkspace(r, 64)) rtol = 1e-13
    # the endpoint s = 0 of a negative weight power is refused (the density is infinite there), and points off [0, 1]
    for bad in (0.0, -1e-300, 1.0 + 1e-15, NaN)
        @test_throws ArgumentError DB.global_interp!(zeros(1, n), r, [bad], ws)
    end
    @test_throws DimensionMismatch DB.global_interp!(zeros(2, n), r, [0.5], ws)
    @test_throws ArgumentError DB.global_interp!(zeros(1, n), r, [0.5], DB.GlobalInterpWorkspace(gcq_tensor_rule(edge_gcq_rule([-0.6]; tol = 1e-8))))
    # Gauss-Legendre: ordinary polynomial interpolation, the identity at the nodes, no allocation
    rg = DB.gl_tensor_rule(10); tg = collect(range(0, 1, length = 33))
    Eg = zeros(33, 10)
    @test DB.global_interp!(Eg, rg, tg, nothing) ≈ DB.tensor_rule_interp(rg, tg) atol = 1e-14
    @test DB.global_interp!(zeros(10, 10), rg, rg.x, nothing) == Matrix(1.0I, 10, 10)
    DB.global_interp!(Eg, rg, tg, nothing)
    @test (@allocated DB.global_interp!(Eg, rg, tg, nothing)) == 0
    # GCQ: no allocation with a workspace
    E = zeros(50, n); DB.global_interp!(E, r, ts, ws)
    @test (@allocated DB.global_interp!(E, r, ts, ws)) == 0
    # read-only sharing: threads with their own workspaces give the serial values
    ref = DB.global_interp!(zeros(50, n), r, ts, ws)
    out = [zeros(50, n) for _ in 1:8]
    Threads.@threads for i in 1:8
        DB.global_interp!(out[i], r, ts, DB.GlobalInterpWorkspace(r))
    end
    @test all(==(ref), out)
    # every GCQ rule of a mesh (edge panels GCQ × GL in either orientation, corner panels GCQ × GCQ) carries its data
    rule = edge_gcq_rule(DB.edge_density_powers(Inf, 1.0; gmax = 2.0); tol = 1e-6, qmax = 12.0)
    cube = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, Inf, 1.0; nlev = 1, n_quad = 6, rule)
    for P in cube.panels, rr in (P.ru, P.rv)
        DB.is_gcq(rr) || continue
        @test DB.gcq_global_interp(rr) isa DB.GCQGlobalInterp
    end
    @test any(P -> P.kind == :corner, cube.panels) && any(P -> DB.is_gcq(P.rv) && !DB.is_gcq(P.ru), cube.panels) ||
          any(P -> DB.is_gcq(P.ru) && !DB.is_gcq(P.rv), cube.panels)
end
