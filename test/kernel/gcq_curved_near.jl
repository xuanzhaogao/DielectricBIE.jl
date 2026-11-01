# The curved GCQ row engine (src/kernel/gcq_curved_near.jl) for targets off the panel: tensor_near_row's quadtree on the
# exact map of a `CurvedPanel`. Checked against tensor_near_row on an affine panel and against brute-force Gauss on the
# lens. No SCTL.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using FastGaussQuadrature: gausslegendre, gaussjacobi
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

# Brute-force rows (:dt, :sl) of P for targets X (P's anchor frame) with normals N: nq-point Gauss-Legendre on each fine
# u-segment × nq points on each of 8 equal v-pieces, Gauss-Jacobi for the weight u^pw on the weighted edge segment (the
# cardinal functions divided by u^pw there), the cardinal functions from tensor_rule_interp and plain differences of
# positions (the targets are far enough away for both)
function gcq_brute_rows(P, X, N; nq = 48)
    ru, rv = P.ru, P.rv
    t, w = gausslegendre(nq)
    us = Float64[]; wu = Float64[]; Eus = Matrix{Float64}[]
    for (sa, sb) in ru.segs
        if sa == 0 && ru.pw != 0
            tj, wj = gaussjacobi(nq, 0.0, ru.pw)
            u = sb .* (1 .+ tj) ./ 2
            append!(us, u); append!(wu, (sb / 2)^(ru.pw + 1) .* wj)
            push!(Eus, BI.tensor_rule_interp(ru, u) ./ u .^ ru.pw)
        else
            u = sa .+ (sb - sa) .* (1 .+ t) ./ 2
            append!(us, u); append!(wu, (sb - sa) / 2 .* w)
            push!(Eus, BI.tensor_rule_interp(ru, u))
        end
    end
    Eu = reduce(vcat, Eus)
    vs = reduce(vcat, [(k - 1 .+ (1 .+ t) ./ 2) ./ 8 for k in 1:8]); wv = repeat(w ./ 16, 8)
    Ev = BI.tensor_rule_interp(rv, vs)
    Y = Matrix{NTuple{3, Float64}}(undef, length(us), length(vs)); JW = zeros(length(us), length(vs))
    Threads.@threads for a in eachindex(us)
        for b in eachindex(vs)
            Y[a, b] = BI.curved_lpoint(P, us[a], vs[b])
            JW[a, b] = BI.curved_frame(P, us[a], vs[b])[4] * wu[a] * wv[b]
        end
    end
    return map(eachindex(X)) do i
        Kd = similar(JW); Ks = similar(JW)
        for j in eachindex(JW)
            d = X[i] .- Y[j]; r = norm(d)
            Kd[j] = JW[j] * dot(N[i], d) / (4π * r^3); Ks[j] = JW[j] / (4π * r)
        end
        (vec(permutedims(Eu' * Kd * Ev)), vec(permutedims(Eu' * Ks * Ev)))
    end
end

@testset "gcq_curved_row: off-panel targets" begin
    # (a) an affine CurvedPanel on a GCQ edge TensorPanel of a box: the same rows as tensor_near_row
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 8,
                                      rule = edge_gcq_rule(10.0, 1.0; tol = 1e-8))
    Pt = first(p for p in itf.panels if p.kind == :edge)
    @test Pt.ru.pw != 0
    o, eu, ev, Lu, Lv, nP = Pt.o, Pt.eu, Pt.ev, Pt.Lu, Pt.Lv, Pt.normal
    Pc = BI.CurvedPanel(Pt.anchor, (u, v) -> o .+ (u * Lu) .* eu .+ (v * Lv) .* ev, Pt.ru, Pt.rv; kind = :edge,
                        flip = dot(cross(collect(eu), collect(ev)), collect(nP)) < 0, dmap = (u, v) -> (Lu .* eu, Lv .* ev))
    @test all(n -> isapprox(collect(n), collect(nP); atol = 1e-15), Pc.normals)
    # Dᵀ: 20 targets on the box's face across the panel's edge (u = 0), 1e-7 … 1e-1 from it, alternately at the panel's
    # node rows and between them; the target normal is that face's (outward, -eu)
    vpos = sort([Pt.rv.x; (Pt.rv.x[1:end-1] .+ Pt.rv.x[2:end]) ./ 2])
    nt = .-eu
    kdt(d) = dot(nt, d) / (4π * norm(d)^3)
    worst_dt = 0.0
    for (i, δ) in enumerate(10.0 .^ range(-7, -1, length = 20))
        x = o .+ (vpos[mod1(i, length(vpos))] * Lv) .* ev .- δ .* nP
        rt = tensor_near_row(kdt, Pt, x)
        rc = BI.gcq_curved_row(:dt, Pc, x; n = nt)
        worst_dt = max(worst_dt, maximum(abs, rc .- rt) / maximum(abs, rt))
    end
    @test worst_dt <= 1e-13                                             # 9.3e-15 measured
    # single layer: 10 targets 1e-6 … 1e-1 above the plane, over points near the edge (one inside the weighted edge
    # segment [0, 2^-K]), in the middle and at the far end, at node rows and between them
    ksl(d) = 1 / (4π * norm(d))
    upos = [5e-10, Pt.ru.x[1], (Pt.ru.x[1] + Pt.ru.x[2]) / 2, Pt.ru.x[4], (Pt.ru.x[9] + Pt.ru.x[10]) / 2, Pt.ru.x[end]]
    @test upos[1] < Pt.ru.segs[1][2]
    worst_sl = 0.0
    for (i, h) in enumerate(10.0 .^ range(-6, -1, length = 10))
        x = o .+ (upos[mod1(i, length(upos))] * Lu) .* eu .+ (vpos[mod1(3i, length(vpos))] * Lv) .* ev .+ h .* nP
        rt = tensor_near_row(ksl, Pt, x)
        rc = BI.gcq_curved_row(:sl, Pc, x)
        worst_sl = max(worst_sl, maximum(abs, rc .- rt) / maximum(abs, rt))
    end
    @test worst_sl <= 1e-13                                             # 2.1e-15

    # (b) the lens edge panel (curved, weighted edge rule) against brute force: 4 nodes of neighbouring panels (the
    # same patch, the patch beside, the cap's top GL panel, the other cap across the rim) and 4 points off the surface,
    # all at least 0.05 × P's diameter from P
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    ps = m.interface.panels
    kP = findfirst(p -> p.kind == :edge, ps)
    P = ps[kP]
    @test P.ru.pw != 0
    diam = BI._curved_diam(P, 0.0, 1.0, 0.0, 1.0)
    offset(T, j) = (T.anchor .- P.anchor) .+ T.lpoints[j]
    X = NTuple{3, Float64}[]; N = NTuple{3, Float64}[]
    for k in (2, 8, 12, 13)
        T = ps[k]
        ds = [BI._curved_closest(P, offset(T, j))[3] for j in eachindex(T.lpoints)]
        @test minimum(ds) < 0.05 * diam                                 # a neighbour of P
        j = argmin(j -> ds[j] >= 0.05 * diam ? ds[j] : Inf, eachindex(ds))
        push!(X, offset(T, j)); push!(N, T.normals[j])
    end
    for (u, v, s) in ((0.02, 0.5, 0.05), (0.5, 0.05, -0.05), (0.3, 0.8, 0.1), (0.9, 0.3, -0.3))
        _, _, nn, _ = BI.curved_frame(P, u, v)
        push!(X, BI.curved_lpoint(P, u, v) .+ (s * diam) .* nn); push!(N, nn)
    end
    ref = gcq_brute_rows(P, X, N)
    worst = [0.0, 0.0]
    for i in eachindex(X)
        rd = BI.gcq_curved_row(:dt, P, X[i]; n = N[i]); rs = BI.gcq_curved_row(:sl, P, X[i])
        worst[1] = max(worst[1], maximum(abs, rd .- ref[i][1]) / maximum(abs, ref[i][1]))
        worst[2] = max(worst[2], maximum(abs, rs .- ref[i][2]) / maximum(abs, ref[i][2]))
    end
    @test worst[1] <= 1e-12                                             # 8.8e-15 measured (Dᵀ)
    @test worst[2] <= 1e-12                                             # 5.0e-15 (single layer)

    # (c) across the rim: the nodes of the other cap's edge panel Q nearest P (as in test/ext/sctl_curved.jl), offsets
    # from the anchors' difference; all rows finite, and the 4 targets farthest from the rim against brute force
    mid(p, u, v) = p.anchor .+ BI.curved_lpoint(p, u, v)
    kQ = argmin(k -> m.panel_patch[k] in 6:10 && ps[k].kind == :edge ?
                     norm(collect(mid(ps[k], 0.0, 0.5) .- mid(P, 0.0, 0.5))) : Inf, eachindex(ps))
    Q = ps[kQ]
    XQ = [offset(Q, j) for j in eachindex(Q.lpoints)]
    rows = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(XQ))
    Threads.@threads for j in eachindex(XQ)
        rows[j] = (BI.gcq_curved_row(:dt, P, XQ[j]; n = Q.normals[j]), BI.gcq_curved_row(:sl, P, XQ[j]))
    end
    @test all(r -> all(isfinite, r[1]) && all(isfinite, r[2]), rows)
    far = sortperm([BI._curved_closest(P, x)[3] for x in XQ])[end-3:end]
    refQ = gcq_brute_rows(P, XQ[far], Q.normals[far]; nq = 64)
    worst_rim = [0.0, 0.0]
    for (i, j) in enumerate(far), s in 1:2
        worst_rim[s] = max(worst_rim[s], abs(sum(rows[j][s]) - sum(refQ[i][s])) / abs(sum(refQ[i][s])))
    end
    @test worst_rim[1] <= 1e-10                                         # 4.5e-16 measured (Dᵀ)
    @test worst_rim[2] <= 1e-10                                         # 5.3e-16 (single layer)

    # (d) Review Focus 2: a target on P (here a node) passed without its parameters throws, never a wrong row
    @test_throws ArgumentError BI.gcq_curved_row(:sl, P, P.lpoints[5])
end
