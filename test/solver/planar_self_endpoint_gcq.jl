using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Random
using Test
include(joinpath(@__DIR__, "planar_test_helpers.jl"))       # ep_rule, ep_panel, pl_*, sp_self_ref

# Single-layer self rows of flat GCQ panels at their own nodes: the node's fine cell C is excluded from the walk, the
# Duffy region R ⊂ C (metric-balanced, around the node) is integrated by duffy_rule and C \ R (at most two strips) by
# the walk with drop = false. With endpoint_rule = :gcq the endpoint clusters of the walk (outside C) use the scaled
# GCQ rules, in every GCQ direction (GCQ × GL and GCQ × GCQ).

const SP_ROT = let ax = normalize([0.3, -0.81, 0.47]), K = [0 -ax[3] ax[2]; ax[3] 0 -ax[1]; -ax[2] ax[1] 0]
    I + sin(0.7310281) * K + (1 - cos(0.7310281)) * K^2
end
const SP_ANCHOR = (37.3, -51.9, 93.1)

# fixture panels on the skew bilinear quadrilateral u A + v B + u v C (rotated, moved): GCQ × GL in both orientations,
# and a GCQ × GCQ corner with unequal rules (exponents, tolerances and fine-segment counts) in u and v
function sp_panels(; scale = 1.0, shape = ([1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.3, -0.25, 0.0]))
    _, r1 = ep_rule(-0.6, 1e-8); _, r2 = ep_rule(-0.268308, 1e-12); rgl = DB.gl_tensor_rule(10)
    A, B, C = (scale .* SP_ROT * x for x in shape)
    return (edge_u = ep_panel(SP_ANCHOR, A, B, C, r1, rgl), edge_v = ep_panel(SP_ANCHOR, B, A, C, rgl, r1),
            corner = ep_panel(SP_ANCHOR, A, B, C, r1, r2))
end

# owned nodes of P: those nearest the edge(s), interior and far ones, as (i_u, j_v)
function sp_nodes(P)
    nu, nv = length(P.ru.x), length(P.rv.x)
    pick(n) = unique([1, 2, n ÷ 2, n])
    return [(i, j) for i in pick(nu) for j in pick(nv)]
end

# the owned targets of P at the nodes `ij`, with their cells (the Duffy path)
function sp_owned(P, ij)
    u = [P.ru.x[i] for (i, _) in ij]; v = [P.rv.x[j] for (_, j) in ij]
    cells = [(DB._pn_cell(P.ru, u[t], 1, "u"), DB._pn_cell(P.rv, v[t], 1, "v")) for t in eachindex(ij)]
    return DB._FlatTargets(u, v, zeros(3, length(ij)), nothing, cells)
end

# max over the targets of max_j |R_j - ref_j| / Σ_j |ref_j|, and the rounding floor 100 ε max_j ∫|kernel φ_j| / Σ_j |ref_j|
sp_card(R, ref) = maximum(maximum(abs.(R[t, :] .- ref[t][1])) / sum(abs, ref[t][1]) for t in axes(R, 1))
sp_floor(ref) = maximum(100eps() * maximum(r[2]) / sum(abs, r[1]) for r in ref)

@testset "self rows: owned nodes take the Duffy path (baseline, :moment)" begin
    for (name, P) in pairs(sp_panels())
        ij = sp_nodes(P); T = sp_owned(P, ij); g = DB._flat_geom(P)
        R = DB._flat_rows(P, g, T, PlanarProductNearQuad(self = :cell, endpoint_rule = :moment))
        # the public path (planar_slp_corrections plus the node rule it subtracts) gives the same rows
        one = DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0])
        X = reduce(hcat, [collect(y) for y in P.points]); nv = length(P.rv.x)
        idx = [(i - 1) * nv + j for (i, j) in ij]
        Rp = Matrix(DB.planar_slp_corrections(one; q = PlanarProductNearQuad(self = :cell, endpoint_rule = :moment)))[idx, :] .+
             DB._pn_slp_node_block(P, X)[idx, :]
        @test maximum(maximum(abs.(Rp[t, :] .- R[t, :])) / sum(abs, R[t, :]) for t in eachindex(idx)) < 1e-13
        # and they are the self rows of the parent interpolant: against the independent reference (owned cell by a
        # graded Duffy fan, the rest by fine-grid boxes), cardinal rows to rounding
        ref = [sp_self_ref(P, T.ur[t], T.vr[t], T.cell[t]...) for t in eachindex(T.ur)]
        @test sp_card(R, ref) < 1e-13 + sp_floor(ref)
        @info "baseline self rows (:moment)" panel = name nodes = length(idx) card = sp_card(R, ref) floor = sp_floor(ref)
        # the public path takes the Duffy route: with the depth cap at 20 halvings it still reaches the reference, and
        # without Duffy (the node's own boxes halved to the cap) it does not
        pub(q) = Matrix(DB.planar_slp_corrections(one; q))[idx, :] .+ DB._pn_slp_node_block(P, X)[idx, :]
        @test sp_card(pub(PlanarProductNearQuad(max_depth = 20, self = :cell, endpoint_rule = :moment)), ref) < 1e-13 + sp_floor(ref)
        @test sp_card(pub(PlanarProductNearQuad(max_depth = 20, duffy = false, self = :cell, endpoint_rule = :moment)), ref) > 1e-9
    end
end

sp_sub(T, ix) = DB._FlatTargets(T.ur[ix], T.vr[ix], T.w[:, ix], nothing, T.cell[ix])
sp_root(r) = length(r.segs) == 1 ? (1, 0.0, 1.0) : (-length(r.segs), 0.0, 1.0)
sp_ws(P) = map(r -> (zeros(DB.TENSOR_NEAR_M), zeros(DB.TENSOR_NEAR_M), zeros(length(r.z0), DB.TENSOR_NEAR_M),
                     zeros(size(r.P, 2), DB.TENSOR_NEAR_M)), (P.ru, P.rv))
sp_vec(R) = vec(permutedims(R, (2, 3, 1)))
sp_close(R, (ref, mag)) = maximum(abs.(R .- ref)) <= 1e-13 * sum(abs, ref) + 100eps() * maximum(mag)

# the skew quadrilateral, and rectangles 3 × 1 and 1 × 3 (the Duffy region's shorter side in u or in v)
const SP_SHAPES = (skew = ([1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.3, -0.25, 0.0]), wide = ([3.0, 0.0, 0.0], [0.0, 1.0, 0.0], zeros(3)),
                   tall = ([1.0, 0.0, 0.0], [0.0, 3.0, 0.0], zeros(3)))

@testset "self rows: the cell, strip and Duffy partition" begin
    seen = Set{Symbol}()
    for (sname, shape) in pairs(SP_SHAPES), (name, P) in pairs(sp_panels(; shape))
        g = DB._flat_geom(P); T = sp_owned(P, sp_nodes(P))
        J(u, v) = norm(cross(collect(g.a .+ v .* g.c), collect(g.b .+ u .* g.c)))
        parea(q) = (z = (1 / sqrt(3)) .* (-1, 1);                # 2-point Gauss-Legendre (J is linear on a flat panel)
                    sum(J(q[1] + (q[2] - q[1]) * (1 + s) / 2, q[3] + (q[4] - q[3]) * (1 + t) / 2) for s in z, t in z) *
                    (q[2] - q[1]) * (q[4] - q[3]) / 4)
        for t in eachindex(T.ur)
            k, l = T.cell[t]; u0, v0 = T.ur[t], T.vr[t]
            (sa, sb), (ta, tb) = P.ru.segs[k], P.rv.segs[l]
            D, strips, G = DB._flat_self_regions(g, P.ru, P.rv, k, l, u0, v0)
            @test sa <= D[1] < u0 < D[2] <= sb && ta <= D[3] < v0 < D[4] <= tb        # R ⊂ C and holds the node
            @test all(s[1] == k && s[4] == l for s in strips)
            pieces = vcat([D], [(s[2], s[3], s[5], s[6]) for s in strips if s[2] < s[3] && s[5] < s[6]])
            @test sum(q -> (q[2] - q[1]) * (q[4] - q[3]), pieces) ≈ (sb - sa) * (tb - ta) rtol = 1e-14
            ov(p, q) = max(0.0, min(p[2], q[2]) - max(p[1], q[1])) * max(0.0, min(p[4], q[4]) - max(p[3], q[3]))
            @test all(ov(pieces[i], pieces[j]) == 0 for i in eachindex(pieces) for j in i + 1:length(pieces))
            @test sum(parea, pieces) ≈ parea((sa, sb, ta, tb)) rtol = 1e-14
            # the shorter side (physically, at the node) whole, the other as long unless clipped
            ushort = (D[1], D[2]) == (sa, sb)
            push!(seen, ushort ? :u_shorter : :v_shorter)
            @test ushort || (D[3], D[4]) == (ta, tb)
            lo, hi, a_, b_ = ushort ? (D[3], D[4], ta, tb) : (D[1], D[2], sa, sb)
            if a_ < lo && hi < b_
                @test ushort ? (D[4] - D[3])^2 * G[3] ≈ (sb - sa)^2 * G[1] : (D[2] - D[1])^2 * G[1] ≈ (tb - ta)^2 * G[3]
            else
                push!(seen, :clipped)
            end
        end
    end
    @test seen == Set([:u_shorter, :v_shorter, :clipped])
end

@testset "self rows: the walk leaves the owned cell out, the strips and R once each" begin
    for (name, P) in pairs(sp_panels()), rule in (:moment,)
        g = DB._flat_geom(P); tu, tv = DB._tensor_near_tables(P.ru), DB._tensor_near_tables(P.rv)
        q = PlanarProductNearQuad(endpoint_rule = rule, self = :cell); ex = DB._flat_cluster_rules(P, tu, tv, rule)
        ws = sp_ws(P); nu, nv = size(P.ru.P, 2), size(P.rv.P, 2)
        pairs_ = [(k, sa, sb, l, ta, tb) for (k, (sa, sb)) in enumerate(P.ru.segs) for (l, (ta, tb)) in enumerate(P.rv.segs)]
        T = sp_owned(P, sp_nodes(P))
        for t in eachindex(T.ur)
            Tt = sp_sub(T, [t]); k, l = T.cell[t]; u0, v0 = T.ur[t], T.vr[t]
            # the walk with drop: Ω \ C (a prefix holding C is peeled, never accepted; no leaf inside C)
            R = zeros(nu, 1, nv)
            DB._flat_walk!(R, P, g, tu, tv, ex, ws, Tt, (sp_root(P.ru)..., sp_root(P.rv)..., 0), [1], true, q)
            F, Fa = pl_fine(P); pl_boxes!(F, Fa, P, g, u0, v0, zeros(3), nothing, filter(b -> (b[1], b[4]) != (k, l), pairs_))
            @test sp_close(sp_vec(R), pl_row(P, F, Fa))
            # the strips C \ R by the walk without drop, and the Duffy region R
            D, strips, _ = DB._flat_self_regions(g, P.ru, P.rv, k, l, u0, v0)
            live = [s for s in strips if s[2] < s[3] && s[5] < s[6]]
            Rs = zeros(nu, 1, nv)
            for s in live
                DB._flat_walk!(Rs, P, g, tu, tv, ex, ws, Tt, (s..., 0), [1], false, q)
            end
            F, Fa = pl_fine(P); pl_boxes!(F, Fa, P, g, u0, v0, zeros(3), nothing, live)
            @test sp_close(sp_vec(Rs), pl_row(P, F, Fa))
            Rself = zeros(nu, 1, nv)
            DB._flat_self!(Rself, P, g, tu, tv, ex, ws, Tt, 1, q)
            F, Fa = pl_fine(P); pl_singular!(F, Fa, P, g, u0, v0, k, l, D)
            @test sp_close(sp_vec(Rself) .- sp_vec(Rs), pl_row(P, F, Fa))
        end
    end
end

# The owned rows of _flat_rows under `rule` against their parts assembled by hand with the scaled-GCQ cluster rules:
# the walk over Ω \ C (drop) plus _flat_self! (strips and Duffy). Returns (max relative mismatch, max relative change
# from :moment).
function sp_gcq_routing(P)
    g = DB._flat_geom(P); tu, tv = DB._tensor_near_tables(P.ru), DB._tensor_near_tables(P.rv)
    qg = PlanarProductNearQuad(endpoint_rule = :gcq, self = :cell)
    exg = DB._flat_cluster_rules(P, tu, tv, :gcq)
    ws = sp_ws(P); nu, nv = size(P.ru.P, 2), size(P.rv.P, 2)
    T = sp_owned(P, sp_nodes(P))
    Rg = DB._flat_rows(P, g, T, qg); Rm = DB._flat_rows(P, g, T, PlanarProductNearQuad(self = :cell, endpoint_rule = :moment))
    mis = 0.0
    for t in eachindex(T.ur)
        Tt = sp_sub(T, [t]); R = zeros(nu, 1, nv)
        DB._flat_walk!(R, P, g, tu, tv, exg, ws, Tt, (sp_root(P.ru)..., sp_root(P.rv)..., 0), [1], true, qg)
        DB._flat_self!(R, P, g, tu, tv, exg, ws, Tt, 1, qg)
        mis = max(mis, maximum(abs.(sp_vec(R) .- Rg[t, :])) / sum(abs, Rm[t, :]))
    end
    return mis, maximum(maximum(abs.(Rg[t, :] .- Rm[t, :])) / sum(abs, Rm[t, :]) for t in axes(Rm, 1))
end

@testset "self rows under :gcq: the walk outside the owned cell takes the scaled GCQ rules (GCQ × GL)" begin
    for name in (:edge_u, :edge_v)
        P = sp_panels()[name]
        mis, change = sp_gcq_routing(P)
        @test mis < 1e-14                     # the owned rows are the :gcq walk outside C plus strips and Duffy
        @test change > 1e-12                  # and the scaled GCQ rules changed them (not the moment rules)
    end
end

@testset "rows do not depend on the targets walked with them" begin
    # owned nodes (Duffy rows) and off-surface targets in one walk, against each row computed alone: a box holding one
    # node's cell peels its cluster for that node only, so every target's leaves are its own (with :gcq the leaves'
    # quadrature error would otherwise move a row by up to ~1e-6 with its batch)
    for (name, P) in pairs(sp_panels()), rule in (:moment, :gcq)
        g = DB._flat_geom(P); q = PlanarProductNearQuad(endpoint_rule = rule, self = :cell)
        To = sp_owned(P, sp_nodes(P)); m = length(To.ur); n = normalize(collect(P.normals[1]))
        T = DB._FlatTargets(vcat(To.ur, [0.0, 0.5, 0.2]), vcat(To.vr, [0.4, 0.0, 0.7]),
                            hcat(To.w, 1e-6 .* n, 1e-3 .* n, 0.2 .* n), nothing, vcat(To.cell, fill((0, 0), 3)))
        R = DB._flat_rows(P, g, T, q)
        for t in eachindex(T.ur)
            r1 = DB._flat_rows(P, g, sp_sub(T, [t]), q)[1, :]
            @test maximum(abs.(R[t, :] .- r1)) <= 1e-14 * sum(abs, r1)
        end
    end
end

# The leaf (ku, ua, ub) × (kv, va, vb) of _flat_leaf! at an off-surface target against the product of its two rules,
# built here: a cluster [0, s] of a GCQ direction under :gcq by the scaled GCQ map H = _flat_endpoint_gcq_map(r, s)
# (H_u K J H_vᵀ), any other side by n-point Gauss-Legendre with the segment's basis (pl_basis)
function sp_leaf_check(P, rule, ku, ua, ub, kv, va, vb, x)
    g = DB._flat_geom(P); tu, tv = DB._tensor_near_tables(P.ru), DB._tensor_near_tables(P.rv)
    ex = DB._flat_cluster_rules(P, tu, tv, rule); ws = sp_ws(P); nu, nv = size(P.ru.P, 2), size(P.rv.P, 2)
    T = DB._FlatTargets([0.0], [0.0], reshape(collect(x .- g.o), 3, 1), nothing, [(0, 0)])
    R = zeros(nu, 1, nv)
    DB._flat_leaf!(R, P, g, tu, tv, ex, ws, T, ku, ua, ub, kv, va, vb, [1])
    side(r, k, a, b) = k < 0 ? DB._flat_endpoint_gcq_map(r, b) :
        (z = DB.gausslegendre(30); y = a .+ (b - a) .* (1 .+ z[1]) ./ 2;
         (y, (DB._tensor_near_tables(r).pk[k] * pl_basis(r, k, y)) .* ((b - a) / 2 .* z[2])'))
    (yu, Hu), (yv, Hv) = side(P.ru, ku, ua, ub), side(P.rv, kv, va, vb)
    K = [norm(cross(collect(g.a .+ v .* g.c), collect(g.b .+ u .* g.c))) /
         (4π * norm(collect(x .- g.o) .- DB._flat_step(g, 0.0, 0.0, u, v))) for u in yu, v in yv]
    return maximum(abs.(sp_vec(R) .- vec(transpose(Hu * K * Hv')))) / sum(abs, Hu * abs.(K) * abs.(Hv)')
end

@testset "the four product leaf types under :gcq" begin
    P = sp_panels()[:corner]; Pe = sp_panels()[:edge_u]; Pv = sp_panels()[:edge_v]
    su(k) = P.ru.segs[k][2]; sv(l) = P.rv.segs[l][2]
    x = collect(DB._flat_geom(P).o) .+ 3.0 .* normalize(collect(P.normals[1]))      # far: every leaf rule is exact to ~1e-15
    ku = length(P.ru.segs) - 2; kv = length(P.rv.segs) - 3                          # clusters [0, 1/4] × [0, 1/8]
    @test sp_leaf_check(P, :gcq, -ku, 0.0, su(ku), -kv, 0.0, sv(kv), x) < 1e-14                   # scaled GCQ × scaled GCQ
    @test sp_leaf_check(P, :gcq, -ku, 0.0, su(ku), kv + 1, sv(kv), sv(kv + 1), x) < 1e-14         # scaled GCQ × segment
    @test sp_leaf_check(P, :gcq, ku + 1, su(ku), su(ku + 1), -kv, 0.0, sv(kv), x) < 1e-14         # segment × scaled GCQ
    @test sp_leaf_check(Pe, :gcq, -ku, 0.0, Pe.ru.segs[ku][2], 1, 0.25, 0.5, x) < 1e-14           # scaled GCQ × GL part
    @test sp_leaf_check(Pv, :gcq, 1, 0.25, 0.5, -ku, 0.0, Pv.rv.segs[ku][2], x) < 1e-14           # GL part × scaled GCQ
end

@testset "GCQ × GCQ: u/v exchange and asymmetric corners" begin
    # the corner with its rules exchanged and its map transposed is the same panel: rows equal after reordering the nodes,
    # owned (self) and off-surface, under both rules
    _, r1 = ep_rule(-0.6, 1e-8); _, r2 = ep_rule(-0.268308, 1e-12)
    A, B, C = (SP_ROT * x for x in SP_SHAPES.skew)
    P = ep_panel(SP_ANCHOR, A, B, C, r1, r2); Q = ep_panel(SP_ANCHOR, B, A, C, r2, r1)
    nu, nv = length(r1.x), length(r2.x)
    perm = [(j - 1) * nu + i for i in 1:nu for j in 1:nv]
    ij = sp_nodes(P); TP = sp_owned(P, ij); TQ = sp_owned(Q, [(j, i) for (i, j) in ij])
    n = normalize(cross(A, B))
    off = DB._FlatTargets([0.0, 0.0, 0.3], [0.0, 0.4, 0.0], hcat(1e-7 .* n, 1e-4 .* n, 0.05 .* n), nothing, fill((0, 0), 3))
    offQ = DB._FlatTargets(off.vr, off.ur, off.w, nothing, off.cell)
    for rule in (:moment, :gcq)
        q = PlanarProductNearQuad(endpoint_rule = rule, self = :cell)
        for (Tp, Tq) in ((TP, TQ), (off, offQ))
            Rp = DB._flat_rows(P, DB._flat_geom(P), Tp, q); Rq = DB._flat_rows(Q, DB._flat_geom(Q), Tq, q)
            @test maximum(maximum(abs.(Rq[t, perm] .- Rp[t, :])) / sum(abs, Rp[t, :]) for t in axes(Rp, 1)) < 1e-13
        end
    end
end

@testset "the walk under :gcq: the owned cell left out, the strips once (physical density)" begin
    for (name, P) in pairs(sp_panels())
        g = DB._flat_geom(P); tu, tv = DB._tensor_near_tables(P.ru), DB._tensor_near_tables(P.rv)
        q = PlanarProductNearQuad(endpoint_rule = :gcq, self = :cell); ex = DB._flat_cluster_rules(P, tu, tv, :gcq)
        ws = sp_ws(P); nu, nv = size(P.ru.P, 2), size(P.rv.P, 2)
        tol = 1e-8                            # the coarser of the fixture rules' tolerances
        σ = [(DB.is_gcq(P.ru) ? u^-0.6 : 1.0) * (DB.is_gcq(P.rv) ? v^-0.268308 : 1.0) * (1 + u * v) for u in P.ru.x for v in P.rv.x]
        pairs_ = [(k, sa, sb, l, ta, tb) for (k, (sa, sb)) in enumerate(P.ru.segs) for (l, (ta, tb)) in enumerate(P.rv.segs)]
        T = sp_owned(P, sp_nodes(P))
        for t in eachindex(T.ur)
            Tt = sp_sub(T, [t]); k, l = T.cell[t]; u0, v0 = T.ur[t], T.vr[t]
            R = zeros(nu, 1, nv)
            DB._flat_walk!(R, P, g, tu, tv, ex, ws, Tt, (sp_root(P.ru)..., sp_root(P.rv)..., 0), [1], true, q)
            F, Fa = pl_fine(P); pl_boxes!(F, Fa, P, g, u0, v0, zeros(3), nothing, filter(b -> (b[1], b[4]) != (k, l), pairs_))
            ref, mag = pl_row(P, F, Fa)
            @test abs(dot(sp_vec(R) .- ref, σ)) <= 20tol * dot(abs.(ref), abs.(σ)) + 100eps() * dot(mag, abs.(σ))
        end
    end
end

@testset "self and corner rows against the independent reference: error budgets" begin
    # Budgets. :moment: rounding (1e-13 + the floor 100 ε ∫|kernel φ|). :gcq: densities of the GCQ family (s^p, and a
    # mixed-sign one) within 20 tol; cardinal rows and random densities within the panel's own far-field node-rule error
    # in the same metric (pl_far_error) — the scaled GCQ rule integrates its family, not every density of the span.
    rgl = DB.gl_tensor_rule(10)
    A, B, C = (SP_ROT * x for x in SP_SHAPES.skew)
    cases = [(ep_rule(p, tol)[2], rgl, tol) for (p, tol) in ((-0.9, 1e-12), (-0.6, 1e-12), (-0.268308, 1e-12), (-0.04, 1e-12),
                                                              (-0.268308, 1e-4), (-0.268308, 1e-8))]
    append!(cases, [(ep_rule(pu, tu)[2], ep_rule(pv, tv)[2], max(tu, tv))
                    for (pu, tu, pv, tv) in ((-0.6, 1e-8, -0.268308, 1e-12), (-0.9, 1e-12, -0.04, 1e-4), (-0.04, 1e-12, -0.9, 1e-8))])
    worst = Dict{Tuple{Symbol, Symbol}, Float64}()
    for (ru, rv, tol) in cases
        P = ep_panel(SP_ANCHOR, A, B, C, ru, rv); Q = ep_panel(SP_ANCHOR, B, A, C, rv, ru)        # Q: u and v exchanged
        g = DB._flat_geom(P); ij = sp_nodes(P); T = sp_owned(P, ij); TQ = sp_owned(Q, [(j, i) for (i, j) in ij])
        nu, nv = length(ru.x), length(rv.x); perm = [(j - 1) * nu + i for i in 1:nu for j in 1:nv]
        # owned nodes (self rows) and, for corners, off-surface targets over the corner and its edges
        n = normalize(collect(P.normals[1])); m = length(T.ur)
        refs = Vector{Any}(undef, m)
        Threads.@threads for t in 1:m
            refs[t] = sp_self_ref(P, T.ur[t], T.vr[t], T.cell[t]...)
        end
        rows = Dict(rule => DB._flat_rows(P, g, T, PlanarProductNearQuad(endpoint_rule = rule, self = :cell)) for rule in (:moment, :gcq))
        if DB.is_gcq(rv)
            To = DB._FlatTargets([0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.6], hcat(1e-2 .* n, 1e-8 .* n, 1e-6 .* n, 1e-5 .* n),
                                 nothing, fill((0, 0), 4))
            ro, mo = ep_ref_rows(P, To)
            append!(refs, [(ro[t, :], mo[t, :]) for t in 1:4])
            for rule in (:moment, :gcq)
                rows[rule] = vcat(rows[rule], DB._flat_rows(P, g, To, PlanarProductNearQuad(endpoint_rule = rule, self = :cell)))
            end
        end
        efar, efar_rnd = pl_far_error(P, randn(MersenneTwister(7), length(ru.x) * length(rv.x)))
        S = [u for u in ru.x for v in rv.x]; V = [v for u in ru.x for v in rv.x]
        pu = DB.is_gcq(ru) ? ru.pw : 0.0; pv = DB.is_gcq(rv) ? rv.pw : 0.0
        dens = (an = S .^ pu .* V .^ pv, mixsign = S .^ pu .* V .^ pv .* (1 .- 2 .* V .* S) .+ 0.5 .* cos.(3 .* V .+ S),
                rnd = randn(MersenneTwister(7), length(S)))
        for rule in (:moment, :gcq)
            R = rows[rule]
            # the exchanged panel: the same self rows after reordering the nodes
            RQ = DB._flat_rows(Q, DB._flat_geom(Q), TQ, PlanarProductNearQuad(endpoint_rule = rule, self = :cell))
            @test maximum(maximum(abs.(RQ[t, perm] .- R[t, :])) / sum(abs, R[t, :]) for t in 1:m) < 1e-13
            for t in eachindex(refs)
                ref, mag = refs[t]
                ec = maximum(abs.(R[t, :] .- ref)) / sum(abs, ref); fc = 100eps() * maximum(mag) / sum(abs, ref)
                bc = rule == :moment ? 1e-13 + fc : efar
                @test ec <= bc
                worst[(rule, :card)] = max(get(worst, (rule, :card), 0.0), ec / bc)
                for (dn, σ) in pairs(dens)
                    sc = dot(abs.(ref), abs.(σ)); ed = abs(dot(R[t, :] .- ref, σ)) / sc
                    bd = (rule == :moment ? 1e-13 : dn == :rnd ? efar_rnd : 20tol) + 100eps() * dot(mag, abs.(σ)) / sc
                    @test ed <= bd
                    worst[(rule, dn)] = max(get(worst, (rule, dn), 0.0), ed / bd)
                end
            end
        end
    end
    for (k, v) in sort(collect(worst))
        @info "self/corner rows: worst error / budget" rule = k[1] metric = k[2] ratio = v
    end
end

@testset "self rows: the reference converges; invariance; the depth cap is not binding" begin
    P = sp_panels()[:corner]; g = DB._flat_geom(P)
    T = sp_owned(P, [(1, 1), (1, length(P.rv.x))])                      # the nodes nearest the corner and an edge
    for t in eachindex(T.ur)
        (r1, m1), (r2, _) = sp_self_ref(P, T.ur[t], T.vr[t], T.cell[t]...),
                            sp_self_ref(P, T.ur[t], T.vr[t], T.cell[t]...; c = 10.0, M = 28, grade = 0.15)
        @test maximum(abs.(r2 .- r1)) <= 1e-14 * sum(abs, r2) + 100eps() * maximum(m1)
    end
    T = sp_owned(P, sp_nodes(P))
    for rule in (:moment, :gcq)
        q = PlanarProductNearQuad(endpoint_rule = rule, self = :cell)
        R = DB._flat_rows(P, g, T, q)
        rel(X) = maximum(maximum(abs.(X[t, :] .- R[t, :])) / sum(abs, R[t, :]) for t in axes(R, 1))
        # uniformly rescaled (the single layer scales as the size), and unrotated at the origin: the same rows
        for s in (1e-3, 1e3)
            Ps = sp_panels(; scale = s)[:corner]
            @test rel(DB._flat_rows(Ps, DB._flat_geom(Ps), T, q) ./ s) < 1e-13
        end
        P0 = ep_panel((0.0, 0.0, 0.0), SP_SHAPES.skew..., P.ru, P.rv)
        @test rel(DB._flat_rows(P0, DB._flat_geom(P0), T, q)) < 1e-13
        # max_depth = 80 is not reached by the supported targets (200 gives the same rows) ...
        @test rel(DB._flat_rows(P, g, T, PlanarProductNearQuad(endpoint_rule = rule, max_depth = 200, self = :cell))) == 0
    end
    # ... and the cap is no certificate: a target 1e-9 over the interior (where boxes are halved; clusters are peeled
    # without counting) with max_depth = 4 is far off
    n = normalize(collect(P.normals[1]))
    To = DB._FlatTargets([0.43], [0.61], reshape(1e-9 .* n, 3, 1), nothing, [(0, 0)])
    ro, _ = ep_ref_rows(P, To)
    r4 = DB._flat_rows(P, g, To, PlanarProductNearQuad(max_depth = 4, self = :cell, endpoint_rule = :moment))[1, :]
    @test maximum(abs.(r4 .- ro[1, :])) > 1e-6 * sum(abs, ro[1, :])
end

@testset "corrections and planar_slp_nodes under :gcq" begin
    # two faces of the octahedron sharing an edge, nlev 1: GL × GL, GCQ × GL and GCQ × GCQ panels
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F0, _ = orient_outward(V, reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                                            for i in (1, 2), j in (3, 4), k in (5, 6)]))
    patches, quads = polyhedron_patches(V, F0; eps_in = 2.0)
    tol = 1e-8; nlev = 1; npq = length(polyhedron_quad_rects(nlev))
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev, n_quad = 5, tol, corner = :gcq)
    ta = 1; tb = findfirst(t -> t != ta && length(intersect(F0[:, t], F0[:, ta])) == 2, axes(F0, 2))
    sel = [K for t in (ta, tb) for k in 3(t - 1) + 1:3t for K in (k - 1) * npq + 1:k * npq]
    sub = DB.DielectricInterface{CurvedPanel, Float64}(itf.panels[sel], itf.eps_in[sel], itf.eps_out[sel])
    ps = sub.panels; @test Set(P.kind for P in ps) == Set([:gl, :edge, :corner])
    qm = PlanarProductNearQuad(self = :cell, endpoint_rule = :moment); qg = PlanarProductNearQuad(endpoint_rule = :gcq, self = :cell)
    Cm = DB.planar_slp_corrections(sub; q = qm)
    # symmetry reuse on and off: the same corrections (single layer and Dᵀ)
    for q in (qm, qg)
        q0 = PlanarProductNearQuad(endpoint_rule = q.endpoint_rule, symmetry = false, self = :cell)
        C1 = DB.planar_slp_corrections(sub; q); C0 = DB.planar_slp_corrections(sub; q = q0)
        @test norm(C1 - C0, Inf) < 1e-12 * norm(C0, Inf)
        D1 = DB.near_corrections(q, sub); D0 = DB.near_corrections(q0, sub)
        @test norm(D1 - D0, Inf) < 1e-12 * norm(D0, Inf)
    end
    # no strategy data is shared: the moment corrections are unchanged after the :gcq ones, which differ
    @test DB.planar_slp_corrections(sub; q = qm) == Cm
    @test DB.planar_slp_corrections(sub; q = qg) != Cm
    # planar_slp_nodes = direct node sum (the coincident node left out) + C σ: at selected nodes, Σ over the panels of the
    # panel's row (_flat_rows at the node, Duffy on its own panel) for those within η diameters, else the node rule
    offs = cumsum([0; DB.num_points.(ps)]); n = offs[end]
    owner = [searchsortedlast(offs, t - 1) for t in 1:n]; loc = [t - offs[owner[t]] for t in 1:n]
    X = reduce(hcat, [collect(y) for P in ps for y in P.points]); gs = DB._flat_geom.(ps)
    cells = [(DB._pn_cell(ps[owner[t]].ru, ps[owner[t]].ru.x[(loc[t] - 1) ÷ length(ps[owner[t]].rv.x) + 1], 1, "u"),
              DB._pn_cell(ps[owner[t]].rv, ps[owner[t]].rv.x[(loc[t] - 1) % length(ps[owner[t]].rv.x) + 1], 1, "v")) for t in 1:n]
    near(k, t) = any(y -> norm(X[:, t] .- collect(y)) <= 1.5 * gs[k].diam, ps[k].points)
    targets = [offs[k] + 1 for k in eachindex(ps) if ps[k].kind != :gl][1:4]          # nodes nearest an edge or corner
    σs = (one = ones(n), rnd = randn(MersenneTwister(5), n))
    for q in (qm, qg), σ in σs
        u = planar_slp_nodes(sub, σ; q)
        for t in targets
            terms = map(eachindex(ps)) do k
                σk = σ[offs[k] + 1:offs[k + 1]]
                near(k, t) || return sum(ps[k].weights[j] * σk[j] / (4π * norm(X[:, t] .- collect(ps[k].points[j]))) for j in eachindex(σk))
                Tk = DB._pn_targets(ps, gs, k, [t], owner, loc, zeros(3, n), cells, :slp)
                return dot(DB._flat_rows(ps[k], gs[k], Tk, q)[1, :], σk)
            end
            @test abs(u[t] - sum(terms)) <= 1e-13 * sum(abs, terms)
        end
    end
    # and against the independent reference (σ = 1, in the GCQ family) at two of those nodes: :moment to rounding,
    # :gcq to 20 tol
    for t in targets[1:2], q in (qm, qg)
        u = planar_slp_nodes(sub, ones(n); q)
        ref = 0.0; mag = 0.0
        for k in eachindex(ps)
            if !near(k, t)
                ref += sum(ps[k].weights[j] / (4π * norm(X[:, t] .- collect(ps[k].points[j]))) for j in eachindex(ps[k].points))
            elseif owner[t] == k
                r, m = sp_self_ref(ps[k], ps[k].ru.x[(loc[t] - 1) ÷ length(ps[k].rv.x) + 1], ps[k].rv.x[(loc[t] - 1) % length(ps[k].rv.x) + 1], cells[t]...)
                ref += sum(r); mag += sum(m)
            else
                r, m = ep_ref_rows(ps[k], DB._pn_targets(ps, gs, k, [t], owner, loc, zeros(3, n), cells, :slp))
                ref += sum(r); mag += sum(m)
            end
        end
        @test abs(u[t] - ref) <= (q.endpoint_rule == :gcq ? 20tol : 1e-13) * abs(ref) + 100eps() * mag
    end
end
