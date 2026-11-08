# Experimental self rows of the single layer from the global GCQ interpolant (plan 2026-11-05-global-gcq-local-duffy,
# Tasks 4-7): Duffy on a rectangle that may cross fine boundaries (planar_global_self.jl), the enlarged-region policy,
# the four-rectangle complement walked with global leaf maps, and the rows against an independent global-density
# reference (gl_self_ref).
using Test, LinearAlgebra, Random
using DielectricBIE
import DielectricBIE as DB
include(joinpath(@__DIR__, "planar_test_helpers.jl"))

const GS_ROT = let ax = normalize([0.3, -0.81, 0.47]), K = [0 -ax[3] ax[2]; ax[3] 0 -ax[1]; -ax[2] ax[1] 0]
    I + sin(0.7310281) * K + (1 - cos(0.7310281)) * K^2
end
# a skew bilinear panel (c ≠ 0, in a plane), rotated and translated; `par` a parallelogram (c = 0), `aniso` stretched
function gs_panel(ru, rv; L = 1.0, par = false, aniso = 1.0)
    A = GS_ROT * [1.0, 0.0, 0.0]; B = GS_ROT * [0.2, 0.9, 0.0] .* aniso; C = par ? zeros(3) : GS_ROT * [0.3, -0.25, 0.0] .* aniso
    return ep_panel((0.3, -0.2, 0.1), Tuple(L .* A), Tuple(L .* B), Tuple(L .* C), ru, rv)
end
gs_cell(r, t) = DB.is_gcq(r) ? searchsortedlast([sa for (sa, _) in r.segs], t) : 1
# the node of a GCQ rule nearest an interior fine boundary (relative to its segment; the panel's side s = 1 is not one)
# and the node nearest the centre of its segment
function gs_nodes(r)
    d = [begin k = gs_cell(r, x); (sa, sb) = r.segs[k]
               min(k >= 2 ? (x - sa) / (sb - sa) : Inf, sb < 1 ? (sb - x) / (sb - sa) : Inf) end for x in r.x]
    return argmin(d), argmax(min.(d, 0.5))
end
# the node nearest the upper end of its (interior) segment, and a node of a small segment (s_k <= 1e-3) nearest a boundary
gs_upper(r) = argmin([begin k = gs_cell(r, x); (sa, sb) = r.segs[k]; sb < 1 ? (sb - x) / (sb - sa) : Inf end for x in r.x])
gs_small(r) = argmin([begin k = gs_cell(r, x); (sa, sb) = r.segs[k]
                      sb <= 1e-3 && k >= 2 ? min((x - sa), (sb - x)) / (sb - sa) : Inf end for x in r.x])
gs_ws(r) = DB.is_gcq(r) ? DB.GlobalInterpWorkspace(r) : nothing
const GS_Q = PlanarProductNearQuad(endpoint_rule = :gcq, self = :cell)
const GS_QM = PlanarProductNearQuad(endpoint_rule = :moment, self = :cell)
# tensor-product physical densities (node values, ordering (i_u - 1) n_v + j_v): s^p (1 + 0.3 s) in a GCQ direction,
# a smooth factor in the other, and a mixed-sign combination
function gs_phys(P)
    f(r, a) = DB.is_gcq(r) ? r.x .^ r.pw .* (1 .+ a .* r.x) : 1 .+ a .* r.x .^ 2
    g(r) = DB.is_gcq(r) ? r.x .^ (r.pw + 1) .* cos.(r.x) : cos.(2 .* r.x)
    t(a, b) = vec(transpose(a * b'))
    return (t(f(P.ru, 0.3), f(P.rv, 0.5)), t(f(P.ru, -0.2), ones(length(P.rv.x))) .- 0.7 .* t(g(P.ru), g(P.rv)),
            ones(length(P.ru.x) * length(P.rv.x)))
end
relerr(Δ, ref, σ) = abs(dot(Δ, σ)) / dot(abs.(ref), abs.(σ))

const GS_RGL = DB.gl_tensor_rule(12)
gs_rules() = (ep_rule(-0.268308, 1e-12)[2], ep_rule(-0.6, 1e-12)[2], ep_rule(-0.04, 1e-8)[2])

@testset "global Duffy rows" begin
    ru, _, _ = gs_rules()
    P = gs_panel(ru, GS_RGL); g = DB._flat_geom(P)
    nu, nv = length(P.ru.x), length(P.rv.x)
    ib, ic = gs_nodes(ru)
    for i in (ib, ic), j in (1, 6)
        u0, v0 = ru.x[i], GS_RGL.x[j]; k = gs_cell(ru, u0)
        R0, _, G = DB._flat_self_regions(g, P.ru, P.rv, k, 1, u0, v0)
        row = zeros(nu, nv)
        DB._global_duffy!(row, P, g, gs_ws(P.ru), gs_ws(P.rv), u0, v0, R0, G)
        # against the graded fan of the same global density on R0
        F = zeros(nu, nv); Fa = zeros(nu, nv)
        gl_singular!(F, Fa, P, g, u0, v0, R0)
        @test maximum(abs.(row - F)) <= gl_noise(P) * sum(Fa)
        for σ in gs_phys(P)
            @test relerr(vec(transpose(row - F)), vec(transpose(F)), σ) < 1e-13
        end
        # the contraction (modal in the GCQ direction) against the density evaluated at the Duffy points directly
        du, dv, w = DB.duffy_rule((u0, v0), R0, G; nρ = DB._global_duffy_nrho(P))
        wt = [w[q] * norm(cross(collect(g.a .+ (v0 + dv[q]) .* g.c), collect(g.b .+ (u0 + du[q]) .* g.c))) /
              (4π * norm(collect(DB._flat_step(g, u0, v0, du[q], dv[q])))) for q in eachindex(w)]
        Σ = randn(MersenneTwister(i + 100j), nu, nv)
        dens = [dot(gl_E(ru, [u0 + du[q]]), Σ * vec(gl_E(GS_RGL, [v0 + dv[q]]))) for q in eachindex(w)]
        @test abs(sum(row .* Σ) - dot(wt, dens)) <= gl_noise(P) * dot(abs.(wt), abs.(dens))
        σp = gs_phys(P)[1]; Σp = transpose(reshape(σp, nv, nu))
        densp = [dot(gl_E(ru, [u0 + du[q]]), Σp * vec(gl_E(GS_RGL, [v0 + dv[q]]))) for q in eachindex(w)]
        @test abs(sum(row .* Σp) - dot(wt, densp)) <= 1e-14 * dot(abs.(wt), abs.(densp))
        # geometry scaling: the single layer of a panel 2× larger (exactly: the same region and points) is 2× the row
        P2 = gs_panel(ru, GS_RGL; L = 2.0); g2 = DB._flat_geom(P2)
        R02, _, G2 = DB._flat_self_regions(g2, P2.ru, P2.rv, k, 1, u0, v0)
        row2 = zeros(nu, nv)
        DB._global_duffy!(row2, P2, g2, gs_ws(P2.ru), gs_ws(P2.rv), u0, v0, R02, G2)
        @test R02 == R0 && maximum(abs.(row2 - 2 .* row)) <= 1e-14 * sum(abs, 2 .* row)
    end
    # orientation: the panel with u and v exchanged (GL × GCQ) gives the transposed row
    Pt = ep_panel((0.3, -0.2, 0.1), Tuple(GS_ROT * [0.2, 0.9, 0.0]), Tuple(GS_ROT * [1.0, 0.0, 0.0]),
                  Tuple(GS_ROT * [0.3, -0.25, 0.0]), GS_RGL, ru); gt = DB._flat_geom(Pt)
    u0, v0 = ru.x[ib], GS_RGL.x[3]; k = gs_cell(ru, u0)
    R0, _, G = DB._flat_self_regions(g, P.ru, P.rv, k, 1, u0, v0)
    R0t, _, Gt = DB._flat_self_regions(gt, Pt.ru, Pt.rv, 1, k, v0, u0)
    row = zeros(nu, nv); rowt = zeros(nv, nu)
    DB._global_duffy!(row, P, g, gs_ws(P.ru), gs_ws(P.rv), u0, v0, R0, G)
    DB._global_duffy!(rowt, Pt, gt, gs_ws(Pt.ru), gs_ws(Pt.rv), v0, u0, R0t, Gt)
    @test maximum(abs.(rowt - transpose(row))) <= gl_noise(P) * sum(abs, row)
    for σ in gs_phys(P)
        @test relerr(vec(rowt - transpose(row)), vec(transpose(row)), σ) < 1e-14
    end
    # a rectangle that crosses fine boundaries: one formula, no splitting at them (against the graded fan)
    sa = ru.segs[k][1]; R = (sa - 0.6 * (sa - ru.segs[k - 1][1]), ru.segs[k][2], R0[3], R0[4])
    row = zeros(nu, nv); DB._global_duffy!(row, P, g, gs_ws(P.ru), gs_ws(P.rv), u0, v0, R, G; nρ = 32, nside = 48)
    F = zeros(nu, nv); Fa = zeros(nu, nv); gl_singular!(F, Fa, P, g, u0, v0, R)
    for σ in gs_phys(P)
        @test relerr(vec(transpose(row - F)), vec(transpose(F)), σ) < 1e-12
    end
    # the global backend refuses the density's singular endpoint in R
    @test_throws ArgumentError DB._global_duffy!(zeros(nu, nv), P, g, gs_ws(P.ru), gs_ws(P.rv), u0, v0, (0.0, R0[2], R0[3], R0[4]), G)
end

@testset "enlarged region policy" begin
    ru, _, _ = gs_rules()
    for P in (gs_panel(ru, GS_RGL), gs_panel(ru, GS_RGL; aniso = 0.05), gs_panel(ru, GS_RGL; aniso = 20.0))
        g = DB._flat_geom(P); ns = length(ru.segs)
        ib, ic = gs_nodes(ru)
        for i in (ib, ic, ib - 1, ib + 1, length(ru.x)), j in (1, 6, 12)
            (1 <= i <= length(ru.x)) || continue
            u0, v0 = ru.x[i], GS_RGL.x[j]; k = gs_cell(ru, u0)
            (sa, sb) = ru.segs[k]
            R0, _, G = DB._flat_self_regions(g, P.ru, P.rv, k, 1, u0, v0)
            prev = R0
            for λ in (0.0, 0.25, 0.5, 1.0)
                R, R0b, Gb = DB._global_self_region(g, P.ru, P.rv, k, 1, u0, v0, λ)
                @test R0b == R0 && Gb == G
                @test R == DB._global_self_region(g, P.ru, P.rv, k, 1, u0, v0, λ)[1]       # deterministic
                ua, ub, va, vb = R
                @test ua < ub && va < vb && ua <= u0 <= ub && va <= v0 <= vb             # positive, holds the target
                @test ua <= R0[1] && ub >= R0[2] && va <= R0[3] && vb >= R0[4]           # never shrinks R0
                @test ua <= prev[1] && ub >= prev[2] && va <= prev[3] && vb >= prev[4]   # grows with λ
                @test ua >= ru.segs[max(k - 1, 1)][1] && ub <= ru.segs[min(k + 1, ns)][2] # 3 × 3 neighbourhood
                @test ua >= 0.25 * u0 && 0 <= va && vb <= 1 && ub <= 1                    # off the GCQ edge, in the panel
                λ == 0 && @test R == R0
                prev = R
            end
            # the Q of λ, in the panel's metric at the target: as long in u as in v (unless capped)
            lu, lv = sqrt(G[1]), sqrt(G[3]); ell = min((sb - sa) * lu, lv)
            R = DB._global_self_region(g, P.ru, P.rv, k, 1, u0, v0, 0.5)[1]
            @test R[1] <= max(u0 - 0.5ell / lu, ru.segs[max(k - 1, 1)][1], 0.25u0) * (1 + 1e-15) &&
                  R[4] >= min(v0 + 0.5ell / lv, 1.0) * (1 - 1e-15)
        end
        # a target beside its cell's lower side extends across it, one beside the upper side across that
        u0 = ru.x[ib]; k = gs_cell(ru, u0); (sa, sb) = ru.segs[k]
        R = DB._global_self_region(g, P.ru, P.rv, k, 1, u0, 0.5, 1.0)[1]
        @test u0 - sa < sb - u0 ? R[1] < sa : R[2] > sb
    end
    # a point of segment 2 (not a node) near its lower end: the cap u >= u0/4 binds at λ = 1 (Q reaches below u0/4)
    P = gs_panel(ru, GS_RGL); g = DB._flat_geom(P)
    u0 = 1.1 * ru.segs[1][2]
    @test DB._global_self_region(g, P.ru, P.rv, 2, 1, u0, 0.5, 1.0)[1][1] == u0 / 4
    # a target well inside R0 keeps R0 at λ = 0.25
    P = gs_panel(ru, GS_RGL); g = DB._flat_geom(P); _, ic = gs_nodes(ru)
    u0 = ru.x[ic]; k = gs_cell(ru, u0)
    @test DB._global_self_region(g, P.ru, P.rv, k, 1, u0, 0.5, 0.25)[1] == DB._flat_self_regions(g, P.ru, P.rv, k, 1, u0, 0.5)[1]
end

@testset "complement partition" begin
    ru, ru2, rv2 = gs_rules()
    for P in (gs_panel(ru, GS_RGL), gs_panel(GS_RGL, ru), gs_panel(ru2, rv2))
        g = DB._flat_geom(P)
        rs = (P.ru, P.rv)
        gdir = DB.is_gcq(P.ru) ? 1 : 2; rg = rs[gdir]
        ib, ic = gs_nodes(rg)
        for i in (ib, ic), λ in (0.0, 0.5, 1.0)
            t0 = rg.x[i]; o = 3 - gdir; s0 = rs[o].x[DB.is_gcq(rs[o]) ? gs_nodes(rs[o])[1] : 3]
            u0, v0 = gdir == 1 ? (t0, s0) : (s0, t0)
            k, l = gs_cell(P.ru, u0), gs_cell(P.rv, v0)
            R = DB._global_self_region(g, P.ru, P.rv, k, l, u0, v0, λ)[1]
            boxes = DB._flat_complement_boxes(P.ru, P.rv, g, R)
            area(b) = (b[3] - b[2]) * (b[6] - b[5])
            @test all(b -> b[2] < b[3] && b[5] < b[6], boxes)
            @test sum(area, boxes) + (R[2] - R[1]) * (R[4] - R[3]) ≈ 1 atol = 1e-15
            ov(a1, b1, a2, b2) = max(0.0, min(b1, b2) - max(a1, a2))
            @test all(ov(b[2], b[3], R[1], R[2]) * ov(b[5], b[6], R[3], R[4]) == 0 for b in boxes)
            @test all(ov(b[2], b[3], c[2], c[3]) * ov(b[5], b[6], c[5], c[6]) == 0 for b in boxes, c in boxes if b !== c)
            # each direction of a box is a cluster [0, s_k] or a part of one fine segment
            inpart(r, kk, a, b) = kk < 0 ? (a == 0 && b == r.segs[-kk][2]) : (r.segs[kk][1] <= a && b <= r.segs[kk][2])
            @test all(inpart(P.ru, b[1], b[2], b[3]) && inpart(P.rv, b[4], b[5], b[6]) for b in boxes)
        end
    end
    # the adapter at R = R0 with the old (piecewise) density reproduces the production self row
    P = gs_panel(ru, GS_RGL); g = DB._flat_geom(P)
    ib, ic = gs_nodes(ru)
    for i in (ib, ic), j in (1, 6)
        u0, v0 = ru.x[i], GS_RGL.x[j]; k = gs_cell(ru, u0)
        T = DB._FlatTargets([u0], [v0], zeros(3, 1), nothing, [(k, 1)])
        rowA = vec(DB._flat_rows(P, g, T, GS_QM))
        rowP = DB.planar_global_self_row(P, u0, v0; λ = 0.0, q = GS_QM, backend = :piecewise)
        @test maximum(abs.(rowP - rowA)) <= 1e-13 * sum(abs, rowA)
    end
    @test_throws ArgumentError DB.planar_global_self_row(P, ru.x[ib], 0.5; q = GS_QM)                  # global needs :gcq
    @test_throws ArgumentError DB.planar_global_self_row(P, ru.x[ib], 0.5; backend = :other)
    @test_throws ArgumentError DB.planar_global_self_row(P, 0.0, 0.5)                                 # on the GCQ edge
    # a GL x GL panel has no GCQ direction; a point in the weighted segment has no self row, whatever the weight power
    @test_throws "no GCQ direction" DB.planar_global_self_row(gs_panel(GS_RGL, GS_RGL), 0.5, 0.5)
    r0 = gcq_tensor_rule(edge_gcq_rule([0.0, 0.37]; tol = 1e-8)); @test r0.pw == 0
    @test_throws "weighted segment" DB.planar_global_self_row(gs_panel(r0, GS_RGL), r0.segs[1][2] / 2, 0.5)
    # pw = 0: the [0, b] part of segment 1 by the scaled GCQ rule (the density has non-integer powers there)
    P0 = gs_panel(r0, GS_RGL); u0 = 1.5 * r0.segs[1][2]
    ref, _ = gl_self_ref(P0, u0, 0.5, 2, 1)
    for σ in gs_phys(P0)
        @test relerr(DB.planar_global_self_row(P0, u0, 0.5; λ = 1.0) - ref, ref, σ) < 1e-10
    end
    # rows computed in threads (task-local workspaces) are the serial rows
    ts = [(ru.x[i], GS_RGL.x[j]) for i in (ib, ic, 5) for j in (1, 6)]
    ser = [DB.planar_global_self_row(P, t...; λ = 0.5) for t in ts]
    par = Vector{Vector{Float64}}(undef, length(ts))
    Threads.@threads for i in eachindex(ts)
        par[i] = DB.planar_global_self_row(P, ts[i]...; λ = 0.5)
    end
    @test par == ser
    # σ = 1 on a parallelogram: the closed form (the global interpolant of 1 is 1 to the rule's tolerance)
    Pp = gs_panel(ru, GS_RGL; par = true)
    corners = [collect(DB.curved_lpoint(Pp, s, t)) for (s, t) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    nrm = normalize(cross(corners[2] - corners[1], corners[4] - corners[1]))
    for i in (ib, ic), j in (1, 6), λ in (0.0, 0.5, 1.0)
        u0, v0 = ru.x[i], GS_RGL.x[j]
        ex = gl_polygon_slp(corners, collect(DB.curved_lpoint(Pp, u0, v0)), nrm)
        @test abs(sum(DB.planar_global_self_row(Pp, u0, v0; λ)) - ex) <= 1e-12 * ex
    end
end

@testset "global self rows against an independent reference" begin
    ru, ru2, rv2 = gs_rules()
    # (name, panel, largest GCQ tolerance): a tol 1e-8 direction's scaled endpoint rule integrates its family to ~tol
    rv12 = ep_rule(-0.04, 1e-12)[2]
    panels = (("GCQ x GL", gs_panel(ru, GS_RGL), 1e-12), ("GL x GCQ", gs_panel(GS_RGL, ru), 1e-12),
              ("GCQ x GCQ", gs_panel(ru2, rv12), 1e-12), ("GCQ x GCQ tol 1e-12 x 1e-8", gs_panel(ru2, rv2), 1e-8),
              ("GCQ x GCQ one rule", gs_panel(ru2, ru2), 1e-12),            # the same rule object in u and v
              ("GCQ x GL aniso", gs_panel(ru, GS_RGL; aniso = 0.1), 1e-12))
    for (name, P, tolmax) in panels
        rs = (P.ru, P.rv)
        targets = Tuple{Float64, Float64}[]
        for d in 1:2
            DB.is_gcq(rs[d]) || continue
            ib, ic = gs_nodes(rs[d]); o = 3 - d
            s0 = DB.is_gcq(rs[o]) ? rs[o].x[gs_nodes(rs[o])[2]] : rs[o].x[4]
            for i in (ib, ic)
                push!(targets, d == 1 ? (rs[d].x[i], s0) : (s0, rs[d].x[i]))
            end
        end
        startswith(name, "GCQ x GCQ") && push!(targets, (P.ru.x[1], P.rv.x[1]))  # the extreme corner node
        # a fine-cell corner (both directions beside a boundary), an upper boundary, a small segment, and a point (not a
        # node) of segment 2: the cap u >= u0/4 binds there and the complement takes the weighted segment's [0, b] part
        startswith(name, "GCQ x GCQ") && push!(targets, (P.ru.x[gs_nodes(P.ru)[1]], P.rv.x[gs_nodes(P.rv)[1]]))
        for d in 1:2
            DB.is_gcq(rs[d]) || continue
            o = 3 - d; s0 = DB.is_gcq(rs[o]) ? rs[o].x[gs_nodes(rs[o])[2]] : rs[o].x[4]
            for t in (rs[d].x[gs_upper(rs[d])], rs[d].x[gs_small(rs[d])], 1.1 * rs[d].segs[1][2])
                push!(targets, d == 1 ? (t, s0) : (s0, t))
            end
        end
        for (u0, v0) in targets
            k, l = gs_cell(P.ru, u0), gs_cell(P.rv, v0)
            ref, mag = gl_self_ref(P, u0, v0, k, l)
            ref2, _ = gl_self_ref(P, u0, v0, k, l; M = 24, grade = 0.2, c = 8.0)
            # the reference's uncertainty: physical densities to rounding, cardinal ones to the Float64 floor
            unc = maximum(relerr(ref2 - ref, ref, σ) for σ in gs_phys(P))
            uncc = maximum(abs.(ref2 - ref)) / sum(mag)
            @info "reference uncertainty $name ($u0, $v0)" physical = unc cardinal = uncc
            @test unc < max(1e-14, 1e-3 * tolmax)          # well below the bound on the rows below
            @test uncc < gl_noise(P)
            for λ in (0.0, 0.25, 0.5, 1.0)
                row = DB.planar_global_self_row(P, u0, v0; λ)
                car = maximum(abs.(row - ref)) / sum(mag)
                phys = maximum(relerr(row - ref, ref, σ) for σ in gs_phys(P))
                @info "global self row $name ($u0, $v0) λ = $λ" cardinal = car physical = phys
                # physical: acceptance 1e-12 (tol 1e-12 rules), held at 1e-14 here (10x the largest measured) for regressions.
                # cardinal (max_j |Δ_j| / Σ_j |row_j|, the far-field budget's convention, pl_far_error): beside a GCQ edge
                # the scaled-GCQ cluster leaves dominate (they integrate the rule's family to tol, not a cardinal function:
                # splitting the clusters drops the extreme corner from 3.6e-5 to 1.6e-7 per entry), elsewhere the Float64
                # evaluation noise; held at 5e-6, 10x the largest measured, far below the node rule's far-field error
                @test phys < (tolmax <= 1e-12 ? 1e-14 : 1e-2 * tolmax)
                @test car < 5e-6
            end
        end
    end
end

@testset "piecewise backend across fine cells (refined interpolant)" begin
    # with a rule whose fine cells resolve the global interpolant (ninterp = 28: jumps ~1e-15), the enlarged Duffy region
    # may evaluate the cheap piecewise interpolant across fine boundaries: the row of that interpolant, against the
    # independent fine-grid reference of the same interpolant (sp_self_ref)
    e28 = edge_gcq_rule([-0.268308]; tol = 1e-12, ninterp = 28); ru = gcq_tensor_rule(e28)
    for P in (gs_panel(ru, GS_RGL), gs_panel(GS_RGL, ru), gs_panel(ru, gcq_tensor_rule(edge_gcq_rule([-0.04]; tol = 1e-12, ninterp = 28))))
        rs = (P.ru, P.rv); d = DB.is_gcq(P.ru) ? 1 : 2; rg = rs[d]
        ib, ic = gs_nodes(rg)
        for i in (ib, ic), λ in (0.5, 1.0)
            o = 3 - d; s0 = DB.is_gcq(rs[o]) ? rs[o].x[gs_nodes(rs[o])[1]] : rs[o].x[4]
            u0, v0 = d == 1 ? (rg.x[i], s0) : (s0, rg.x[i])
            k, l = gs_cell(P.ru, u0), gs_cell(P.rv, v0)
            row = DB.planar_global_self_row(P, u0, v0; λ, q = GS_QM, backend = :piecewise)
            ref, mag = sp_self_ref(P, u0, v0, k, l)
            @test maximum(abs.(row - ref)) <= 1e-13 * sum(mag)
            for σ in gs_phys(P)
                @test relerr(row - ref, ref, σ) < 1e-14
            end
        end
    end
end
