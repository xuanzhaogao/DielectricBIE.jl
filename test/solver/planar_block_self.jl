# The default self rows of the single layer on flat panels (PlanarProductNearQuad(self = :block)): one Duffy rule around
# the node over the 3 × 3 fine cells around its own (the density from each point's own cell polynomial), the rest of the
# panel by the walk; against the closed form for σ = 1 and against the own-cell rows (self = :cell) for a smooth density.
using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra, SparseArrays, Random
using Test

# ∫ dA / (4π |x0 - y|) over the convex polygon c (vertices in order) for x0 in its plane (n its normal): signed triangles
function bs_polygon_slp(c, x0, n)
    s = 0.0
    for k in eachindex(c)
        a = c[k]; b = c[mod1(k + 1, length(c))]
        e = (b - a) / norm(b - a); f = a + dot(x0 - a, e) * e; h = norm(x0 - f)
        h == 0 && continue
        s += sign(dot(cross(a - x0, b - x0), n)) * h * (asinh(dot(b - f, e) / h) - asinh(dot(a - f, e) / h))
    end
    return abs(s) / (4π)
end

# the single layer at a lone panel's own nodes: node rule (coincident node skipped) + planar_slp_corrections
function bs_self(P, σ; q = PlanarProductNearQuad())
    one = DB.DielectricInterface{CurvedPanel, Float64}([P], [Inf], [1.0])
    X = reduce(hcat, [collect(y) for y in P.points])
    return DB._pn_direct_slp(X, DB.all_weights(one) .* σ, X) .+ planar_slp_corrections(one; q) * σ
end

@testset "polyhedron pipeline defaults" begin
    # polyhedron_mesh: GCQ rules whose fine grid resolves the family to tol (grid_tol = :auto -> tol)
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                      for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, F; eps_in = Inf)
    g = CurvedGeometry(patches)
    _, rules, _ = polyhedron_mesh(g, quads; nlev = 1, n_quad = 6, tol = 1e-10, corner = :gcq)
    for r in values(rules)
        c = DB.gcq_certificate(r)
        @test c["grid_tol"] == 1e-10 && c["component_resolution"] <= 1e-10
    end
    _, rules0, _ = polyhedron_mesh(g, quads; nlev = 1, n_quad = 6, tol = 1e-10, corner = :gcq, grid_tol = nothing)
    @test all(r -> isempty(DB.gcq_certificate(r)), values(rules0))
    @test_throws ArgumentError polyhedron_mesh(g, quads; nlev = 1, tol = 1e-10, grid_tol = :fine)
    # PlanarProductNearQuad: the scaled-GCQ endpoint clusters and the 3 × 3 Duffy self rows
    q = PlanarProductNearQuad()
    @test q.endpoint_rule == :gcq
    @test q.self == :block
    @test PlanarProductNearQuad(self = :cell).self == :cell
    @test_throws ArgumentError PlanarProductNearQuad(self = :global)
end

@testset "3 × 3 Duffy self rows" begin
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                      for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, F; eps_in = Inf)
    # tol 1e-12 (the rows follow the GCQ tolerance: σ = 1 to 5e-14 / 2e-13 of the closed form on edge / corner panels at
    # tol 1e-10, against 9e-14 / 4e-13 for the own-cell rows)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 2, n_quad = 8, tol = 1e-12, corner = :gcq)
    for kind in (:gl, :edge, :corner)
        P = itf.panels[findfirst(P -> P.kind == kind, itf.panels)]
        n = collect(P.normals[1]); a = collect(P.anchor)
        c = [a .+ collect(curved_lpoint(P, u, v)) for (u, v) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
        exact = [bs_polygon_slp(c, collect(y), n) for y in P.points]
        m = DB.num_points(P)
        @test maximum(abs.(bs_self(P, ones(m)) .- exact) ./ exact) < 5e-14
        # a smooth density s^pw (1 + 0.3 s)(1 + 0.2 t): the 3 × 3 rows agree with the own-cell rows
        sf(r, x) = DB.is_gcq(r) ? x^r.pw * (1 + 0.3x) : 1 + 0.2x
        σ = [sf(P.ru, u) * sf(P.rv, v) for u in P.ru.x for v in P.rv.x]
        ub = bs_self(P, σ); uc = bs_self(P, σ; q = PlanarProductNearQuad(self = :cell))
        @test maximum(abs.(ub .- uc)) / maximum(abs.(uc)) < 1e-14
    end
end

@testset "complement of the Duffy region, walked from the whole panel" begin
    # the walk roots of Ω \ D (study 75): boxes disjoint from D, a cluster [0, s_k] or a part of one fine segment per
    # direction, tiling Ω \ D; the cells at an edge stay clustered past D (a cluster reaching beyond D covers the boxes
    # above or beside it), instead of four rectangles cut at the fine grid
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                      for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, F; eps_in = Inf)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 2, n_quad = 8, tol = 1e-12, corner = :gcq)
    inpart(r, kk, a, b) = kk < 0 ? (a == 0 && b == r.segs[-kk][2]) : (r.segs[kk][1] <= a && b <= r.segs[kk][2])
    ov(a1, b1, a2, b2) = max(0.0, min(b1, b2) - max(a1, a2))
    area(b) = (b[3] - b[2]) * (b[6] - b[5])
    for (kind, iu, iv) in ((:corner, 1, 1), (:corner, 5, 9), (:edge, 10, 4), (:gl, 3, 3))
        P = itf.panels[findfirst(P -> P.kind == kind, itf.panels)]
        g = DB._flat_geom(P); ru, rv = P.ru, P.rv
        u0, v0 = ru.x[iu], rv.x[iv]
        D, _ = DB._flat_block_region(g, ru, rv, DB._pn_cell(ru, u0, 1, "u"), DB._pn_cell(rv, v0, 1, "v"), u0, v0)
        boxes = DB._flat_complement_boxes(ru, rv, g, D)
        @test all(b -> b[2] < b[3] && b[5] < b[6], boxes)
        @test sum(area, boxes; init = 0.0) + (D[2] - D[1]) * (D[4] - D[3]) ≈ 1 atol = 1e-15
        @test all(ov(b[2], b[3], D[1], D[2]) * ov(b[5], b[6], D[3], D[4]) == 0 for b in boxes)
        @test all(ov(b[2], b[3], c[2], c[3]) * ov(b[5], b[6], c[5], c[6]) == 0 for b in boxes, c in boxes if b !== c)
        @test all(inpart(ru, b[1], b[2], b[3]) && inpart(rv, b[4], b[5], b[6]) for b in boxes)
        if kind == :corner && iu == 1
            @test any(b -> b[1] < 0 && b[3] > D[2] && b[5] >= D[4], boxes)     # a u-cluster past D, above it
            @test any(b -> b[4] < 0 && b[6] > D[4] && b[2] >= D[2], boxes)     # a v-cluster past D, beside it
        end
    end
end
