# The half-cluster split of PlanarProductNearQuad (split = :half, the default; study 99 of dielectric_edge_singularity):
# a GCQ direction's endpoint cluster [0, s_k] splits at the segment end nearest below s_k / 2 into a cluster and an
# interval across fine segments (a 12-point Gauss-Legendre leaf, the density from each point's own segment), instead of
# peeling one segment (split = :peel). Checked: the pieces tile [0, 1]; an interval leaf integrates a resolved density
# as the segments' own leaves do; near rows of GCQ-family densities agree with :peel's; the far end of a panel is no
# longer cut at the fine grid.
using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test

function hs_mesh()
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                      for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, F; eps_in = Inf)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 2, n_quad = 10, tol = 1e-12, corner = :gcq)
    return itf
end

# a target hrel × the panel's diameter above (u, v)
function hs_above(P, g, u, v, hrel)
    n = DB._cross3(g.a .+ v .* g.c, g.b .+ u .* g.c); n = n ./ DB._norm3(n)
    return DB._FlatTargets([u], [v], reshape(collect(hrel * g.diam .* n), 3, 1), nothing, [(0, 0)])
end

@testset "half-cluster split" begin
    itf = hs_mesh()
    pe = findfirst(P -> P.kind == :edge && DB.is_gcq(P.ru) && !DB.is_gcq(P.rv), itf.panels)
    pc = findfirst(P -> P.kind == :corner, itf.panels)
    r = itf.panels[pe].ru
    # the far fine segments are halved (n_max powers near s = 1): :half splits the root at 1/2, :peel at the last one
    @test r.segs[end - 1][2] > 0.5
    c1, c2 = DB._flat_dir_split(r, -length(r.segs), 0.0, 1.0, true)
    @test c1[1] < 0 && c1[3] == 0.5 && c2 == (0, 0.5, 1.0)
    p1, p2 = DB._flat_dir_split(r, -length(r.segs), 0.0, 1.0, false)
    @test p1[3] == r.segs[end - 1][2] && p2 == (length(r.segs), r.segs[end]...)
    # splitting every direction to depth 8 tiles [0, 1]: clusters from 0, parts inside one segment, intervals across
    pieces = [(-length(r.segs), 0.0, 1.0)]
    for _ in 1:8
        pieces = reduce(vcat, [collect(DB._flat_dir_split(r, p..., true)) for p in pieces])
    end
    sort!(pieces; by = p -> p[2])
    @test pieces[1][2] == 0 && pieces[end][3] == 1 && all(pieces[i][3] == pieces[i + 1][2] for i in 1:length(pieces) - 1)
    @test all(p -> p[1] >= 0 || p[2] == 0, pieces)
    @test all(p -> p[1] <= 0 || r.segs[p[1]][1] <= p[2] && p[3] <= r.segs[p[1]][2], pieces)
    @test all(p -> p[1] != 0 || DB._flat_piece(r, p[2], p[3])[1] == 0, pieces)
    # an interval leaf against the segments' own leaves on a resolved density s^pw (1 + s^2) e^-s times a smooth f
    tb = DB._tensor_near_tables(r)
    buf = (zeros(DB.TENSOR_NEAR_M), zeros(DB.TENSOR_NEAR_M), zeros(length(r.z0), DB.TENSOR_NEAR_M),
           zeros(size(r.P, 2), DB.TENSOR_NEAR_M))
    σ = [x^r.pw * (1 + x^2) * exp(-x) for x in r.x]
    f(s) = 1 / (1.3 - s)
    a, b = 0.5, 1.0
    c, off, G = DB._flat_interval_map!(buf, r, tb, a, b)
    I12 = dot(σ, G * f.(c .+ off))
    ks = [k for k in eachindex(r.segs) if a <= r.segs[k][1] && r.segs[k][2] <= b]
    Iseg = sum(ks) do k
        pts, sh, Gk, _ = DB._tensor_leaf_map!(buf, r, tb, 1.0, k, r.segs[k][1], 0.0, r.segs[k][2] - r.segs[k][1])
        dot(σ, Gk * f.(r.segs[k][1] .+ (pts .- sh)))
    end
    @test length(ks) > 1 && abs(I12 - Iseg) <= 1e-12 * abs(Iseg)
    # near rows of GCQ-family densities: :half against :peel (η_leaf = 3, the reference), and fewer kernel points
    for (pid, u, v, h) in ((pe, 0.0, 0.62, 0.035), (pe, 0.62, 0.5, 0.04), (pc, 0.0, 0.0, 0.035), (pc, 0.62, 0.5, 0.04))
        P = itf.panels[pid]; g = DB._flat_geom(P); T = hs_above(P, g, u, v, h); pu, pv = P.ru.pw, P.rv.pw
        S = [[x^pu * y^pv * exp(0.6x - 0.4y) for x in P.ru.x for y in P.rv.x],
             [x^pu * y^pv * (cos(3x + 2y) + 2) + (1 + x * y) + x^-pu * y^-pv * (1 - y) for x in P.ru.x for y in P.rv.x]]
        ref = DB._flat_rows(P, g, T, PlanarProductNearQuad(split = :peel, η_leaf = 3.0))
        R = DB._flat_rows(P, g, T, PlanarProductNearQuad())
        @test abs(only(R * S[1]) - only(ref * S[1])) <= 1e-14 * abs(only(ref * S[1]))
        @test abs(only(R * S[2]) - only(ref * S[2])) <= 2e-12 * abs(only(ref * S[2]))
    end
end
