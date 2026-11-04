using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test

# the regular octahedron conv(±e_i): 8 triangles, 12 edges with dihedral angle arccos(-1/3)
const OCT_V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
# outward: (i, j, k) is positively oriented when the signs of ±x, ±y, ±z multiply to +1
const OCT_F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                           for i in (1, 2), j in (3, 4), k in (5, 6)])

@testset "OBJ input and orientation" begin
    f = tempname() * ".obj"
    open(f, "w") do io
        for x in eachcol(OCT_V)
            println(io, "v ", join(x .* 0.618034, " "))         # six-digit golden ratio, snapped below
        end
        for t in eachcol(OCT_F)
            println(io, "f ", join(["$i/$i" for i in t], " "))
        end
    end
    V, F = read_obj(f; snap = [(sqrt(5) - 1) / 2])
    @test V == OCT_V .* ((sqrt(5) - 1) / 2)
    F2, vol = orient_outward(V, F)
    @test vol ≈ 4 / 3 * ((sqrt(5) - 1) / 2)^3
    @test orient_outward(V, F2[[1, 3, 2], :])[1] == F2
    @test_throws ArgumentError orient_outward(V, F[:, 1:7])            # not closed
end

@testset "polyhedron mesh and planar near quadrature" begin
    F, _ = orient_outward(OCT_V, OCT_F)
    patches, quads = polyhedron_patches(OCT_V, F; eps_in = 2.0)
    g = CurvedGeometry(patches)
    @test length(polyhedron_quad_rects(2)) == 7
    itf, rules, powers = polyhedron_mesh(g, quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    @test length(itf.panels) == 8 * 3 * 4
    @test sum(DB.all_weights(itf)) ≈ 8 * sqrt(3) / 2 rtol = 1e-7         # area (8 triangles of side √2), to the rule's tol
    @test length(rules) == 1                                              # one junction for all 12 edges
    # rows reused between congruent panels agree with rows computed per panel: on the 6 corner panels of two
    # triangles sharing an edge (the three quadrilaterals of an equilateral triangle are congruent)
    m0, _, _ = polyhedron_mesh(g, quads; nlev = 0, tol = 1e-4, corner = :gcq)
    itf0 = DB.DielectricInterface{CurvedPanel, Float64}(m0.panels[1:6], m0.eps_in[1:6], m0.eps_out[1:6])
    C1 = DB.near_corrections(PlanarProductNearQuad(), itf0)
    C0 = DB.near_corrections(PlanarProductNearQuad(symmetry = false), itf0)
    @test norm(C1 - C0, Inf) < 1e-12 * norm(C0, Inf)
    # a charge inside: the total induced charge is 1 - 1/ε
    src = PointSource((0.1, 0.05, 0.02), 1.0)
    A = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = PlanarProductNearQuad())
    σ, st = solve_dielectric_box3d_gcq(A, rhs_dielectric_box3d(itf, src, 2.0), itf; rtol = 1e-12)
    @test st.solved
    @test abs(dot(DB.all_weights(itf), σ) / 0.5 - 1) < 1e-4
    # single-layer potential: continuous across a face, and the node rule far away
    c = sum(OCT_V[:, F[:, 1]]; dims = 2)[:] ./ 3; n = normalize(c)
    u = planar_slp_potential(itf, σ, hcat(c .+ 1e-8 .* n, c .- 1e-8 .* n, [0.0, 0.0, 20.0]))
    @test abs(u[1] - u[2]) < 1e-7 * abs(u[1])
    X = reduce(vcat, [p.points for p in itf.panels]); w = DB.all_weights(itf)
    @test u[3] ≈ sum(w[j] * σ[j] / (4π * norm(collect(X[j]) .- [0.0, 0.0, 20.0])) for j in eachindex(w)) rtol = 1e-13
end

# ∫ dA / (4π |x0 - y|) over the convex polygon c (vertices in order) for x0 in its plane (n its normal): signed triangles
# (x0, a, b), each h (asinh(t_b / h) - asinh(t_a / h)), h the distance of x0 to the line ab, t along it from the foot
function polygon_slp(c, x0, n)
    s = 0.0
    for k in eachindex(c)
        a = c[k]; b = c[mod1(k + 1, length(c))]
        e = (b - a) / norm(b - a)
        f = a + dot(x0 - a, e) * e
        h = norm(x0 - f)
        h == 0 && continue
        s += sign(dot(cross(a - x0, b - x0), n)) * h * (asinh(dot(b - f, e) / h) - asinh(dot(a - f, e) / h))
    end
    return abs(s) / (4π)
end

@testset "single layer on the surface" begin
    # a GL × GL panel, σ = 1: against the closed form at its own nodes
    F, _ = orient_outward(OCT_V, OCT_F)
    patches, quads = polyhedron_patches(OCT_V, F; eps_in = 2.0)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    P = first(P for P in itf.panels if P.kind == :gl)
    one = DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0])
    u = planar_slp_nodes(one, ones(DB.num_points(P)))
    c = [collect(DB.curved_lpoint(P, s, t)) for (s, t) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    ue = [polygon_slp(c, collect(y .- P.anchor), collect(P.normals[1])) for y in P.points]
    @test maximum(abs.(u .- ue) ./ ue) < 1e-12
    # GCQ edge and corner panels (PEC rules of a cube, GCQ in one or both directions) as CurvedPanels of the same map
    # and rules, a random density: against potential_tensor's product integration on the TensorPanels
    rule = DB.edge_gcq_rule(DB.edge_density_powers(Inf, 1.0; gmax = 2.0); tol = 1e-6, qmax = 12.0)
    cube = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, Inf, 1.0; nlev = 1, n_quad = 6, rule)
    for kind in (:edge, :corner)
        T = first(P for P in cube.panels if P.kind == kind)
        C = CurvedPanel(T.anchor, (u, v) -> T.o .+ (u * T.Lu) .* T.eu .+ (v * T.Lv) .* T.ev, T.ru, T.rv; kind,
                        dmap = (u, v) -> (T.Lu .* T.eu, T.Lv .* T.ev))
        @test maximum(norm(collect(x .- y)) for (x, y) in zip(C.points, T.points)) < 1e-14
        σ = randn(DB.num_points(T))
        ut = DB.potential_tensor(DB.DielectricInterface{typeof(T), Float64}([T], [Inf], [1.0]), σ, T.points; quad_tol = 1e-16)
        uc = planar_slp_nodes(DB.DielectricInterface{CurvedPanel, Float64}([C], [Inf], [1.0]), σ)
        @test maximum(abs.(uc .- ut)) < 1e-12 * maximum(abs.(ut))
    end
end
