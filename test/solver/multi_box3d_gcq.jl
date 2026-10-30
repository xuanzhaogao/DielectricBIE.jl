using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
Base.find_package("SCTL") === nothing || @eval using SCTL

cube(c; L = 1.0) = (center = c, Lx = L, Ly = L, Lz = L)
# nodes, weights, normals and (eps_in, eps_out) of every quadrature node
nodeset(itf) = sort([(round.(p, digits = 13), round(w, digits = 15), pn.normal, itf.eps_in[k], itf.eps_out[k])
                     for (k, pn) in enumerate(itf.panels) for (p, w) in zip(pn.points, pn.weights)])

@testset "multi_dielectric_box3d_gcq: mesh" begin
    # one box: the same panels, nodes and weights as single_dielectric_box3d_gcq
    m1 = multi_dielectric_box3d_gcq([cube((0.0, 0.0, 0.0))], [10.0]; nlev = 2, n_quad = 4, tol = 1e-6)
    s1 = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 4, tol = 1e-6)
    @test length(m1.interface.panels) == length(s1.panels)
    @test nodeset(m1.interface) == nodeset(s1)                  # outward normals, eps_in = box
    @test m1.bodies == [1:DB.num_points(s1)] && m1.clusters == [[1]]

    # two cubes sharing the face x = 0: the shared face is meshed once, as εA | εB
    boxes = [cube((-0.5, 0.0, 0.0)), cube((0.5, 0.0, 0.0))]
    m = multi_dielectric_box3d_gcq(boxes, [2.0, 10.0]; nlev = 1, n_quad = 3, tol = 1e-6)
    w = DB.all_weights(m.interface)
    @test sum(w) ≈ 11 rtol = 1e-6                              # 10 outer faces + 1 shared face
    shared = [k for (k, p) in enumerate(m.interface.panels) if all(abs(x[1]) < 1e-14 for x in p.points)]
    @test sum(sum(m.interface.panels[k].weights) for k in shared) ≈ 1 rtol = 1e-6
    # normal out of the box with the larger index (B), as in multi_dielectric_box3d
    @test all(m.interface.panels[k].normal == (-1.0, 0.0, 0.0) for k in shared)
    @test all(m.interface.eps_in[k] == 10.0 && m.interface.eps_out[k] == 2.0 for k in shared)
    @test m.clusters == [[1, 2]] && m.bodies == [1:DB.num_points(m)]
    # every edge gets the rule of its own junction: the cube edges of ε = 2 and ε = 10, and the
    # triple junction (vacuum π, εA π/2, εB π/2) where the shared face meets the outside
    pws = sort(unique(round.([r.pw for p in m.interface.panels for r in (p.ru, p.rv) if DB.is_gcq(r)], digits = 10)))
    ex = sort(round.([edge_density_powers(2.0, 1.0)[1], edge_density_powers(10.0, 1.0)[1],
                      junction_powers([pi, pi / 2, pi / 2], [1.0, 2.0, 10.0])[1] - 1], digits = 10))
    @test pws ≈ ex atol = 1e-8
    # equal junctions share one rule object (near-row reuse keys on it)
    @test length(unique(objectid, [p.ru for p in m.interface.panels if DB.is_gcq(p.ru)])) == 3

    # a smaller box on top of a larger one: the grid line through the big top face is not singular,
    # the small box's bottom edges are triple junctions (big π, small π/2, vacuum π/2)
    mt = multi_dielectric_box3d_gcq([(center = (0.0, 0.0, 0.0), Lx = 2.0, Ly = 2.0, Lz = 1.0),
                                     (center = (0.0, 0.0, 1.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)], [5.0, 3.0];
                                    nlev = 1, n_quad = 3, tol = 1e-6)
    @test sum(DB.all_weights(mt.interface)) ≈ (2 * 4 + 4 * 2) + (4 + 1) rtol = 1e-6
    pws = unique(round.([r.pw for p in mt.interface.panels for r in (p.ru, p.rv) if DB.is_gcq(r)], digits = 10))
    @test any(isapprox(junction_powers([pi, pi / 2, pi / 2], [5.0, 3.0, 1.0])[1] - 1), pws)
    @test length(pws) == 3                                     # big edge, small edge, triple junction

    # two cubes touching only along an edge stay two bodies; the shared edge has four sectors
    me = multi_dielectric_box3d_gcq([cube((0.5, 0.5, 0.5)), cube((-0.5, -0.5, 0.5))], [10.0, 10.0];
                                    nlev = 1, n_quad = 3, tol = 1e-6)
    @test me.clusters == [[1], [2]] && length(me.bodies) == 2
    pcb = junction_powers(fill(pi / 2, 4), [10.0, 1.0, 10.0, 1.0])[1] - 1
    @test any(p -> any(r -> DB.is_gcq(r) && isapprox(r.pw, pcb; atol = 1e-8), (p.ru, p.rv)), me.interface.panels)

    # perfect conductor touching a dielectric: Dirichlet on the conductor sector
    mc = multi_dielectric_box3d_gcq(boxes, [Inf, 10.0]; nlev = 1, n_quad = 3, tol = 1e-6)
    pws = sort(unique(round.([r.pw for p in mc.interface.panels for r in (p.ru, p.rv) if DB.is_gcq(r)], digits = 10)))
    @test any(isapprox(-1 / 3; atol = 1e-8), pws)               # conductor cube edge, γ = 2/3
    @test any(isapprox(junction_powers([pi / 2, pi, pi / 2], [10.0, 1.0, Inf])[1] - 1; atol = 1e-8), pws)
    @test DB._junction_density_powers([3pi / 2, pi / 2], [1.0, Inf]; gmax = 4.0) ≈ edge_density_powers(Inf, 1.0; gmax = 3.9)

    # source location and constraints
    @test source_permittivity(m, (-0.4, 0.1, 0.1)) == 2.0
    @test source_permittivity(m, (0.4, 0.1, 0.1)) == 10.0
    @test source_permittivity(m, (0.0, 0.9, 0.1)) == 1.0
    @test_throws ArgumentError source_permittivity(m, (0.0, 0.1, 0.1))     # on the shared face
    @test_throws ArgumentError source_permittivity(m, (1.0, 0.1, 0.1))     # on an outer face
    cons = box_charge_constraints(m, [PointSource((-0.4, 0.1, 0.1), 1.0), PointSource((0.4, 0.1, 0.1), 2.0)])
    @test length(cons) == 1 && cons[1][1] == 1:DB.num_points(m)
    @test cons[1][2] ≈ (1 - 1 / 2) + 2 * (1 - 1 / 10)
    @test_throws ArgumentError box_charge_constraints(mc, PointSource((-0.4, 0.1, 0.1), 1.0))
    @test_throws ArgumentError multi_dielectric_box3d_gcq([cube((0.0, 0.0, 0.0)), cube((0.5, 0.0, 0.0))], [2.0, 3.0])
end

@testset "multi_dielectric_box3d_gcq: solve" begin
    # one box: the same system as single_dielectric_box3d_gcq (same iterations and flux)
    src1 = PointSource((0.1, 0.15, 0.2), 1.0)
    m1 = multi_dielectric_box3d_gcq([cube((0.0, 0.0, 0.0))], [2.0]; nlev = 1, n_quad = 3, tol = 1e-3)
    s1 = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 2.0, 1.0; nlev = 1, n_quad = 3, tol = 1e-3)
    a = solve_dielectric_boxes_gcq(m1, src1; deflate = false, rtol = 1e-12)
    b = solve_dielectric_boxes_gcq([s1], src1; deflate = false, rtol = 1e-12)
    @test a.stats.niter == b.stats.niter
    @test dot(DB.all_weights(a.interface), a.σ) ≈ dot(DB.all_weights(b.interface), b.σ) rtol = 1e-12
    # two cubes sharing a face, charge in A: Gauss's law ∫σ = 1 - 1/ε_A, improving with nlev;
    # the cluster constraint is consistent with it (rule tol 1e-4: errs ≈ [9e-7, 6e-7], deflated vs plain ≈ 2e-6)
    boxes = [cube((-0.5, 0.0, 0.0)), cube((0.5, 0.0, 0.0))]
    src = PointSource((-0.4, 0.1, 0.15), 1.0)
    errs = map((0, 1)) do nlev
        m = multi_dielectric_box3d_gcq(boxes, [2.0, 10.0]; nlev, n_quad = 4, tol = 1e-4)
        sol = solve_dielectric_boxes_gcq(m, src; deflate = false, rtol = 1e-8)
        @test sol.stats.solved
        if nlev == 0
            sd = solve_dielectric_boxes_gcq(m, src; rtol = 1e-8)
            @test sd.stats.solved && maximum(abs.(sd.σ .- sol.σ)) / maximum(abs.(sol.σ)) < 1e-4
        end
        abs(dot(DB.all_weights(sol.interface), sol.σ) - (1 - 1 / 2))
    end
    @test errs[2] < errs[1] && errs[2] < 1e-5
end
