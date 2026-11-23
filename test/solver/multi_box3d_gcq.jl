using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
using SCTL

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
    @test length(cons) == 1 && cons[1].nodes == 1:DB.num_points(m)
    @test cons[1].Q ≈ (1 - 1 / 2) + 2 * (1 - 1 / 10)
    @test_throws ArgumentError box_charge_constraints(mc, PointSource((-0.4, 0.1, 0.1), 1.0))
    @test_throws ArgumentError multi_dielectric_box3d_gcq([cube((0.0, 0.0, 0.0)), cube((0.5, 0.0, 0.0))], [2.0, 3.0])
end

@testset "multi_dielectric_box3d_gcq: nlev per vertex" begin
    # nlev may be a function (vertex, a1, a2) -> levels of a true vertex and its quadrant's sides
    b = [cube((0.0, 0.0, 0.0))]
    m2 = multi_dielectric_box3d_gcq(b, [10.0]; nlev = 2, n_quad = 3, tol = 1e-6)
    f2 = multi_dielectric_box3d_gcq(b, [10.0]; nlev = (v, a1, a2) -> 2, n_quad = 3, tol = 1e-6)
    @test nodeset(f2.interface) == nodeset(m2.interface)          # a constant function is the Int
    # grade the vertex (0.5, 0.5, 0.5) by 3 levels, the others not at all
    seen = Set{NTuple{3, Float64}}(); sides = Set{Tuple{Float64, Float64}}()
    lv(v, a1, a2) = (push!(seen, v); push!(sides, (a1, a2)); v == (0.5, 0.5, 0.5) ? 3 : 0)
    g = multi_dielectric_box3d_gcq(b, [10.0]; nlev = lv, n_quad = 3, tol = 1e-6)
    @test length(seen) == 8 && sides == Set([(0.5, 0.5)])          # every vertex, with its quadrant's sides
    m0 = multi_dielectric_box3d_gcq(b, [10.0]; nlev = 0, n_quad = 3, tol = 1e-6)
    # the 3 quadrants at the vertex gain 3 panels per level each
    @test length(g.interface.panels) == length(m0.interface.panels) + 3 * 3 * 3
    @test minimum(max(p.Lu, p.Lv) for p in g.interface.panels if p.kind == :corner) ≈ 0.5 / 2^3
    @test sum(DB.all_weights(g.interface)) ≈ 6 rtol = 1e-6
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

@testset "multi_dielectric_box3d_gcq: conductors" begin
    # conductors: perfectly conducting boxes joined by a shared face or a shared edge; a vertex contact keeps them apart
    gm(cs, es) = multi_dielectric_box3d_gcq(cube.(cs), es; nlev = 1, n_quad = 3, tol = 1e-4)
    row = gm([(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [Inf, 2.0, Inf])
    @test row.clusters == [[1, 2, 3]] && row.conductors == [[1], [3]]
    ml = gm([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], [Inf, 2.0, Inf])     # edge contact x = y = 1/2
    @test ml.conductors == [[1, 3]]
    mv = gm([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [Inf, 2.0, Inf])     # vertex contact (1/2, 1/2, 1/2)
    @test sort(mv.conductors) == [[1], [3]] && sort(mv.clusters) == [[1, 2], [3]]
    mf = gm([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)], [Inf, Inf])                          # face contact: one body
    @test mf.conductors == [[1, 2]]
    @test sum(DB.all_weights(mf.interface)) ≈ 10 rtol = 1e-4                    # no interface between them
    @test gm([(0.0, 0.0, 0.0)], [2.0]).conductors == Vector{Int}[]

    # the box on the eps_in side and on the eps_out side of every panel (0: the exterior)
    epsof(m, b) = b == 0 ? m.eps_out : m.epses[b]
    @test length(row.panel_boxes) == length(row.interface.panels)
    @test all(row.interface.eps_in[k] == epsof(row, bi) && row.interface.eps_out[k] == epsof(row, bo)
              for (k, (bi, bo)) in enumerate(row.panel_boxes))

    # one constraint per conductor on its own surface: the free charge Σ ε_d ∫ σ (ε_d across the face), normalized
    # by the conductor's mean ε_d; u carries the sign of the node's row (+ where the normal leaves the conductor), so
    # ℓᵀu = 1 for the conductor's left null vector ℓ = s w. A cluster holding a conductor has no row of its own.
    cons = box_charge_constraints(row, PointSource((-0.3, 1.5, 0.1), 1.0))
    @test length(cons) == 2 && all(c -> c isa ChargeConstraint && c.Q == 0, cons)
    w = DB.all_weights(row.interface)
    offs = cumsum([0; DB.num_points.(row.interface.panels)])
    for (c, K) in zip(cons, row.conductors)
        on = [k for (k, (bi, bo)) in enumerate(row.panel_boxes) if bi in K || bo in K]
        @test sort(c.nodes) == sort(reduce(vcat, [collect(offs[k]+1:offs[k+1]) for k in on]))
        @test sum(w[c.nodes]) ≈ 6 rtol = 1e-4                                       # the cube's surface
        εd = [let k = searchsortedlast(offs, i - 1); row.panel_boxes[k][1] in K ? row.interface.eps_out[k] :
                                                                               row.interface.eps_in[k] end for i in c.nodes]
        @test c.g ≈ εd .* w[c.nodes] ./ (sum(εd .* w[c.nodes]) / sum(w[c.nodes]))   # 5 vacuum faces, 1 face on ε 2
        @test sort(unique(εd)) == [1.0, 2.0]
        s = [let k = searchsortedlast(offs, i - 1); row.panel_boxes[k][1] in K ? 1.0 : -1.0 end for i in c.nodes]
        @test c.u ≈ s ./ sum(w[c.nodes]) && dot(s .* w[c.nodes], c.u) ≈ 1
    end
    @test count(<(0), cons[1].u) > 0                     # box 1's shared face: normal out of the ε box, into the conductor
    @test length(box_charge_constraints(ml, PointSource((-0.3, 1.0, 0.15), 1.0))) == 1
    @test length(box_charge_constraints(mv, PointSource((-0.3, 1.0, 0.15), 1.0))) == 2
    # a conductor alone in vacuum: the plain total charge, as before
    c1_m = gm([(0.0, 0.0, 0.0)], [Inf])
    c1 = only(box_charge_constraints(c1_m, PointSource((0.0, 1.5, 0.0), 1.0)))
    @test c1.g ≈ DB.all_weights(c1_m.interface)[c1.nodes] && c1.u ≈ fill(1 / sum(c1.g), length(c1.g))
    # dielectric clusters keep their Gauss total charge (constraints as tuples still work in the solver)
    pair = gm([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)], [2.0, 10.0])
    cp = only(box_charge_constraints(pair, PointSource((-0.4, 0.1, 0.1), 1.0)))
    @test cp.nodes == collect(1:DB.num_points(pair)) && cp.Q ≈ 1 - 1 / 2 && cp.g == DB.all_weights(pair.interface)
end

@testset "multi_dielectric_box3d_gcq: conductor solves" begin
    # inside a conductor the total potential u_inc + Sσ is constant; the charge q = 1 is outside, so every conductor is
    # neutral. Coarse mesh (rules to 1e-4): the free charges and the potential spreads are ~1e-5 of their scales.
    cin(c) = [c .+ s .* ntuple(i -> Float64(i == ax), 3) for ax in 1:3 for s in (-0.25, 0.25)]
    free(c, σ) = abs(dot(c.g, σ[c.nodes])) / dot(abs.(c.g), abs.(σ[c.nodes]))
    function check(boxes, epses, src)
        m = multi_dielectric_box3d_gcq(cube.(boxes), epses; nlev = 0, n_quad = 3, tol = 1e-4)
        A = dielectric_operator(m.interface; far = FMM3DFarField(1e-12), near = GCQNearQuad(quad_tol = 1e-12))
        b = rhs_dielectric_box3d(m.interface, PointSource(src, 1.0), 1.0)
        cons = box_charge_constraints(m, PointSource(src, 1.0))
        σ, st = solve_dielectric_box3d_gcq(A, b, m.interface; rtol = 1e-10, constraints = cons)
        V = map(m.conductors) do K
            T = cin(boxes[K[1]])
            u = potential_tensor(m.interface, σ, T) .+ [1 / (4π * norm(x .- src)) for x in T]
            (sum(u) / length(u), maximum(u) - minimum(u))
        end
        return m, A, b, cons, σ, st, V
    end
    # two conductors joined through a dielectric (one cluster): each neutral, each at its own potential
    m, A, b, cons, σ, st, V = check([(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [Inf, 2.0, Inf], (-0.3, 1.5, 0.1))
    @test st.solved && length(cons) == 2 && all(free(c, σ) < 1e-4 for c in cons)
    @test all(v[2] < 1e-5 * abs(v[1]) for v in V) && abs(V[1][1] - V[2][1]) > 0.1 * abs(V[1][1])
    # the cluster's total alone (the earlier default) leaves the split between the two conductors free: ~10% off
    σc, _ = solve_dielectric_box3d_gcq(A, b, m.interface; rtol = 1e-10, constraints = [(1:DB.num_points(m), 0.0)])
    @test maximum(free(c, σc) for c in cons) > 1e-2
    # conductors touching along an edge are one conductor: one row, one potential, charge flows between the boxes
    m, A, b, cons, σ, st, V = check([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], [Inf, 2.0, Inf], (-0.3, 1.0, 0.15))
    @test st.solved && free(only(cons), σ) < 1e-4 && only(V)[2] < 1e-4 * abs(only(V)[1])
end
