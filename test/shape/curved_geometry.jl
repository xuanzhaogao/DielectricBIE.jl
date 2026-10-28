using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "lens_patches.jl"))
include(joinpath(@__DIR__, "cylinders_patches.jl"))

capR(a, θ) = a / sin(θ)
patch_area(msh, k) = sum(BI.all_weights(msh.interface)[msh.ranges[k]])
nonsmooth(g) = [c for c in g.curves if !c.smooth]

@testset "curved geometry" begin
    # one ε, θ1 = θ2 = π/2: a sphere; every shared curve is a smooth seam
    g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
    @test isempty(nonsmooth(g))
    @test all(c -> length(c.sides) == 2 && all(isapprox.(c.angles, π; atol = 1e-9)), g.curves)
    msh = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-12)
    @test sum(BI.all_weights(msh.interface)) ≈ 4π rtol = 1e-10
    @test all(p -> p.kind == :gl, msh.interface.panels)
    # grading toward a chosen curve (the equator), with the GCQ rule of its (smooth) junction
    eq = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-8, grade = c -> abs(c.midpoint[3]) < 1e-12)
    @test count(p -> p.kind == :edge, eq.interface.panels) == 2 * 4 * 2
    @test sum(BI.all_weights(eq.interface)) ≈ 4π rtol = 1e-7

    # lens: four rim curves (one per side chart), each a dielectric wedge of angle θ1 + θ2
    θ1, θ2 = π / 3, π / 4
    l = CurvedGeometry(lens_patches(1.0, θ1, θ2, 4.0, 4.0, 1.0))
    rim = nonsmooth(l)
    @test length(rim) == 4
    for c in rim
        @test length(c.sides) == 2
        @test sort(c.angles) ≈ sort([θ1 + θ2, 2π - θ1 - θ2]) atol = 1e-10
        @test c.eps[argmin(c.angles)] == 4.0 && c.eps[argmax(c.angles)] == 1.0
        @test abs(c.midpoint[3]) < 1e-14 && norm(c.midpoint) ≈ 1
    end
    lm = curved_gcq_mesh(l; nlev = 3, n_quad = 8, tol = 1e-12)
    @test all(isapprox.(lm.powers[findfirst(c -> !c.smooth, l.curves)], edge_density_powers(4.0, 1.0; angle = θ1 + θ2); atol = 1e-10))
    @test count(p -> p.kind == :edge, lm.interface.panels) == 2 * 4 * 2
    up = sum(patch_area(lm, k) for k in 1:5); lo = sum(patch_area(lm, k) for k in 6:10)
    @test up ≈ 2π * capR(1.0, θ1)^2 * (1 - cos(θ1)) rtol = 1e-10
    @test lo ≈ 2π * capR(1.0, θ2)^2 * (1 - cos(θ2)) rtol = 1e-10
    # outward normals on the upper cap
    c1 = (0.0, 0.0, -capR(1.0, θ1) * cos(θ1))
    @test all(dot(n, x .- c1) > 0 for p in lm.interface.panels[findall(==(1), lm.panel_patch)] for (x, n) in zip(p.points, p.normals))

    # split sphere: three-sided rim curves (outside π, upper π/2, lower π/2)
    s = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
    rim = nonsmooth(s)
    @test length(rim) == 4 && all(c -> length(c.sides) == 3, rim)
    for c in rim
        @test sort(c.angles) ≈ [π / 2, π / 2, π] atol = 1e-10
        @test sort(c.eps) == [1.0, 2.0, 8.0]
        @test c.eps[argmax(c.angles)] == 1.0
    end
    sm = curved_gcq_mesh(s; nlev = 3, n_quad = 8, tol = 1e-12)
    k = findfirst(c -> !c.smooth, s.curves)
    @test all(isapprox.(sm.powers[k], BI._junction_density_powers([π, π / 2, π / 2], [1.0, 2.0, 8.0]; gmax = 20.0); atol = 1e-10))
    @test sum(patch_area(sm, k) for k in 11:15) ≈ π rtol = 1e-10            # the disk
    @test count(p -> p.kind == :edge, sm.interface.panels) == 3 * 4 * 2
    @test all(p -> p.normals[1] == (0.0, 0.0, 1.0), sm.interface.panels[findall(>=(11), sm.panel_patch)])

    # panel offsets near the edge are cancellation-free for a plainly written map: u = 1e-12 vs BigFloat
    p = first(p for p in lm.interface.panels if p.kind == :edge)
    k = lm.panel_patch[findfirst(==(p), lm.interface.panels)]
    pa = lm.panel_param[findfirst(==(p), lm.interface.panels)]            # (u, v) -> (U, V) of the patch
    U, V = pa(1e-12, 0.3); Ua, Va = pa(0.0, 0.0)
    o = BI.curved_lpoint(p, 1e-12, 0.3)
    ob = setprecision(BigFloat, 256) do
        f = l.patches[k].map
        Float64.(f(big(U), big(V)) .- f(big(Ua), big(Va)))
    end
    @test all(isapprox.(o, ob; rtol = 1e-13, atol = 1e-30))
    # the panels' tangents come from the patch map by the chain rule; they agree with differentiating the offsets
    for q in (p, lm.interface.panels[end]), (u, v) in ((1e-9, 0.3), (0.4, 0.8))
        @test q.dmap !== nothing
        fa = BI.curved_frame(q, u, v); fb = BI._curved_frame(q.lmap, q.flip, u, v)
        @test all(isapprox.(fa[3], fb[3]; atol = 1e-12))
        @test fa[4] ≈ fb[4] rtol = 1e-12
    end

    # errors: an unshared side, inconsistent materials, a corner (two graded sides meet)
    @test_throws ArgumentError CurvedGeometry(lens_patches(1.0, θ1, θ2, 4.0, 4.0, 1.0)[1:end-1])
    bad = lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0)
    bad[1] = SurfacePatch(bad[1].map; eps_in = 3.0, eps_out = 1.0, flip = bad[1].flip)
    @test_throws ArgumentError CurvedGeometry(bad)
    cube = CurvedGeometry(cube_patches(4.0, 1.0))
    @test length(cube.curves) == 12 && all(c -> sort(c.angles) ≈ [π / 2, 3π / 2], cube.curves)
    @test_throws ArgumentError curved_gcq_mesh(cube; nlev = 2, n_quad = 6)
end

# every panel's normal is its patch's normal at the panel's parameters, whichever side the edge is on: the split
# sphere with its charts re-parametrized so that the rim lies on side 1, 2, 3 or 4 meshes to the same nodes with
# the same normals and the same junctions
@testset "curved mesh: edge on every patch side" begin
    function onside(P, s)
        f = P.map
        s == 1 && return P
        s == 2 && return SurfacePatch((U, V) -> f(1 - U, V); eps_in = P.eps_in, eps_out = P.eps_out, flip = !P.flip)
        s == 3 && return SurfacePatch((U, V) -> f(V, U); eps_in = P.eps_in, eps_out = P.eps_out, flip = !P.flip)
        return SurfacePatch((U, V) -> f(1 - V, U); eps_in = P.eps_in, eps_out = P.eps_out, flip = P.flip)
    end
    ps = lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0)
    rs = [onside(P, mod1(k, 4)) for (k, P) in enumerate(ps)]
    g0 = CurvedGeometry(ps); g1 = CurvedGeometry(rs)
    @test sort([sort(c.angles) for c in g1.curves if !c.smooth]) ≈ sort([sort(c.angles) for c in g0.curves if !c.smooth])
    m0 = curved_gcq_mesh(g0; nlev = 3, n_quad = 6, tol = 1e-8)
    m1 = curved_gcq_mesh(g1; nlev = 3, n_quad = 6, tol = 1e-8)
    # the patch side each edge panel's u = 0 lies on: all four occur
    side(U, V) = U == 0 ? 1 : U == 1 ? 2 : V == 0 ? 3 : V == 1 ? 4 : 0
    @test Set(side(m1.panel_param[k](0.0, 0.5)...) for k in eachindex(m1.panel_param) if m1.interface.panels[k].kind == :edge) == Set(1:4)
    @test count(p -> p.kind == :edge, m1.interface.panels) == count(p -> p.kind == :edge, m0.interface.panels)
    # each panel normal is the patch normal at (U, V) = panel_param(u, v)
    for (k, p) in enumerate(m1.interface.panels)
        P = g1.patches[m1.panel_patch[k]]
        for (i, (u, v)) in enumerate((u, v) for u in p.ru.x for v in p.rv.x)
            @test all(isapprox.(p.normals[i], patch_frame(P, m1.panel_param[k](u, v)...)[3]; atol = 1e-12))
        end
    end
    # the same nodes, normals and weights as the mesh with every rim on side 1
    pts(m) = reduce(vcat, [p.points for p in m.interface.panels]); nrm(m) = reduce(vcat, [p.normals for p in m.interface.panels])
    X0 = pts(m0); X1 = pts(m1); N0 = nrm(m0); N1 = nrm(m1); w0 = BI.all_weights(m0.interface); w1 = BI.all_weights(m1.interface)
    @test length(X0) == length(X1)
    tree = BI.NearestNeighbors.KDTree(reduce(hcat, collect.(X0)))
    idx, d = BI.NearestNeighbors.nn(tree, reduce(hcat, collect.(X1)))
    @test maximum(d) < 1e-12 && sort(idx) == 1:length(X0)
    @test maximum(norm(collect(N1[i] .- N0[idx[i]])) for i in eachindex(X1)) < 1e-12
    @test maximum(abs.(w1 .- w0[idx]) ./ w0[idx]) < 1e-10
end

@testset "curved mesh: uniform panels across a graded curve" begin
    # across = a splits a graded direction into a equal panels before the nlev dyadic levels toward the curve: nlev = 0
    # is uniform h-refinement with the GCQ rule on the panel of width 1/a at the curve
    g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
    m0 = curved_gcq_mesh(g; nlev = 2, n_quad = 6, tol = 1e-6)
    m1 = curved_gcq_mesh(g; nlev = 2, n_quad = 6, tol = 1e-6, across = 1)
    @test BI.num_points(m1) == BI.num_points(m0)                           # across = 1: the default mesh
    @test BI._graded_breaks(true, false, 0, 2, 3) ≈ [0, 1/3, 2/3, 1]
    @test BI._graded_breaks(true, false, 2, 2, 2) ≈ [0, 1/8, 1/4, 1/2, 1]
    @test BI._graded_breaks(false, true, 1, 2, 2) ≈ [0, 1/2, 3/4, 1]
    @test BI._graded_breaks(true, true, 1, 2, 3) ≈ [0, 1/6, 1/3, 2/3, 5/6, 1]
    for a in (1, 3)
        m = curved_gcq_mesh(g; nlev = 0, n_quad = 6, tol = 1e-6, across = a)
        # every patch with a graded side: a panels across it, the one at the curve with the GCQ rule
        for k in 1:4                                                    # the upper cap's side charts
            ks = findall(==(k), m.panel_patch)
            edges = [j for j in ks if m.interface.panels[j].kind == :edge]
            width(j) = norm(collect(m.panel_param[j](1.0, 0.5) .- m.panel_param[j](0.0, 0.5)))
            @test length(ks) == 2a && length(edges) == 2                      # a across × m = 2 along
            @test all(j -> abs(width(j) - 1 / a) < 1e-12, ks)
        end
    end
end

@testset "curved geometry: edge of varying angle (two cylinders)" begin
    r, z1, z2 = 1.0, 0.0, 0.8
    ps = cylinders_patches(r, z1, z2, 4.0, 1.0)
    @test_throws ArgumentError CurvedGeometry(ps)                 # the angle varies: refused by default
    g = CurvedGeometry(ps; varying = true)
    edges = [i for (i, c) in enumerate(g.curves) if !c.smooth]
    @test length(g.curves) == 20 && length(edges) == 4
    # interior angle π - acos((z - z1)(z - z2)/r²) along every edge curve; the body (ε = 4) is the smaller sector
    for i in edges, t in (0.0, 0.3, 0.5, 1.0)
        ang, eps = junction_angles(g, i, t)
        k, s = g.curves[i].sides[1]
        z = BI._side_point(g.patches[k], s, t)[3]
        α = π - acos((z - z1) * (z - z2) / r^2)
        @test eps[argmin(ang)] == 4.0 && eps[argmax(ang)] == 1.0
        @test minimum(ang) ≈ α atol = 1e-12
    end
    @test junction_angles(g, edges[1], 0.5)[1] ≈ g.curves[edges[1]].angles
    # one universal rule on every edge panel; the GL-only mesh integrates area and volume of the body
    ru = universal_gcq_rule(-0.21; qmax = 8.0, tol = 1e-8)
    m = curved_gcq_mesh(g; nlev = 1, n_quad = 6, m = 2, edge_rule = ru)
    eps_panels = [p for p in m.interface.panels if p.kind == :edge]
    @test length(eps_panels) == 8 * 2 && all(p -> p.ru.x ≈ BI.gcq_tensor_rule(ru).x, eps_panels)
    @test curved_gcq_mesh(g; nlev = 1, n_quad = 6, m = 2, edge_rule = c -> ru).interface.panels[1].ru.x ≈ eps_panels[1].ru.x
    a(z) = sqrt(max(0.0, r^2 - (z - z2)^2)); b(z) = sqrt(max(0.0, r^2 - (z - z1)^2))
    t, w = BI.gausslegendre(200)                                  # V = ∫ 4ab dz, ab ~ sqrt at both ends: z = mid + half sin(πτ/2)
    zl, zh = z2 - r, z1 + r
    V = sum(w[j] * 4a(zm) * b(zm) * (zh - zl) / 2 * π / 2 * cos(π * t[j] / 2)
            for j in eachindex(t) for zm in ((zl + zh) / 2 + (zh - zl) / 2 * sin(π * t[j] / 2),))
    mg = curved_gcq_mesh(g; nlev = 0, across = 2, m = 2, n_quad = 12, edge = :gl)
    X = reduce(vcat, [p.points for p in mg.interface.panels]); nr = reduce(vcat, [p.normals for p in mg.interface.panels])
    @test sum(BI.all_weights(mg.interface) .* [dot(collect(x), collect(n)) for (x, n) in zip(X, nr)]) / 3 ≈ V rtol = 1e-8
end
