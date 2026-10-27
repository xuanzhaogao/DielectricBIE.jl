using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "lens_patches.jl"))

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
