using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
Base.find_package("SCTL") === nothing || @eval using SCTL

@testset "box node orbits" begin
    r = edge_gcq_rule(Inf, 1.0; tol = 1e-6)
    for (L, order) in (((1.0, 1.0, 1.0), 48), ((1.0, 1.0, 2.0), 16), ((1.0, 0.8, 0.6), 8))
        itf = single_dielectric_box3d_gcq(L..., Inf, 1.0; nlev = 1, n_quad = 4, rule = r)
        orbit, reps = box_node_orbits(itf, L...)
        sz = [count(==(o), orbit) for o in eachindex(reps)]
        @test all(s -> order % s == 0, sz) && maximum(sz) == order
        @test orbit[reps] == eachindex(reps)
        @test sum(sz) == DB.num_points(itf)
    end
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, Inf, 1.0; nlev = 1, n_quad = 4, rule = r)
    @test_throws ArgumentError box_node_orbits(itf, 1.0, 1.0, 1.0; center = (0.1, 0.0, 0.0))
end

@testset "symmetric reduction matches the full dense system" begin
    r = edge_gcq_rule(Inf, 1.0; tol = 1e-6)
    for L in ((1.0, 1.0, 1.0), (1.0, 0.8, 0.6))
        itf = single_dielectric_box3d_gcq(L..., Inf, 1.0; nlev = 1, n_quad = 3, rule = r)
        w = DB.all_weights(itf); area = sum(w)
        A = lhs_dielectric_box3d_gcq_dense(itf)
        σf = (A .- (0.5 / area) .* ones(length(w)) * w') \ fill(-0.5 / area, length(w))
        orbit, reps = box_node_orbits(itf, L...)
        W = [sum(w[orbit .== o]) for o in eachindex(reps)]
        M = symmetric_dense_operator(itf, orbit, reps)
        s = (M .- (0.5 / area) .* W') \ fill(-0.5 / area, length(reps))
        @test norm(s[orbit] .- σf, Inf) < 1e-10 * norm(σf, Inf)
        @test M ≈ [sum(A[reps[o], orbit .== o′]) for o in eachindex(reps), o′ in eachindex(reps)] rtol = 1e-13
    end
end

@testset "capacitance of the unit cube" begin
    C_HP = 0.66067815409957                # Helsing & Perfekt (2013), relative error ~1e-13
    c = box_capacitance_gcq(1.0, 1.0, 1.0; nlev = 2, n_quad = 8, tol = 1e-12, quad_tol = 1e-12,
                            targets = [(0.0, 0.0, 0.0), (0.1, 0.2, 0.3)])
    @test abs(c.C / C_HP - 1) < 1e-10      # the Thomson energy: second order in the error of σ
    @test all(abs.(c.C_point ./ C_HP .- 1) .< 1e-8)
    @test abs(c.Q - 1) < 1e-8
    # C scales with the size: the same mesh, scaled
    c2 = box_capacitance_gcq(2.0, 2.0, 2.0; nlev = 2, n_quad = 8, tol = 1e-12, quad_tol = 1e-12)
    @test c2.C ≈ 2 * c.C rtol = 1e-12
end
