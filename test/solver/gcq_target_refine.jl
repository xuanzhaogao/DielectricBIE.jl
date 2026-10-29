using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, Random
using Test

@testset "GCQ: target-driven refinement" begin
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0), (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    src = BI.PointSource((0.3, 0.1, 0.8), 1.0)
    rhs(p, n) = -BI.laplace3d_grad(src.point, p, n)
    # targets inside box 1 under its top face and next to the shared face, and above the top face
    rng = MersenneTwister(7)
    X = hcat([0.1, -0.2, 0.3] .+ [0.3, 0.4, 0.1] .* rand(rng, 3, 40),
             [0.2, -0.2, 0.6] .+ [0.6, 0.4, 0.1] .* rand(rng, 3, 20))
    kw = (; nlev = 1, n_quad = 4, tol = 1e-6, rhs, rhs_atol = 1e-5)
    isfar(p, t) = BI.tensor_isfar(p, (X[1, t], X[2, t], X[3, t]) .- p.anchor; quad_tol = 1e-8, gcq_far = :direction)
    m0 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; kw...)
    m1 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; kw..., targets = X, target_tol = 1e-8)
    p0, p1 = m0.interface.panels, m1.interface.panels
    @test length(p1) > length(p0)
    @test sum(BI.all_weights(m1.interface)) ≈ sum(BI.all_weights(m0.interface)) rtol = 1e-10
    # every target is far from every panel, so GCQPotential is the plain FMM
    @test all(isfar(p, t) for p in p1, t in axes(X, 2))
    @test !all(isfar(p, t) for p in p0, t in axes(X, 2))
    ev = GCQPotential(quad_tol = 1e-8, gcq_far = :direction, far = FMM3DFarField(1e-13))
    σ = randn(rng, BI.num_points(m1.interface))
    @test potential_operator(ev, m1.interface, X) * σ ≈ BI.laplace3d_pottrg_fmm3d(m1.interface, X, 1e-13) * σ rtol = 1e-12
    # a smallest side stops the refinement
    m2 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; kw..., targets = X, target_tol = 1e-8, target_min_size = 0.1)
    @test length(p0) < length(m2.interface.panels) < length(p1)
    side(ps) = minimum(p -> min(p.Lu, p.Lv), ps)
    @test side(m2.interface.panels) >= min(side(p0), 0.05) - 1e-12

    # one box: the potential of the solution is the same from the refined and the unrefined mesh
    b1 = boxes[1:1]
    kw1 = (; nlev = 1, n_quad = 4, tol = 1e-4, rhs, rhs_atol = 1e-3)
    n0 = multi_dielectric_box3d_gcq(b1, [4.0], 1.0; kw1...)
    n1 = multi_dielectric_box3d_gcq(b1, [4.0], 1.0; kw1..., targets = X[:, 1:40], target_tol = 1e-6)
    u = map((n0, n1)) do m
        s = solve_dielectric_boxes_gcq(m, src; quad_tol = 1e-8, gcq_far = :direction, rtol = 1e-8)
        potential_operator(ev, m.interface, X[:, 1:40]) * s.σ
    end
    @test norm(u[2] - u[1]) / norm(u[1]) < 1e-4
end

@testset "GCQ: target-driven refinement for many targets (target_skin)" begin
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0), (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    rng = MersenneTwister(5)
    X = [0.1, -0.2, 0.3] .+ [0.3, 0.4, 0.12] .* rand(rng, 3, 8000)          # a dense cloud under the top face
    kw = (; nlev = 1, n_quad = 4, tol = 1e-4, targets = X, target_tol = 1e-8)
    isfar(p, t) = BI.tensor_isfar(p, (X[1, t], X[2, t], X[3, t]) .- p.anchor; quad_tol = 1e-8, gcq_far = :direction)
    nnear(ps) = sum(fetch, [Threads.@spawn count(t -> !isfar(p, t), axes(X, 2)) for p in ps])
    m0 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; nlev = 1, n_quad = 4, tol = 1e-4)
    m1 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; kw...)
    m2 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; kw..., target_skin = 0.02)
    # the skin is a small subset; the mesh is nearly the same, and nearly no target is left near
    sk = BI._target_skin(X, BI._panel_planes(m0.interface.panels), 0.02)
    @test size(sk, 2) < size(X, 2) / 5
    @test all(any(sk[:, j] == X[:, t] for t in axes(X, 2)) for j in 1:25:size(sk, 2))
    @test abs(length(m2.interface.panels) - length(m1.interface.panels)) <= length(m1.interface.panels) ÷ 100
    @test nnear(m1.interface.panels) == 0
    @test nnear(m2.interface.panels) <= 1e-4 * nnear(m0.interface.panels)
    # the pairs left near are corrected by GCQPotential
    p2 = m2.interface.panels; σ = randn(rng, BI.num_points(m2.interface))
    T = X[:, 1:40:end]
    uref = BI.potential_tensor(m2.interface, σ, [Tuple(T[:, t]) for t in axes(T, 2)]; quad_tol = 1e-8, gcq_far = :direction)
    u = potential_operator(GCQPotential(quad_tol = 1e-8, gcq_far = :direction, far = FMM3DFarField(1e-13)), m2.interface, T) * σ
    @test maximum(abs, u - uref) / maximum(abs, uref) < 1e-10
end
