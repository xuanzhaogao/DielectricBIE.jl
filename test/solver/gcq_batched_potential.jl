using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using Test

@testset "GCQ: batched RHS refinement and GCQPotential" begin
    # batched refinement gives the same panels as the pointwise one
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0), (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    src = BI.PointSource((0.3, 0.1, 0.62), 1.0)
    rhs(p, n) = -BI.laplace3d_grad(src.point, p, n)
    m1 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; nlev = 1, n_quad = 4, rhs = rhs, rhs_atol = 1e-5)
    nb = Ref(0)
    rhsb(P, N) = (nb[] += 1; [rhs(p, n) for (p, n) in zip(P, N)])
    m2 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; nlev = 1, n_quad = 4, rhs_batch = rhsb, rhs_atol = 1e-5)
    key(p) = (round.(p.anchor .+ p.o, digits = 12), round(p.Lu, digits = 12), round(p.Lv, digits = 12), p.kind)
    @test sort(key.(m1.interface.panels)) == sort(key.(m2.interface.panels))
    @test length(m1.interface.panels) > 96             # it refined
    @test nb[] < 30                                    # one call per level

    # GCQPotential == potential_tensor (direct), on and near edges and corners
    ε = 10.0
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = 4, rule = edge_gcq_rule(ε, 1.0; tol = 1e-6))
    X, Nn = BI._interface_targets_normals(itf)
    rng = MersenneTwister(4)
    σ = randn(rng, BI.num_points(itf))
    T = hcat(rand(rng, 3, 40) .* 2 .- 1, [0.5 + 1e-3, 0.5 + 2e-4, 0.3], [0.501, 0.499, 0.4999], X[:, 1:97:end] .+ 1e-3 .* Nn[:, 1:97:end])
    uref = BI.potential_tensor(itf, σ, [Tuple(T[:, t]) for t in axes(T, 2)])
    P = potential_operator(GCQPotential(far = FMM3DFarField(1e-13)), itf, T)
    @test size(P) == (size(T, 2), BI.num_points(itf))
    @test maximum(abs, P * σ - uref) / maximum(abs, uref) < 1e-11

    # gcq_far = :direction: the near distance of edge panels against their GCQ direction; same results at the tolerance
    Pd = potential_operator(GCQPotential(far = FMM3DFarField(1e-13), gcq_far = :direction), itf, T)
    @test maximum(abs, Pd * σ - uref) / maximum(abs, uref) < 1e-9
    ud = BI.potential_tensor(itf, σ, [Tuple(T[:, t]) for t in axes(T, 2)]; gcq_far = :direction)
    @test maximum(abs, ud - uref) / maximum(abs, uref) < 1e-9
    Cm = BI.near_corrections(GCQNearQuad(), itf)
    Cd = BI.near_corrections(GCQNearQuad(gcq_far = :direction), itf)
    @test nnz(Cd) <= nnz(Cm)
    A = BI.laplace3d_DT(itf)                       # node rule, dense
    @test norm((A + Cd) * σ - (A + Cm) * σ) / norm((A + Cm) * σ) < 1e-9
    L1 = first(BI.lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-13))
    L2 = first(BI.lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-13; gcq_far = :direction))
    @test norm(L2 * σ - L1 * σ) / norm(L1 * σ) < 1e-9
    @test_throws ArgumentError BI.tensor_isfar(itf.panels[1], (5.0, 5.0, 5.0); gcq_far = :nearest)
end
