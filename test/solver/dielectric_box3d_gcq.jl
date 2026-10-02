using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

@testset "dielectric box3d with GCQ edge panels" begin
    ε = 10.0
    r = edge_gcq_rule(ε, 1.0; tol = 1e-4)
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = 4, rule = r)
    b = rhs_dielectric_box3d(itf, PointSource((0.1, 0.15, 0.2), 1.0), ε)
    # FMM-corrected operator agrees with the dense corrected matrix
    A, st = lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-12)
    Ad = lhs_dielectric_box3d_gcq_dense(itf)
    x = randn(length(b))
    @test norm(A * x - Ad * x) / norm(Ad * x) < 1e-10
    @test st.unique < st.pairs                       # near rows are reused
    # reuse does not change the corrections
    C1, _ = laplace3d_DT_tensor_corrections(itf; reuse = true)
    C2, _ = laplace3d_DT_tensor_corrections(itf; reuse = false)
    @test maximum(abs.(C1 - C2)) < 1e-12
    # Gauss: ∫σ = 1 - 1/ε for a unit charge inside
    σ, ks = solve_dielectric_box3d_gcq(A, b, itf; rtol = 1e-12)
    @test abs(dot(BI.all_weights(itf), σ) - (1 - 1 / ε)) < 1e-4
end
