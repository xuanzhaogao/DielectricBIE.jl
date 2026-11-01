# The old entry points kept for scripts (src/deprecated.jl): thin wrappers over dielectric_operator,
# sctl_gl_corrections, SCTLPotential and GCQNearQuad. The in-house upsampling, adaptive-quadtree and HCubature near
# code they used to call is gone, and so is the SCTL path of SCTLCurvedNearQuad's GCQ panels.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, Random
using SCTL
using Test

@testset "wrappers kept for old scripts" begin
    rng = MersenneTwister(5)
    itf = single_dielectric_box3d(1.0, 1.0, 1.0, 8, 0.5, 4.0, 1.0, Float64)
    N = BI.num_points(itf)
    x = randn(rng, N)
    X = randn(rng, N, 3)
    A = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = SCTLNearCorrection(tol = 1e-10))
    Ax = A * x

    # the operator: dielectric_operator with SCTL's near correction; the old keywords are accepted and ignored
    @test lhs_dielectric_box3d_fmm3d_corrected(itf, 1e-12, 1e-10, 8; correct_edges = true) * x ≈ Ax rtol = 1e-12
    @test lhs_dielectric_box3d_fmm3d_corrected(itf, 1e-12, 1e-10, 8; correct_edges = false, adaptive_atol = 1e-6,
                                               adaptive_n_GL = 4) * x ≈ Ax rtol = 1e-12
    @test Lhs_dielectric_box3d_fmm3d_corrected === lhs_dielectric_box3d_fmm3d_corrected
    @test batched_lhs_dielectric_box3d_fmm3d_corrected(itf, 1e-12, 1e-10, 8) * X ≈ A * X rtol = 1e-12

    # Dᵀ and D: the FMM node rule plus SCTL's corrections, N × N maps
    DT = laplace3d_DT_fmm3d_corrected(itf, 1e-12, 1e-10, 8; correct_edges = false)
    DTh = laplace3d_DT_fmm3d_corrected_hcubature(itf, 1e-12, 1e-10; correct_edges = true)
    D = laplace3d_D_fmm3d_corrected(itf, 1e-12, 1e-10, 8)
    @test size(DT) == size(DTh) == size(D) == (N, N)
    τ = dielectric_tau(4.0, 1.0)
    @test DT * x .+ τ .* x ≈ Ax rtol = 1e-12
    @test DTh * x ≈ DT * x rtol = 1e-14
    # Gauss: D 1 = 1/2 at every node of a closed surface
    @test maximum(abs, D * ones(N) .- 0.5) < 1e-9

    # the off-surface single layer: SCTLPotential
    T = [0.1 0.6 -0.2; 0.2 0.0 0.3; 0.9 0.1 0.5005]
    P = potential_operator(SCTLPotential(tol = 1e-10), itf, T)
    @test laplace3d_pottrg_near(itf, (0.1, 0.2, 0.9), x, 1e-10) ≈
          (potential_operator(SCTLPotential(tol = 1e-10), itf, [0.1; 0.2; 0.9;;]) * x)[1]
    @test laplace3d_pottrg_near(itf, (0.1, 0.2, 0.9), x, 1e-10; range_factor = 2.0) ≈ (P * x)[1]
    @test laplace3d_pottrg_fmm3d_corrected_hcubature(itf, T, 1e-12, 1e-10, 5.0; correct_edges = false) * x ≈ P * x

    # the curved near quadrature: GCQNearQuad(quad_tol = tol), its other keywords ignored (@test_deprecated checks the
    # warning under --depwarn=yes, and only runs the call otherwise)
    @test_deprecated r"GCQNearQuad" SCTLCurvedNearQuad(tol = 1e-10)
    @test SCTLCurvedNearQuad(tol = 1e-8, order = 16, η = 3.0, ε_rel = 1e-6) === GCQNearQuad(quad_tol = 1e-8)
    @test SCTLCurvedNearQuad() === GCQNearQuad(quad_tol = 1e-10)               # its tol's default
    # (one GCQ and one GL panel of a curved sphere: 8 s per call at -t 2, the whole sphere 51 s)
    sph = BI._precompile_sphere_mesh().interface
    sub = [findfirst(!is_gl_panel, sph.panels), findfirst(is_gl_panel, sph.panels)]
    two = DielectricInterface(sph.panels[sub], sph.eps_in[sub], sph.eps_out[sub])
    @test near_corrections(SCTLCurvedNearQuad(tol = 1e-10), two) == near_corrections(GCQNearQuad(quad_tol = 1e-10), two)

    # the in-house near code and its entry points without a wrapper are gone
    for name in (:UpsampledNearQuad, :HCubaturePotential, :SCTLProxyNearQuad, :build_neighbor_list,
                 :build_target_neighbor_list, :AdaptiveConfig, :laplace3d_DT_corrections,
                 :laplace3d_pottrg_corrections_hcubature, :_refine_interface_for_targets,
                 :_refined_interface_prolongation)
        @test !isdefined(DielectricBIE, name)
    end
end
