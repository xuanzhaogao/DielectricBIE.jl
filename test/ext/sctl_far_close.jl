# SCTLFarField's close node pairs (summed directly instead of by PVFMM) and the default far field with SCTL loaded.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

const HAVE_SCTL_FAR = Base.find_package("SCTL") !== nothing
HAVE_SCTL_FAR && @eval using SCTL
include(joinpath(@__DIR__, "..", "shape", "cylinders_patches.jl"))

if HAVE_SCTL_FAR
    @testset "SCTL far field: close node pairs summed directly" begin
        @test default_far_field() isa SCTLFarField && default_far_field().close === nothing
        @test SCTLFarField(1e-9).tol == 1e-9
        g = CurvedGeometry(cylinders_patches(1.0, 0.0, 0.8, 4.0, 1.0); varying = true)
        # above 40000 nodes (41664), so that PVFMM runs (below it ParticleFMM is a direct sum)
        m = curved_gcq_mesh(g; nlev = 0, across = 7, m = 7, n_quad = 8, tol = 1e-12,
                            edge_rule = universal_gcq_rule(-0.21; qmax = 20.0, tol = 1e-12))
        itf = m.interface; N = BI.num_points(itf)
        @test N > 40000
        X, Nr = BI._interface_targets_normals(itf); w = BI.all_weights(itf)
        idx, d = BI.knn(BI.KDTree(X), X, 2, true)
        i = argmin([d[k][2] for k in 1:N]); j = idx[i][2]
        dk = X[:, i] .- X[:, j]
        exact = w[j] * dot(Nr[:, i], dk) / (4π * norm(dk)^3)
        e = zeros(N, 1); e[j] = 1.0
        F0 = BI.far_field_operator(SCTLFarField(tol = 1e-12, close = 0.0), itf); F1 = BI.far_field_operator(SCTLFarField(tol = 1e-12), itf)
        for (F, ok) in ((F0, false), (F1, true))
            Y = zeros(N, 1); BI.apply_far!(Y, F, e)
            ok ? (@test Y[i] ≈ exact rtol = 1e-13) : (@test abs(Y[i] / exact - 1) > 1e-8)
            ok && @test i in F.close && j in F.close && length(F.close) < N ÷ 4
        end
        # away from the close nodes PVFMM with and without them agree on a generic density (the same operators)
        x = randn(N, 1)
        Y0 = zeros(N, 1); BI.apply_far!(Y0, F0, x); Y1 = zeros(N, 1); BI.apply_far!(Y1, F1, x)
        far = setdiff(1:N, F1.close)
        @test maximum(abs.(Y0[far] .- Y1[far])) <= 1e-9 * maximum(abs.(Y1[far]))
    end
end
