# SCTLFarField's close node pairs (through ParticleFMM with SCTL.has_feature(:EXACT_NEAR), else summed directly instead
# of by PVFMM) and the default far field with SCTL loaded.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using SCTL
using Test

include(joinpath(@__DIR__, "..", "shape", "cylinders_patches.jl"))

@testset "SCTL far field: close node pairs" begin
    exact_near = SCTL.has_feature(:EXACT_NEAR)
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
    if exact_near
        # ParticleFMM sums PVFMM's near field from the original coordinates: the pair to rounding, no close nodes
        for F in (F0, F1)
            Y = zeros(N, 1); BI.apply_far!(Y, F, e)
            @test Y[i] ≈ exact rtol = 1e-13
            @test isempty(F.close)
        end
    else
        # PVFMM's own sum, in its rounded unit-box coordinates, is off; the default sums the close nodes directly
        for (F, ok) in ((F0, false), (F1, true))
            Y = zeros(N, 1); BI.apply_far!(Y, F, e)
            ok ? (@test Y[i] ≈ exact rtol = 1e-13) : (@test abs(Y[i] / exact - 1) > 1e-8)
            ok && @test i in F.close && j in F.close && length(F.close) < N ÷ 4
        end
    end
    # away from the close nodes PVFMM with and without them agree on a generic density (the same operators)
    x = randn(N, 1)
    Y0 = zeros(N, 1); BI.apply_far!(Y0, F0, x); Y1 = zeros(N, 1); BI.apply_far!(Y1, F1, x)
    far = setdiff(1:N, F1.close)
    @test maximum(abs.(Y0[far] .- Y1[far])) <= 1e-9 * maximum(abs.(Y1[far]))
end

@testset "planar_slp_nodes with an SCTL far field" begin
    # the node rule by ParticleFMM (here below 40000 nodes: SCTL's direct sum) as by direct summation, the coincident
    # node skipped by both
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F0 = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                       for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, orient_outward(V, F0)[1]; eps_in = Inf)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    σ = 1 .+ 0.3 .* sin.(1:BI.num_points(itf))
    u = planar_slp_nodes(itf, σ)
    @test planar_slp_nodes(itf, σ; far = SCTLFarField(tol = 1e-12)) ≈ u rtol = 1e-13
    @test_throws ArgumentError planar_slp_nodes(itf, σ; far = :fmm)
end
