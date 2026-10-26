# SCTL near and on-surface quadrature for curved GCQ panels (lens_gcq), and solves on curved bodies.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays
using Test

const HAVE_SCTL_CURVED = Base.find_package("SCTL") !== nothing
HAVE_SCTL_CURVED && @eval using SCTL

if HAVE_SCTL_CURVED
    # Dᵀ alone: FMM far field plus the near correction
    function dt_operator(itf, q)
        F = BI.laplace3d_DT_fmm3d(itf, 1e-12)
        C = BlockCorrection(near_corrections(q, itf), itf)
        return x -> F * x .+ C * x
    end
    nodes_of(itf) = reduce(vcat, [p.points for p in itf.panels])

    @testset "SCTLCurvedNearQuad: Dᵀ on the unit sphere" begin
        # θ1 = θ2 = π/2, one ε: the unit sphere, with GCQ edge panels at the equator
        m = lens_gcq(1.0, π / 2, π / 2, 4.0, 4.0, 1.0; nlev = 3, n_quad = 8, tol = 1e-8)
        itf = m.interface
        X = nodes_of(itf); z = [x[3] for x in X]
        isedge = reduce(vcat, [fill(p.kind == :edge, BI.num_points(p)) for p in itf.panels])
        Dt = dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-10))
        # Dᵀ Y_l = Y_l / (2(2l + 1)) on the unit sphere (DielectricBIE's sign)
        e0 = Dt(ones(length(X))) .- 0.5
        @test maximum(abs, e0) < 1e-7
        @test maximum(abs, e0[isedge]) < 1e-7                         # GCQ edge-panel nodes included
        @test maximum(abs, Dt(z) .- z ./ 6) < 1e-7
        y2 = (3 .* z .^ 2 .- 1) ./ 2
        @test maximum(abs, Dt(y2) .- y2 ./ 10) < 1e-7
        # the panel-level near criterion is needed: without it, near neighbours get the node rule
        D0 = dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-10, η = 0.0))
        @test maximum(abs, D0(z) .- z ./ 6) > 1e-5
    end
end
