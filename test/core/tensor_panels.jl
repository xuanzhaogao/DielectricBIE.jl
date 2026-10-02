using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

@testset "tensor panels" begin
    r = edge_gcq_rule(10.0, 1.0; tol = 1e-8)
    itf = single_dielectric_box3d_gcq(1.0, 2.0, 1.5, 10.0, 1.0; nlev = 2, n_quad = 5, rule = r)
    w = BI.all_weights(itf)
    @test sum(w) ≈ 2 * (1.0 * 2.0 + 2.0 * 1.5 + 1.0 * 1.5) atol = 1e-10
    @test BI.num_points(itf) == length(w)
    # every point lies on its face plane and inside the box
    for p in itf.panels, pt in p.points
        @test abs(dot(pt, p.normal)) ≈ abs(dot(p.corners[1], p.normal)) atol = 1e-14
        @test all(abs.(pt) .<= (0.5, 1.0, 0.75) .+ 1e-14)
    end
    kinds = Set(p.kind for p in itf.panels)
    @test kinds == Set([:edge, :gl, :corner])
    # eachpoint works through the interface iterator (used by FMM and rhs)
    @test length(collect(BI.eachpoint(itf))) == length(w)
end
