# The pure-Julia curved-geometry helpers of GCQ panels (src/kernel/curved_geometry_ops.jl): differences of panel points
# from the tangents, the closest point. No SCTL.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

@testset "curved geometry ops" begin
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    P = first(p for p in m.interface.panels if p.kind == :edge)

    # y(b) - y(a) by Gauss along the parameter segment of the tangents
    d = BI._curved_diff(P, (0.2, 0.3), (0.7, 0.9))
    @test d isa NTuple{3, Float64}
    @test maximum(abs.(d .- (BI.curved_lpoint(P, 0.7, 0.9) .- BI.curved_lpoint(P, 0.2, 0.3)))) <= 1e-14

    # the closest point to x = y(0.4, 0.6) + 1e-3 n
    _, _, n, _ = BI.curved_frame(P, 0.4, 0.6)
    x = BI.curved_lpoint(P, 0.4, 0.6) .+ 1e-3 .* n
    u, v, dist = BI._curved_closest(P, x)
    @test isapprox(u, 0.4; atol = 1e-10) && isapprox(v, 0.6; atol = 1e-10)
    @test isapprox(dist, 1e-3; rtol = 1e-8)
end
