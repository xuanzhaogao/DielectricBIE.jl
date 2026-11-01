# The pure-Julia curved-geometry helpers of GCQ panels (src/kernel/curved_geometry_ops.jl): differences of panel points
# from the tangents, the closest point, Chebyshev series along u-lines. No SCTL.
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

    # the Chebyshev series along the u-line u = 0.3: y(u, v0) - y(u, v) for v0 = 0.5, v = 0.45
    G = BI._curved_vgeom(P, 0.3)
    @test size(G) == (BI._CHEB_N + 1, 4)
    c = BI._cheb_diff(G, 2 * 0.5 - 1, 2 * 0.45 - 1)
    @test maximum(abs.(c[1:3] .- (BI.curved_lpoint(P, 0.3, 0.5) .- BI.curved_lpoint(P, 0.3, 0.45)))) <= 1e-14
end
