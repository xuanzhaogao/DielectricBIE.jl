# The sinh-mapped polar (Duffy) rule on a parameter rectangle (src/kernel/duffy.jl): a target at c inside the rectangle,
# the single layer 1/(4π r) against the closed form of a flat rectangle. No SCTL.
using DielectricBIE
import DielectricBIE as BI
using FastGaussQuadrature: gausslegendre
using Test

# ∫_0^a ∫_0^b dx dy / sqrt(x² + y²) (0 when a rectangle is empty), and the single-layer potential of the rectangle
# [0, Lx] × [0, Ly] (σ = 1) at the in-plane point (x0, y0): the four corner rectangles of the target
duffy_Fq(a, b) = (a == 0 || b == 0) ? 0.0 : a * asinh(b / a) + b * asinh(a / b)
duffy_slp_exact(Lx, Ly, x0, y0) = sum(duffy_Fq(a, b) for a in (x0, Lx - x0), b in (y0, Ly - y0)) / 4π

@testset "duffy_rule: flat single layer, closed form" begin
    z14, _ = gausslegendre(14)
    node(k) = (1 + z14[k]) / 2
    # (label, Lx, Ly, x0, y0): the unit square as the Duffy square, targets at 14-point GL positions in u (centred in
    # v), the corner (node 1, node 1), the square clipped to a 1 × 0.52 rectangle, and a target 1e-4 from the u = 0 side
    targets = [("u = node 1, v = 1/2", 1.0, 1.0, node(1), 0.5),
               ("u = node 2, v = 1/2", 1.0, 1.0, node(2), 0.5),
               ("u = node 7, v = 1/2", 1.0, 1.0, node(7), 0.5),
               ("corner: u = v = node 1", 1.0, 1.0, node(1), node(1)),
               ("clipped 1 x 0.52: u = node 1, v = 0.02", 1.0, 0.52, node(1), 0.02),
               ("u = 1e-4, v = 1/2", 1.0, 1.0, 1e-4, 0.5)]
    # measured relative errors for (nρ, nside) = (8, 8) / (16, 32) / (32, 32), at the targets in order (all at rounding,
    # the bound 1e-13 is the spec's):
    #   node 1:   3.0e-15 / 2.8e-15 / 1.3e-15      node 2:  1.3e-16 / 5.9e-15 / 3.9e-15
    #   node 7:   1.4e-15 / 2.0e-16 / 4.0e-16      corner:  1.9e-15 / 4.3e-15 / 3.4e-15
    #   clipped:  1.3e-16 / 2.5e-15 / 4.8e-15      1e-4:    1.2e-14 / 2.5e-15 / 4.3e-16
    for (label, Lx, Ly, x0, y0) in targets, (nρ, nside) in ((8, 8), (16, 32), (32, 32))
        du, dv, w = BI.duffy_rule((x0, y0), (0.0, Lx, 0.0, Ly), (1.0, 0.0, 1.0); nρ, nside)
        @test length(du) == length(dv) == length(w)
        val = sum(w[q] / (4π * hypot(du[q], dv[q])) for q in eachindex(w))
        ex = duffy_slp_exact(Lx, Ly, x0, y0)
        err = abs(val - ex) / abs(ex)
        @test err <= 1e-13
    end

    # the defaults are the constants
    @test BI.DUFFY_NRHO == 16 && BI.DUFFY_NSIDE == 32
    du, dv, w = BI.duffy_rule((0.3, 0.5), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    @test length(w) == 4 * BI.DUFFY_NRHO * BI.DUFFY_NSIDE
    du2, dv2, w2 = BI.duffy_rule((0.3, 0.5), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0); nρ = 16, nside = 32)
    @test (du, dv, w) == (du2, dv2, w2)
    # the weights are the parameter measure: Σ w = area of the rectangle (measured: 2.2e-16 off)
    @test sum(w) ≈ 1.0 atol = 1e-14
    # the points lie in the rectangle
    @test all(0.0 .<= 0.3 .+ du .<= 1.0) && all(0.0 .<= 0.5 .+ dv .<= 1.0)

    # a target on the boundary: the side through it is degenerate and contributes nothing, the other three still integrate
    du, dv, w = BI.duffy_rule((0.0, 0.5), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    @test length(w) == 3 * BI.DUFFY_NRHO * BI.DUFFY_NSIDE
    val = sum(w[q] / (4π * hypot(du[q], dv[q])) for q in eachindex(w))
    @test isapprox(val, duffy_slp_exact(1.0, 1.0, 0.0, 0.5); rtol = 1e-13)    # measured relative error 2.5e-15
end

@testset "duffy_rule: anisotropic first fundamental form" begin
    # G = (4, 0, 1): the unit parameter square is the physical rectangle [0, 2] × [0, 1] (x = 2u, y = v), r =
    # sqrt(4 du² + dv²), and ∫ f dA = 2 ∫ f du dv, so Σ w / (4π r) is the physical closed form divided by √det G = 2.
    # Measured relative errors at (u0, v0) = (0.0069, 1/2), (0.3, 0.3), (1e-4, 1/2), (1/2, 1/2): 6.5e-16, 2.9e-15, 1.6e-15, 3.3e-15
    for (u0, v0) in ((0.0069, 0.5), (0.3, 0.3), (1e-4, 0.5), (0.5, 0.5))
        du, dv, w = BI.duffy_rule((u0, v0), (0.0, 1.0, 0.0, 1.0), (4.0, 0.0, 1.0))
        val = sum(w[q] / (4π * sqrt(4 * du[q]^2 + dv[q]^2)) for q in eachindex(w))
        ex = duffy_slp_exact(2.0, 1.0, 2u0, v0) / 2
        err = abs(val - ex) / abs(ex)
        @test err <= 1e-13
    end
end

@testset "duffy_rule: sheared first fundamental form (F ≠ 0)" begin
    # x = (u + 0.7 v, 0.8 v), G = (1, 0.7, 1.13): no closed form, so the default rule is compared with a 48 × 96 rule of the
    # same G. Measured relative differences 2.4e-14 (16, 32) and 1.8e-14 (8, 8); with the isotropic G in place of the
    # sheared one, the 48 × 96 rule is 1.5e-11 off, so the foot and distance in G do matter. The bound is the rounding floor.
    G = (1.0, 0.7, 1.13)
    f(du, dv) = 1 / sqrt(G[1] * du^2 + 2G[2] * du * dv + G[3] * dv^2)
    integral(; kw...) = ((du, dv, w) = BI.duffy_rule((1e-3, 0.5), (0.0, 1.0, 0.0, 1.0), G; kw...);
                         sum(w[q] * f(du[q], dv[q]) for q in eachindex(w)))
    ref = integral(nρ = 48, nside = 96)
    @test abs(integral() - ref) / ref <= 1e-13
    @test abs(integral(nρ = 8, nside = 8) - ref) / ref <= 1e-13
end
