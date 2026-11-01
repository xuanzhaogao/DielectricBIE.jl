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
               ("u = 1e-4, v = 1/2", 1.0, 1.0, 1e-4, 0.5),
               # very close to a side: s = (E Gg - F²)^(1/2) det / A from the Lagrange identity, not C - B²/A, which
               # lost all digits (error 1.5e-12 at 1e-7, 4.3e-11 at 1e-8) and threw below about 3e-9
               ("u = 1e-8, v = 1/2", 1.0, 1.0, 1e-8, 0.5),
               ("u = 1e-12, v = 1/2", 1.0, 1.0, 1e-12, 0.5),
               ("u = 1 - 1e-9, v = 1/2 (the u = 1 side)", 1.0, 1.0, 1 - 1e-9, 0.5),
               ("corner: u = v = 1e-9", 1.0, 1.0, 1e-9, 1e-9)]
    # measured relative errors for (nρ, nside) = (8, 8) / (16, 32) / (32, 32), at the targets in order (all at rounding,
    # the bound 1e-13 is the spec's; every target but node 7 has a side within DUFFY_NEAR, with 2 nside points):
    #   node 1:   7.0e-16 / 1.7e-15 / 4.6e-15      node 2:  1.2e-15 / 1.4e-15 / 7.8e-16
    #   node 7:   1.4e-15 / 2.0e-16 / 4.0e-16      corner:  1.5e-15 / 3.2e-15 / 7.5e-15
    #   clipped:  1.1e-15 / 4.4e-15 / 1.4e-15      1e-4:    2.9e-16 / 3.6e-15 / 6.5e-15
    #   1e-8:     1.9e-15 / 5.4e-15 / 7.0e-15      1e-12:   1.3e-15 / 5.8e-16 / 4.1e-15
    #   1 - 1e-9: 5.8e-16 / 4.3e-16 / 7.2e-16      corner 1e-9: 2.0e-16 / 2.0e-16 / 6.7e-15
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
    # Measured relative errors at (u0, v0) = (0.0069, 1/2), (0.3, 0.3), (1e-4, 1/2), (1/2, 1/2): 1.3e-15, 2.9e-15, 2.0e-15, 3.3e-15
    for (u0, v0) in ((0.0069, 0.5), (0.3, 0.3), (1e-4, 0.5), (0.5, 0.5))
        du, dv, w = BI.duffy_rule((u0, v0), (0.0, 1.0, 0.0, 1.0), (4.0, 0.0, 1.0))
        val = sum(w[q] / (4π * sqrt(4 * du[q]^2 + dv[q]^2)) for q in eachindex(w))
        ex = duffy_slp_exact(2.0, 1.0, 2u0, v0) / 2
        err = abs(val - ex) / abs(ex)
        @test err <= 1e-13
    end
end

# ∫_0^A ∫_0^B du dv / sqrt(E u² + 2F u v + Gg v²) in BigFloat: split at the diagonal, with m = v/u (resp. u/v) the part
# below (above) it is A ∫_0^(B/A) dm / sqrt(Gg m² + 2F m + E), a logarithm
function duffy_Iq(A, B, E, F, Gg)
    (A == 0 || B == 0) && return big(0.0)
    J(M, a, f, g) = (log(2 * sqrt(g) * sqrt(g * M^2 + 2f * M + a) + 2g * M + 2f) - log(2 * sqrt(g) * sqrt(a) + 2f)) / sqrt(g)
    return A * J(B / A, E, F, Gg) + B * J(A / B, Gg, F, E)
end
# ∫ du dv / r over the unit parameter square at c = (u0, v0): the four corner rectangles, F changing sign with the quadrant
function duffy_exact(c, G)
    u0, v0 = big.(c); E, F, Gg = big.(G)
    return sum(duffy_Iq(a, b, E, su * sv * F, Gg) for (a, su) in ((u0, -1), (1 - u0, 1)), (b, sv) in ((v0, -1), (1 - v0, 1)))
end

@testset "duffy_rule: sheared first fundamental form (F ≠ 0), closed form" begin
    # G = (1, ±0.7, 1.13) (x = (u ± 0.7 v, 0.8 v)): the exact integral of 1/r over the unit parameter square is the closed
    # form duffy_Iq above (BigFloat), so the foot and distance in G are checked, not only the F = 0 shortcut.
    # Measured relative errors, at (u0, v0) = (1e-3, 1/2), (0.3, 0.6), (1e-9, 1/2), (1e-12, 1/2), (1e-9, 1e-9),
    # (1 - 1e-9, 1e-12):
    #   F = +0.7:  4.6e-17, 3.0e-15, 4.6e-15, 6.8e-15, 2.0e-15, 3.3e-16
    #   F = -0.7:  6.3e-16, 2.9e-17, 3.1e-15, 2.4e-15, 4.7e-17, 9.5e-16
    # (a scratch run down to a target 1e-15 from the side or the corner stayed within 1.3e-14, for F = 0, 0.7, -0.7 and
    # G = (4, 0, 1))
    for F in (0.7, -0.7), c in ((1e-3, 0.5), (0.3, 0.6), (1e-9, 0.5), (1e-12, 0.5), (1e-9, 1e-9), (1 - 1e-9, 1e-12))
        G = (1.0, F, 1.13)
        du, dv, w = BI.duffy_rule(c, (0.0, 1.0, 0.0, 1.0), G)
        val = sum(w[q] / sqrt(G[1] * du[q]^2 + 2G[2] * du[q] * dv[q] + G[3] * dv[q]^2) for q in eachindex(w))
        ex = duffy_exact(c, G)
        @test Float64(abs(val - ex) / ex) <= 1e-13
    end
end

@testset "duffy_rule: a side close to the target gets 2 nside points" begin
    # g / r on [0.5, 1] × [0, 0.26] (a curved GCQ panel's last fine segment: g like its degree-13 basis times u^pw)
    # with the target close to the v = 0 side, against nρ = 64, nside = 256; errors relative to ∫ |g| / r. Measured with
    # the side at 2 nside / at nside (before): (0.77, 1e-5) 3.0e-15 / 6.9e-12, (0.95, 1e-7) 3.4e-15 / 3.2e-11,
    # (0.55, 1e-3) 8.8e-16 / 6.7e-13; at (0.77, 0.03) the side is 0.06 × its length away (refined), at (0.77, 0.06)
    # 0.12 × (not refined, 5.0e-15)
    g(u, v) = (2u - 1.5)^13 * (1 + v)^7 / u^0.27
    rect = (0.5, 1.0, 0.0, 0.26); G = (1.0, 0.0, 1.0)
    quad(c, f; kw...) = ((du, dv, w) = BI.duffy_rule(c, rect, G; kw...);
                         sum(w[q] * f(c[1] + du[q], c[2] + dv[q]) / hypot(du[q], dv[q]) for q in eachindex(w)))
    for (c, nsides) in (((0.77, 1e-5), 5), ((0.95, 1e-7), 5), ((0.55, 1e-3), 5), ((0.77, 0.03), 4 + 1), ((0.77, 0.06), 4))
        @test length(BI.duffy_rule(c, rect, G)[3]) == nsides * BI.DUFFY_NRHO * BI.DUFFY_NSIDE
        ref = quad(c, g; nρ = 64, nside = 256)
        @test abs(quad(c, g) - ref) / quad(c, (u, v) -> abs(g(u, v)); nρ = 64, nside = 256) <= 5e-14
    end
end

@testset "duffy_rule: G must be positive definite" begin
    rect = (0.0, 1.0, 0.0, 1.0)
    @test_throws ArgumentError BI.duffy_rule((0.5, 0.5), rect, (1.0, 1.0, 1.0))     # det G = 0
    @test_throws ArgumentError BI.duffy_rule((0.5, 0.5), rect, (1.0, 2.0, 1.0))     # indefinite
    @test_throws ArgumentError BI.duffy_rule((0.5, 0.5), rect, (-1.0, 0.0, -1.0))   # negative definite
end
