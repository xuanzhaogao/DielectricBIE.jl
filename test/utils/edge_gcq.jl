using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

@testset "edge GCQ rule" begin
    # right-angle dielectric edge: density powers {2m ± q} ∪ {2m+1}, q = (2/π) asin(λ/2)
    for ε in (2.0, 10.0, 200.0)
        λ = (ε - 1) / (ε + 1)
        q = 2 / π * asin(λ / 2)
        p = edge_density_powers(ε, 1.0; gmax = 4.0)
        @test p[1] ≈ -q atol = 1e-10
        @test p[2] ≈ q atol = 1e-10
        @test any(x -> abs(x - 1) < 1e-8, p)        # double root γ = 2 is found
        @test any(x -> abs(x - (2 - q)) < 1e-9, p)
    end
    # the weighted rule integrates its whole family, has positive weights
    r = edge_gcq_rule(10.0, 1.0; tol = 1e-10)
    @test edge_gcq_family_error(r) < 1e-9
    @test all(r.w .> 0)
    @test all(0 .< r.x .< 1)
    @test sum(r.w) ≈ 1 atol = 1e-10                 # ∫₀¹ 1 = 1 (s⁰ is in the family)
    # interpolation matrix reproduces a family member on the fine grid
    p1 = r.pw
    f(s) = s^p1 * (1 + 0.3s) + s^(-p1) - 0.2s^2
    xf = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* r.z0 for (a, b) in r.segs])
    vf = r.P * f.(r.x)
    @test maximum(abs.(vf .* xf .^ p1 .- f.(xf)) ./ abs.(f.(xf))) < 1e-6
    # a power below -1/2 (four-material junction) still gives an accurate rule
    pj = junction_powers(fill(π / 2, 4), [1.0, 50.0, 2.0, 10.0]; gmax = 12.0) .- 1
    @test pj[1] < -0.5
    rj = edge_gcq_rule(pj; tol = 1e-10)
    @test edge_gcq_family_error(rj) < 1e-9
    # qmax: the family is set by the largest exponent; extra powers below gmax add nothing
    d2 = edge_density_powers(10.0, 1.0; gmax = 2.0); d20 = edge_density_powers(10.0, 1.0; gmax = 20.0)
    ra = edge_gcq_rule(d2; tol = 1e-12, qmax = 10.3); rb = edge_gcq_rule(d20; tol = 1e-12, qmax = 10.3)
    @test maximum(ra.pows) <= 10.3 && ra.pows ≈ rb.pows && ra.x ≈ rb.x
    @test edge_gcq_family_error(ra) < 1e-11
    # right-angle junctions keep the γ ∈ {2m ± γ1} ∪ {2m} structure
    γs = junction_powers(fill(π / 2, 4), [1.0, 50.0, 2.0, 10.0]; gmax = 13.0)
    γs = γs[γs .<= 11.5]
    ex = sort(vcat([2m + s * γs[1] for m in 0:6 for s in (-1, 1) if 0 < 2m + s * γs[1] <= 11.5], 2.0 .* (1:5)))
    @test length(γs) == length(ex) && maximum(abs.(γs .- ex)) < 1e-8
    # nearly coincident simple roots (γ1 ≈ 1, so γ1 and 2 - γ1 share one grid cell):
    # low contrast, and a seam between two nearly equal permittivities
    classes(γ1, gmax) = sort(vcat([2m + s * γ1 for m in 0:ceil(Int, gmax / 2) for s in (-1, 1)
                                   if 0 < 2m + s * γ1 <= gmax], 2.0 .* (1:floor(Int, gmax / 2))))
    for ε in (1.0001, 1.00001)
        λ = (ε - 1) / (ε + 1)
        q = 2 / π * asin(λ / 2)
        γ = filter(<=(5.9), edge_density_powers(ε, 1.0; gmax = 6.0) .+ 1)
        @test γ[1] ≈ 1 - q rtol = 1e-6
        @test length(γ) == length(classes(1 - q, 5.9)) && maximum(abs.(γ .- classes(1 - q, 5.9))) < 1e-9
    end
    γn = filter(<=(5.9), junction_powers([π, π / 2, π / 2], [1.0, 10.0, 10.001]; gmax = 6.0))
    @test 1 - 1e-4 < γn[1] < 1
    @test length(γn) == length(classes(γn[1], 5.9)) && maximum(abs.(γn .- classes(γn[1], 5.9))) < 1e-9
    # p1 → -1 (checkerboard edge at contrast ~1e6): graded grid, nodes kept off the edge
    rs = edge_gcq_rule([-0.999, 0.999, 1.0]; tol = 1e-12, qmax = 12.0)
    @test edge_gcq_family_error(rs) < 1e-10
    @test rs.x[1] > 1e-12
    @test_throws ArgumentError edge_gcq_rule([-0.5]; grid = :nope)
    ra2 = edge_gcq_rule(d2; tol = 1e-12, qmax = 10.3, grid = :adaptive)
    @test length(ra2.x) == length(ra.x) && edge_gcq_family_error(ra2) < 1e-11
end

@testset "universal GCQ rule" begin
    pmin = -0.3
    ru = universal_gcq_rule(pmin; qmax = 12.0, tol = 1e-12)
    @test ru.pw == pmin && sum(abs, ru.w) < 1.01      # a few small negative weights, no cancellation
    # exponents off the family's grid, anywhere in [pmin, 10], and the log terms of a varying power
    qs = range(pmin + 0.00037, 10 - 0.0037, length = 97)
    @test maximum(abs(sum(ru.w .* ru.x .^ q) * (q + 1) - 1) for q in qs) < 1e-10
    for q in (pmin, pmin + 0.123, 0.4567)       # ∫₀¹ s^q log s = -1/(q+1)², ∫₀¹ s^q log² s = 2/(q+1)³
        @test sum(ru.w .* ru.x .^ q .* log.(ru.x)) * (q + 1)^2 ≈ -1 atol = 1e-9
        @test sum(ru.w .* ru.x .^ q .* log.(ru.x) .^ 2) * (q + 1)^3 / 2 ≈ 1 atol = 1e-8
    end
    @test minimum(ru.x) >= 1e-10                       # the default node floor
    @test minimum(universal_gcq_rule(pmin; qmax = 12.0, tol = 1e-12, floor = 0.0).x) < 1e-15
    @test_throws ArgumentError universal_gcq_rule(-1.0)
end
