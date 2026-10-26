using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

capR(a, θ) = a / sin(θ)
capc(a, θ, sgn) = (0.0, 0.0, -sgn * capR(a, θ) * cos(θ))        # centre of the cap sphere
nodes(itf, rg) = (pts = reduce(vcat, [p.points for p in itf.panels]); pts[rg])
normals(itf, rg) = (ns = reduce(vcat, [p.normals for p in itf.panels]); ns[rg])
area(m, s) = sum(BI.all_weights(m.interface)[m.ranges[s]])

@testset "lens3d_gcq" begin
    # single lens: one ε, edge of angle θ1 + θ2, no disk
    a, θ1, θ2 = 1.0, π / 3, π / 4
    m = lens_gcq(a, θ1, θ2, 4.0, 4.0, 1.0; nlev = 3, n_quad = 8, tol = 1e-12)
    @test !haskey(m.ranges, :disk)
    @test area(m, :upper) ≈ 2π * capR(a, θ1)^2 * (1 - cos(θ1)) rtol = 1e-10
    @test area(m, :lower) ≈ 2π * capR(a, θ2)^2 * (1 - cos(θ2)) rtol = 1e-10
    @test m.powers == edge_density_powers(4.0, 1.0; angle = θ1 + θ2)
    for (s, sgn, θ) in ((:upper, 1, θ1), (:lower, -1, θ2))
        c = capc(a, θ, sgn)
        @test all(dot(n, x .- c) > 0 for (x, n) in zip(nodes(m.interface, m.ranges[s]), normals(m.interface, m.ranges[s])))
        @test all(abs(norm(x .- c) - capR(a, θ)) < 1e-13 for x in nodes(m.interface, m.ranges[s]))
    end
    @test all(p -> p.kind == :edge ? minimum(p.ru.x) < 1e-3 : true, m.interface.panels)
    @test count(p -> p.kind == :edge, m.interface.panels) == 2 * 4 * 2       # 2 caps × 4 side charts × nv = 2
    @test all(m.interface.eps_in .== 4.0) && all(m.interface.eps_out .== 1.0)

    # split sphere: two ε, disk interface, triple junction (sectors π, π/2, π/2)
    s = lens_gcq(1.0, π / 2, π / 2, 2.0, 8.0, 1.0; nlev = 3, n_quad = 8, tol = 1e-12)
    @test area(s, :disk) ≈ π rtol = 1e-10
    @test area(s, :upper) ≈ 2π rtol = 1e-10
    @test area(s, :lower) ≈ 2π rtol = 1e-10
    @test s.powers == BI._junction_density_powers([π, π / 2, π / 2], [1.0, 2.0, 8.0]; gmax = 20.0)
    @test all(n -> n == (0.0, 0.0, 1.0), normals(s.interface, s.ranges[:disk]))
    @test all(x -> abs(x[3]) < 1e-15, nodes(s.interface, s.ranges[:disk]))
    pereps(v) = reduce(vcat, [fill(v[k], BI.num_points(p)) for (k, p) in enumerate(s.interface.panels)])
    @test all(pereps(s.interface.eps_in)[s.ranges[:disk]] .== 8.0)       # normal +z: in = lower, out = upper
    @test all(pereps(s.interface.eps_out)[s.ranges[:disk]] .== 2.0)
    @test all(pereps(s.interface.eps_in)[s.ranges[:upper]] .== 2.0) && all(pereps(s.interface.eps_in)[s.ranges[:lower]] .== 8.0)
    @test count(p -> p.kind == :edge, s.interface.panels) == 3 * 4 * 2

    # h-graded reference: no GCQ panel
    g = lens_gcq(1.0, π / 2, π / 2, 2.0, 8.0, 1.0; nlev = 3, n_quad = 8, edge = :gl)
    @test all(p -> p.kind == :gl, g.interface.panels)
    @test area(g, :disk) ≈ π rtol = 1e-10

    # edge-relative chart offsets are cancellation-free: U = 1e-14 against BigFloat
    for (θ, sgn) in ((π / 3, 1), (2π / 3, -1))
        o = BI._cap_side_offset(1.0, θ, 1, 1e-14, 0.3, 0.25, sgn)
        ob = setprecision(BigFloat, 256) do
            p = BI._cap_side_point(big(1.0), big(θ), 1, big(1e-14), big(0.3), sgn)
            q = BI._cap_side_point(big(1.0), big(θ), 1, big(0.0), big(0.25), sgn)
            Float64.(p .- q)
        end
        @test all(isapprox.(o, ob; rtol = 1e-13, atol = 1e-30))
        @test abs(o[3] - ob[3]) <= 1e-13 * abs(ob[3])                   # the normal offset ~1e-14
    end
    od = BI._disk_side_offset(1.0, 2, 1e-14, 0.7, 0.75)
    odb = setprecision(BigFloat, 256) do
        Float64.(BI._disk_side_point(big(1.0), 2, big(1e-14), big(0.7)) .- BI._disk_side_point(big(1.0), 2, big(0.0), big(0.75)))
    end
    @test all(isapprox.(od, odb; rtol = 1e-13, atol = 1e-30))
end
