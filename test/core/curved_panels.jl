using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

@testset "curved panels" begin
    # flat 2 × 3 rectangle: area, constant normal, flip
    r6 = gl_tensor_rule(6)
    p = CurvedPanel((1.0, 0.0, 0.0), (u, v) -> (2u, 3v, 0.0), r6, r6)
    @test sum(p.weights) ≈ 6
    @test all(n -> n == (0.0, 0.0, 1.0), p.normals)
    @test p.points[1] == (1.0, 0.0, 0.0) .+ curved_lpoint(p, r6.x[1], r6.x[1])
    pf = CurvedPanel((1.0, 0.0, 0.0), (u, v) -> (2u, 3v, 0.0), r6, r6; flip = true)
    @test all(n -> n == (0.0, 0.0, -1.0), pf.normals)

    # unit-sphere rectangle θ ∈ [π/4, π/2], φ ∈ [0, π/2]: area and radial normals
    sph(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
    a0 = sph(π / 4, 0.0)
    r10 = gl_tensor_rule(10)
    q = CurvedPanel(a0, (u, v) -> sph(π / 4 + u * π / 4, v * π / 2) .- a0, r10, r10)
    @test sum(q.weights) ≈ (cos(π / 4) - cos(π / 2)) * π / 2 rtol = 1e-12
    for (x, n) in zip(q.points, q.normals)
        @test norm(n) ≈ 1 atol = 1e-14
        @test abs(dot(n, x)) ≈ 1 atol = 1e-13
    end
    # the frame at a parameter point: unit normal along ±(X_u × X_v), J = |X_u × X_v|
    Xu, Xv, n, J = BI.curved_frame(q, 0.3, 0.6)
    @test J ≈ norm(cross(collect(Xu), collect(Xv)))
    @test abs(dot(n, sph(π / 4 + 0.3 * π / 4, 0.6 * π / 2))) ≈ 1 atol = 1e-13

    # eachpoint: one item per node, per-node normals
    pts = collect(BI.eachpoint(q))
    @test length(pts) == BI.num_points(q) == 100
    @test pts[1].normal != pts[end].normal
    @test pts[7].weight == q.weights[7]

    # a GCQ rule in u (edge at u = 0): weights carry the singular weight, the area is still exact
    gr = gcq_tensor_rule(edge_gcq_rule(4.0, 1.0; angle = 2π / 3, tol = 1e-8))
    e = CurvedPanel(a0, (u, v) -> sph(π / 4 + u * π / 4, v * π / 2) .- a0, gr, r10; kind = :edge)
    @test e.kind == :edge
    @test sum(e.weights) ≈ (cos(π / 4) - cos(π / 2)) * π / 2 rtol = 1e-7
end
