using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using Test

@testset "tensor_near_row: clustered quadtree == fine-grid product integration" begin
    # edge and corner panels with the GCQ rule (dyadic segments), GL panels; targets near and far, near edges and
    # corners, above and beside the panel
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 5, rule = edge_gcq_rule(10.0, 1.0; tol = 1e-8))
    rng = MersenneTwister(11)
    nx = (0.3, -0.5, 0.8) ./ norm((0.3, -0.5, 0.8))
    kerns = (d -> 1 / (4π * norm(d)), d -> BI._tensor_dnG(nx, d))
    worst = 0.0
    for kind in (:gl, :edge, :corner), k in 1:2
        p = rand(rng, filter(q -> q.kind == kind, itf.panels))
        for _ in 1:25
            u, v = rand(rng) * 1.4 - 0.2, rand(rng) * 1.4 - 0.2
            h = 10.0^(-rand(rng) * 5) * max(p.Lu, p.Lv)
            u = rand(rng) < 0.3 ? 10.0^(-rand(rng) * 8) : u          # close to the edge at u = 0
            xl = p.o .+ (u * p.Lu) .* p.eu .+ (v * p.Lv) .* p.ev .+ (rand(rng, (-1, 1)) * h) .* p.normal
            for kern in kerns
                a = tensor_near_row(kern, p, xl); b = BI._tensor_near_row_fine(kern, p, xl)
                worst = max(worst, maximum(abs, a - b) / sum(abs, b))
            end
        end
    end
    @test worst < 1e-12
    # a reused workspace gives the same rows
    p = first(filter(q -> q.kind == :corner, itf.panels)); ws = BI._TensorRowWS(itf.panels)
    xl = p.o .+ (0.3 * p.Lu) .* p.eu .+ (0.2 * p.Lv) .* p.ev .+ (0.05 * p.Lu) .* p.normal
    @test tensor_near_row(kerns[1], p, xl; ws) == tensor_near_row(kerns[1], p, xl)

    # the corrections convert to blocks by the equal-rows fast path; the general path agrees
    C, _ = laplace3d_DT_tensor_corrections(itf; quad_tol = 1e-8)
    B = BI.BlockCorrection(C, itf)
    @test sparse(B) == C
    x = randn(rng, size(C, 2))
    @test B * x ≈ C * x rtol = 1e-13
    Cp = copy(C); Cp.nzval[1] = 0.0; dropzeros!(Cp)               # one column with fewer rows: general path
    @test sparse(BI.BlockCorrection(Cp, itf)) == Cp
end
