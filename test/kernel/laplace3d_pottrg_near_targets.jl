using DielectricBIE
import DielectricBIE as BI
using HCubature, StaticArrays, LinearAlgebra, Random
using Test

# Off-surface potential at targets close to the surface near panel edges and corners (where the
# centre-based neighbour range of laplace3d_pottrg_fmm3d_corrected_hcubature used to leave pairs
# uncorrected, and edge panels were never corrected), against adaptive cubature over every panel.
@testset "pottrg corrected: targets near panel edges and corners" begin
    itf = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.3, 4.0, 1.0, Float64)
    X, Nn = BI._interface_targets_normals(itf)
    n = BI.num_points(itf)
    σ = [cos(2X[1, i]) + X[2, i] * X[3, i] + 0.5 for i in 1:n]
    rng = MersenneTwister(11)
    # targets: off panel corners (all panels), and a few random nodes, at 1e-3 .. 1e-1 off the surface
    T = Matrix{Float64}(undef, 3, 0)
    for p in itf.panels[1:7:end]
        c = collect(p.corners[1] .+ 0.02 .* (p.corners[3] .- p.corners[1]))
        T = hcat(T, c .+ 10.0^(-1 - 2rand(rng)) .* collect(p.normal))
    end
    ref(x) = sum(itf.panels) do p
        k = findfirst(==(p), itf.panels); off = sum(BI.num_points, itf.panels[1:k-1]; init = 0)
        nq = p.n_quad; vals = permutedims(reshape(σ[off+1:off+nq^2], nq, nq))
        a, b, c, d = p.corners; cc = collect((a .+ b .+ c .+ d) ./ 4); bma = collect(b .- a); dma = collect(d .- a)
        area = norm(bma) * norm(dma); rx = zeros(nq); ry = zeros(nq)
        f = uv -> (BI.barycentric_row!(rx, p.gl_xs, p.bary_weights, uv[1]); BI.barycentric_row!(ry, p.gl_xs, p.bary_weights, uv[2]);
                   dot(rx, vals * ry) / (4π * norm(x .- (cc .+ bma .* (uv[1] / 2) .+ dma .* (uv[2] / 2)))) * area / 4)
        hcubature(f, SVector(-1.0, -1.0), SVector(1.0, 1.0); rtol = 1e-12, atol = 1e-15, maxevals = 10^7)[1]
    end
    uref = [ref(T[:, t]) for t in axes(T, 2)]
    u = BI.laplace3d_pottrg_fmm3d_corrected_hcubature(itf, T, 1e-10, 1e-10, 5.0) * σ
    @test maximum(abs, u - uref) / maximum(abs, uref) < 1e-8
    # the previous behaviour (no edge panels, centre range only) is far off at these targets
    nl_old = BI.build_target_neighbor_list(itf, T, false; range_factor = 5.0)
    nl_new = BI.build_target_neighbor_list(itf, T, true; range_factor = 5.0, atol = 1e-10)
    @test sum(length, values(nl_new)) > sum(length, values(nl_old))
    u_old = BI.laplace3d_pottrg_fmm3d_corrected_hcubature(itf, T, 1e-10, 1e-10, 5.0; correct_edges = false) * σ
    @test maximum(abs, u_old - uref) / maximum(abs, uref) > 1e-6
end
