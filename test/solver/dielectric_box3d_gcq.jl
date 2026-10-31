using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
using SCTL

@testset "dielectric box3d with GCQ edge panels" begin
    ε = 10.0
    r = edge_gcq_rule(ε, 1.0; tol = 1e-3)
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = 3, rule = r)
    b = rhs_dielectric_box3d(itf, PointSource((0.1, 0.15, 0.2), 1.0), ε)
    # FMM-corrected operator agrees with the dense corrected matrix
    A, st = lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-12)
    Ad = lhs_dielectric_box3d_gcq_dense(itf)
    x = randn(length(b))
    @test norm(A * x - Ad * x) / norm(Ad * x) < 1e-10
    @test st.unique < st.pairs                       # near rows are reused
    # reuse does not change the corrections
    C1, _ = laplace3d_DT_tensor_corrections(itf; reuse = true)
    C2, _ = laplace3d_DT_tensor_corrections(itf; reuse = false)
    @test maximum(abs.(C1 - C2)) < 1e-12
    # Gauss: ∫σ = 1 - 1/ε for a unit charge inside (≈ 3e-6 here)
    σ, ks = solve_dielectric_box3d_gcq(A, b, itf; rtol = 1e-12)
    @test abs(dot(BI.all_weights(itf), σ) - (1 - 1 / ε)) < 1e-4
    # block GMRES: the same solutions, column by column, in blocks of 2 (a zero column stays zero)
    B = reduce(hcat, [rhs_dielectric_box3d(itf, PointSource(x0, 1.0), ε)
                      for x0 in ((0.1, 0.15, 0.2), (-0.3, 0.2, 0.0), (0.0, -0.1, 0.35))])
    B = [B zeros(length(b)) 1e-3 .* B[:, 1]]
    Σ, it = solve_dielectric_box3d_gcq_block(Ad, B, itf; rtol = 1e-10, block = 2)
    @test it > 0 && iszero(Σ[:, 4])
    for j in (1, 2, 3, 5)
        σj, _ = solve_dielectric_box3d_gcq(Ad, B[:, j], itf; rtol = 1e-12)
        @test norm(Σ[:, j] - σj) < 1e-7 * norm(σj)
    end
end

@testset "GL columns come from SCTL" begin
    # laplace3d_DT_tensor_corrections takes the columns of the GL×GL source panels from sctl_gl_corrections at
    # quad_tol, those of the GCQ edge and corner panels from the product integration. n_quad = 3 and 5 are not SCTL
    # orders: the GL panels are proxy elements of order 4 and 8, the density interpolated
    gauss(Dt, w, σ) = abs(dot(w, Dt(σ)) - dot(w, σ) / 2) / dot(w, abs.(σ))
    # Gauss identity wᵀ Dᵀ σ = ½ wᵀ σ on the closed box: for a random σ (limited by the GCQ edge panels) and a random
    # σ on the GL panels' nodes only (the GL columns, limited by the node rule over the targets); bounds: 1.5 times
    # the values with the product-integration GL columns (before the GL panels moved to SCTL, commit 844db1c)
    # ggl at nq = 3 (was 8e-7): the random σg is another stream on Julia 1.10 (lts) than on 1.13 (randn(MersenneTwister(4))
    # differs), 8.99e-7 on lts against 5.35e-7 on 1.13, and 2e-7 to 3.1e-6 over seeds 1:12 on 1.13: 1.5e-6 covers lts
    for (nq, tol, gall, ggl) in ((3, 1e-3, 5e-5, 1.5e-6),   # before: 3.55e-5, 5.35e-7; SCTL: 3.55e-5, 5.35e-7
                                 (5, 1e-6, 7e-5, 1.3e-9))   # before: 4.81e-5, 8.44e-10; SCTL: 4.81e-5, 7.54e-10
        itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 1, n_quad = nq, rule = edge_gcq_rule(10.0, 1.0; tol))
        gl = findall(is_gl_panel, itf.panels); gcq = findall(!is_gl_panel, itf.panels)
        @test !isempty(gl) && !isempty(gcq)
        N = BI.num_points(itf)
        glc = falses(N); foreach(r -> glc[r] .= true, panel_column_ranges(itf)[gl])
        C, _ = laplace3d_DT_tensor_corrections(itf)
        Cg = sctl_gl_corrections(itf, gl; tol = 1e-12)
        @test nnz(Cg[:, glc]) > 0 && C[:, glc] == Cg[:, glc]                   # same tolerance, same code
        @test C[:, .!glc] == first(laplace3d_DT_tensor_corrections(itf; sources = gcq))[:, .!glc]
        F = BI.laplace3d_DT_fmm3d(itf, 1e-13)
        Dt = σ -> F * σ .+ C * σ
        w = BI.all_weights(itf)
        σg = zeros(N); σg[glc] .= randn(MersenneTwister(4), count(glc))
        @test gauss(Dt, w, randn(MersenneTwister(3), N)) < gall
        @test gauss(Dt, w, σg) < ggl
    end
end
