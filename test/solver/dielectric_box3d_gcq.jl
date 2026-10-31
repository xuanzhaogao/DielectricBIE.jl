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
    # Gauss identity wᵀ Dᵀ σ = ½ wᵀ σ on the closed box, for the rough σ = sin(97x - 89y + 83z + 1) at the nodes (the
    # phase: an odd σ gives 0 = 0 by the box's symmetry): on every node (limited by the GCQ edge panels), and on the GL
    # panels' nodes only (the GL columns, limited by the node rule over the targets). Bounds: about twice the values
    # measured. It replaced random densities (randn(MersenneTwister(3)), (4)), another stream on Julia 1.10 than on
    # 1.13, whose ggl at nq = 3 ranged over 2e-7 to 3.1e-6 with the seed. Old bounds (random σ, 1.5 times the values
    # of the product-integration GL columns of commit 844db1c, ggl at nq = 3 raised to cover lts): gall 5e-5, 7e-5;
    # ggl 1.5e-6, 1.3e-9.
    for (nq, tol, gall, ggl) in ((3, 1e-3, 4.5e-5, 1.1e-6),   # 2.23e-5, 5.49e-7 measured (random σ: 3.55e-5, 5.35e-7)
                                 (5, 1e-6, 3e-5, 4e-10))      # 1.47e-5, 1.91e-10 (random σ: 4.81e-5, 7.54e-10)
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
        σ = [sin(97y[1] - 89y[2] + 83y[3] + 1) for P in itf.panels for y in P.points]
        σg = zeros(N); σg[glc] .= σ[glc]
        @test gauss(Dt, w, σ) < gall
        @test gauss(Dt, w, σg) < ggl
    end
end

@testset "GL TensorPanel columns against tensor_near_row" begin
    # the GL columns (node rule plus sctl_gl_corrections, through _panel_param and _gl_tangents of TensorPanels)
    # against tensor_near_row's product integration, which is independent of SCTL: the 8 closest near targets off
    # each GL panel's plane (Dᵀ vanishes in it), with n_quad = 4 (the panel its own SCTL element) and 5 (a proxy of
    # order 8). The error is relative to the row's largest entry.
    for (nq, bound) in ((4, 1e-12),     # 1.8e-15 measured
                        (5, 1e-12))     # 1.0e-14
        itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 1, n_quad = nq,
                                          rule = edge_gcq_rule(10.0, 1.0; tol = 1e-6))
        ps = itf.panels; gl = findall(is_gl_panel, ps)
        X, Nrm = BI._interface_targets_normals(itf)
        cols = panel_column_ranges(itf)
        C = sctl_gl_corrections(itf, gl; tol = 1e-12)
        err = 0.0; rows = 0
        for k in gl
            P = ps[k]
            dist(t) = minimum(norm(X[:, t] .- collect(y)) for y in P.points)
            ts = filter(t -> abs(dot(collect(P.normal), X[:, t] .- collect(P.points[1]))) > 1e-12,
                        unique(rowvals(C[:, cols[k]])))
            for t in sort(ts; by = dist)[1:min(8, length(ts))]
                nx = (Nrm[1, t], Nrm[2, t], Nrm[3, t])
                ref = -BI.tensor_near_row(d -> BI._tensor_dnG(nx, d), P, (X[1, t], X[2, t], X[3, t]) .- P.anchor)
                row = dt_node_rule_block(itf, k, [t])[1, :] .+ Vector(C[t, cols[k]])
                err = max(err, maximum(abs, row .- ref) / maximum(abs, ref)); rows += 1
            end
        end
        @test rows == 8 * length(gl)
        @test err < bound
    end
end
