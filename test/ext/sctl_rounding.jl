# Nodes and targets that SCTL sees only to rounding. SCTL matches a target to an element node by exact equality
# (r² == 0), and a flat panel's kernels vanish in its plane only up to the rounding of the coordinates.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays
using SCTL
using Test

@testset "SCTLPotential: targets on interface nodes" begin
    # A panel whose n_quad is an SCTL order is its own SCTL element, and a target within max(1e-13 × the panel's
    # size, 2^-48 × box) of one of its nodes is moved onto the element's node: SCTL's Duffy self row. On this box
    # with n_quad = 8, 630 of the 2,560 interface nodes are SCTL's element nodes only to rounding; unmoved, SCTL's
    # off-node rule left the node rule's 1/r term uncancelled there: off by up to 0.53 on the nodes, 1.8e-3 at 2e-15
    # off them (|u| ≤ 0.85), and 0.29 on the nodes at n_quad = 4. At n_quad = 6 the elements are order-8 proxies,
    # whose nodes are not the panel's: no move, and SCTL's off-node rule is accurate there (the single layer is
    # continuous). The reference: the potential 1e-7 inside along the normal.
    ext = Base.get_extension(DielectricBIE, :DielectricBIESCTLExt)
    for (nq, bound) in ((8, 3e-7),     # on the nodes 6.0e-8, 2e-15 off 6.0e-8 measured
                        (4, 3e-7),     # 4.5e-8, 4.5e-8
                        (6, 3e-7))     # 5.1e-8, 5.1e-8 (as before)
        box = BI.single_dielectric_box3d(1.2, 0.8, 0.6, nq, 0.4, 4.0, 1.0, Float64)
        N = BI.num_points(box)
        X, Nn = BI._interface_targets_normals(box)
        X = Matrix{Float64}(X); Nn = Matrix{Float64}(Nn)
        σ = [cos(2X[1, j]) + X[2, j] * X[3, j] + 0.5 for j in 1:N]
        ids = 1:7:N
        m = length(ids)
        T = hcat(X[:, ids], X[:, ids] .+ 2e-15 .* Nn[:, ids])
        u = potential_operator(SCTLPotential(tol = 1e-10), box, T) * σ
        uo = potential_operator(SCTLPotential(tol = 1e-10), box, X[:, ids] .- 1e-7 .* Nn[:, ids]) * σ
        @test maximum(abs, u[1:m] .- uo) < bound
        @test maximum(abs, u[m+1:2m] .- uo) < bound
        if nq == 8
            # 88 of these targets are SCTL's element nodes only to rounding
            el = QuadElemList(8, length(box.panels)) do e, u, v
                ext._panel_param(box.panels[e])[1](u, v)
            end
            Xs = first(node_coord(el)); close(el)
            @test count(t -> Xs[:, t] != X[:, t], ids) > 0
        end
    end
end

@testset "sctl_gl_corrections: rotated flat box far from the origin" begin
    # A flat panel's D, and its Dᵀ at targets with its normal, vanish in its plane: those targets get no rows, its
    # own nodes by index and the others when |n·(x - y₁)| ≤ 1e-12 × its size + 8 eps (|x| + |y₁|), the rounding of
    # n·(x - y₁). With 1e-12 × its size alone, a panel's own nodes failed the test once the box was turned off the
    # axes and moved to (1e4, 3e3, 0): at n_quad = 6 (proxy elements, nodes not moved onto them) SCTL's off-node rule
    # then gave the own rows the one-sided limit, own blocks of 0.5 and max|Cx - C₀x| = 1.3 max|C₀x|. What is left is
    # the rounding of the absolute coordinates, ~eps |x| / (distance to the nearest node) (2e-9 at 1e5).
    θ, φ = 0.37, 0.81
    R = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1] * [1 0 0; 0 cos(φ) -sin(φ); 0 sin(φ) cos(φ)]
    S = [1e4, 3e3, 0.0]
    tr(x) = Tuple(R * collect(x) .+ S)
    for (nq, bound) in ((6, 1e-9),     # Dᵀ 2.2e-10, D 2.2e-10 measured (1.3, 1.6 before)
                        (8, 1e-9))     # Dᵀ 2.8e-10, D 4.0e-10 (the same before, with 1,024 own-block entries)
        itf0 = BI.single_dielectric_box3d(1.0, 1.0, 1.0, nq, 0.5, 4.0, 1.0, Float64)
        itf = DielectricInterface([BI.rect_panel3d_discretize(tr.(p.corners)..., p.gl_xs, p.gl_ws,
                                                              Tuple(R * collect(p.normal)); is_edge = p.is_edge)
                                   for p in itf0.panels], itf0.eps_in, itf0.eps_out)
        np = length(itf.panels); N = BI.num_points(itf)
        cols = panel_column_ranges(itf)
        x = [sin(3i) for i in 1:N]
        for kernel in (:DT, :D)
            C0 = sctl_gl_corrections(itf0, 1:np; tol = 1e-10, kernel)
            C = sctl_gl_corrections(itf, 1:np; tol = 1e-10, kernel)
            @test sum(k -> nnz(C[cols[k], cols[k]]), 1:np) == 0
            @test nnz(C) == nnz(C0)                     # the same targets are in a plane (7% more before at n_quad 6)
            @test maximum(abs, C * x - C0 * x) < bound * maximum(abs, C0 * x)
        end
    end
end
