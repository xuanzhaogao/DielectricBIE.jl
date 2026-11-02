# SCTL's near and Duffy self rows for GL×GL panels: sctl_gl_corrections (Dᵀ, D) and sctl_gl_potential_corrections
# (single layer).
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays
using SCTL
using HCubature
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

gl_sphere = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
gl_nodes(itf) = reduce(hcat, [collect(y) for p in itf.panels for y in p.points])

# node rule (FMM3D) plus the correction, for Dᵀ or D
function gl_apply(itf, C; kernel = :DT)
    F = kernel == :DT ? BI.laplace3d_DT_fmm3d(itf, 1e-13) : BI.laplace3d_D_fmm3d(itf, 1e-13)
    B = BlockCorrection(C, itf)
    return x -> F * x .+ B * x
end

# today's SCTLNearCorrection assembly for a FlatPanel interface (copied, so that this reference stays independent
# of the GL routine): one QuadElemList per n_quad, own-node targets with SCTL's node coordinates, SCTL's near
# correction mapped back to interface order with DielectricBIE's sign (kernel = Laplace3D_FxdU: -Dᵀ with target
# normals; Laplace3D_DxU: -D)
function gl_sctl_reference(itf, kernel; tol)
    ps = itf.panels
    offs = cumsum([0; [BI.num_points(p) for p in ps]])
    X, Nrm = BI._interface_targets_normals(itf)
    X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
    groups = map(sort!(unique(p.n_quad for p in ps))) do q
        ids = findall(p -> p.n_quad == q, ps)
        el = QuadElemList(q, length(ids)) do e, u, v
            a, b, c, d = ps[ids[e]].corners
            collect((a .+ b .+ c .+ d) ./ 4 .+ (b .- a) .* (u - 0.5) .+ (d .- a) .* (v - 0.5))
        end
        perm = reduce(vcat, [collect(offs[k]+1:offs[k+1]) for k in ids])
        @assert maximum(abs, first(node_coord(el)) .- X[:, perm]) < 1e-12
        (el = el, perm = perm)
    end
    Xt = copy(X)
    foreach(g -> Xt[:, g.perm] .= first(node_coord(g.el)), groups)
    I = Int[]; J = Int[]; V = Float64[]
    for g in groups
        C, _ = kernel isa Laplace3D_FxdU ? SCTL.near_correction(kernel, g.el; targets = Xt, normals = Nrm, tol) :
                                           SCTL.near_correction(kernel, g.el; targets = Xt, tol)
        i, j, v = findnz(C)
        append!(I, i); append!(J, g.perm[j]); append!(V, .-v)
    end
    return sparse(I, J, V, size(X, 2), size(X, 2))
end

# the Dᵀ row of FlatPanel P for target x with normal nx, ∫_P nx·(x - y)/(4π|x - y|³) ℓ_j(y) dS_y for each node j,
# by adaptive cubature (as _ref_potential in sctl.jl)
function gl_dt_ref_row(P, x, nx; rtol = 1e-12)
    nq = P.n_quad
    a, b, c, d = P.corners
    cc = collect((a .+ b .+ c .+ d) ./ 4); bma = collect(b .- a); dma = collect(d .- a)
    area = norm(bma) * norm(dma)
    f = uv -> begin
        rx = zeros(nq); ry = zeros(nq)
        BI.barycentric_row!(rx, P.gl_xs, P.bary_weights, uv[1]); BI.barycentric_row!(ry, P.gl_xs, P.bary_weights, uv[2])
        r = x .- (cc .+ bma .* (uv[1] / 2) .+ dma .* (uv[2] / 2))
        vec(permutedims((dot(nx, r) / (4π * norm(r)^3) * area / 4) .* (rx * ry')))    # node (i, j): (i-1) nq + j
    end
    return hcubature(f, BI.StaticArrays.SVector(-1.0, -1.0), BI.StaticArrays.SVector(1.0, 1.0);
                     rtol, atol = 1e-15, maxevals = 10^7)[1]
end

# the single-layer potential of the density σ on FlatPanel P at x, by adaptive cubature
function gl_sl_ref(P, σ, x; rtol = 1e-12)
    nq = P.n_quad; vals = permutedims(reshape(σ, nq, nq))
    a, b, c, d = P.corners
    cc = collect((a .+ b .+ c .+ d) ./ 4); bma = collect(b .- a); dma = collect(d .- a)
    area = norm(bma) * norm(dma)
    f = uv -> begin
        rx = zeros(nq); ry = zeros(nq)
        BI.barycentric_row!(rx, P.gl_xs, P.bary_weights, uv[1]); BI.barycentric_row!(ry, P.gl_xs, P.bary_weights, uv[2])
        dot(rx, vals * ry) / (4π * norm(x .- (cc .+ bma .* (uv[1] / 2) .+ dma .* (uv[2] / 2)))) * area / 4
    end
    return hcubature(f, BI.StaticArrays.SVector(-1.0, -1.0), BI.StaticArrays.SVector(1.0, 1.0);
                     rtol, atol = 1e-15, maxevals = 10^7)[1]
end

# node-rule single layer Σ_j w_j σ_j / (4π |x - y_j|), the coincident node skipped (what the far field adds)
function gl_sl_nodes(itf, σ, T)
    Y = gl_nodes(itf); w = BI.all_weights(itf)
    return [sum((r = norm(T[:, t] .- Y[:, j]); r == 0 ? 0.0 : w[j] * σ[j] / (4π * r)) for j in axes(Y, 2))
            for t in axes(T, 2)]
end

@testset "sctl_gl_corrections: sphere, Dᵀ with Duffy self" begin
    # every panel of the sphere is GL×GL (cube-sphere charts, half of them flipped: the Duffy self rows use SCTL's
    # element normal, which the routine turns to the panel's). Dᵀ Y_l = Y_l / (2(2l + 1)) on the unit sphere, and
    # so is D Y_l (the two kernels are equal there). The element is the panel, its geometry SCTL's order-n_quad
    # interpolant of the map: on these ~50° panels (m = 2) that is off by 4e-6 at n_quad = 8, its normals at the
    # nodes by 3.5e-4, and that bounds the near and self rows: Dᵀ1 - 1/2 is 2.3e-4 at n_quad = 8, 2.9e-7 at 12,
    # 4.9e-9 at 16, 9.2e-12 at 20 (m = 4: 8.4e-7 at 8, 1.8e-9 at 12, 6.9e-13 at 16), tracking the normals' error.
    # An own node that SCTL did not match to its element would be off by ~1.
    for (nq, m, t1, tz) in ((8, 2, 5e-4, 5e-4),     # Dᵀ: 2.3e-4, 2.2e-4; D: 1.7e-4, 1.2e-4 measured
                            (20, 2, 1e-10, 1e-6))   # Dᵀ: 9.2e-12, 9.0e-12; D: 5.2e-12, 3.6e-12
        itf = curved_gcq_mesh(gl_sphere; n_quad = nq, m).interface
        @test all(is_gl_panel, itf.panels) && any(p -> p.flip, itf.panels)
        np = length(itf.panels); N = BI.num_points(itf)
        z = gl_nodes(itf)[3, :]
        C = sctl_gl_corrections(itf, 1:np; tol = 1e-12)
        @test size(C) == (N, N) && all(isfinite, nonzeros(C))
        Dt = gl_apply(itf, C)
        @test maximum(abs, Dt(ones(N)) .- 0.5) < t1
        @test maximum(abs, Dt(z) .- z ./ 6) < tz
        CD = sctl_gl_corrections(itf, 1:np; tol = 1e-12, kernel = :D)
        D = gl_apply(itf, CD; kernel = :D)
        @test maximum(abs, D(ones(N)) .- 0.5) < t1
        @test maximum(abs, D(z) .- z ./ 6) < tz
    end
end

@testset "sctl_gl_corrections: snapped own nodes get SCTL's Duffy self rows" begin
    # the sphere's rules moved by one ulp: the panels' nodes are then SCTL's element nodes only to rounding (with
    # FastGaussQuadrature's nodes, as the meshers use, they happen to be equal), and SCTL matches a node only
    # exactly: unless the routine passes the element's coordinates, an own node gets SCTL's off-node rule, a
    # one-sided limit (Dᵀ1 then off by ~1)
    itf0 = curved_gcq_mesh(gl_sphere; n_quad = 8, m = 2).interface
    r = itf0.panels[1].ru
    @test all(p -> p.ru.x == r.x && p.rv.x == r.x, itf0.panels)
    rn = TensorRule1D(nextfloat.(r.x), r.w, r.segs, r.z0, r.wz, r.P, 0.0)
    itf = DielectricInterface([CurvedPanel(P.anchor, P.lmap, rn, rn; flip = P.flip, dmap = P.dmap) for P in itf0.panels],
                              itf0.eps_in, itf0.eps_out)
    N = BI.num_points(itf)
    C = sctl_gl_corrections(itf, 1:length(itf.panels); tol = 1e-12)
    @test maximum(abs, gl_apply(itf, C)(ones(N)) .- 0.5) < 5e-4                    # 2.3e-4, as unmoved
    P = itf.panels[1]; n = BI.num_points(P)
    el = QuadElemList(8, 1) do e, u, v
        collect(P.anchor .+ curved_lpoint(P, u, v))
    end
    X, Nel = node_coord(el)
    @test count(t -> X[:, t] != collect(P.points[t]), 1:n) > n ÷ 2
    q = ElemQuadrature(Laplace3D_FxdU())
    S = self_matrices(q, el; tol = 1e-12, trg_normal_dot_prod = true)[:, :, 1]
    close(q); close(el)
    # SCTL's self rows are dotted with the element's normal (along ∂u × ∂v; the panel's is opposite when flipped)
    # and are -Dᵀ
    s = sign(dot(Nel[:, 1], collect(P.normals[1])))
    ref = -s .* S .- dt_node_rule_block(itf, 1, 1:n)
    @test maximum(abs, Matrix(C[1:n, 1:n]) .- ref) < 1e-12 * maximum(abs, ref)        # 0 measured
end

@testset "sctl_gl_corrections: flat box equals SCTL's near_correction" begin
    box = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.4, 4.0, 1.0, Float64)
    np = length(box.panels)
    R = gl_sctl_reference(box, Laplace3D_FxdU(); tol = 1e-10)
    C = sctl_gl_corrections(box, 1:np; tol = 1e-10)
    @test norm(C - R) / norm(R) < 1e-12                        # 1.4e-14 measured (D: 1.7e-14)
    RD = gl_sctl_reference(box, Laplace3D_DxU(); tol = 1e-10)
    @test norm(sctl_gl_corrections(box, 1:np; tol = 1e-10, kernel = :D) - RD) / norm(RD) < 1e-12
    # only the columns of the source panels
    C3 = sctl_gl_corrections(box, [3, 7]; tol = 1e-10)
    cols = panel_column_ranges(box)
    A3 = C3[:, vcat(cols[3], cols[7])]; A = C[:, vcat(cols[3], cols[7])]
    @test findnz(A3)[1:2] == findnz(A)[1:2] && norm(A3 - A) < 1e-13 * norm(A)    # SCTL's rule: rounding apart
    @test nnz(C3[:, setdiff(1:BI.num_points(box), vcat(cols[3], cols[7]))]) == 0
    @test nnz(sctl_gl_corrections(box, Int[]; tol = 1e-10)) == 0
end

@testset "sctl_gl_corrections: proxy near radius" begin
    # n_quad = 5 under an order-8 proxy element: near targets are those of the 5-point rule (the proxy's own
    # radius, 2.5 times smaller here, leaves the node rule's 9.4e-8 error on the targets beyond the diameter)
    b5 = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 5, 0.4, 4.0, 1.0, Float64)
    P = b5.panels[1]; n = BI.num_points(P)
    X, Nrm = BI._interface_targets_normals(b5)
    C = sctl_gl_corrections(b5, 1:length(b5.panels); tol = 1e-10)
    # 20 targets off panel 1's plane (in it, Dᵀ vanishes): the 10 closest, and the 10 closest beyond its diameter
    dist(t) = minimum(norm(X[:, t] .- collect(y)) for y in P.points)
    cand = sort!([t for t in axes(X, 2) if abs(dot(collect(P.normal), X[:, t] .- collect(P.points[1]))) > 1e-12];
                 by = dist)
    diam = maximum(norm(collect(a .- b)) for a in P.corners for b in P.corners)
    ts = vcat(cand[1:10], filter(t -> dist(t) > diam, cand)[1:10])
    err = map(ts) do t
        row = dt_node_rule_block(b5, 1, [t])[1, :] .+ Vector(C[t, 1:n])
        maximum(abs, row .- gl_dt_ref_row(P, X[:, t], Nrm[:, t]))
    end
    @test maximum(err) < 1e-9                                    # 1.2e-14 measured (beyond the diameter 2.2e-17)

    # the proxies' radius is SCTL's formula in Julia (far_field_nodes clamps tol to 2^-53, and the order-p radius at
    # the n-point rule's ρ would need tol' far below it): the accuracy ellipse of the n-point rule at tol, its
    # distance at the element's nodes times the tangent lengths. For an element of the rule's own order it is
    # far_field_nodes' radius, the clamp included
    ext = Base.get_extension(DielectricBIE, :DielectricBIESCTLExt)
    for p in (4, 8, 12), tol in (1e-6, 1e-12, 1e-17)
        el = QuadElemList(p, 1) do e, u, v
            [0.4u + 0.1v, 0.3v, 0.05u]
        end
        @test ext._sctl_radius(param_nodes(p), p, p, norm([0.4, 0.0, 0.05]), norm([0.1, 0.3, 0.0]), tol) ≈
              far_field_nodes(el; tol).radius rtol = 1e-13
        close(el)
    end
    # tol = 1e-12: the 5-point rule needs ρ = 18.3, which SCTL's order-8 radius reaches only at tol' = 2.6e-20 (it
    # stops at 2^-53: radius 0.9-1.06 instead of 1.65-1.8 here). The targets between (12, at 1.06-1.16) are near
    # panel 1, and the single layer and Dᵀ of smooth densities there match cubature (3e-15 measured; the node rule
    # alone, as with the clamped radius: 8.6e-12 and 4.3e-10)
    tol = 1e-12
    band = filter(t -> 1.06 < dist(t) < 1.5, cand)
    band = unique(band[round.(Int, range(1, length(band), 12))])
    Y = reduce(hcat, collect.(P.points))
    C1 = sctl_gl_corrections(b5, [1]; tol)
    Cp = sctl_gl_potential_corrections(b5, [1], X[:, band]; tol)
    @test all(t -> nnz(C1[t:t, 1:n]) > 0, band) && all(r -> nnz(Cp[r:r, 1:n]) > 0, eachindex(band))
    w = P.weights
    for σ in (ones(n), [1 + 2Y[1, j] - Y[2, j] + 0.5Y[3, j] for j in 1:n])
        u = [sum(w[j] * σ[j] / (4π * norm(X[:, t] .- Y[:, j])) for j in 1:n) for t in band] .+ Cp[:, 1:n] * σ
        uref = [gl_sl_ref(P, σ, X[:, t]; rtol = 1e-14) for t in band]
        @test maximum(abs.(u .- uref) ./ abs.(uref)) < 1e-13
        # Dᵀ: relative to the row's scale Σ |entry σ_j| (n_t·(x - y) can nearly cancel)
        e = map(band) do t
            ref = gl_dt_ref_row(P, X[:, t], Nrm[:, t]; rtol = 1e-14)
            row = dt_node_rule_block(b5, 1, [t])[1, :] .+ Vector(C1[t, 1:n])
            abs(dot(row .- ref, σ)) / dot(abs.(ref), abs.(σ))
        end
        @test maximum(e) < 1e-13
    end
end

@testset "sctl_gl_potential_corrections: single layer" begin
    # on a node of a proxied panel SCTL has no self row (its nodes are not the panel's): an error
    b5 = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 5, 0.4, 4.0, 1.0, Float64)
    X5 = reduce(hcat, collect.(b5.panels[1].points))
    @test_throws ArgumentError sctl_gl_potential_corrections(b5, [1], X5; tol = 1e-10)
    @test_throws ArgumentError sctl_gl_potential_corrections(b5, [1], X5[:, 3:3] .+ 1e-15; tol = 1e-10)
    # n_quad = 8: on a node the Duffy self row, just off it the near rule; the single layer is continuous
    box = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.4, 4.0, 1.0, Float64)
    np = length(box.panels); N = BI.num_points(box)
    P = box.panels[1]; n = BI.num_points(P)
    X1 = reduce(hcat, collect.(P.points))
    T = hcat(X1, X1 .+ 1e-9 .* collect(P.normal))
    Cp = sctl_gl_potential_corrections(box, 1:np, T; tol = 1e-10)
    @test size(Cp) == (2n, N) && all(isfinite, nonzeros(Cp))
    Y = gl_nodes(box)
    σ = [cos(2Y[1, j]) + Y[2, j] * Y[3, j] + 0.5 for j in 1:N]
    u = gl_sl_nodes(box, σ, T) .+ Cp * σ
    @test maximum(abs, u[1:n] .- u[n+1:2n]) < 1e-8                # 3.1e-9 measured (|u| ≤ 0.8)
    # the rows of panel 1's nodes are SCTL's Duffy self rows: 8 of its 64 nodes are SCTL's only to rounding, and
    # there SCTL's off-node rule would be off by 3% of the row
    cols = panel_column_ranges(box)[1]
    el = QuadElemList(8, 1) do e, u, v
        a, b, c, d = P.corners
        collect((a .+ b .+ c .+ d) ./ 4 .+ (b .- a) .* (u - 0.5) .+ (d .- a) .* (v - 0.5))
    end
    @test count(t -> first(node_coord(el))[:, t] != X1[:, t], 1:n) > 0
    q = ElemQuadrature(Laplace3D_FxU())
    S = self_matrices(q, el; tol = 1e-10)[:, :, 1]
    close(q); close(el)
    w1 = BI.all_weights(box)[cols]
    ref = S .- [i == j ? 0.0 : w1[j] / (4π * norm(X1[:, i] .- X1[:, j])) for i in 1:n, j in 1:n]
    @test maximum(abs, Matrix(Cp[1:n, cols]) .- ref) < 1e-12 * maximum(abs, ref)        # 0 measured
    # against cubature over panel 1 (columns of panel 1 only): above its nodes and across its edge
    Xb, _ = BI._interface_targets_normals(box)
    across = argmin(t -> abs(dot(collect(P.normal), Xb[:, t] .- X1[:, 1])) > 1e-12 ?
                         minimum(norm(Xb[:, t] .- X1[:, j]) for j in 1:n) : Inf, axes(Xb, 2))
    Tq = hcat(X1[:, [1, 20, 37]] .+ [1e-2, 1e-4, 1e-7]' .* collect(P.normal), Xb[:, across])
    C1 = sctl_gl_potential_corrections(box, [1], Tq; tol = 1e-10)
    @test nnz(C1[:, setdiff(1:N, cols)]) == 0
    σ1 = σ[cols]
    w1 = BI.all_weights(box)[cols]
    u1 = [sum(w1[j] * σ1[j] / (4π * norm(Tq[:, t] .- X1[:, j])) for j in 1:n) for t in axes(Tq, 2)] .+ C1[:, cols] * σ1
    uref = [gl_sl_ref(P, σ1, Tq[:, t]) for t in axes(Tq, 2)]
    @test maximum(abs, u1 .- uref) < 1e-10 * maximum(abs, uref)                         # 4.7e-12 measured
end

@testset "sctl_gl: argument errors" begin
    gcq = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 4.0, 1.0; nlev = 1, n_quad = 4)
    both(itf, ks) = (@test_throws ArgumentError sctl_gl_corrections(itf, ks; tol = 1e-10);
                     @test_throws ArgumentError sctl_gl_potential_corrections(itf, ks, [2.0, 2.0, 2.0][:, :]; tol = 1e-10))
    # a GCQ edge panel
    both(gcq, [findfirst(p -> p.kind == :edge, gcq.panels)])
    # a curved panel whose order is not one of SCTL's (6), or not the same in u and v
    s6 = curved_gcq_mesh(gl_sphere; n_quad = 6, m = 1).interface
    both(s6, [1])
    P = curved_gcq_mesh(gl_sphere; n_quad = 8, m = 2).interface.panels[1]
    uneven = CurvedPanel(P.anchor, P.lmap, BI.gl_tensor_rule(8), BI.gl_tensor_rule(4); flip = P.flip, dmap = P.dmap)
    both(DielectricInterface([uneven], [4.0], [1.0]), [1])
    # a GL TensorPanel of aspect ratio 5 (above GL_MAX_ASPECT)
    k = findfirst(is_gl_panel, gcq.panels); p = gcq.panels[k]
    ps = copy(gcq.panels)
    ps[k] = TensorPanel(p.normal, p.anchor, p.o, p.eu, p.ev, p.Lu, p.Lu / 5, p.ru, p.rv, :gl, p.plane)
    both(DielectricInterface(ps, gcq.eps_in, gcq.eps_out), [k])
    # a CurvedPanel whose Gauss-Legendre nodes moved by 1e-7: they are not SCTL's element nodes
    r = BI.gl_tensor_rule(8)
    rm = TensorRule1D(r.x .+ 1e-7, r.w, r.segs, r.z0, r.wz, r.P, 0.0)
    moved = CurvedPanel(P.anchor, P.lmap, rm, rm; flip = P.flip, dmap = P.dmap)
    both(DielectricInterface([moved], [4.0], [1.0]), [1])
    # more than 20 nodes per direction
    ns, ws = BI.gausslegendre(21)
    big = BI.rect_panel3d_discretize((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), ns, ws,
                                     (0.0, 0.0, 1.0))
    both(DielectricInterface([big], [4.0], [1.0]), [1])
    # an unknown kernel
    box = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.4, 4.0, 1.0, Float64)
    @test_throws ArgumentError sctl_gl_corrections(box, [1]; tol = 1e-10, kernel = :S)
end
