using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
Base.find_package("SCTL") === nothing || @eval using SCTL

@testset "GCQ: batched RHS refinement and GCQPotential" begin
    # batched refinement gives the same panels as the pointwise one
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0), (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    src = BI.PointSource((0.3, 0.1, 0.62), 1.0)
    rhs(p, n) = -BI.laplace3d_grad(src.point, p, n)
    m1 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; nlev = 1, n_quad = 4, rhs = rhs, rhs_atol = 1e-5)
    nb = Ref(0)
    rhsb(P, N) = (nb[] += 1; [rhs(p, n) for (p, n) in zip(P, N)])
    m2 = multi_dielectric_box3d_gcq(boxes, [4.0, 2.0], 1.0; nlev = 1, n_quad = 4, rhs_batch = rhsb, rhs_atol = 1e-5)
    key(p) = (round.(p.anchor .+ p.o, digits = 12), round(p.Lu, digits = 12), round(p.Lv, digits = 12), p.kind)
    @test sort(key.(m1.interface.panels)) == sort(key.(m2.interface.panels))
    @test length(m1.interface.panels) > 96             # it refined
    @test nb[] < 30                                    # one call per level

    # GCQPotential == potential_tensor (direct), on and near edges and corners
    ε = 10.0
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = 3, rule = edge_gcq_rule(ε, 1.0; tol = 1e-4))
    X, Nn = BI._interface_targets_normals(itf)
    rng = MersenneTwister(4)
    σ = randn(rng, BI.num_points(itf))
    T = hcat(rand(rng, 3, 40) .* 2 .- 1, [0.5 + 1e-3, 0.5 + 2e-4, 0.3], [0.501, 0.499, 0.4999], X[:, 1:47:end] .+ 1e-3 .* Nn[:, 1:47:end])
    uref = BI.potential_tensor(itf, σ, [Tuple(T[:, t]) for t in axes(T, 2)])
    P = potential_operator(GCQPotential(far = FMM3DFarField(1e-13)), itf, T)
    @test size(P) == (size(T, 2), BI.num_points(itf))
    @test maximum(abs, P * σ - uref) / maximum(abs, uref) < 1e-11

    # gcq_far = :direction: the near distance of edge panels against their GCQ direction; same results at the tolerance
    Pd = potential_operator(GCQPotential(far = FMM3DFarField(1e-13), gcq_far = :direction), itf, T)
    @test maximum(abs, Pd * σ - uref) / maximum(abs, uref) < 1e-9
    ud = BI.potential_tensor(itf, σ, [Tuple(T[:, t]) for t in axes(T, 2)]; gcq_far = :direction)
    @test maximum(abs, ud - uref) / maximum(abs, uref) < 1e-9
    Cm = BI.near_corrections(GCQNearQuad(), itf)
    Cd = BI.near_corrections(GCQNearQuad(gcq_far = :direction), itf)
    @test nnz(Cd) <= nnz(Cm)
    A = BI.laplace3d_DT(itf)                       # node rule, dense
    @test norm((A + Cd) * σ - (A + Cm) * σ) / norm((A + Cm) * σ) < 1e-9
    L1 = first(BI.lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-13))
    L2 = first(BI.lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-13; gcq_far = :direction))
    @test norm(L2 * σ - L1 * σ) / norm(L1 * σ) < 1e-9
    @test_throws ArgumentError BI.tensor_isfar(itf.panels[1], (5.0, 5.0, 5.0); gcq_far = :nearest)
end

@testset "GL panels: potential from SCTL" begin
    ε = 10.0
    G(d) = 1 / (4π * sqrt(d[1]^2 + d[2]^2 + d[3]^2))
    rng = MersenneTwister(11)
    # GL panels as proxies of order 4 (n_quad 3) and as SCTL elements themselves (n_quad 4)
    boxes = Dict(nq => single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = nq,
                                                   rule = edge_gcq_rule(ε, 1.0; tol = 1e-4)) for nq in (3, 4))
    for nq in (3, 4)
        itf = boxes[nq]
        ps = itf.panels
        offs = cumsum([0; BI.num_points.(ps)])
        gl = findall(is_gl_panel, ps); gcqp = setdiff(eachindex(ps), gl)
        @test !isempty(gl) && !isempty(gcqp)
        X, Nn = BI._interface_targets_normals(itf)
        σ = randn(rng, BI.num_points(itf))
        # 50 targets within 1e-3 of GL panels: GL nodes moved by 5e-4 to 1e-3 along the normal, to either side
        glnodes = reduce(vcat, [collect(offs[k]+1:offs[k+1]) for k in gl])
        idx = glnodes[rand(rng, 1:length(glnodes), 50)]
        T = X[:, idx] .+ (rand(rng, (-1, 1), 50) .* (5e-4 .+ 5e-4 .* rand(rng, 50)))' .* Nn[:, idx]
        Tt = [Tuple(T[:, t]) for t in axes(T, 2)]
        nr(x, k) = sum(ps[k].weights[j] * σ[offs[k] + j] * G(x .- ps[k].points[j]) for j in eachindex(ps[k].points))
        # the GCQ panels as before: their node rule where tensor_isfar accepts the pair, else the product-integration row
        gcq_part(x, qt) = sum(gcqp) do k
            xl = x .- ps[k].anchor
            BI.tensor_isfar(ps[k], xl; quad_tol = qt) ? nr(x, k) : dot(BI.tensor_near_row(G, ps[k], xl), σ[offs[k]+1:offs[k+1]])
        end
        # quad_tol 1e-6: SCTL's rows differ from tensor_near_row's (by 2e-6 to 3e-6 of max|u|)
        for qt in (1e-12, 1e-6)
            u = BI.potential_tensor(itf, σ, Tt; quad_tol = qt)
            # the GL panels: their node rule plus SCTL's correction. The two cancel down to max|u| ~ 0.06 from terms of
            # up to max|Cσ| ~ 2: the rounding is measured against that (4e-15 of it, 1.3e-13 of max|u|)
            Cσ = sctl_gl_potential_corrections(itf, gl, T; tol = qt) * σ
            ref = [sum(k -> nr(x, k), gl) for x in Tt] .+ Cσ .+ [gcq_part(x, qt) for x in Tt]
            @test maximum(abs, u - ref) < 1e-13 * maximum(abs, Cσ)
        end
        u = BI.potential_tensor(itf, σ, Tt; quad_tol = 1e-12)
        P = potential_operator(GCQPotential(quad_tol = 1e-12), itf, T)
        @test maximum(abs, P * σ - u) / maximum(abs, u) < 1e-10
    end

    # on the surface: a target on a GL node and the same node moved by 1e-15 of the panel's size are snapped to the node,
    # so the node rule (direct or by the FMM) and SCTL's correction skip the same pair (n_quad 4: SCTL's Duffy self row)
    itf = boxes[4]
    ps = itf.panels
    σ = randn(rng, BI.num_points(itf))
    gl = findall(is_gl_panel, ps)
    diam(p) = norm(maximum(reduce(hcat, collect.(p.points)); dims = 2) - minimum(reduce(hcat, collect.(p.points)); dims = 2))
    e = normalize([0.6, -0.48, 0.64])
    for k in gl[[1, end ÷ 2, end]], j in (1, 6, 16)
        y = collect(ps[k].points[j])
        Tn = [y y .+ 1e-15 * diam(ps[k]) .* e]
        @test Tn[:, 2] != Tn[:, 1]
        un = BI.potential_tensor(itf, σ, [Tuple(Tn[:, t]) for t in 1:2]; quad_tol = 1e-12)
        ug = potential_operator(GCQPotential(quad_tol = 1e-12, far = FMM3DFarField(1e-13)), itf, Tn) * σ
        @test all(isfinite, un) && all(isfinite, ug)
        @test abs(un[2] - un[1]) <= 1e-10 * abs(un[1])
        @test abs(ug[2] - ug[1]) <= 1e-10 * abs(ug[1])
        @test abs(ug[1] - un[1]) <= 1e-10 * abs(un[1])
    end
    # a GL panel much smaller than the box of nodes and targets (a far target makes it ~1000 here): FMM3D skips pairs
    # closer than ~2^-51 × box, far above 1e-13 of the panel's size (2e-14), so the snap radius is at least 2^-48 × box
    # (3.6e-12): a node moved by 1e-13 (FMM3D skips it) or by 1e-12 (~1e-15 × box) gives the value on the node
    far = [1000.0, 0.1, 0.2]
    k = gl[end ÷ 2]
    y = collect(ps[k].points[6])
    @test 1e-13 * diam(ps[k]) < 1e-13 < 2.0^-51 * 1000
    for δ in (1e-13, 1e-12)
        Tn = [y y .+ δ .* e far]
        @test Tn[:, 2] != Tn[:, 1]
        un = BI.potential_tensor(itf, σ, [Tuple(Tn[:, t]) for t in axes(Tn, 2)]; quad_tol = 1e-12)
        ug = potential_operator(GCQPotential(quad_tol = 1e-12, far = FMM3DFarField(1e-13)), itf, Tn) * σ
        @test all(isfinite, un) && all(isfinite, ug)
        @test abs(un[2] - un[1]) <= 1e-10 * abs(un[1])
        @test abs(ug[2] - ug[1]) <= 1e-10 * abs(ug[1])
        @test abs(ug[1] - un[1]) <= 1e-10 * abs(un[1])
    end
    # on a node of a proxied GL panel (n_quad 3: proxy elements of order 4) SCTL has no self row
    p3 = boxes[3].panels[findfirst(is_gl_panel, boxes[3].panels)]
    @test_throws ArgumentError BI.potential_tensor(boxes[3], randn(rng, BI.num_points(boxes[3])), [p3.points[1]])
end
