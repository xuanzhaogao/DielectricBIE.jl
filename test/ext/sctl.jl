# Operator backends (far field x near quadrature), with the SCTL.jl extension when SCTL is installed.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using Test

const HAVE_SCTL = Base.find_package("SCTL") !== nothing
HAVE_SCTL && @eval using SCTL

relerr(a, b) = norm(a - b) / norm(b)

# single-layer potential of the interpolated density at x, by adaptive cubature over every panel (reference)
function _ref_potential(itf, σ, x; rtol = 1e-12)
    acc = 0.0; off = 0
    for p in itf.panels
        nq = p.n_quad; vals = permutedims(reshape(σ[off+1:off+nq^2], nq, nq)); off += nq^2
        a, b, c, d = p.corners; cc = collect((a .+ b .+ c .+ d) ./ 4); bma = collect(b .- a); dma = collect(d .- a)
        area = norm(bma) * norm(dma); rx = zeros(nq); ry = zeros(nq)
        f = uv -> (BI.barycentric_row!(rx, p.gl_xs, p.bary_weights, uv[1]); BI.barycentric_row!(ry, p.gl_xs, p.bary_weights, uv[2]);
                   dot(rx, vals * ry) / (4π * norm(x .- (cc .+ bma .* (uv[1] / 2) .+ dma .* (uv[2] / 2)))) * area / 4)
        acc += BI.HCubature.hcubature(f, BI.StaticArrays.SVector(-1.0, -1.0), BI.StaticArrays.SVector(1.0, 1.0);
                                      rtol = rtol, atol = 1e-15, maxevals = 10^7)[1]
    end
    return acc
end

# a self-defined near quadrature: any NearQuadrature with a near_corrections method
struct WrappedQuad <: BI.NearQuadrature
    inner::BI.NearQuadrature
end
BI.near_corrections(q::WrappedQuad, interface) = 2 .* BI.near_corrections(q.inner, interface)

@testset "operator backends" begin
    rng = MersenneTwister(7)
    itf = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.2, 4.0, 1.0, Float64)
    n = BI.num_points(itf)
    X = randn(rng, n, 3)

    # FMM3D + DielectricBIE's quadrature reproduces the batched operator
    ref_op = BI.batched_lhs_dielectric_box3d_fmm3d_corrected(itf, 1e-12, 1e-10, 16)
    A = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = UpsampledNearQuad(up_tol = 1e-10, max_order = 16))
    Yref = ref_op * X
    @test relerr(A * X, Yref) < 1e-12
    @test relerr(A * X[:, 2], Yref[:, 2]) < 1e-12            # vector = one column of the block

    q = UpsampledNearQuad(up_tol = 1e-10, max_order = 16)
    C = BI.near_corrections(q, itf)
    A2 = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = WrappedQuad(q))
    @test relerr(A2 * X - A * X, C * X) < 1e-10
    # ... or a matrix, or a function of the interface
    @test relerr(dielectric_operator(itf; far = FMM3DFarField(1e-12), near = C) * X, A * X) < 1e-14
    @test relerr(dielectric_operator(itf; far = FMM3DFarField(1e-12), near = i -> C) * X, A * X) < 1e-14
    @test_throws DimensionMismatch dielectric_operator(itf; near = spzeros(3, 3))
    @test_throws ErrorException dielectric_operator(itf; far = FMM3DFarField(1e-12), near = SCTLNearQuad())
    @test default_far_field() isa SCTLFarField              # with SCTL loaded: the default pairs with SCTLNearQuad

    # GCQ interface: same operator as lhs_dielectric_box3d_gcq_fmm3d
    ε = 10.0
    gcq = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, ε, 1.0; nlev = 1, n_quad = 4,
                                      rule = edge_gcq_rule(ε, 1.0; tol = 1e-4))
    xg = randn(rng, BI.num_points(gcq))
    Ag_ref, _ = lhs_dielectric_box3d_gcq_fmm3d(gcq, 1e-12)
    Ag = dielectric_operator(gcq; far = FMM3DFarField(1e-12))           # default near: GCQNearQuad()
    @test relerr(Ag * xg, Ag_ref * xg) < 1e-12
    # corrections restricted to source panels: the GCQ method and the generic fallback (drop the other columns)
    gl = findall(p -> p.kind == :gl, gcq.panels); other = setdiff(eachindex(gcq.panels), gl)
    Cg_all = BI.near_corrections(GCQNearQuad(), gcq)
    @test BI.near_corrections(GCQNearQuad(), gcq, gl) + BI.near_corrections(GCQNearQuad(), gcq, other) ≈ Cg_all
    @test BI.near_corrections(CorrectionMatrix(Cg_all), gcq, other) == BI.near_corrections(GCQNearQuad(), gcq, other)
    @test default_polynomial_panel(gcq.panels[gl[1]]) && !default_polynomial_panel(gcq.panels[other[1]])

    # campaign [backend] table
    mktempdir() do dir
        base = read(joinpath(@__DIR__, "..", "fixtures", "system_small.toml"), String)
        cfg(extra) = (f = joinpath(dir, "c.toml"); write(f, base * extra); load_campaign(f))
        @test BI.campaign_backend(cfg("")) == (nothing, nothing)
        far, near = BI.campaign_backend(cfg("\n[backend]\nfar = \"sctl\"\nnear = \"sctl\"\n"))
        @test far isa SCTLFarField && near isa SCTLNearQuad
        @test BI.campaign_uses_sctl(cfg("\n[backend]\nfar = \"sctl\"\n"))
        @test_throws ErrorException BI.campaign_backend(cfg("\n[backend]\nnear = \"sctl\"\n"))
        @test_throws ErrorException BI.campaign_backend(cfg("\n[backend]\nfar = \"fmm2d\"\n"))
        @test BI.campaign_potential(cfg("")) === nothing
        @test BI.campaign_potential(cfg("\n[backend]\npotential = \"sctl\"\n")) isa SCTLPotential
        @test BI.campaign_uses_sctl(cfg("\n[backend]\npotential = \"sctl\"\n"))
        @test_throws ErrorException BI.campaign_potential(cfg("\n[backend]\npotential = \"fmm\"\n"))
    end

    if HAVE_SCTL
        # SCTL's FMM for the node rule: a direct sum below 40000 nodes, so equal to FMM3D at tight tolerance
        As = dielectric_operator(itf; far = SCTLFarField(tol = 1e-12), near = UpsampledNearQuad(up_tol = 1e-10, max_order = 16))
        @test relerr(As * X, A * X) < 1e-10
        # all of SCTL: its own singular quadrature (an independent discretization of the same operator)
        Aq = dielectric_operator(itf; far = SCTLFarField(tol = 1e-10), near = SCTLNearQuad())
        @test relerr(Aq * X, A * X) < 1e-7
        # mixed orders (one element list per order, nodes permuted back)
        two = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 2.0, 1.0, Float64)
        far_box = BI.DielectricInterface([BI.rect_panel3d_discretize(p.corners[1] .+ (3.0, 0.0, 0.0), p.corners[2] .+ (3.0, 0.0, 0.0),
                                              p.corners[3] .+ (3.0, 0.0, 0.0), p.corners[4] .+ (3.0, 0.0, 0.0),
                                              p.gl_xs, p.gl_ws, p.normal; is_edge = p.is_edge) for p in itf.panels],
                                         itf.eps_in, itf.eps_out)
        mixed = BI.DielectricInterface(vcat(far_box.panels, two.panels), vcat(far_box.eps_in, two.eps_in),
                                       vcat(far_box.eps_out, two.eps_out))
        xm = randn(rng, BI.num_points(mixed))
        Am_ref = dielectric_operator(mixed; far = FMM3DFarField(1e-12), near = UpsampledNearQuad(up_tol = 1e-11, max_order = 16))
        Am = dielectric_operator(mixed; far = SCTLFarField(tol = 1e-11), near = SCTLNearQuad())
        @test relerr(Am * xm, Am_ref * xm) < 1e-7
        @test_throws ArgumentError dielectric_operator(BI.single_dielectric_box3d(1.0, 1.0, 1.0, 6, 0.5, 2.0, 1.0, Float64);
                                                       far = SCTLFarField(), near = SCTLNearQuad())
        # GCQ with SCTL's FMM
        Ags = dielectric_operator(gcq; far = SCTLFarField(tol = 1e-12))
        @test relerr(Ags * xg, Ag_ref * xg) < 1e-10

        # SCTL's near correction as a matrix: with FMM3D's far field it is SCTL's operator again
        Anc = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = SCTLNearCorrection(tol = 1e-10))
        @test relerr(Anc * X, Aq * X) < 1e-8          # quadratures at 1e-10 vs 1e-12
        @test relerr(dielectric_operator(mixed; far = FMM3DFarField(1e-12), near = SCTLNearCorrection(tol = 1e-11)) * xm,
                     Am_ref * xm) < 1e-7

        # a near quadrature written in Julia on SCTL's classification: DielectricBIE's adaptive quadtree for the
        # accurate rows (its moments use the panel's Lagrange basis, so they act on nodal values), the node rule where
        # D^T vanishes (targets in the panel's plane)
        small = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 4.0, 1.0, Float64)
        nl = sctl_near_list(small; tol = 1e-10)
        @test nl isa SCTL.NearList && SCTL.nelem(nl) == length(small.panels)
        Xs, Ns = BI._interface_targets_normals(small)
        cfg = BI.AdaptiveConfig(atol = 1e-12)
        offs = cumsum([0; [BI.num_points(p) for p in small.panels]])
        owner = reduce(vcat, [fill(k, BI.num_points(p)) for (k, p) in enumerate(small.panels)])
        acc = map(eachindex(small.panels)) do k
            p = small.panels[k]
            ts = SCTL.near_targets(nl, k)
            @test all(t -> t in ts, offs[k]+1:offs[k+1])      # a panel's own nodes are near it
            B = dt_node_rule_block(small, k, ts)
            for k2 in unique(owner[ts])
                coplanar = abs(dot(p.normal, small.panels[k2].normal)) > 1 - 1e-12 &&
                           abs(dot(p.normal, small.panels[k2].points[1] .- p.points[1])) < 1e-12
                coplanar && continue
                rows = findall(t -> owner[t] == k2, ts)
                K = zeros(length(rows), BI.num_points(p))
                BI.adaptive_block_moments_tree!(K, p, [Tuple(Xs[:, t]) for t in ts[rows]], small.panels[k2].normal, :DT, cfg)
                B[rows, :] = K
            end
            B
        end
        plain = [dt_node_rule_block(small, k, SCTL.near_targets(nl, k)) for k in eachindex(small.panels)]
        Cj = SCTL.assemble_near_correction(nl, acc, [BI.num_points(p) for p in small.panels]; subtract = plain)
        xs = randn(rng, BI.num_points(small))
        Aj = dielectric_operator(small; far = FMM3DFarField(1e-13), near = Cj)
        As_ = dielectric_operator(small; far = FMM3DFarField(1e-13), near = SCTLNearCorrection(tol = 1e-12))
        @test relerr(Aj * xs, As_ * xs) < 1e-9
        @test sctl_near_list(small; radius = 0.0).trg == Int[]
        @test_throws DimensionMismatch sctl_near_list(small; radius = [1.0])

        # SCTL's quadrature for self-defined panels through proxies: GCQ's Gauss-Legendre panels by SCTL, the edge and
        # corner panels by the GCQ fallback; FlatPanels all by SCTL (= SCTLNearCorrection)
        Agp = dielectric_operator(gcq; far = FMM3DFarField(1e-12), near = SCTLProxyNearQuad(tol = 1e-12))
        @test relerr(Agp * xg, Ag_ref * xg) < 1e-10
        Afp = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = SCTLProxyNearQuad(tol = 1e-10))
        @test relerr(Afp * X, Anc * X) < 1e-12
        # a predicate choosing no panel leaves everything to the fallback
        Agn = dielectric_operator(gcq; far = FMM3DFarField(1e-12), near = SCTLProxyNearQuad(polynomial = p -> false))
        @test relerr(Agn * xg, Ag_ref * xg) < 1e-12
        @test_throws ArgumentError BI.near_corrections(SCTLProxyNearQuad(order = 6), itf)

        # off-surface potential: SCTL's operator against an independent cubature, down to 1e-4 from the surface
        Xp, Np = BI._interface_targets_normals(small)
        σp = [cos(2Xp[1, i]) + Xp[2, i] * Xp[3, i] + 0.5 for i in 1:BI.num_points(small)]
        idp = [1, 17, 40, 77, 123, 200, 301, 333]
        Tp = hcat(Xp[:, idp] .+ (10.0 .^ -(1:length(idp)) ./ 2)' .* Np[:, idp], [0.1 0.9; 0.2 -0.3; 0.05 1.4])
        Pp = potential_operator(SCTLPotential(tol = 1e-10), small, Tp)
        @test size(Pp) == (size(Tp, 2), BI.num_points(small))
        up = Pp * σp
        uref = [_ref_potential(small, σp, Tp[:, t]) for t in axes(Tp, 2)]
        @test maximum(abs, up - uref) / maximum(abs, uref) < 1e-9
        Σp = [σp randn(rng, length(σp))]
        # linear up to the rounding of node rule minus correction, ~1e-16 / (distance to the nearest node): the targets
        # here reach 5e-9 from a node
        @test Pp * (Σp[:, 1] + 2Σp[:, 2]) ≈ Pp * Σp[:, 1] + 2 * (Pp * Σp[:, 2]) rtol = 1e-8
        @test isempty(potential_operator(SCTLPotential(), small, zeros(3, 0)) * σp)
        # n_quad without an SCTL element order: proxies of order 8, the density interpolated (exact)
        six = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 6, 0.5, 4.0, 1.0, Float64)
        X6, N6 = BI._interface_targets_normals(six)
        σ6 = [cos(2X6[1, i]) + X6[2, i] * X6[3, i] + 0.5 for i in 1:BI.num_points(six)]
        T6 = X6[:, [3, 50, 101]] .+ [1e-2, 1e-4, 1e-6]' .* N6[:, [3, 50, 101]]
        u6 = potential_operator(SCTLPotential(tol = 1e-10), six, T6) * σ6
        @test maximum(abs, u6 - [_ref_potential(six, σ6, T6[:, t]) for t in 1:3]) / maximum(abs, u6) < 1e-9
        @test_throws ArgumentError potential_operator(SCTLPotential(), gcq, zeros(3, 1))   # GCQ edge panels
        # the HCubature evaluator is the previous function
        Ph = potential_operator(HCubaturePotential(1e-10), small, Tp[:, end-1:end])
        @test Ph * σp ≈ BI.laplace3d_pottrg_fmm3d_corrected_hcubature(small, Tp[:, end-1:end], 1e-10, 1e-10, 5.0) * σp

        # GCQPotential with SCTL's far field
        Tg = hcat(rand(rng, 3, 20) .* 2 .- 1, [0.501, 0.499, 0.4999])
        ug = potential_operator(GCQPotential(far = SCTLFarField(tol = 1e-12)), gcq, Tg) * xg
        @test maximum(abs, ug - BI.potential_tensor(gcq, xg, [Tuple(Tg[:, t]) for t in axes(Tg, 2)])) / maximum(abs, ug) < 1e-10

        # batch mode: block GMRES through solve_dielectric_box3d_block (two sources, one solve)
        xs = -0.3:0.3:0.3
        P = [(x, y, z) for x in xs for y in xs for z in xs]
        pos = reduce(hcat, collect.(P))
        vss = [VolumeSource(pos, fill(0.027, length(P)), [exp(-((p[1] - c)^2 + p[2]^2 + p[3]^2) / 0.1) for p in P])
               for c in (0.0, 0.2)]
        bitf = multi_dielectric_box3d_rhs_adaptive(4, 2.0, [(center = (0.0, 0.0, 0.0), Lx = 4.0, Ly = 4.0, Lz = 4.0)],
                                                   [11.7], vss, 1e-2; eps_out = 1.0)
        kw = (fmm_tol = 1e-10, up_tol = 1e-10, max_order = 16, rtol = 1e-11, itmax = 300)
        Σ0, _ = solve_dielectric_box3d_block(bitf, vss; kw...)
        Σ1, _ = solve_dielectric_box3d_block(bitf, vss; kw..., far = SCTLFarField(tol = 1e-10))
        Σ2, _ = solve_dielectric_box3d_block(bitf, vss; kw..., far = SCTLFarField(tol = 1e-10), near = SCTLNearQuad())
        @test size(Σ1) == size(Σ0) == (BI.num_points(bitf), 2)
        @test relerr(Σ1, Σ0) < 1e-8
        @test relerr(Σ2, Σ0) < 1e-6
        # ... and the potential at the sources' grid points with either evaluator: the results differ by exactly the
        # evaluators' difference, which is now rounding (before the HCubature path corrected every near pair it was off
        # by ~5e-4 on this coarse 24-panel interface)
        Ph_b = potential_operator(HCubaturePotential(1e-10), bitf, pos)
        Ps_b = potential_operator(SCTLPotential(tol = 1e-10), bitf, pos)
        dU = reduce(hcat, [Ps_b * Σ0[:, a] - Ph_b * Σ0[:, a] for a in 1:2])
        Φh = BI.evaluate_batch_potential(bitf, Σ0, vss, pos; lhs_tol = 1e-10, volume_tol = 1e-8)
        Φs = BI.evaluate_batch_potential(bitf, Σ0, vss, pos; lhs_tol = 1e-10, volume_tol = 1e-8,
                                         potential = SCTLPotential(tol = 1e-10))
        @test maximum(abs, (Φs - Φh) - dU) < 1e-12 * maximum(abs, Φs)
        @test maximum(abs, Φs - Φh) < 1e-9 * maximum(abs, Φs)
        Vh = BI.four_index_matrix(bitf, vss, Σ0; lhs_tol = 1e-10, volume_tol = 1e-8)
        Vs = BI.four_index_matrix(bitf, vss, Σ0; lhs_tol = 1e-10, volume_tol = 1e-8, potential = SCTLPotential(tol = 1e-10))
        tw = [v.weights .* v.density for v in vss]
        @test maximum(abs, (Vs - Vh) - [dot(tw[a], dU[:, b]) for a in 1:2, b in 1:2]) < 1e-12 * maximum(abs, Vs)
        sref = [_ref_potential(bitf, Σ0[:, 1], pos[:, t]) for t in 1:4:size(pos, 2)]
        @test maximum(abs, Ps_b * Σ0[:, 1] - [_ref_potential(bitf, Σ0[:, 1], pos[:, t]) for t in axes(pos, 2)]) /
              maximum(abs, sref) < 1e-12
    else
        @test_throws ErrorException dielectric_operator(itf; far = SCTLFarField())
        @test_throws ErrorException dielectric_operator(itf; near = SCTLNearCorrection())
        @test_throws ErrorException dielectric_operator(gcq; near = SCTLProxyNearQuad())
        @test_throws ErrorException potential_operator(SCTLPotential(), itf, zeros(3, 1))
        @info "SCTL.jl not installed: skipping the SCTL backend tests"
    end
end

@testset "point_charge_field: SCTL vs FMM3D" begin
    rng = MersenneTwister(21)
    S = rand(rng, 3, 50000); q = randn(rng, 50000); T = rand(rng, 3, 45000) .+ [1.5, 0.0, 0.0]   # 95k points: the FMM
    if HAVE_SCTL
        for tol in (1e-6, 1e-10)
            u0 = point_charge_field(FMM3DFarField(1e-14), S, q, T); g0 = point_charge_field(FMM3DFarField(1e-14), S, q, T; grad = true)
            u = point_charge_field(SCTLFarField(tol = tol), S, q, T); g = point_charge_field(SCTLFarField(tol = tol), S, q, T; grad = true)
            @test size(u) == size(u0) && size(g) == size(g0)
            @test maximum(abs, u - u0) < tol * maximum(abs, u0)
            @test maximum(abs, g - g0) < tol * maximum(abs, g0)
        end
        @test size(point_charge_field(SCTLFarField(), S, q, zeros(3, 0); grad = true)) == (3, 0)
        # the RHS of a volume source through either evaluator
        gsrc = BI.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
        iface = BI.single_dielectric_box3d(8.0, 8.0, 8.0, 3, 1.0, 4.0, 1.0)
        r0 = rhs_dielectric_box3d(iface, gsrc, 2.0, FMM3DFarField(1e-12))
        @test maximum(abs, rhs_dielectric_box3d(iface, gsrc, 2.0, SCTLFarField(tol = 1e-10)) - r0) < 1e-10 * maximum(abs, r0)
    else
        @test_throws ErrorException point_charge_field(SCTLFarField(), S, q, T)
    end
end

@testset "GCQ campaign mode: SCTL and FMM3D far fields agree" begin
    if HAVE_SCTL
        include(joinpath(@__DIR__, "..", "campaign", "fixture_campaign.jl"))
        dir = mktempdir(); toml = write_fixture_campaign(dir); txt = read(toml, String)
        two = "[[orbital]]\ntype = 1\nx = 5.4\ny = 1.8\nz = 1.8\n"
        write(toml, replace(txt, two => "[[orbital]]\ntype = 1\nx = 3.6\ny = 1.8\nz = 1.8\n\n" * two))
        c = load_campaign(toml)
        kw = (; nlev = 1, n_quad = 4, rhs_tol = 1e-4, lhs_tol = 1e-9, gmres_rtol = 1e-10, volume_tol = 1e-8,
                quad_tol = 1e-6, edge_tol = 1e-6, tau = 1e-10, fg_tol = 1e-9, verbose = false)
        rf = gcq_four_index(c; kw..., far = :fmm3d)
        rs = gcq_four_index(c; kw..., far = :sctl)
        @test rs.stats.N == rf.stats.N
        @test maximum(abs, rs.V - rf.V) < 1e-6 * maximum(abs, rf.V)
    else
        @test_throws ErrorException point_charge_field(SCTLFarField(), zeros(3, 1), [1.0], ones(3, 1))
    end
end

# the surface corrections of cut densities in the GCQ campaign mode (SCTL's quadrature)
HAVE_SCTL && include(joinpath(@__DIR__, "sctl_surface.jl"))
