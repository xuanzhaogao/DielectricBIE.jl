# Accuracy of a GCQ rule on its own training family (plan 2026-11-06-gcq-family-accuracy-and-stability, Tasks 1-3):
# - training-grid resolution of every original regularized component f_j(s) = s^q_j before compression;
# - analytic reconstruction of every f_j by the retained interpolant from its exact nodal values, on independent holdouts
#   (domains: the selected-node range [x_min, x_max]; all positive cells; the terminal element, reported apart).
# The default construction's known failures are @test_broken; the opt-in `grid_tol` / `density_tol` must pass.
using Test, LinearAlgebra, Random
using DielectricBIE
import DielectricBIE as DB

const FA_DP840 = edge_density_powers(2.0, 1.0; angle = acos(-1 / 3))     # an octahedron edge: 840 components
fa_q(er) = er.pows .- er.pw                                               # the regularized exponents
fa_cheb2(N) = [-cos(π * j / (N - 1)) for j in 0:N-1]
fa_holdout() = vcat(-1.0, [-cos((2j - 1) * π / 82) for j in 1:41], 1.0)    # 43 per cell, independent of 2nd-kind points

# max over positive cells and components of |I_cell f_j - f_j| (degree len(z0)-1 interpolation at the cell's GL points), at
# N independent second-kind Chebyshev points; components are normalized by their maximum on [0, 1] (= 1)
function fa_component_resolution(er; N = 65)
    q = fa_q(er); bw = DB.gl_barycentric_weights(er.z0, er.wz)
    τ = fa_cheb2(N); L = DB._tensor_lagrange_mat(er.z0, bw, τ)'
    e = 0.0
    for (a, b) in er.segs[2:end]
        xn = a .+ (b - a) .* (1 .+ er.z0) ./ 2; xt = a .+ (b - a) .* (1 .+ τ) ./ 2
        e = max(e, maximum(abs.(L * [x^u for x in xn, u in q] - [x^u for x in xt, u in q])))
    end
    return e
end

# analytic reconstruction errors of every component by the retained interpolant h(t) Hn⁻¹ f(x), h = f V, on the holdouts;
# prec = 0 Float64, else BigFloat arithmetic for h and the nodal solve
function fa_reconstruction(er; prec = 0)
    g = DB.gcq_global_interp(er); q = g.q; x = g.x; xmin, xmax = extrema(x)
    τ = fa_holdout(); zt, _ = DB.gausslegendre(33)
    th = reduce(vcat, [a .+ (b - a) .* (1 .+ τ) ./ 2 for (a, b) in er.segs[2:end]]); tt = er.segs[1][2] .* (1 .+ zt) ./ 2
    ts = vcat(th, tt); term = vcat(falses(length(th)), trues(length(tt)))
    if prec == 0
        H = [t^u for t in ts, u in q] * g.V; C = ([xi^u for xi in x, u in q] * g.V) \ [xi^u for xi in x, u in q]
    else
        H, C = setprecision(BigFloat, prec) do
            qb = big.(q); Vb = big.(g.V); xb = big.(x)
            Hb = [big(t)^u for t in ts, u in qb] * Vb
            Float64.(Hb), Float64.(([xi^u for xi in xb, u in qb] * Vb) \ [xi^u for xi in xb, u in qb])
        end
    end
    E = abs.(H * C - [t^u for t in ts, u in q])
    inr = .!term .& (xmin .<= ts .<= xmax)
    return (; node = maximum(E[inr, :]), positive = maximum(E[.!term, :]), terminal = maximum(E[term, :]))
end

@testset "GCQ family accuracy: training-grid resolution of the original components" begin
    e0 = edge_gcq_rule(FA_DP840; tol = 1e-12)
    r0 = fa_component_resolution(e0)
    @test r0 > 1e-6                                                   # the default grid: 7.5e-5 (study 63)
    @test_broken r0 <= 1e-13                                          # known failure of the default construction
    for (dp, tol) in ((FA_DP840, 1e-12), ([-0.6], 1e-12), ([-0.04], 1e-8))
        er = edge_gcq_rule(dp; tol, grid_tol = tol / 10)
        @test fa_component_resolution(er) <= tol / 10
        # the rule is a GCQ rule on [0, 1]: segments tile it from 0, finite map and weights, the family integrated
        @test er.segs[1][1] == 0 && er.segs[end][2] == 1 && all(er.segs[k][1] == er.segs[k - 1][2] for k in 2:length(er.segs))
        @test all(isfinite, er.P) && all(isfinite, er.w)
        @test edge_gcq_family_error(er) < 100 * tol
        @test length(er.segs) > length(edge_gcq_rule(dp; tol).segs) || dp != FA_DP840   # (the 840 family needed refinement)
    end
    # an unresolvable request is refused, not silently accepted
    @test_throws ArgumentError edge_gcq_rule([-0.6]; tol = 1e-12, grid_tol = 1e-30)
end

# the rule's actual evaluator (14-point Lagrange per fine cell through P) at independent holdouts: 31 Gauss-Legendre points
# per positive cell (neither the refinement's 33 second-kind points nor the certificate's 43 first-kind points), max over
# the selected-node range of |local(P) f_j(x) - f_j(t)| for every original component (full nodal values x^(pw+q))
function fa_evaluator(er)
    q = fa_q(er); x = er.x; xmin, xmax = extrema(x)
    bw = DB.gl_barycentric_weights(er.z0, er.wz); nz = length(er.z0)
    zg, _ = DB.gausslegendre(31)
    Fx = [xi^(er.pw + u) for xi in x, u in q]
    e = 0.0
    for (k, (a, b)) in enumerate(er.segs)
        k == 1 && continue
        t = a .+ (b - a) .* (1 .+ zg) ./ 2; keep = xmin .<= t .<= xmax
        any(keep) || continue
        L = DB._tensor_lagrange_mat(er.z0, bw, zg[keep])'
        e = max(e, maximum(abs.(L * (er.P[(k-1)*nz .+ (1:nz), :] * Fx) - [ti^u for ti in t[keep], u in q])))
    end
    return e
end

@testset "GCQ family accuracy: analytic reconstruction and the density_tol rank" begin
    # at the singular-value rank, a component-resolved 840 rule reconstructs its family to ~1e-10 only (study 66)
    er = edge_gcq_rule(FA_DP840; tol = 1e-12, grid_tol = 1e-13)
    @test fa_reconstruction(er).node > 1e-11
    @test_broken fa_reconstruction(er).node <= 2.5e-13                # known failure: weighted SVD tol is not pointwise
    # density_tol: the smallest rank (within the conditioning limits) whose model AND evaluator meet density_tol / 4
    for (dp, tol, dtol) in ((FA_DP840, 1e-12, 1e-12), ([-0.04], 1e-12, 1e-12), ([-0.6], 1e-8, 1e-8), (FA_DP840, 1e-8, 1e-8))
        ed = edge_gcq_rule(dp; tol, grid_tol = tol / 10, density_tol = dtol)
        c = DB.gcq_certificate(ed)
        @info "density_tol rank" dp = length(dp) tol rank = c["rank"] rank_sv = c["rank_singular_value"] model = c["model_error_node_range"] evaluator = c["evaluator_error_node_range"] terminal = c["evaluator_error_terminal"] lebesgue = c["lebesgue_node_range"] negw = c["negative_weight_fraction"]
        @test c["certified"] == true && c["domain"] == "selected-node range"
        @test c["rank"] == length(ed.x) >= c["rank_singular_value"]
        # the rule's own evaluator meets the gate at holdouts independent of the certificate's
        @test fa_evaluator(ed) <= dtol / 4
        @test c["evaluator_error_node_range"] <= dtol / 4 && c["model_error_node_range"] <= dtol / 4
        # the conditioning limits hold for a certified rule
        @test c["lebesgue_node_range"] <= c["lebesgue_max"] && c["negative_weight_fraction"] <= c["negative_weight_max"]
        # minimality: one rank less is not certified (or is excluded by the limits)
        if c["rank"] > c["rank_singular_value"]
            em = edge_gcq_rule(dp; tol, grid_tol = tol / 10, density_tol = dtol, rank_max = c["rank"] - 1)
            @test DB.gcq_certificate(em)["certified"] == false
        end
        @test all(isfinite, ed.w) && edge_gcq_family_error(ed) < 100 * tol
    end
    # density_tol alone refines the training grid (grid_tol = density_tol / 10): an unresolved grid is never certified
    ea = edge_gcq_rule(FA_DP840; tol = 1e-8, density_tol = 1e-8)
    @test fa_component_resolution(ea) <= 1e-9 && haskey(DB.gcq_certificate(ea), "cells")
    @test fa_evaluator(ea) <= 2.5e-9
    # a family whose low ranks bring large negative weights (a cube edge, ε = 10): the certified rank respects the limit
    ec = edge_gcq_rule(10.0, 1.0; tol = 1e-12, grid_tol = 1e-13, density_tol = 1e-12)
    cc = DB.gcq_certificate(ec)
    @test cc["certified"] == false || cc["negative_weight_fraction"] <= cc["negative_weight_max"]
    # 128-bit spot check of the Float64 model certificate (840 family, 1e-12)
    ed = edge_gcq_rule(FA_DP840; tol = 1e-12, grid_tol = 1e-13, density_tol = 1e-12)
    @test fa_reconstruction(ed; prec = 128).node <= 2.5e-13
    # the map of a density_tol rule reproduces nodal values (cardinal directions included): at its high rank the Float64
    # FV/S construction does not (nodal identity error 3.8e-2), so P samples the retained model in 128-bit arithmetic
    for (dp, tol) in ((FA_DP840, 1e-12), ([-0.04], 1e-12))
        ed = edge_gcq_rule(dp; tol, grid_tol = tol / 10, density_tol = tol)
        bw = DB.gl_barycentric_weights(ed.z0, ed.wz); nz = length(ed.z0)
        R = reduce(vcat, [begin k = clamp(searchsortedlast(ed.segs, (xi, Inf)), 1, length(ed.segs)); (sa, sb) = ed.segs[k]
                              DB._tensor_lagrange_mat(ed.z0, bw, [2 * (xi - sa) / (sb - sa) - 1])' * ed.P[(k-1)*nz .+ (1:nz), :] end
                          for xi in ed.x]) * Diagonal(ed.x .^ ed.pw)
        @test maximum(abs.(R - I)) <= 1e-12
        @test DB.gcq_certificate(ed)["map"] == "128-bit sampling of the retained model"
        @test DB.gcq_certificate(ed)["fvs_nodal_identity"] > 1e-6        # the recorded reason
    end
    # an unreachable request returns the best rank found, marked uncertified, with its measured errors
    eu = edge_gcq_rule(FA_DP840; tol = 1e-12, grid_tol = 1e-13, density_tol = 1e-15, rank_max = 30)
    cu = DB.gcq_certificate(eu)
    @test cu["certified"] == false && cu["model_error_node_range"] > 2.5e-16 && length(eu.x) <= 30
    # the default construction has no certificate request: unchanged rule, empty certificate; bad requests are refused
    @test isempty(DB.gcq_certificate(edge_gcq_rule([-0.6]; tol = 1e-8)))
    @test_throws ArgumentError edge_gcq_rule([-0.6]; tol = 1e-8, density_tol = 0.0)
    @test_throws ArgumentError edge_gcq_rule([-0.6]; tol = 1e-8, density_tol = 1e-8, rank_max = 0)
    @test_throws ArgumentError edge_gcq_rule([-0.6]; tol = 1e-8, density_tol = 1e-8, grid = :adaptive)
end

@testset "GCQ family accuracy: tensor products of components with the rule's evaluator" begin
    # the actual two-sided contraction E_u Σ E_v' of a product of original components against the analytic product
    # s^(pu+qi) t^(pv+qj), with the improved rules, on node-range holdouts of both directions (2e + e^2 for the gate e)
    eu = edge_gcq_rule([-0.6]; tol = 1e-12, grid_tol = 1e-13, density_tol = 1e-12)
    ev = edge_gcq_rule([-0.04]; tol = 1e-12, grid_tol = 1e-13, density_tol = 1e-12)
    ru, rv = gcq_tensor_rule(eu), gcq_tensor_rule(ev)
    tsu = filter(t -> minimum(ru.x) <= t <= maximum(ru.x), sort(rand(MersenneTwister(5), 60)))
    tsv = filter(t -> minimum(rv.x) <= t <= maximum(rv.x), sort(rand(MersenneTwister(6), 60)))
    Eu = DB.tensor_rule_interp(ru, tsu) ./ tsu .^ ru.pw; Ev = DB.tensor_rule_interp(rv, tsv) ./ tsv .^ rv.pw
    qu, qv = fa_q(eu), fa_q(ev)
    for (i, j) in ((1, 1), (5, 17), (length(qu), 3), (argmin(abs.(qu .- 2.0)), length(qv)))
        Σ = (ru.x .^ (ru.pw + qu[i])) * (rv.x .^ (rv.pw + qv[j]))'
        @test maximum(abs.(Eu * Σ * Ev' - (tsu .^ qu[i]) * (tsv .^ qv[j])')) <= 2 * 2.5e-13 + 2.5e-13^2
    end
end

@testset "GCQ family accuracy: polyhedron_mesh passes the options" begin
    OV = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    OF = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j] for i in (1, 2), j in (3, 4), k in (5, 6)])
    Fo, _ = orient_outward(OV, OF); patches, quads = polyhedron_patches(OV, Fo; eps_in = 2.0)
    m0, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 0, n_quad = 5, tol = 1e-6, corner = :gcq)
    m1, rules, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 0, n_quad = 5, tol = 1e-6, corner = :gcq,
                                   grid_tol = 1e-7, density_tol = 1e-6)
    @test all(DB.gcq_certificate(r)["certified"] for r in values(rules))
    @test any(length(P.ru.x) != length(Q.ru.x) for (P, Q) in zip(m0.panels, m1.panels) if DB.is_gcq(P.ru))
end
