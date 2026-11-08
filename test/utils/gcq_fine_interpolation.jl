# The fine-grid density interpolant of a GCQ rule (plan 2026-11-06-gcq-fine-grid-interpolation-audit): P (node values
# -> regularized values at the fine nodes) and the local polynomial of each fine cell, against g*, the rule's
# retained-space global interpolant evaluated in BigFloat (the same exponents, right singular vectors and nodes).
# Regularized densities g = σ / s^pw; errors normalized by G = max |g*| over the node range [x_min, x_max].
using Test, LinearAlgebra
using DielectricBIE
import DielectricBIE as DB

# regularized g* map (BigFloat arithmetic, Float64 result): full nodal values -> g at ts
function gf_gstar(r, ts; prec = 128)
    g = DB.gcq_global_interp(r)
    setprecision(BigFloat, prec) do
        q = big.(g.q); V = big.(g.V); x = big.(g.x)
        Float64.(([big(t)^qi for t in ts, qi in q] * V) * (([xi^qi for xi in x, qi in q] * V) \ Diagonal(x .^ (-big(g.pw)))))
    end
end
gf_cheb(N) = [-cos(π * j / (N - 1)) for j in 0:N-1]               # Chebyshev points of the 2nd kind, ±1 included
gf_block(r, k) = (nz = length(r.z0); r.P[(k - 1) * nz .+ (1:nz), :])
# max over the holdouts of the cells above the weighted one, within [x_min, x_max], of |local - g*| applied to σ, / G
function gf_holdout_error(r, σ; N = 17)
    xmin, xmax = extrema(r.x); e = 0.0; G = 0.0
    for (k, (sa, sb)) in enumerate(r.segs)
        k < 2 && continue
        τ = gf_cheb(N); t = sa .+ (sb - sa) .* (1 .+ τ) ./ 2; t[1] = sa; t[end] = sb
        keep = xmin .<= t .<= xmax
        any(keep) || continue
        loc = DB._tensor_lagrange_mat(r.z0, r.bw, τ[keep])' * (gf_block(r, k) * σ)
        ref = gf_gstar(r, t[keep]) * σ
        e = max(e, maximum(abs.(loc - ref))); G = max(G, maximum(abs.(ref)))
    end
    return e / G
end
# regularized jumps L_k(+1) z_k - L_(k+1)(-1) z_(k+1) at the interior boundaries inside [x_min, x_max], / G
function gf_jumps(r, σ)
    xmin, xmax = extrema(r.x)
    e1 = DB._tensor_lagrange_mat(r.z0, r.bw, [1.0])'; e0 = DB._tensor_lagrange_mat(r.z0, r.bw, [-1.0])'
    J = [only(e1 * (gf_block(r, k) * σ) - e0 * (gf_block(r, k + 1) * σ)) for k in 1:length(r.segs)-1 if xmin <= r.segs[k][2] <= xmax]
    G = maximum(abs.(gf_gstar(r, r.x) * σ))
    return maximum(abs.(J)) / G
end

const GF_DP = edge_density_powers(2.0, 1.0; angle = acos(-1 / 3))      # an octahedron edge: 39 powers, m = 840
gf_f(s) = s^GF_DP[1] * (1 + 0.3s) + 0.5 * s^GF_DP[2] * (1 - s) - 0.25 * s^GF_DP[3] * cos(s)   # a family member

@testset "GCQ fine interpolation: the node-to-fine map P is accurate" begin
    for (name, dp, f) in (("dedicated", [-0.268308], s -> s^-0.268308 * (1 + 0.3s - 0.2s^2)), ("octahedron edge", GF_DP, gf_f))
        r = gcq_tensor_rule(edge_gcq_rule(dp; tol = 1e-12))
        σ = f.(r.x)
        xf = vec(r.fx); keep = xf .>= minimum(r.x)                   # fine nodes from the smallest original node up
        ref = gf_gstar(r, xf[keep]) * σ
        @test maximum(abs.(r.P[keep, :] * σ - ref)) <= 1e-13 * maximum(abs.(ref))
    end
end

@testset "GCQ fine interpolation: degree-13 cells miss 1e-12, cells of 20 and 28 points do not" begin
    e0 = edge_gcq_rule(GF_DP; tol = 1e-12); r0 = gcq_tensor_rule(e0)
    σ = gf_f.(r0.x)
    @test gf_holdout_error(r0, σ) > 1e-12                            # the diagnosed failure (polynomial stage)
    for (n, bound) in ((20, 1e-12), (28, 1e-13))
        en = edge_gcq_rule(GF_DP; tol = 1e-12, ninterp = n); rn = gcq_tensor_rule(en)
        # the same rule: nodes, weights, weight power, family, segments; its own fine grid of n points per cell
        @test en.x == e0.x && en.w == e0.w && en.pw == e0.pw && en.pows == e0.pows && en.segs == e0.segs && en.tol == e0.tol
        @test length(en.z0) == n && size(en.P) == (n * length(en.segs), length(en.x))
        @test DB.gcq_global_interp(rn).V == DB.gcq_global_interp(r0).V   # the same retained subspace (and its data)
        # P holds g* at the new fine nodes, sampled in BigFloat (cardinal columns included)
        xf = vec(rn.fx); keep = xf .>= minimum(rn.x)
        Gs = gf_gstar(rn, xf[keep])
        @test maximum(abs.(rn.P[keep, :] - Gs)) <= 1e-14 * maximum(abs.(Gs))
        e = gf_holdout_error(rn, σ)
        @info "fine interpolation, $n points per cell" holdout = e jump = gf_jumps(rn, σ)
        @test e <= bound
        @test gf_jumps(rn, σ) <= 2 * bound
    end
    @test gf_jumps(r0, σ) > 1e-13                                     # (and the default cells jump visibly)
    # ninterp = 14 is the default construction, bit for bit; a degenerate order is refused
    @test edge_gcq_rule(GF_DP; tol = 1e-12, ninterp = 14).P == e0.P
    @test_throws ArgumentError edge_gcq_rule([-0.5]; tol = 1e-6, ninterp = 1)
    # a universal rule passes the order through
    @test length(universal_gcq_rule(-0.3; qmax = 6.0, tol = 1e-8, ninterp = 20).z0) == 20
end

@testset "GCQ fine interpolation: refined rules in meshes and the near machinery" begin
    include(joinpath(@__DIR__, "..", "solver", "planar_test_helpers.jl"))
    # polyhedron_mesh passes the order to its GCQ rules; the nodes (and the far-field rule) are unchanged
    OV = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    OF = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j] for i in (1, 2), j in (3, 4), k in (5, 6)])
    Fo, _ = orient_outward(OV, OF); patches, quads = polyhedron_patches(OV, Fo; eps_in = 2.0)
    m14, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 0, n_quad = 5, tol = 1e-6, corner = :gcq)
    m20, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 0, n_quad = 5, tol = 1e-6, corner = :gcq, ninterp = 20)
    @test all(P.points == Q.points && P.weights == Q.weights for (P, Q) in zip(m14.panels, m20.panels))
    @test all(length(r.z0) == (DB.is_gcq(r) ? 20 : 5) for P in m20.panels for r in (P.ru, P.rv))
    # a self row of a panel with refined rules: the production row against the independent fine-grid reference of the
    # same (refined) interpolant, and the global row on a physical density
    ru = gcq_tensor_rule(edge_gcq_rule([-0.268308]; tol = 1e-12, ninterp = 20)); rv = DB.gl_tensor_rule(8)
    P = ep_panel((0.3, -0.2, 0.1), (1.0, 0.0, 0.1), (0.2, 0.9, 0.0), (0.3, -0.25, 0.0), ru, rv); g = DB._flat_geom(P)
    for i in (3, 11, 19)
        u0, v0 = ru.x[i], rv.x[3]; k = DB._pn_cell(ru, u0, 0, "u")
        T = DB._FlatTargets([u0], [v0], zeros(3, 1), nothing, [(k, 1)])
        row = vec(DB._flat_rows(P, g, T, PlanarProductNearQuad(self = :cell, endpoint_rule = :moment)))
        ref, mag = sp_self_ref(P, u0, v0, k, 1)
        @test maximum(abs.(row - ref)) <= 1e-13 * sum(mag)
        σ = vec(transpose((ru.x .^ ru.pw .* (1 .+ 0.3 .* ru.x)) * (1 .+ 0.5 .* rv.x .^ 2)'))
        rg = DB.planar_global_self_row(P, u0, v0; λ = 0.0)
        @test abs(dot(row - rg, σ)) <= 1e-14 * dot(abs.(rg), abs.(σ))
    end
end
