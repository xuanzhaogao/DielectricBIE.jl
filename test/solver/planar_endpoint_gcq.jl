using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Random
using Test
include(joinpath(@__DIR__, "planar_test_helpers.jl"))       # ep_rule, ep_panel, ep_ref_rows

# PlanarProductNearQuad(endpoint_rule = :gcq): an endpoint cluster [0, a] of a GCQ direction is integrated by the panel's
# GCQ rule scaled to [0, a] (nodes a x, weights a w, the parent basis interpolated at a x) instead of the cluster's
# 18-point moment rule (endpoint_rule = :moment, the default).

# ∫_0^a φ_j(s) s^m ds and ∫_0^a |φ_j(s)| s^m ds over the parent basis φ_j (each fine segment's Lagrange polynomials
# through P, times s^pw), by 40-point Gauss-Legendre per fine segment (Gauss-Jacobi for s^pw on the one at s = 0)
function ep_moments(r, a, m)
    nz = length(r.z0); out = zeros(size(r.P, 2)); abs_ = zeros(size(r.P, 2))
    tg, wg = DB.gausslegendre(40); tj, wj = DB.gaussjacobi(40, 0.0, r.pw)
    for (k, (sa, sb)) in enumerate(r.segs)
        sa >= a && break
        b = min(sb, a)
        y, w = sa == 0 ? (b .* (1 .+ tj) ./ 2, (b / 2)^(r.pw + 1) .* wj) :
                         (sa .+ (b - sa) .* (1 .+ tg) ./ 2, (b - sa) / 2 .* wg .* (sa .+ (b - sa) .* (1 .+ tg) ./ 2) .^ r.pw)
        V = DB._tensor_lagrange_mat(r.z0, r.bw, 2 .* (y .- sa) ./ (sb - sa) .- 1)' * r.P[(k - 1) * nz .+ (1:nz), :]
        out .+= V' * (w .* y .^ m); abs_ .+= abs.(V)' * (abs.(w) .* y .^ m)
    end
    return out, abs_
end

@testset "endpoint GCQ map" begin
    for p in (-0.9, -0.6, -0.268308, -0.04), tol in (1e-4, 1e-8, 1e-12)
        e, r = ep_rule(p, tol); n = length(r.x)
        rel(xs, ws, a, q) = sum(ws .* xs .^ q) * (q + 1) / a^(q + 1) - 1
        σ = randn(MersenneTwister(1), n)
        for a in (1.0, 2.0^-4, 2.0^-12, 2.0^-30)
            xs, H = DB._flat_endpoint_gcq_map(r, a)
            @test xs == a .* r.x && size(H) == (n, n)
            # H[j, l] = a w_l φ_j(a x_l): the weights once, the factor s^pw inside φ_j
            @test H' * σ ≈ (a .* r.w) .* (DB.tensor_rule_interp(r, xs) * σ) rtol = 1e-14
            # the family s^q on [0, a]: the scaled rule as accurate as the rule itself (a power of 2: scaling is exact)
            @test maximum(abs(rel(xs, a .* r.w, a, q) - rel(r.x, r.w, 1.0, q)) for q in e.pows) < 1e-13
            # an analytic density s^p (1 + s + s²/2) through the parent interpolant, times s^m: against the interpolant's
            # moments (quadrature only; the interpolant is the density to the rule's tol, not exactly)
            σa = r.x .^ p .* (1 .+ r.x .+ r.x .^ 2 ./ 2)
            for m in 0:4
                ref, _ = ep_moments(r, a, m)
                @test abs(dot(H * xs .^ m, σa) - dot(ref, σa)) <= 20 * tol * abs(dot(ref, σa))
            end
        end
        # inside the weighted segment at the edge, [0, a] ⊂ [0, δ], every basis function is s^pw times a polynomial, a
        # member of the scaled family: the cardinal functions themselves, to the rule's tol of ∫|φ_j| s^m
        a = r.segs[1][2] / 4
        xs, H = DB._flat_endpoint_gcq_map(r, a)
        for m in 0:4
            ref, mag = ep_moments(r, a, m)
            @test maximum(abs.(H * xs .^ m .- ref) ./ mag) <= 20 * tol
        end
    end
    _, r = ep_rule(-0.6, 1e-8)
    @test_throws ArgumentError DB._flat_endpoint_gcq_map(r, 0.0)
    @test_throws ArgumentError DB._flat_endpoint_gcq_map(r, 1.5)
    @test_throws ArgumentError DB._flat_endpoint_gcq_map(r, 2.0^-1070)          # the scaled nodes underflow
    @test_throws ArgumentError DB._flat_endpoint_gcq_map(DB.gl_tensor_rule(8), 0.5)
    @test PlanarProductNearQuad(endpoint_rule = :gcq).endpoint_rule == :gcq
    @test PlanarProductNearQuad().endpoint_rule == :gcq                  # the default (study 73)
    @test_throws ArgumentError PlanarProductNearQuad(endpoint_rule = :scaled)
end

# targets beside a panel with the GCQ edge at s = 0 (parameters (s, t), s the GCQ direction): (s, t, offset) with the
# offset along the unit normal n, or in the plane across the edge (-e, e the inward unit normal of the edge), or both
function ep_targets(n, e, f)
    T = Tuple{Float64, Float64, Vector{Float64}}[]
    for h in (1e-1, 1e-3, 1e-6, 1e-9)
        push!(T, (0.0, 0.37, h .* n))                       # above the edge
    end
    for h in (1e-2, 1e-5, 1e-8)
        push!(T, (0.0, 0.37, -h .* e))                      # in the plane, across the edge
    end
    push!(T, (0.43, 0.61, 1e-2 .* n), (0.43, 0.61, 1e-6 .* n))    # over the interior
    push!(T, (1.0, 0.29, 1e-3 .* n))                        # over the far end
    push!(T, (0.0, 0.0, 1e-4 .* normalize(n .- e .- f)))    # beside the corner, outside
    push!(T, (0.5, 0.5, 0.8 .* n), (0.0, 0.37, -0.3 .* e))  # farther away
    return T
end

@testset "endpoint GCQ rows against a fine-grid reference" begin
    ax = normalize([0.3, -0.81, 0.47]); Kx = [0 -ax[3] ax[2]; ax[3] 0 -ax[1]; -ax[2] ax[1] 0]
    rot = I + sin(0.7310281) * Kx + (1 - cos(0.7310281)) * Kx^2
    shapes = (("affine rectangle", [1.3, 0.0, 0.0], [0.0, 0.8, 0.0], zeros(3)),
              ("bilinear quadrilateral", [1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.3, -0.25, 0.0]))
    anchor = (37.3, -51.9, 93.1)
    rgl = DB.gl_tensor_rule(10)
    cases = vcat([(p, 1e-12) for p in (-0.9, -0.6, -0.268308, -0.04)], [(-0.268308, 1e-4), (-0.268308, 1e-8)])
    worst = Dict{Tuple{Symbol, Symbol, Symbol}, Float64}()
    for (p, tol) in cases, (name, a0, b0, c0) in shapes
        _, rg = ep_rule(p, tol)
        A, B, C = rot * a0, rot * b0, rot * c0
        n = normalize(cross(A, B)); e = normalize(cross(B, n)); f = normalize(B)       # e inward across the edge
        @assert dot(e, A) > 0
        P1 = ep_panel(anchor, A, B, C, rg, rgl)                  # GCQ in u, the edge u = 0
        P2 = ep_panel(anchor, B, A, C, rgl, rg)                  # the same panel, GCQ in v
        tg = ep_targets(n, e, f); m = length(tg)
        nrm = reduce(hcat, [normalize(randn(MersenneTwister(k), 3)) for k in 1:m])
        nu, nv = length(rg.x), length(rgl.x)
        perm = [(j - 1) * nu + i for i in 1:nu for j in 1:nv]  # node (i, j) of P1 is node (j, i) of P2
        # densities at the nodes of P1: s^p, the mixture s^p (1 + s/2 + t²) + 0.3 cos 2t (1 + s), and random
        S = [s for s in rg.x for t in rgl.x]; Tt = [t for s in rg.x for t in rgl.x]
        dens = (an = S .^ p, mix = S .^ p .* (1 .+ S ./ 2 .+ Tt .^ 2) .+ 0.3 .* cos.(2 .* Tt) .* (1 .+ S),
                rnd = randn(MersenneTwister(7), nu * nv))
        for kern in (:slp, :dt)
            N = kern == :slp ? nothing : nrm
            T1 = DB._FlatTargets([x[1] for x in tg], [x[2] for x in tg], reduce(hcat, [x[3] for x in tg]), N, fill((0, 0), m))
            T2 = DB._FlatTargets([x[2] for x in tg], [x[1] for x in tg], reduce(hcat, [x[3] for x in tg]), N, fill((0, 0), m))
            ref, absref = ep_ref_rows(P1, T1)
            mag = abs.(ref) * ones(nu * nv)                     # Σ_j |row_j|
            # :gcq on cardinal rows and random densities: within the panel's far-field node-rule error (same metric)
            efar, efar_rnd = pl_far_error(P1, dens.rnd; nrm = N === nothing ? nothing : N[:, 1])
            # the rounding floor of a row (or a density's sum): 100 ε ∫|kernel × basis| relative to the result's scale
            floor_c = 100eps() .* maximum(absref; dims = 2)[:] ./ mag
            for rule in (:moment, :gcq)
                q = PlanarProductNearQuad(endpoint_rule = rule, self = :cell)
                R1 = DB._flat_rows(P1, DB._flat_geom(P1), T1, q)
                R2 = DB._flat_rows(P2, DB._flat_geom(P2), T2, q)
                # both orientations: the same rows to rounding
                @test maximum(abs.(R2[:, perm] .- R1) ./ mag) < 1e-13
                # the cardinal rows (the moment rules exact for them), and the densities, against the reference
                ec = maximum(abs.(R1 .- ref); dims = 2)[:] ./ mag
                rule == :moment && @test all(ec .<= 1e-13 .+ floor_c)
                rule == :gcq && @test all(ec .<= efar)
                rule == :gcq && (worst[(rule, kern, :card_far)] = max(get(worst, (rule, kern, :card_far), 0.0), maximum(ec) / efar))
                key = (rule, kern, :card); worst[key] = max(get(worst, key, 0.0), maximum(ec ./ (1e-13 .+ floor_c)))
                for (dn, σ) in pairs(dens)
                    sc = abs.(ref) * abs.(σ)
                    ed = abs.((R1 .- ref) * σ) ./ sc
                    bound = (rule == :gcq && dn != :rnd ? 20tol : 1e-13) .+ 100eps() .* (absref * abs.(σ)) ./ sc
                    (rule == :moment || dn != :rnd) && @test all(ed .<= bound)
                    if rule == :gcq && dn == :rnd
                        @test all(ed .<= efar_rnd .+ bound)
                        worst[(rule, kern, :rnd_far)] = max(get(worst, (rule, kern, :rnd_far), 0.0), maximum(ed) / efar_rnd)
                    end
                    key = (rule, kern, dn); worst[key] = max(get(worst, key, 0.0), maximum(ed ./ bound))
                end
            end
        end
    end
    # the reference converges: stricter boxes and more points change the rows of the hardest targets by rounding only
    _, rg = ep_rule(-0.9, 1e-12)
    P = ep_panel(anchor, rot * [1.0, 0.0, 0.0], rot * [0.2, 0.9, 0.0], rot * [0.3, -0.25, 0.0], rg, rgl)
    g = DB._flat_geom(P); n = normalize(collect(P.normals[1])); e = normalize(cross(collect(g.b), n))
    for N in (nothing, hcat(normalize([0.3, 0.5, -0.8]), normalize([-0.6, 0.2, 0.7])))
        T = DB._FlatTargets([0.0, 0.0], [0.37, 0.37], hcat(1e-9 .* n, -1e-8 .* e), N, [(0, 0), (0, 0)])
        (r6, a6), (r10, _) = ep_ref_rows(P, T), ep_ref_rows(P, T; c = 10.0, M = 28)
        @test all(abs.(r10 .- r6) .<= 1e-14 .* abs.(r10) .+ 100eps() .* a6)
    end
    for (k, v) in sort(collect(worst))
        @info "endpoint GCQ rows: worst error / bound (the gcq bound 20 tol for :an, :mix; 1e-13 else; plus the floor)" rule = k[1] kernel = k[2] density = k[3] ratio = v
    end
end

@testset "endpoint GCQ on polyhedron panels: GL × GL unchanged, self rows accurate" begin
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]      # the octahedron conv(±e_i)
    F, _ = orient_outward(V, reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                                           for i in (1, 2), j in (3, 4), k in (5, 6)]))
    patches, quads = polyhedron_patches(V, F; eps_in = 2.0)
    tol = 1e-10
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 6, tol, corner = :gcq)
    qm = PlanarProductNearQuad(self = :cell, endpoint_rule = :moment); qg = PlanarProductNearQuad(endpoint_rule = :gcq, self = :cell)
    # GL × GL panels have no endpoint clusters: the same rows under either rule (self and off-surface)
    P = first(P for P in itf.panels if P.kind == :gl)
    one = DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0])
    σ = randn(MersenneTwister(3), DB.num_points(P))
    @test planar_slp_nodes(one, σ; q = qg) == planar_slp_nodes(one, σ; q = qm)
    g = DB._flat_geom(P); nrm = collect(P.normals[1])
    T = DB._FlatTargets([0.0, 0.3, 0.9], [0.0, 0.6, 0.2], hcat(1e-3 .* nrm, 0.1 .* nrm, 1e-6 .* nrm),
                        [0.0 0.6 0.0; 0.8 0.0 0.0; 0.6 0.8 1.0], fill((0, 0), 3))
    @test DB._flat_rows(P, g, T, qg) == DB._flat_rows(P, g, T, qm)
    # GCQ × GL and GCQ × GCQ self rows under :gcq: σ = 1 (in the GCQ family) at nodes nearest the edge, interior and far,
    # against the independent self reference (sp_self_ref), to 20 tol of ∫|row|
    for kind in (:edge, :corner)
        P = first(P for P in itf.panels if P.kind == kind)
        one = DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0])
        u = planar_slp_nodes(one, ones(DB.num_points(P)); q = qg)
        nu, nv = length(P.ru.x), length(P.rv.x)
        for i in unique([1, nu ÷ 2, nu]), j in unique([1, nv])
            u0, v0 = P.ru.x[i], P.rv.x[j]
            ref, mag = sp_self_ref(P, u0, v0, DB._pn_cell(P.ru, u0, 1, "u"), DB._pn_cell(P.rv, v0, 1, "v"))
            @test abs(u[(i - 1) * nv + j] - sum(ref)) <= 20tol * sum(abs, ref) + 100eps() * sum(mag)
        end
    end
end
