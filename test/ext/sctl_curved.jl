# GCQNearQuad on curved interfaces: the GCQ edge panels' near and self rows in Julia (gcq_curved_row), SCTL's for the
# Gauss-Legendre × Gauss-Legendre panels, and solves on curved bodies (patches from test/shape/lens_patches.jl).
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays
using SCTL
using HCubature
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

# Dᵀ alone: FMM far field plus the near correction
function dt_operator(itf, q::BI.NearQuadrature)
    return dt_operator(itf, near_corrections(q, itf))
end
function dt_operator(itf, Cm::AbstractMatrix)
    F = BI.laplace3d_DT_fmm3d(itf, 1e-12)
    C = BlockCorrection(Cm, itf)
    return x -> F * x .+ C * x
end
nodes_of(itf) = reduce(vcat, [p.points for p in itf.panels])
# the lens (θ1 = θ2 = π/3, singular rim) with its corrections at tol = 1e-12: built once, used by the Gauss-identity
# test, the source-column test and the solves (the most expensive object here: the GCQ edge panels' own-node rows)
sctl_curved_lens_cache = Dict{Symbol, Any}()
sctl_curved_lens() = get!(sctl_curved_lens_cache, :coarse) do
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    (g, m, near_corrections(GCQNearQuad(quad_tol = 1e-12), m.interface))
end

@testset "GCQNearQuad (curved): Dᵀ on the unit sphere, one panel per patch" begin
    # θ1 = θ2 = π/2, one ε: the unit sphere; the equator graded, with GCQ edge panels. One panel per patch: the
    # caps' top charts have their pole at the centre of a panel, where the tangents vanish, and panels span ~90° of
    # the sphere. The GCQ panels' rows are gcq_curved_row's on the exact maps (all finite). The GL×GL panels are SCTL
    # elements of their own order, their geometry SCTL's order-n_quad interpolant of the map, which on these ~90°
    # panels bounds every row near them (the GCQ nodes next to a GL panel included): Dᵀ1 - 1/2 is 6.3e-2 at
    # n_quad = 8, 9.8e-5 at 16, 4.8e-6 at 20, 7.0e-9 with m = 2 at 16. Hence n_quad = 16 here (8 before SCTL's GL
    # routine). The values below are unchanged, to 4 digits, from SCTLCurvedNearQuad's (SCTL proxies for the GCQ panels)
    g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, across = 1, m = 1, n_quad = 16, tol = 1e-6, grade = c -> abs(c.midpoint[3]) < 1e-12)
    itf = m.interface
    X = nodes_of(itf); z = [x[3] for x in X]
    isedge = reduce(vcat, [fill(p.kind == :edge, BI.num_points(p)) for p in itf.panels])
    Cm = near_corrections(GCQNearQuad(quad_tol = 1e-10), itf)
    @test all(isfinite, nonzeros(Cm))
    Dt = dt_operator(itf, Cm)
    # Dᵀ Y_l = Y_l / (2(2l + 1)) on the unit sphere (DielectricBIE's sign), limited by the GL panels' geometry
    # (n_quad = 8 with DielectricBIE's split proxies for the GL panels, before SCTL's GL routine: 2.5e-11 for Dᵀ1,
    # 2.0e-6 for Y_1, 1.1e-5 for Y_2)
    e0 = Dt(ones(length(X))) .- 0.5
    @test maximum(abs, e0) < 2e-4                                 # 9.8e-5 measured
    @test maximum(abs, e0[isedge]) < 8e-5                         # GCQ edge-panel nodes: 4.0e-5
    @test maximum(abs, Dt(z) .- z ./ 6) < 1.5e-4                  # 7.6e-5
    y2 = (3 .* z .^ 2 .- 1) ./ 2
    @test maximum(abs, Dt(y2) .- y2 ./ 10) < 8e-5                 # 4.0e-5
    # the GL×GL panels' columns are sctl_gl_corrections' at quad_tol (η_gcq and gcq_far concern the GCQ panels only)
    gl = findall(is_gl_panel, itf.panels)
    glc = falses(length(X)); foreach(r -> glc[r] .= true, panel_column_ranges(itf)[gl])
    @test !isempty(gl) && !all(glc)
    @test Cm[:, glc] == sctl_gl_corrections(itf, gl; tol = 1e-10)[:, glc]
end

@testset "GCQNearQuad (curved): Gauss identity on the lens (singular edge rules)" begin
    # D1 = 1/2 on a closed surface, so wᵀ Dᵀ σ = ½ wᵀ σ for every σ; with σ ~ s^p₁ at the rim (s the distance to
    # it) this exercises the weighted GCQ edge panels (pw ≠ 0): strip, Gauss-Jacobi weights, s^pw interpolation
    g, m, Cm = sctl_curved_lens()
    itf = m.interface; w = BI.all_weights(itf)
    p1 = m.powers[findfirst(c -> !c.smooth, g.curves)][1]
    @test all(p -> p.kind != :edge || p.ru.pw ≈ p1, itf.panels) && p1 < -0.1
    s = [hypot(hypot(x[1], x[2]) - 1, x[3]) for x in nodes_of(itf)]
    gauss(Dt, σ) = abs(dot(w, Dt(σ)) - dot(w, σ) / 2) / dot(abs.(w), abs.(σ))
    Dt = dt_operator(itf, Cm)
    # limited by the GL panels' geometry at n_quad = 8 (SCTL's order-8 elements; with DielectricBIE's split proxies
    # for them, before SCTL's GL routine: 3.0e-8 and 4.5e-8; without the panel-level near criterion it is 0.2-0.4)
    @test gauss(Dt, s .^ p1) < 5e-6                                 # 2.7e-6 measured
    @test gauss(Dt, s .^ (p1 + 1)) < 1e-5                           # 5.7e-6
end

@testset "GCQNearQuad (curved): the columns of some source panels" begin
    # with `sources` (panel indices, in any order) only their columns are computed: exactly the full call's, for the
    # GCQ edge panels (gcq_curved_row) and the GL×GL panels (sctl_gl_corrections) alike. GCQNearQuad() has
    # quad_tol = 1e-12, as the lens's corrections
    g, m, Cm = sctl_curved_lens()
    itf = m.interface
    gcq = findall(!is_gl_panel, itf.panels); gl = findall(is_gl_panel, itf.panels)
    srcs = [gcq[end], gl[1], gcq[1], gl[end], gcq[cld(length(gcq), 2)]]
    Cs = near_corrections(GCQNearQuad(), itf, srcs)
    @test Cs isa SparseMatrixCSC{Float64, Int} && nnz(Cs) > 0
    @test Cs == BI._keep_source_columns(Cm, itf, srcs)
end

@testset "GCQNearQuad (curved): the near test" begin
    # a GCQ panel P's columns hold exactly its own nodes and the nodes within η_gcq × L of P's nearest node, L its
    # larger mean chord (gcq_far = :max) or the one across u (:direction); the rows do not depend on the near set
    g, m, Cm = sctl_curved_lens()
    itf = m.interface
    k = findfirst(!is_gl_panel, itf.panels); P = itf.panels[k]
    cols = panel_column_ranges(itf)[k]
    X = nodes_of(itf)
    own = falses(length(X)); own[cols] .= true
    dist = [minimum(norm(collect(x .- y)) for y in P.points) for x in X]
    su, sv = BI._mean_chords((u, v) -> BI.curved_lpoint(P, u, v))
    @test su < sv                                     # (so that :direction measures with a shorter L than :max)
    support(C) = sort(unique(rowvals(C[:, cols])))
    @test support(Cm) == findall(own .| (dist .<= 2.0 * max(su, sv)))              # the default: η_gcq = 2, :max
    Cd = near_corrections(GCQNearQuad(quad_tol = 1e-12, η_gcq = 1.0, gcq_far = :direction), itf, [k])
    @test support(Cd) == findall(own .| (dist .<= 1.0 * su))
    @test length(support(Cd)) < length(support(Cm)) && Cd[support(Cd), cols] == Cm[support(Cd), cols]
end

# ∫ over the parameter triangle (c, p, q), Duffy-mapped so that a 1/r singularity at c cancels: f(z, s, dz) is the
# integrand at z = c + s dz (dz = p - c + t (q - p)), which for s small should take r from the tangents at c (two
# positions that close round to the same point)
function duffy_tri(f, c, p, q)
    J = abs((p[1] - c[1]) * (q[2] - p[2]) - (p[2] - c[2]) * (q[1] - p[1]))
    g = st -> (dz = (p .- c) .+ st[2] .* (q .- p); st[1] * J * f(c .+ st[1] .* dz, st[1], dz))
    return hcubature(g, [0.0, 0.0], [1.0, 1.0]; rtol = 1e-11, atol = 1e-19, maxevals = 50_000)[1]
end
# ∫ over the unit parameter square of f, singular at c when c is given
square_int(f, c) = c === nothing ?
    hcubature(uv -> f((uv[1], uv[2]), nothing, nothing), [0.0, 0.0], [1.0, 1.0]; rtol = 1e-11,
              atol = 1e-19, maxevals = 250_000)[1] :
    sum(duffy_tri(f, c, (o[1], c[2]), o) + duffy_tri(f, c, o, (c[1], o[2])) for o in ((0, 0), (1, 0), (0, 1), (1, 1)))

@testset "GCQNearQuad (curved): rows on thin panels at the edge (exact sphere integrals)" begin
    # On the unit sphere n_x·(x - y) = ±|x - y|²/2, so the Dᵀ row of a source panel P for σ = 1 is ±∫_P dS/(8π r),
    # computed independently (Duffy on P itself). Grading and a tight rule make P's elements next to the equator
    # very thin (u-widths down to 6e-15 against a v-length ~0.76) and put nodes within 6e-8 of it. Rows: the
    # nodes nearest the edge and the farthest, at both ends in v (the worst ones of a 5 × 3 grid)
    g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-12)
    ps = m.interface.panels
    mid(p, u, v) = p.anchor .+ BI.curved_lpoint(p, u, v)
    kP = findfirst(k -> m.panel_patch[k] <= 5 && ps[k].kind == :edge, eachindex(ps))                 # upper cap
    P = ps[kP]
    kQ = argmin(k -> m.panel_patch[k] in 6:10 && ps[k].kind == :edge ?                               # lower cap,
                     norm(collect(mid(ps[k], 0.0, 0.5) .- mid(P, 0.0, 0.5))) : Inf, eachindex(ps))  # across the edge
    kG = argmin(k -> m.panel_patch[k] == m.panel_patch[kP] && ps[k].kind == :gl ?                    # P's graded
                     norm(collect(mid(ps[k], 0.5, 0.5) .- mid(P, 1.0, 0.5))) : Inf, eachindex(ps))  # neighbour
    sub = [kP, kQ, kG]
    itf = DielectricInterface(ps[sub], m.interface.eps_in[sub], m.interface.eps_out[sub])
    C = Matrix(near_corrections(GCQNearQuad(quad_tol = 1e-12), itf))
    offs = cumsum([0; [BI.num_points(p) for p in itf.panels]])
    X, N = BI._interface_targets_normals(itf)
    worst = zeros(2, 2)               # sources (the GCQ panel P, the GL panel G) × targets (same surface, across)
    for (s, S) in enumerate(itf.panels[[1, 3]])
        js = offs[2s - 1] .+ (1:BI.num_points(S))
        A = C[:, js] .+ BI.dt_node_rule_block(itf, 2s - 1, X, N)
        for (t, T) in enumerate(itf.panels), a in unique([1, 2, length(T.ru.x)]), b in (1, length(T.rv.x))
            r = offs[t] + (a - 1) * length(T.rv.x) + b
            x = T.anchor .- S.anchor .+ T.lpoints[r - offs[t]]
            c = T === S ? (T.ru.x[a], T.rv.x[b]) : nothing
            f = (z, s, dz) -> begin
                ρ = s !== nothing && s < 1e-6 ?
                    (fr = BI.curved_frame(S, c...); s * norm(collect(fr[1] .* dz[1] .+ fr[2] .* dz[2]))) :
                    norm(collect(x .- BI.curved_lpoint(S, z...)))
                BI.curved_frame(S, z...)[4] / (8π * ρ)
            end
            ref = sign(dot(N[:, r], X[:, r])) * square_int(f, c)
            k = t == 2 ? 2 : 1
            worst[s, k] = max(worst[s, k], abs(sum(A[r, :]) - ref) / abs(ref))
        end
    end
    # P's own rows and G's nodes: gcq_curved_row, 4.1e-13 measured (2.3e-12 with SCTLCurvedNearQuad's proxies and
    # thin rows, under a bound of 1e-9; now ~10× the measured value)
    @test worst[1, 1] < 4e-12
    # across the edge the target's offset from P is formed from two patch maps (two anchors, ~2e-16 each) that meet
    # at the equator only to rounding: for Q's nodes 6e-8 from it that limits the geometry, and so the row (and
    # the identity behind the reference; 1.5e-9 at nlev = 6, n_quad = 12, Q's nodes 1.6e-8 from it)
    @test worst[1, 2] < 1e-8                                          # 7.5e-11 measured (as SCTLCurvedNearQuad's)
    # the GL panel G is an SCTL element of order 8, its geometry the order-8 interpolant of the map (before SCTL's
    # GL routine, with DielectricBIE's split proxies, at most 9.2e-12 and 7.5e-11, as for P). These are G's columns,
    # sctl_gl_corrections' under either quadrature: the worst, 4.6e-6, is P's node next to G (a = n_u)
    @test worst[2, 1] < 1e-5                                          # 4.6e-6 measured
    @test worst[2, 2] < 5e-8                                          # 2.0e-8 measured
end

# induced potential u(x) = Σ w σ / (4π |x - y|), by direct summation (targets well away from the surface)
function u_ind(itf, σ, T)
    X = nodes_of(itf); w = BI.all_weights(itf)
    return [sum(w[j] * σ[j] / (4π * norm(collect(t .- X[j]))) for j in eachindex(X)) for t in T]
end
curved_operator(m; tol = 1e-10) =
    dielectric_operator(m.interface; far = FMM3DFarField(1e-12), near = GCQNearQuad(quad_tol = tol))
function solve_point(A, m, src)
    itf = m.interface
    b = BI.rhs_dielectric_box3d(itf, PointSource(src, 1.0), 1.0)
    σ, st = BI.solve_dielectric_box3d_gcq(A, b, itf; rtol = 1e-12)
    @test st.solved
    return σ
end
# 20 directions on the sphere of radius r (Fibonacci)
fib(r, n = 20) = [(φ = π * (3 - sqrt(5)) * i; z = 1 - 2 * (i + 0.5) / n; ρ = sqrt(1 - z^2);
                   (r * ρ * cos(φ), r * ρ * sin(φ), r * z)) for i in 0:n-1]
# the induced potential of a unit charge at distance d on the z axis outside a dielectric sphere (radius a)
function sphere_series(eps_in, eps_out, a, d, x; lmax = 80)
    r = norm(x); c = x[3] / r
    P0, P1 = 1.0, c; s = 0.0
    for l in 1:lmax
        Pl = l == 1 ? P1 : ((2l - 1) * c * P1 - (l - 1) * P0) / l
        l == 1 || ((P0, P1) = (P1, Pl))
        s += (eps_out - eps_in) * l / (eps_in * l + eps_out * (l + 1)) * a^(2l + 1) / (d^(l + 1) * r^(l + 1)) * Pl
    end
    return s / (4π * eps_out)
end

@testset "curved solves: sphere, lens, split sphere" begin
    # dielectric sphere (smooth: no graded curve) against the analytic series
    g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; n_quad = 8)
    σ = solve_point(curved_operator(m), m, (0.0, 0.0, 2.5))
    T = fib(1.8)
    uref = [sphere_series(4.0, 1.0, 1.0, 2.5, collect(t)) for t in T]
    # 4.5e-5 measured: the panels' geometry at n_quad = 8 (SCTL's order-8 elements; 4.5e-8 with DielectricBIE's
    # split proxies before SCTL's GL routine)
    @test maximum(abs, u_ind(m.interface, σ, T) .- uref) / maximum(abs, uref) < 1e-4

    x1 = (0.3, -0.2, 1.7); x2 = (-1.4, 0.5, -0.9)
    for name in ("lens", "split sphere")
        if name == "lens"                             # the Gauss-identity test's mesh and corrections
            g, m, Cm = sctl_curved_lens()
            A = dielectric_operator(m.interface; far = FMM3DFarField(1e-12), near = Cm)
        else
            g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
            m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
            A = curved_operator(m)
        end
        w = BI.all_weights(m.interface)
        σ1 = solve_point(A, m, x1); σ2 = solve_point(A, m, x2)
        # every body is neutral; neutrality and reciprocity are limited by the GL panels' geometry at n_quad = 8
        # (SCTL's order-8 elements; lens: 7.0e-6 and 4.6e-5, split sphere: 1.8e-5 and 1.4e-4; with DielectricBIE's
        # split proxies before SCTL's GL routine: 9.7e-8 and 3.2e-6, 1.8e-7 and 1.5e-5)
        tn, tr = name == "lens" ? (2e-5, 1e-4) : (1e-4, 1e-3)
        @test abs(sum(w .* σ1)) / sum(w .* abs.(σ1)) < tn
        # reciprocity of the induced potential: u(x2; charge at x1) = u(x1; charge at x2)
        a12 = u_ind(m.interface, σ1, [x2])[1]; a21 = u_ind(m.interface, σ2, [x1])[1]
        @test a12 ≈ a21 rtol = tr
        # (GCQ self-convergence, one more level and a tighter rule, changes the lens's u by 7.2e-7; not checked
        # here: its near corrections cost ~45 s; neutrality and reciprocity above bound the error)
    end
end
