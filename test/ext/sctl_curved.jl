# SCTL near and on-surface quadrature for curved GCQ panels, and solves on curved bodies (patches from
# test/shape/lens_patches.jl).
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays
using Test

const HAVE_SCTL_CURVED = Base.find_package("SCTL") !== nothing
HAVE_SCTL_CURVED && @eval using SCTL
include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

if HAVE_SCTL_CURVED
    # Dᵀ alone: FMM far field plus the near correction
    function dt_operator(itf, q)
        F = BI.laplace3d_DT_fmm3d(itf, 1e-12)
        C = BlockCorrection(near_corrections(q, itf), itf)
        return x -> F * x .+ C * x
    end
    nodes_of(itf) = reduce(vcat, [p.points for p in itf.panels])

    @testset "SCTLCurvedNearQuad: Dᵀ on the unit sphere" begin
        # θ1 = θ2 = π/2, one ε: the unit sphere; the equator graded, with GCQ edge panels
        g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
        m = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-8, grade = c -> abs(c.midpoint[3]) < 1e-12)
        itf = m.interface
        X = nodes_of(itf); z = [x[3] for x in X]
        isedge = reduce(vcat, [fill(p.kind == :edge, BI.num_points(p)) for p in itf.panels])
        Dt = dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-10))
        # Dᵀ Y_l = Y_l / (2(2l + 1)) on the unit sphere (DielectricBIE's sign)
        e0 = Dt(ones(length(X))) .- 0.5
        @test maximum(abs, e0) < 1e-7
        @test maximum(abs, e0[isedge]) < 1e-7                         # GCQ edge-panel nodes included
        @test maximum(abs, Dt(z) .- z ./ 6) < 1e-7
        y2 = (3 .* z .^ 2 .- 1) ./ 2
        @test maximum(abs, Dt(y2) .- y2 ./ 10) < 1e-7
        # the panel-level near criterion is needed: without it, near neighbours get the node rule
        D0 = dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-10, η = 0.0))
        @test maximum(abs, D0(z) .- z ./ 6) > 1e-5
    end

    @testset "SCTLCurvedNearQuad: Gauss identity on the lens (singular edge rules)" begin
        # D1 = 1/2 on a closed surface, so wᵀ Dᵀ σ = ½ wᵀ σ for every σ; with σ ~ s^p₁ at the rim (s the distance to
        # it) this exercises the weighted GCQ edge panels (pw ≠ 0): strip, Gauss-Jacobi weights, s^pw interpolation
        g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
        m = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-8)
        itf = m.interface; w = BI.all_weights(itf)
        p1 = m.powers[findfirst(c -> !c.smooth, g.curves)][1]
        @test all(p -> p.kind != :edge || p.ru.pw ≈ p1, itf.panels) && p1 < -0.1
        s = [hypot(hypot(x[1], x[2]) - 1, x[3]) for x in nodes_of(itf)]
        gauss(Dt, σ) = abs(dot(w, Dt(σ)) - dot(w, σ) / 2) / dot(abs.(w), abs.(σ))
        Dt = dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-12))
        @test gauss(Dt, s .^ p1) < 1e-9                                 # 9e-11 measured
        @test gauss(Dt, s .^ (p1 + 1)) < 1e-8                           # 3e-9 (1e-10 at n_quad = 12)
        # without the panel-level near criterion it fails badly
        @test gauss(dt_operator(itf, SCTLCurvedNearQuad(tol = 1e-12, η = 0.0)), s .^ p1) > 1e-2
    end

    # ∫ over the parameter triangle (c, p, q), Duffy-mapped so that a 1/r singularity at c cancels: f(z, s, dz) is the
    # integrand at z = c + s dz (dz = p - c + t (q - p)), which for s small should take r from the tangents at c (two
    # positions that close round to the same point)
    function duffy_tri(f, c, p, q)
        J = abs((p[1] - c[1]) * (q[2] - p[2]) - (p[2] - c[2]) * (q[1] - p[1]))
        g = st -> (dz = (p .- c) .+ st[2] .* (q .- p); st[1] * J * f(c .+ st[1] .* dz, st[1], dz))
        return BI.HCubature.hcubature(g, [0.0, 0.0], [1.0, 1.0]; rtol = 1e-13, atol = 1e-19, maxevals = 2_000_000)[1]
    end
    # ∫ over the unit parameter square of f, singular at c when c is given
    square_int(f, c) = c === nothing ?
        BI.HCubature.hcubature(uv -> f((uv[1], uv[2]), nothing, nothing), [0.0, 0.0], [1.0, 1.0]; rtol = 1e-13,
                               atol = 1e-19, maxevals = 10_000_000)[1] :
        sum(duffy_tri(f, c, (o[1], c[2]), o) + duffy_tri(f, c, o, (c[1], o[2])) for o in ((0, 0), (1, 0), (0, 1), (1, 1)))

    @testset "SCTLCurvedNearQuad: rows on thin panels at the edge (exact sphere integrals)" begin
        # On the unit sphere n_x·(x - y) = ±|x - y|²/2, so the Dᵀ row of a source panel P for σ = 1 is ±∫_P dS/(8π r),
        # computed independently (Duffy on P itself). Deep grading and a tight rule make P's elements next to the
        # equator very thin (u-widths down to 1e-14 against a v-length ~0.4) and put nodes within 2e-8 of it
        g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
        m = curved_gcq_mesh(g; nlev = 6, n_quad = 12, tol = 1e-12)
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
        C = Matrix(near_corrections(SCTLCurvedNearQuad(tol = 1e-12), itf))
        offs = cumsum([0; [BI.num_points(p) for p in itf.panels]])
        X, N = BI._interface_targets_normals(itf)
        worst = zeros(2)                                                  # same surface, across the edge
        for (s, S) in enumerate(itf.panels[[1, 3]])                       # sources: the GCQ panel P and the thin GL G
            js = offs[2s - 1] .+ (1:BI.num_points(S))
            A = C[:, js] .+ BI.dt_node_rule_block(itf, 2s - 1, X, N)
            for (t, T) in enumerate(itf.panels), a in unique([1, 2, 3, length(T.ru.x) ÷ 2, length(T.ru.x)]),
                b in (1, length(T.rv.x) ÷ 2, length(T.rv.x))
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
                worst[k] = max(worst[k], abs(sum(A[r, :]) - ref) / abs(ref))
            end
        end
        @test worst[1] < 3e-10                                            # 5e-11 measured
        # across the edge the target's offset from P is formed from two patch maps (two anchors, ~2e-16 each) that meet
        # at the equator only to rounding: for Q's nodes 1.6e-8 from it that limits the geometry, and so the row (and
        # the identity behind the reference)
        @test worst[2] < 1e-8                                             # 1.5e-9 measured
    end

    @testset "SCTLCurvedNearQuad: a chart's pole inside a panel" begin
        # one panel per patch: the caps' polar charts have their pole at the centre of a panel, where the tangents
        # vanish (panel scales and closest points must not be taken there)
        g = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
        m = curved_gcq_mesh(g; nlev = 0, across = 1, m = 1, n_quad = 4, tol = 1e-8)
        C = near_corrections(SCTLCurvedNearQuad(tol = 1e-10), m.interface)
        @test all(isfinite, nonzeros(C))
    end

    # induced potential u(x) = Σ w σ / (4π |x - y|), by direct summation (targets well away from the surface)
    function u_ind(itf, σ, T)
        X = nodes_of(itf); w = BI.all_weights(itf)
        return [sum(w[j] * σ[j] / (4π * norm(collect(t .- X[j]))) for j in eachindex(X)) for t in T]
    end
    curved_operator(m; tol = 1e-10) =
        dielectric_operator(m.interface; far = FMM3DFarField(1e-12), near = SCTLCurvedNearQuad(; tol))
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
        @test maximum(abs, u_ind(m.interface, σ, T) .- uref) / maximum(abs, uref) < 1e-6

        x1 = (0.3, -0.2, 1.7); x2 = (-1.4, 0.5, -0.9)
        for (name, ps) in (("lens", lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0)),
                           ("split sphere", lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0)))
            g = CurvedGeometry(ps)
            m = curved_gcq_mesh(g; nlev = 3, n_quad = 8, tol = 1e-8)
            w = BI.all_weights(m.interface)
            A = curved_operator(m)
            σ1 = solve_point(A, m, x1); σ2 = solve_point(A, m, x2)
            # every body is neutral; neutrality and reciprocity are limited by n_quad = 8 (split sphere: 1.4e-7 and
            # 1.4e-6 at n_quad = 8, 1e-10 and 1e-9 at n_quad = 12, the same at nlev 3 and 4)
            @test abs(sum(w .* σ1)) / sum(w .* abs.(σ1)) < 1e-6
            # reciprocity of the induced potential: u(x2; charge at x1) = u(x1; charge at x2)
            a12 = u_ind(m.interface, σ1, [x2])[1]; a21 = u_ind(m.interface, σ2, [x1])[1]
            @test a12 ≈ a21 rtol = 1e-5
            # GCQ self-convergence: one more level and a tighter rule change u by < 1e-6
            mf = curved_gcq_mesh(g; nlev = 4, n_quad = 8, tol = 1e-10)
            σf = solve_point(curved_operator(mf), mf, x1)
            T = fib(2.2)
            uf = u_ind(mf.interface, σf, T)
            @test maximum(abs, u_ind(m.interface, σ1, T) .- uf) / maximum(abs, uf) < 1e-6
        end
    end
end
