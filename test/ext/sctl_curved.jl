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
