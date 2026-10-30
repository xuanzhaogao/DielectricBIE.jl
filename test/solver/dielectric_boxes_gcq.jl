using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Test
# the near corrections of GL×GL panels are SCTL's (sctl_gl_corrections): they need SCTL.jl
Base.find_package("SCTL") === nothing || @eval using SCTL

@testset "dielectric_tau and conductor edge powers" begin
    @test dielectric_tau(10.0, 1.0) ≈ 0.5 * (1.0 + 10.0) / (1.0 - 10.0)
    @test dielectric_tau(Inf, 1.0) == -0.5
    @test dielectric_tau(1.0, Inf) == 0.5
    @test_throws ArgumentError dielectric_tau(Inf, Inf)
    @test_throws ArgumentError dielectric_tau(2.0, 2.0)
    # perfect conductor at a right-angle edge: γ_k = 2k/3, the limit of the dielectric roots
    pc = edge_density_powers(Inf, 1.0; gmax = 4.0)
    @test pc ≈ [2k / 3 - 1 for k in 1:6]
    @test edge_density_powers(1e10, 1.0; gmax = 3.9) ≈ pc[1:5] atol = 1e-4
    @test edge_gcq_family_error(edge_gcq_rule(Inf, 1.0; tol = 1e-10)) < 1e-8
end

@testset "bodies and total-charge constraints" begin
    r = edge_gcq_rule(10.0, 1.0; tol = 1e-4)
    parts = [single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 0, n_quad = 3, rule = r, center = c)
             for c in ((-0.75, 0.0, 0.0), (0.75, 0.0, 0.0))]
    itf, bodies = join_bodies(parts)
    @test bodies == [1:DB.num_points(parts[1]), DB.num_points(parts[1])+1:DB.num_points(itf)]
    @test source_permittivity(parts, (0.7, 0.1, 0.0)) == 10.0
    @test source_permittivity(parts, (0.0, 0.9, 0.0)) == 1.0
    cons = box_charge_constraints(parts, PointSource((0.7, 0.1, 0.0), 1.0))
    @test cons[1][2] == 0.0 && cons[2][2] ≈ 1 - 1 / 10
    @test all(c[2] == 0.0 for c in box_charge_constraints(parts, [PointSource((0.0, 0.9, 0.0), 1.0)]))
    @test_throws ArgumentError box_charge_constraints(parts, PointSource((1.25, 0.0, 0.0), 1.0))   # on a face
    pinf = [single_dielectric_box3d_gcq(1.0, 1.0, 1.0, Inf, 1.0; nlev = 0, n_quad = 3, rule = r)]
    @test_throws ArgumentError box_charge_constraints(pinf, PointSource((0.1, 0.1, 0.1), 1.0))      # inside a conductor
end

@testset "high contrast and perfect conductors: deflation by total charge" begin
    src = PointSource((0.0, 0.9, 0.1), 1.0)                       # outside: the cube is neutral
    qerr(sol) = abs(dot(DB.all_weights(sol.interface), sol.σ)) / dot(DB.all_weights(sol.interface), abs.(sol.σ))
    # R = 2: a constraint shift of +1 would make the deflated system singular here
    # (nlev = 0, rule tol 1e-6: qerr(sd) ≈ 2e-7 for R = 1e6 and Inf; the plain system ≈ 7e-2 at R = 1e6)
    for R in (2.0, 1e6, Inf)
        r = edge_gcq_rule(R, 1.0; tol = 1e-6)
        parts = [single_dielectric_box3d_gcq(1.0, 1.0, 1.0, R, 1.0; nlev = 0, n_quad = 4, rule = r)]
        sd = solve_dielectric_boxes_gcq(parts, src; rtol = 1e-10)
        @test sd.stats.solved
        @test qerr(sd) < 1e-6
        if R == 2.0
            sp = solve_dielectric_boxes_gcq(parts, src; deflate = false, rtol = 1e-10)
            @test qerr(sd) < 10 * qerr(sp)                         # low contrast: no worse than the plain system
        elseif isfinite(R)
            sp = solve_dielectric_boxes_gcq(parts, src; deflate = false, rtol = 1e-10)
            @test qerr(sp) > 100 * qerr(sd)                        # the plain system's charge error grows ~ R
        else
            # perfect conductor: the total potential inside is constant
            pts = [(0.3cos(t), 0.3sin(t), 0.2) for t in range(0, 2pi, length = 9)[1:8]]
            u = potential_tensor(sd.interface, sd.σ, pts) .+ [1 / (4pi * norm(p .- src.point)) for p in pts]
            ui = [1 / (4pi * norm(p .- src.point)) for p in pts]
            @test (maximum(u) - minimum(u)) / (maximum(ui) - minimum(ui)) < 1e-4
        end
    end
end

@testset "GCQ solve: right contrast preconditioner" begin
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0), (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    m = multi_dielectric_box3d_gcq(boxes, [12.0, 3.0], 1.0; nlev = 0, n_quad = 3, tol = 1e-3)
    itf = m.interface
    src = DielectricBIE.PointSource((0.3, 0.1, 0.8), 1.0)
    A = first(DielectricBIE.lhs_dielectric_box3d_gcq_fmm3d(itf, 1e-12; quad_tol = 1e-8))
    b = DielectricBIE.rhs_dielectric_box3d(itf, src, 1.0)
    σ0, st0 = solve_dielectric_box3d_gcq(A, b, itf; rtol = 1e-10, precondition = false)
    σ1, st1 = solve_dielectric_box3d_gcq(A, b, itf; rtol = 1e-10)
    @test st1.niter < st0.niter
    # both to residual 1e-10; without the charge constraint σ is set only to about κ × 1e-10
    @test norm(σ1 - σ0) / norm(σ0) < 1e-6
end
