using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, Krylov
using Random
using SpecialFunctions: erf
using SCTL       # the near corrections of the corrected operator (SCTLNearCorrection)
using Test

@testset "dielectric_box3d" begin
    eps_box = 4.0
    # coarse mesh (1008 points): this checks dense == FMM and LU/GMRES convergence; the
    # flux check is a loose sanity bound.
    interface = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 3, 0.3, eps_box, 1.0, Float64; alpha = sqrt(2))

    lhs = BI.lhs_dielectric_box3d(interface)
    lhs_fmm3d = BI.lhs_dielectric_box3d_fmm3d(interface, 1e-12)
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x = BI.solve_lu(lhs, rhs)
    @test norm(lhs * x - rhs) < 1e-10

    x_trial = randn(BI.num_points(interface))
    @test norm(lhs * x_trial - lhs_fmm3d * x_trial) < 1e-8

    x_gmres = BI.solve_gmres(lhs_fmm3d, rhs, 1e-12, 1e-12)
    @test norm(lhs_fmm3d * x_gmres - rhs) < 1e-10

    total_flux = dot(ws, x)
    @test isapprox(total_flux + 1.0 / eps_box, 1.0, atol = 1e-1)

    total_flux_gmres = dot(ws, x_gmres)
    @test isapprox(total_flux_gmres + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "dielectric_box3d corrected" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    # FMM3D plus SCTL's near correction. (The uncorrected FMM operator's GMRES solve and
    # flux are already checked, more tightly, in the "dielectric_box3d" testset above.)
    lhs_corrected = dielectric_operator(interface; far = FMM3DFarField(1e-6), near = SCTLNearCorrection(tol = 1e-6))
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x_gmres_corrected = BI.solve_gmres(lhs_corrected, rhs, 1e-6, 1e-6)
    @test norm(lhs_corrected * x_gmres_corrected - rhs) < 1e-5

    total_flux_corrected = dot(ws, x_gmres_corrected)
    @test isapprox(total_flux_corrected + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "dielectric_box3d rhs adaptive" begin
    eps_box = 4.0
    ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        eps_box,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.lhs_dielectric_box3d(interface)
    rhs = BI.rhs_dielectric_box3d(interface, ps, eps_box)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x) +  1.0 / eps_box - 1.0) < 1e-2

    ps = BI.PointSource((1.0, 1.0, 1.0), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        1.0,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.lhs_dielectric_box3d(interface)
    rhs = BI.rhs_dielectric_box3d(interface, ps, 1.0)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x)) < 1e-2
end

@testset "dielectric_box3d rhs volume source" begin
    eps_box = 4.0
    eps_src = 2.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    xs = [2.0, 2.5]
    ys = [2.0, 2.5]
    zs = [2.0, 2.5]
    weights = fill(1.0, 2, 2, 2)
    density = fill(1.0, 2, 2, 2)
    vs = BI.VolumeSource{Float64, 3}((xs, ys, zs), weights, density)

    rhs_direct = BI.rhs_dielectric_box3d(interface, vs, eps_src)
    rhs_manual = zeros(Float64, BI.num_points(interface))
    for (i, point) in enumerate(BI.eachpoint(interface))
        acc = 0.0
        for ix in eachindex(xs), iy in eachindex(ys), iz in eachindex(zs)
            pos = (xs[ix], ys[iy], zs[iz])
            acc += weights[ix, iy, iz] * density[ix, iy, iz] *
                BI.laplace3d_grad(pos, point.panel_point.point, point.panel_point.normal)
        end
        rhs_manual[i] = -acc / eps_src
    end
    @test norm(rhs_direct - rhs_manual) <= 1e-12

    rhs_fmm = BI.rhs_dielectric_box3d_fmm3d(interface, vs, eps_src, 1e-8)
    @test norm(rhs_fmm - rhs_direct) / norm(rhs_direct) < 1e-6
end

@testset "dielectric_box3d rhs volume source hybrid near/far" begin
    eps_box = 4.0
    eps_src = 1.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 3, 0.3, eps_box, 1.0, Float64; alpha = sqrt(2))
    vs = BI.GaussianVolumeSource((0.0, 0.0, 0.6), 0.1, 40, 1e-8)

    n_points = BI.num_points(interface)
    targets = Matrix{Float64}(undef, 3, n_points)
    for (i, point) in enumerate(BI.eachpoint(interface))
        targets[:, i] .= point.panel_point.point
    end
    h = BI._estimate_source_spacing(vs)
    is_near = BI._classify_near_far_targets(targets, vs)
    @test any(is_near)
    @test !all(is_near)

    rhs_hybrid = BI.rhs_dielectric_box3d_hybrid(interface, vs, eps_src, 1e-8)

    # use the analytical results for Gaussian to validate the rhs
    rhs_exact = BI.Rhs_dielectric_box3d_gaussian(interface, (0.0, 0.0, 0.6), 0.1, 1.0)

    @test norm(rhs_hybrid - rhs_exact) / norm(rhs_exact) < 1e-8
end

@testset "screened_volume_source" begin
    eps_in = 4.0
    eps_out = 1.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_in, eps_out, Float64; alpha = sqrt(2))

    # mode is required (no default)
    vs_small = BI.VolumeSource([(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)], [0.5, 0.25], [2.0, -3.0])
    @test_throws MethodError BI.screened_volume_source(interface, vs_small)
    @test_throws MethodError BI.screened_volume_source(1.0, 1.0, 1.0, vs_small, eps_in, eps_out)

    # bandwidth validation
    @test_throws ArgumentError BI.SoftMixPermittivity(0.0)
    @test_throws ArgumentError BI.SoftMixInversePermittivity(-0.1)

    # sharp screening
    rho0 = 2.0
    points = [
        (0.0, 0.0, 0.0),   # interior
        (0.5, 0.0, 0.0),   # face
        (0.5, 0.5, 0.0),   # edge
        (0.5, 0.5, 0.5),   # corner
        (1.5, 0.0, 0.0),   # exterior
    ]
    vs = BI.VolumeSource(points, ones(length(points)), fill(rho0, length(points)))

    sharp = BI.screened_volume_source(interface, vs, BI.SharpScreening())
    sharp_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SharpScreening())
    @test sharp.density ≈ sharp_box.density
    @test sharp.density ≈ [rho0/eps_in, rho0/eps_in, rho0/eps_in, rho0/eps_in, rho0/eps_out]

    # soft mix — reference formulas
    bandwidth = 0.05
    min_corner = (-0.5, -0.5, -0.5)
    max_corner = (0.5, 0.5, 0.5)
    H(t) = (1 + erf(t / bandwidth)) / 2
    screen(p) =
        H(p[1] - min_corner[1]) * H(max_corner[1] - p[1]) *
        H(p[2] - min_corner[2]) * H(max_corner[2] - p[2]) *
        H(p[3] - min_corner[3]) * H(max_corner[3] - p[3])

    # SoftMixPermittivity
    soft_eps = BI.screened_volume_source(interface, vs, BI.SoftMixPermittivity(bandwidth))
    soft_eps_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SoftMixPermittivity(bandwidth))
    @test soft_eps.density ≈ soft_eps_box.density
    ref_eps = [rho0 / (eps_in * screen(p) + eps_out * (1 - screen(p))) for p in points]
    @test soft_eps.density ≈ ref_eps atol = 1e-12
    @test soft_eps.density[1] ≈ rho0 / eps_in atol = 1e-12
    @test soft_eps.density[end] ≈ rho0 / eps_out atol = 1e-12

    # SoftMixInversePermittivity
    soft_inv = BI.screened_volume_source(interface, vs, BI.SoftMixInversePermittivity(bandwidth))
    soft_inv_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SoftMixInversePermittivity(bandwidth))
    @test soft_inv.density ≈ soft_inv_box.density
    ref_inv = [rho0 * (screen(p) / eps_in + (1 - screen(p)) / eps_out) for p in points]
    @test soft_inv.density ≈ ref_inv atol = 1e-12
    @test soft_inv.density[1] ≈ rho0 / eps_in atol = 1e-12
    @test soft_inv.density[end] ≈ rho0 / eps_out atol = 1e-12

    # RHS convenience overloads use sharp screening internally
    vs_sharp = BI.screened_volume_source(interface, vs_small, BI.SharpScreening())
    @test BI.rhs_dielectric_box3d(interface, vs_small) ≈ BI.rhs_dielectric_box3d(interface, vs_sharp, 1.0)
end

@testset "dielectric_box3d volume backend wiring" begin
    root = pkgdir(DielectricBIE)
    entrypoint = read(joinpath(root, "src", "DielectricBIE.jl"), String)
    shape_dir = joinpath(root, "src", "shape")
    shape_sources = join([read(joinpath(shape_dir, f), String)
                          for f in readdir(shape_dir) if endswith(f, ".jl")], "\n")

    @test occursin("using .TKM3D", entrypoint)
    @test !occursin("using FBCPoisson", entrypoint)
    @test occursin("ltkm3dc", shape_sources)
    @test !occursin("lfbc3d", shape_sources)
end
