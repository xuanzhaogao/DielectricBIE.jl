using DielectricBIE, Test, LinearAlgebra, HCubature, SpecialFunctions
import DielectricBIE as BI
include(joinpath(@__DIR__, "fixture_cut.jl"))

@testset "GCQ campaign mode: densities cut by a face" begin
    σ = 0.5; xc = (0.13, -0.07, 0.15)                     # a Gaussian, cut off-centre by the plane z = 0
    ρ(x) = exp(-((x[1] - xc[1])^2 + (x[2] - xc[2])^2 + (x[3] - xc[3])^2) / (2σ^2)) / (2π * σ^2)^1.5

    # the cut-cell rule on a lattice line: int_{z < 0} of a Gaussian, the face between levels or on one
    h = 0.1
    for θ in (0.0, 0.37, 0.9)
        δ, mlo = BI._cutcell_correction(θ)
        @test length(δ) == 8 && mlo == -3
        I = sum(((m >= 1) + (mlo <= m <= mlo + 7 ? δ[m - mlo + 1] : 0.0)) * h * exp(-((m - 1 + θ) * h + xc[3])^2 / (2σ^2))
                for m in -60:200)
        Iref = σ * sqrt(π / 2) * (1 - erf(xc[3] / (sqrt(2) * σ)))
        @test abs(I - Iref) < 1e-7 * Iref                 # degree 7, h / σ = 0.2
    end

    # dz N[chi rho] on the plane and above it, chi = [z < 0], on a hexagonal lattice with a tilted third vector,
    # against adaptive cubature: depth u of the source, polar angle α about the vertical through y,
    # dz N[chi rho](y) = -int du dα dφ sin(α) / (4π) rho
    ref(y) = -hcubature(v -> begin
            u, α, ϕ = v; r = (y[3] + u) * tan(α)
            sin(α) / (4π) * ρ((y[1] + r * cos(ϕ), y[2] + r * sin(ϕ), -u))
        end, (0.0, 0.0, 0.0), (8σ, π / 2, 2π); rtol = 1e-10, atol = 1e-14, maxevals = 10^7)[1]
    ea = [h, 0.0, 0.0]; eb = [-h / 2, h * sqrt(3) / 2, 0.0]; ec = [0.07h, -0.04h, 0.9h]
    R = 7σ; n = ceil(Int, 2R / h) + 1; nz = ceil(Int, 2R / ec[3]) + 1
    o = [xc[1] - R + (n ÷ 2) * h / 2, xc[2] - R, xc[3] - R - 0.37 * ec[3]]
    Fd = [ρ(o .+ (i - 1) .* ea .+ (j - 1) .* eb .+ (k - 1) .* ec) for i in 1:n, j in 1:n, k in 1:nz]
    Y = [0.2 -0.4 0.3; -0.1 0.3 -0.6; 0.0 0.0 0.6]
    A = BI._gcq_halfspace_dz(Fd, o, ea, eb, ec, 0.0, -1, Y)
    refs = [ref(Y[:, j]) for j in axes(Y, 2)]
    @test maximum(abs, A - refs) < 1e-7 * maximum(abs, refs)
    # on the plane the two sides add up to dz phi0, phi0 = N[rho]
    Yp = Y[:, 1:2]
    B = BI._gcq_halfspace_dz(Fd, o, ea, eb, ec, 0.0, 1, Yp)
    dzφ0(y) = (d = y .- xc; r = norm(d);
               (sqrt(2 / π) / σ * exp(-r^2 / (2σ^2)) * r - erf(r / (sqrt(2) * σ))) / (4π * r^2) * d[3] / r)
    @test maximum(abs, A[1:2] + B - dzφ0.(eachcol(Yp))) < 1e-6 * maximum(abs, refs)
    # far from the density, point charges with the cut-cell weights of the side agree
    Yf = [0.4 -2.5; 3.1 0.2; 0.9 2.4]
    Af = BI._gcq_halfspace_dz(Fd, o, ea, eb, ec, 0.0, -1, Yf)
    dv = abs(det(hcat(ea, eb, ec)))
    zk = [o[3] + (k - 1) * ec[3] for k in 1:nz]
    θ = mod(-zk[1] / ec[3], 1.0); δ, mlo = BI._cutcell_correction(θ)
    pc = zeros(size(Yf, 2))
    for i in 1:n, j in 1:n, k in 1:nz
        m = round(Int, -zk[k] / ec[3] - θ) + 1
        wgt = (m >= 1) + (mlo <= m <= mlo + 7 ? δ[m - mlo + 1] : 0.0)
        wgt == 0 && continue
        x = o .+ (i - 1) .* ea .+ (j - 1) .* eb .+ (k - 1) .* ec
        for t in axes(Yf, 2)
            r = Yf[:, t] .- x
            pc[t] += -dv * wgt * Fd[i, j, k] * r[3] / (4π * norm(r)^3)
        end
    end
    @test maximum(abs, Af - pc) < 1e-7 * maximum(abs, refs)        # (the periodic images, at pad = 4)

    # a campaign whose substrate face cuts three orbitals (a fourth above it): V converges with the grid, the same
    # Gaussian blob (to 6 sigma) sampled at h = 0.2 and 0.133 (TKM on the screened samples rho / eps is off by 2e-2 to
    # 6e-2 at any h here)
    kw = (; far = :fmm3d, nlev = 1, quad_tol = 1e-8, edge_tol = 1e-8, tau = 1e-10, fg_tol = 1e-10, verbose = false)
    rs = map((31, 46)) do n
        gcq_four_index(load_campaign(write_cut_campaign(mktempdir(), n; L = 6.0)); kw...)
    end
    @test rs[1].pair_ids == rs[2].pair_ids
    @test all(r -> r.stats.cut_pairs == 8 && r.stats.cut_classes == 5, rs)
    sc = maximum(abs, rs[2].V)
    @test maximum(abs, rs[1].V - rs[2].V) < 3e-4 * sc
    @test all(r -> r.stats.asym < 1e-4, rs)
    @test_throws ArgumentError gcq_four_index(load_campaign(write_cut_campaign(mktempdir(), 21)); kw..., refine_targets = true)
    # other cuts (a vertical face through the orbitals, a slab cutting them twice) are classes without a plane (the
    # surface corrections, test/ext/sctl.jl); a vertical cut needs one lattice vector across the face
    function screening(c)
        cs = BI.enumerate_centers(c); shapes, shape_of, delta, dg = gcq_pair_shapes(c, cs, BI.enumerate_pairs(cs, c.neighbor_cutoff))
        return BI._gcq_screening(c, dg, shapes, shape_of, delta)..., shapes, dg
    end
    for box in ([2.0 + 1.6 + 6.0 - 0.07, 2.0, 2.0, 12.0, 12.0, 12.0, 4.0], [3.6, 2.0, 2.0, 12.0, 12.0, 0.5, 4.0])
        cls, _, classes = screening(load_campaign(write_cut_campaign(mktempdir(), 21; box)))
        @test count(>(0), cls) > 0 && any(cl -> !cl.plane, classes)
    end
    @test_throws ErrorException screening(load_campaign(write_cut_campaign(mktempdir(), 21; shear = 0.3,
                                                        box = [2.0 + 1.6 + 6.0 - 0.07, 2.0, 2.0, 12.0, 12.0, 12.0, 4.0])))
    # the cut-cell screening of one horizontal face (the tensor rule) is the 1D rule of the face:
    # m = (a_lo or a_hi by side) + δ_m (a_lo - a_hi), δ from the depth below the face
    cls, inveps, classes, shapes, dg = screening(load_campaign(write_cut_campaign(mktempdir(), 21)))
    o, ea, eb, ec = BI._grid_frame(dg)
    @test all(cl -> cl.plane, classes)
    for cl in classes
        t = [(cl.zf - (o[3] + (g[3] - 1) * ec[3])) / abs(ec[3]) for g in shapes[cl.shape].gidx]
        θ = mod(t[1], 1.0); δ, mlo = BI._cutcell_correction(θ)
        m1 = map(t) do tt
            mi = round(Int, tt - θ) + 1
            (mi >= 1 ? cl.a_lo : cl.a_hi) + (mlo <= mi <= mlo + 7 ? δ[mi - mlo + 1] : 0.0) * (cl.a_lo - cl.a_hi)
        end
        @test maximum(abs, cl.m - m1) < 1e-14
    end
end
