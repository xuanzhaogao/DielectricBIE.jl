# The incident field g = grad N[a rho] by the surface corrections of the GCQ campaign mode (gcq_surface.jl, SCTL's
# quadrature) at points on the faces of boxes, against an independent reference: for a Gaussian rho and a box,
# 1/(4 pi r) = int_0^inf (4 pi t)^(-3/2) exp(-r^2 / 4t) dt gives N[chi_box rho](y) = int_0^inf prod_d E_d(t, y_d) dt,
# E_d a Gaussian times an erf difference. (Included by test/ext/sctl.jl when SCTL is installed.)
using DielectricBIE, SCTL, Test, LinearAlgebra
import DielectricBIE as BI

@testset "GCQ campaign mode: surface corrections of cut densities (SCTL)" begin
    erf = BI.erf
    σ = 0.5; cρ = (0.6, 0.05, 0.1)                                    # next to the face z = 0 and the edge x = 0.8, z = 0
    ρf(x) = exp(-((x[1] - cρ[1])^2 + (x[2] - cρ[2])^2 + (x[3] - cρ[3])^2) / (2σ^2)) / (2π * σ^2)^1.5
    fp(r) = (sqrt(2 / π) / σ * exp(-r^2 / (2σ^2)) * r - erf(r / (sqrt(2) * σ))) / (4π * r^2)
    ψex(x) = (d = collect(x .- cρ); r = norm(d); fp(r) .* d ./ r)
    function hrow(x)
        d = collect(x .- cρ); r = norm(d); u = d ./ r
        H = (-ρf(x) - 2fp(r) / r) .* (u * u') .+ (fp(r) / r) .* (I - u * u')
        return [H[1, 1], H[2, 2], H[3, 3], H[1, 2], H[1, 3], H[2, 3]]
    end
    function E1(t, y, c, lo, hi)
        v = σ^2 + 2t; s2 = 1 / (1 / (2t) + 1 / σ^2); s = sqrt(s2); m = s2 * (y / (2t) + c / σ^2)
        return exp(-(y - c)^2 / (2v)) / sqrt(2π * v) * (erf((hi - m) / (sqrt(2) * s)) - erf((lo - m) / (sqrt(2) * s))) / 2
    end
    gl = BI.gausslegendre(16)
    Nbox(y, lo, hi) = sum(0.5w * exp(p + 0.5 + 0.5x) * prod(E1(exp(p + 0.5 + 0.5x), y[d], cρ[d], lo[d], hi[d]) for d in 1:3)
                          for p in -80:79 for (x, w) in zip(gl...))
    gradN(y, lo, hi) = BI.ForwardDiff.gradient(z -> Nbox(z, lo, hi), collect(y))
    box(lo, hi) = (center = Tuple((lo .+ hi) ./ 2), Lx = hi[1] - lo[1], Ly = hi[2] - lo[2], Lz = hi[3] - lo[3])

    # probes on the faces z = 0 and x = 0.8 near the density, down to 1e-6 from the edge between them
    xe = 0.8
    Y = Vector{Float64}[]; Nn = Vector{Float64}[]
    for x in (0.0, 0.4, 0.7, xe - 1e-2, xe - 1e-4, xe - 1e-6, xe + 1e-3, 1.2), y in (0.05, 0.6)
        push!(Y, [x, y, 0.0]); push!(Nn, [0.0, 0.0, 1.0])
    end
    for z in (-0.6, -0.2, -1e-3, -1e-6, 1e-4, 0.3)
        push!(Y, [xe, 0.05, z]); push!(Nn, [1.0, 0.0, 0.0])
    end
    Y = reduce(hcat, Y); Nn = reduce(hcat, Nn); nt = size(Y, 2)
    # psi and H from samples of rho (spacing 0.1) by TKM near the density and point charges elsewhere
    h = 0.1; R = 7σ; n = ceil(Int, 2R / h) + 1
    Xs = reduce(hcat, [[cρ[1] - R + (i - 0.63) * h, cρ[2] - R + (j - 0.79) * h, cρ[3] - R + (k - 0.57) * h] for i in 1:n, j in 1:n, k in 1:n])
    sv = BI.VolumeSource(Xs, fill(h^3, size(Xs, 2)), [ρf(x) for x in eachcol(Xs)]; lattice_basis = ((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h)))
    ψt(Z) = BI._gcq_volume_gradient(sv, Z; tol = 1e-12, c_pad = 5.0, far = BI._gcq_far(:fmm3d))
    Ht(Z) = BI._gcq_volume_hessian(sv, Z; tol = 1e-12, c_pad = 5.0)
    # the same from one window-derivative field (one scalar grid, reused for every batch)
    fw = PrecomputedVolumeField(sv; tol = 1e-12, c_pad = 5.0, window_derivatives = true)
    ψw(Z) = volume_field_derivatives(fw, Z; pot = false, hess = false).grad
    Hw(Z) = volume_field_derivatives(fw, Z; pot = false, grad = false).hess
    # the Hessian against the exact one
    Hy = reduce(hcat, hrow.(eachcol(Y)))
    @test maximum(abs, Ht(Y) - Hy) < 1e-9 * maximum(abs, Hy)

    for (lohi, epses) in (([((-2.0, -2.0, -2.0), (xe, 2.0, 0.0))], [4.0]),
                          ([((-2.0, -2.0, -2.0), (xe, 2.0, 0.0)), ((xe, -2.0, -2.0), (3.0, 2.0, 0.0)),
                            ((-2.0, -2.0, 0.0), (xe, 2.0, 2.0)), ((xe, -2.0, 0.0), (3.0, 2.0, 2.0))], [4.0, 12.0, 2.0, 3.0]))
        boxes = [box(l...) for l in lohi]; eps_out = 1.0
        Fref = [dot(ψex(Y[:, j]) .+ sum((1 / e - 1 / eps_out) .* gradN(Y[:, j], l...) for (l, e) in zip(lohi, epses)), Nn[:, j])
                for j in 1:nt]
        rects = BI._gcq_surface_rects(BI._gcq_interface_rects(boxes, epses, eps_out), [(cρ, cρ)]; κ = 0.25, L0 = σ)
        Ys = BI._gcq_surface_snap(rects, Y, Nn)                      # onto the planes of their faces (rounding)
        S = BI.gcq_surface_corrections(rects, Ys; order = 8, tol = 1e-10)
        atr = [(BI._surf_a(boxes, epses, eps_out, Y[:, j] .+ 1e-9 .* Nn[:, j]) +
                BI._surf_a(boxes, epses, eps_out, Y[:, j] .- 1e-9 .* Nn[:, j])) / 2 for j in 1:nt]
        F(ΨY, Ψs, Hs) = (g = atr' .* ΨY .+ BI._gcq_surface_layers(S, Ys, Ψs, Hs); [dot(g[:, j], Nn[:, j]) for j in 1:nt])
        Fe = F(reduce(hcat, ψex.(eachcol(Y))), reduce(hcat, ψex.(eachcol(S.X))), reduce(hcat, hrow.(eachcol(S.X))))
        Ft = F(ψt(Y), ψt(S.X), Ht(S.X))
        Fw = F(ψw(Y), ψw(S.X), Hw(S.X))
        @test maximum(abs, Fw - Fref) < 1e-8 * maximum(abs, Fref)
        @info "surface corrections, $(length(boxes)) box(es), $(size(S.X, 2)) source nodes: error $(maximum(abs, Fe - Fref) / maximum(abs, Fref)) (exact psi, H), $(maximum(abs, Ft - Fref) / maximum(abs, Fref)) (TKM)"
        @test maximum(abs, Fe - Fref) < 1e-8 * maximum(abs, Fref)
        @test maximum(abs, Ft - Fref) < 1e-8 * maximum(abs, Fref)
    end
end

include(joinpath(@__DIR__, "..", "campaign", "fixture_cut.jl"))
@testset "GCQ campaign mode: cut densities, :surface (SCTL)" begin
    # four of the pairs (cut ones and the uncut onsite pair of the orbital above the face), to keep the solves few
    kw = (; far = :fmm3d, nlev = 1, quad_tol = 1e-8, edge_tol = 1e-8, tau = 1e-10, fg_tol = 1e-10, verbose = false,
          pairs = [(1, 1), (1, 2), (2, 4), (4, 4)])
    # one horizontal face cutting the orbitals: the surface corrections and the half-space formula agree
    c = load_campaign(write_cut_campaign(mktempdir(), 21))
    rs = gcq_four_index(c; kw..., cut_rhs = :surface); rh = gcq_four_index(c; kw..., cut_rhs = :halfspace)
    @test rs.stats.cut_pairs == 3
    @test maximum(abs, rs.V - rh.V) < 1e-4 * maximum(abs, rh.V)
    # an edge of the box through the orbitals (its faces x = c0 + 1.53 and z = c0 + 0.13): :auto takes :surface
    c0 = 2.0
    c = load_campaign(write_cut_campaign(mktempdir(), 21; box = [c0 + 1.53 + 6.0, c0, c0 + 0.13 - 6.0, 12.0, 12.0, 12.0, 4.0]))
    re = gcq_four_index(c; kw...)
    @test re.stats.cut_pairs > 0 && re.stats.asym < 1e-4
    # the window-derivative fields give the same V
    rw = gcq_four_index(c; kw..., field_derivs = :window)
    @test maximum(abs, rw.V - re.V) < 1e-8 * maximum(abs, re.V)
    @test_throws ErrorException gcq_four_index(c; kw..., cut_rhs = :halfspace)
end
