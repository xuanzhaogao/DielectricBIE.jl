# Window-derivative evaluation of PrecomputedVolumeField (window_derivatives = true): potential, gradient and Hessian
# from one scalar fine grid by differentiating the gathering window.
using DielectricBIE, Test, LinearAlgebra, Random
import DielectricBIE as BI

@testset "PrecomputedVolumeField: window derivatives" begin
    rel(a, b) = maximum(abs, a - b) / maximum(abs, b)
    pr = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

    # the window: analytic derivatives (against forward-mode AD of its value), vanishing smoothly at the support ends
    win = BI._field_window_params(1e-14, 1.5)
    wv(t) = first(BI._field_window_eval(win, t))
    for t in (-win.ns / 2 + 0.3, -1.7, 0.0, 0.41, win.ns / 2 - 1e-3)
        w, d1, d2 = BI._field_window_eval(win, t)
        @test d1 ≈ BI.ForwardDiff.derivative(wv, t) rtol = 1e-12 atol = 1e-300
        @test d2 ≈ BI.ForwardDiff.derivative(s -> BI.ForwardDiff.derivative(wv, s), t) rtol = 1e-10 atol = 1e-300
    end
    @test all(abs.(BI._field_window_eval(win, win.ns / 2 - 1e-6)) .< 1e-20)    # w, w', w'' -> 0 at the ends
    @test BI._field_window_eval(win, win.ns / 2) == (0.0, 0.0, 0.0)

    # gathering against exact Fourier sums, on the scaled coordinates x in [-π, π) (α = nf / 2π, C = 1)
    Random.seed!(11)
    m = (24, 20, 28)
    ks = [collect(Float64, (-(m[d] ÷ 2)):((m[d] - 1) ÷ 2)) for d in 1:3]
    function exact(coeff, X)
        n = size(X, 2); P = zeros(n); G = zeros(3, n); H = zeros(6, n)
        for I in findall(!iszero, coeff), t in 1:n
            k = (ks[1][I[1]], ks[2][I[2]], ks[3][I[3]])
            v = coeff[I] * cis(dot(collect(k), X[:, t]))
            P[t] += real(v)
            for d in 1:3; G[d, t] += real(im * k[d] * v); end
            for (q, (a, b)) in enumerate(pr); H[q, t] += real(-k[a] * k[b] * v); end
        end
        return P, G, H
    end
    function gather(coeff, X; tw = 1e-14, σ = 1.5)
        grid, ns, w, nf = BI._field_window_grid(coeff, tw, σ)
        n = size(X, 2); P = zeros(n); G = zeros(3, n); H = zeros(6, n)
        BI._field_window_gather!(P, G, H, grid, X[1, :], X[2, :], X[3, :], ns, w, ntuple(d -> nf[d] / (2π), 3), 1.0)
        return P, G, H, nf, ns
    end
    # targets: random, at the periodic wrap, and exactly at stencil transitions (u - ns/2 an integer) and next to them
    _, _, _, nf, ns = gather(zeros(ComplexF64, m...), zeros(3, 1))
    X = (rand(3, 120) .* 2 .- 1) .* π
    X = hcat(X, π .* (1 .- 1e-12 .* rand(3, 20)) .* sign.(randn(3, 20)), fill(-π, 3, 1))
    xt(d, j) = 2π * ((j + ns / 2) / nf[d] - 0.5)                               # a transition along axis d
    for j in (3, 17), δ in (0.0, 1e-13, -1e-13, 1e-9)
        X = hcat(X, [xt(1, j) + δ, xt(2, j + 1) - δ, xt(3, j + 2) + δ])
    end
    # single modes: low and mixed (errors relative to the mode's own derivatives, which are small for low k), and one
    # at the edge of the mode box, where the window's Fourier correction is largest (a field's spectrum has decayed
    # there: kmax is chosen for it)
    for (k, b) in (((1, 0, 0), 1e-12), ((3, -2, 5), 1e-12), ((11, -9, 13), 1e-10))
        c = zeros(ComplexF64, m...); c[(k .+ m .÷ 2 .+ 1)...] = cis(0.7)
        P0, G0, H0 = exact(c, X); P, G, H = gather(c, X)
        @test rel(P, P0) < b && rel(G, G0) < b && rel(H, H0) < b
    end
    # a decayed random spectrum, against the exact sums and the ik route on the same coefficients
    c = zeros(ComplexF64, m...)
    for i in 1:m[1], j in 1:m[2], l in 1:m[3]
        c[i, j, l] = exp(-(ks[1][i]^2 + ks[2][j]^2 + ks[3][l]^2) / 9.0) * cis(2π * rand())
    end
    P0, G0, H0 = exact(c, X); P, G, H = gather(c, X)
    @test rel(P, P0) < 1e-12 && rel(G, G0) < 1e-12 && rel(H, H0) < 1e-12
    Gc = BI.TKM3D._spectral_gradient_coeffs_3d(c, ks...)
    Gs = permutedims(real.(BI.TKM3D._finufft_type2_eval_3d(X[1, :], X[2, :], X[3, :], 1, 1e-14, Gc)))
    @test rel(G, Gs) < 1e-12
    @test maximum(abs, (H[1, :] + H[2, :] + H[3, :]) - (H0[1, :] + H0[2, :] + H0[3, :])) < 1e-12 * maximum(abs, H0)

    # a physical Gaussian: anisotropic lattice, translated source box; against the analytic fields and the spectral route
    σg = 0.5; cρ = (3.3, -1.7, 0.4)
    ρf(x) = exp(-((x[1] - cρ[1])^2 + (x[2] - cρ[2])^2 + (x[3] - cρ[3])^2) / (2σg^2)) / (2π * σg^2)^1.5
    fp(r) = (sqrt(2 / π) / σg * exp(-r^2 / (2σg^2)) * r - BI.erf(r / (sqrt(2) * σg))) / (4π * r^2)
    function ex(x)
        d = collect(x .- cρ); r = norm(d); u = d ./ r
        Hm = (-ρf(x) - 2fp(r) / r) .* (u * u') .+ (fp(r) / r) .* (I - u * u')
        return BI.erf(r / (sqrt(2) * σg)) / (4π * r), fp(r) .* u, [Hm[a, b] for (a, b) in pr]
    end
    h = (0.12, 0.15, 0.1); R = 6σg; n = ceil.(Int, 2R ./ h) .+ 1
    Xs = reduce(hcat, [[cρ[1] - R + (i - 0.63) * h[1], cρ[2] - R + (j - 0.79) * h[2], cρ[3] - R + (k - 0.57) * h[3]]
                       for i in 1:n[1], j in 1:n[2], k in 1:n[3]])
    sv = BI.VolumeSource(Xs, fill(prod(h), size(Xs, 2)), [ρf(x) for x in eachcol(Xs)];
                         lattice_basis = ((h[1], 0.0, 0.0), (0.0, h[2], 0.0), (0.0, 0.0, h[3])))
    Y = hcat(collect(cρ) .+ (rand(3, 400) .- 0.5) .* 5.0, collect(cρ) .+ [6.0, 0.0, 0.0] .+ rand(3, 40))   # in and out of the box
    E = ex.(eachcol(Y)); Pe = first.(E); Ge = reduce(hcat, getindex.(E, 2)); He = reduce(hcat, getindex.(E, 3))
    fs = PrecomputedVolumeField(sv; tol = 1e-10, c_pad = 5.0)
    fw = PrecomputedVolumeField(sv; tol = 1e-10, c_pad = 5.0, window_derivatives = true)
    @test fw.coeff === nothing && fw.grad_coeff === nothing && fw.wd_grid isa Array{Float64, 3}
    r = volume_field_derivatives(fw, Y)
    es = (rel(volume_field_potential(fs, Y), Pe), rel(volume_field_gradient(fs, Y), Ge), rel(volume_field_hessian(fs, Y), He))
    ew = (rel(r.pot, Pe), rel(r.grad, Ge), rel(r.hess, He))
    @test all(ew .< 2 .* es .+ 1e-12)                                  # as accurate as the spectral route
    # tr H = -rho to the accuracy of the field (that of the spectral route on the same field)
    trw = maximum(abs, r.hess[1, :] + r.hess[2, :] + r.hess[3, :] + ρf.(eachcol(Y)))
    Hs = volume_field_hessian(fs, Y); trs = maximum(abs, Hs[1, :] + Hs[2, :] + Hs[3, :] + ρf.(eachcol(Y)))
    @test trw < 2 * trs + 1e-12 * maximum(abs, He)
    # the per-quantity evaluators dispatch to the same gathering (in the box; outside, one FMM per quantity instead of
    # one for all), and batches reuse the grid: the results agree to rounding (rtol 1e-13), not bitwise, because the
    # summation order is threaded (an exact == failed intermittently at -t 2)
    inb = 1:400
    @test isapprox(volume_field_gradient(fw, Y[:, inb]), r.grad[:, inb]; rtol = 1e-13) &&
          isapprox(volume_field_hessian(fw, Y[:, inb]), r.hess[:, inb]; rtol = 1e-13) &&
          isapprox(volume_field_potential(fw, Y[:, inb]), r.pot[inb]; rtol = 1e-13)
    @test volume_field_hessian(fw, Y) ≈ r.hess rtol = 1e-10
    r2 = volume_field_derivatives(fw, Y[:, 101:200]; pot = false, hess = false)
    @test r2.pot === nothing && r2.hess === nothing && isapprox(r2.grad, r.grad[:, 101:200]; rtol = 1e-13)
    @test_throws ArgumentError PrecomputedVolumeField(sv; tol = 1e-10, window_derivatives = true, cache_fft = true)
end

@testset "PrecomputedVolumeField: prolate (PSWF) window option" begin
    rel(a, b) = maximum(abs, a - b) / maximum(abs, b)
    pr = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    # psi_0^c: the prolate ODE (1 - z²) psi'' - 2 z psi' + (chi - c² z²) psi = 0 to rounding
    c = 1.15 * π * 15 * (1 - 1 / 3); w = BI._field_window_pswf(15, c; taper = 0)
    f0, _, f2 = BI._legendre_eval3(w.a, 0.0); χ = -f2 / f0
    @test maximum(z -> (v = BI._legendre_eval3(w.a, z); abs((1 - z^2) * v[3] - 2z * v[2] + (χ - c^2 * z^2) * v[1])), range(-0.99, 0.99; length = 101)) <
          1e-12 * c^2
    # analytic derivatives of the implemented window (tapered), against forward-mode AD of its value
    wt = BI._field_window_pswf(15, c; taper = 1)
    wv(t) = first(BI._field_window_eval(wt, t))
    for t in (-7.2, -1.3, 0.0, 2.9, 7.4)
        _, d1, d2 = BI._field_window_eval(wt, t)
        @test d1 ≈ BI.ForwardDiff.derivative(wv, t) rtol = 1e-10 atol = 1e-14
        @test d2 ≈ BI.ForwardDiff.derivative(s -> BI.ForwardDiff.derivative(wv, s), t) rtol = 1e-9 atol = 1e-14
    end
    @test abs(BI._field_window_eval(wt, 7.5 - 1e-12)[1]) < 1e-20 && BI._field_window_eval(wt, 7.5) == (0.0, 0.0, 0.0)
    # gathering against exact Fourier sums: a decayed spectrum and single modes, random / wrap / transition targets
    Random.seed!(13)
    m = (24, 20, 28); ks = [collect(Float64, (-(m[d] ÷ 2)):((m[d] - 1) ÷ 2)) for d in 1:3]
    function exact(coeff, X)
        n = size(X, 2); P = zeros(n); G = zeros(3, n); H = zeros(6, n)
        for I in findall(!iszero, coeff), t in 1:n
            k = (ks[1][I[1]], ks[2][I[2]], ks[3][I[3]]); v = coeff[I] * cis(dot(collect(k), X[:, t]))
            P[t] += real(v)
            for d in 1:3; G[d, t] += real(im * k[d] * v); end
            for (q, (a, b)) in enumerate(pr); H[q, t] += real(-k[a] * k[b] * v); end
        end
        return P, G, H
    end
    win = BI._field_window_choice(:pswf, 1e-14, 1.5)
    @test win isa BI._PSWFWindow && win.ns == 15 && win.m == 1
    function gather(coeff, X)
        grid, ns, _, nf = BI._field_window_grid(coeff, win, 1.5)
        n = size(X, 2); P = zeros(n); G = zeros(3, n); H = zeros(6, n)
        BI._field_window_gather!(P, G, H, grid, X[1, :], X[2, :], X[3, :], ns, win, ntuple(d -> nf[d] / (2π), 3), 1.0)
        return P, G, H, nf
    end
    _, _, _, nf = gather(zeros(ComplexF64, m...), zeros(3, 1))
    X = hcat((rand(3, 100) .* 2 .- 1) .* π, π .* (1 .- 1e-12 .* rand(3, 10)) .* sign.(randn(3, 10)))
    for j in (4, 13), δ in (0.0, 1e-13, -1e-13)
        X = hcat(X, [2π * ((j + win.ns / 2) / nf[d] - 0.5) + δ for d in 1:3])
    end
    # a spectrum decayed at the edge of the mode box, as TKM fields are (the default width is tuned for that; an
    # edge-heavy spectrum, ~1e-7 at the edge, gives Hessian errors ~1e-9 at this width)
    cs = zeros(ComplexF64, m...)
    for i in 1:m[1], j in 1:m[2], l in 1:m[3]
        cs[i, j, l] = exp(-(ks[1][i]^2 + ks[2][j]^2 + ks[3][l]^2) / 5.0) * cis(2π * rand())   # ~3e-13 at the box edge
    end
    P0, G0, H0 = exact(cs, X); P, G, H = gather(cs, X)
    @test rel(P, P0) < 1e-11 && rel(G, G0) < 1e-11 && rel(H, H0) < 1e-11
    for k in ((1, 0, 0), (3, -2, 5))
        c1 = zeros(ComplexF64, m...); c1[(k .+ m .÷ 2 .+ 1)...] = cis(0.7)
        P0, G0, H0 = exact(c1, X); P, G, H = gather(c1, X)
        @test maximum(abs, P - P0) < 1e-11 && maximum(abs, G - G0) < 1e-11 && maximum(abs, H - H0) < 1e-10
    end
    # through PrecomputedVolumeField: a physical Gaussian, as accurate as the spectral route
    σg = 0.5; ρf(x) = exp(-(x[1]^2 + x[2]^2 + x[3]^2) / (2σg^2)) / (2π * σg^2)^1.5
    h = 0.12; R = 6σg; n = ceil(Int, 2R / h) + 1
    Xs = reduce(hcat, [[-R + (i - 0.63) * h, -R + (j - 0.79) * h, -R + (k - 0.57) * h] for i in 1:n, j in 1:n, k in 1:n])
    sv = BI.VolumeSource(Xs, fill(h^3, size(Xs, 2)), [ρf(x) for x in eachcol(Xs)]; lattice_basis = ((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h)))
    Y = (rand(3, 300) .- 0.5) .* 4.0
    fs = PrecomputedVolumeField(sv; tol = 1e-10, c_pad = 5.0)
    fp = PrecomputedVolumeField(sv; tol = 1e-10, c_pad = 5.0, window_derivatives = true, window = :pswf)
    @test fp.wd_hc isa BI._PSWFWindow && fp.wd_nspread == 15
    r = volume_field_derivatives(fp, Y)
    @test rel(r.grad, volume_field_gradient(fs, Y)) < 1e-9 && rel(r.hess, volume_field_hessian(fs, Y)) < 1e-9
    @test_throws ArgumentError PrecomputedVolumeField(sv; tol = 1e-10, window_derivatives = true, window = :kb)
end
