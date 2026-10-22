"""
    PrecomputedVolumeField(vs; tol, kmax = nothing, c_pad = 5.0,
                           compute_pot = true, compute_grad = true,
                           cache_fft = false, cache_fft_pad = 1.25,
                           nthreads = min(TKM3D._ltkm_default_nthreads(), Sys.CPU_THREADS))

Target-independent precomputed spectral representation of the free-space
Laplace potential (`1/(4π r)` convention) of a `VolumeSource`, for evaluation
at many target batches without redoing per-call setup.

Construction (once): fix the evaluation box B_pad = source box B (Eq. 3.7)
padded by `c_pad * h` where `h = ||A_rho||_2` (Eq. 3.3); the Fourier box
follows Eqs. (3.11) and (3.16) via `near_field_geometry`. On that Fourier
box, run the type-1 NUFFT of all source charges, apply the truncated
Laplace kernel `TKM3D.truncated_laplace3d_hat`, and (optionally) materialize
the spectral gradient coefficients.

Evaluation (per batch): targets inside B use a type-2 NUFFT on the stored
coefficients — near the source support this spectral evaluation is the only
accurate option (point-charge summation has O(1%) quadrature error there);
the FFT dimensions never change, so FFTW planning is reused. Targets outside
B are well separated from the support and are evaluated with the FMM
(`lfmm3d` at the field tolerance) — equivalent to the exact point sum within
tolerance and asymptotically scalable in the batch size.

This removes the dominant cost of the RHS-adaptive mesh build, where the
target-dependent Fourier box previously forced a fresh type-1 NUFFT + FFT
plan at every refinement depth (~23 s per call on the production monolayer
density; measured 16x build speedup with max rel deviation 1.4e-5 and no
refinement-decision changes).

`nthreads` caps the internal FINUFFT transforms (both the construction-time
type-1 and every evaluation's type-2), mirroring `TKM3D.ltkm3dc`'s
`_ltkm_default_nthreads()` default of 16 (override via env
`TKM3D_FINUFFT_NTHREADS`): FINUFFT's default FFTW planner picks a
pathological high-thread-count plan for these mode grids otherwise, and
without this cap raising Julia's thread count degrades this path rather than
speeding it up. The default additionally floors that 16 at `Sys.CPU_THREADS`
so a small Slurm allocation (4-8 CPUs) doesn't oversubscribe — a strict
improvement over `ltkm3dc`'s unconditional default, not a divergence from
it; `Sys.CPU_THREADS` reports the machine total and can over-report inside a
cgroup-limited allocation, so `TKM3D_FINUFFT_NTHREADS` remains the precise
control for batch jobs. The FMM branch (`lfmm3d`, out-of-box targets) is
unaffected.

Memory: `coeff` holds `prod(nmodes)` complex doubles and `grad_coeff` three
times that (≈1 GB and ≈3 GB at the production kmax); construction with
`compute_grad = true` transiently holds both even if `compute_pot = false`.
Use the `compute_pot`/`compute_grad` flags to store only what the consumer
needs.

Cached-FFT mode (`cache_fft = true`, experimental): the standard in-box path
re-runs the FFT of the full coefficient grid inside FINUFFT on EVERY type-2
exec (~0.4 s at production scale, independent of the target count). With
`cache_fft = true` that FFT is done once at construction: the spectral
coefficients are divided by the FINUFFT spreading-kernel Fourier factors
(pre-deconvolution: the interpolation kernel's smoothing is corrected per
mode up front), zero-padded onto a fine grid of `cache_fft_pad * nmodes`
per axis, and inverse-FFT'd; per evaluation a native threaded ES-kernel
interpolation reads the stored fine grid (cost ∝ targets, not modes —
equivalent to FINUFFT's `spreadinterponly` type-2 exec, see
`_field_interponly_type2` for why it is not routed through finufft itself).
The coefficient arrays are dropped and replaced by the fine grids
(`cache_fft_pad^3` ≈ 2x their size at the default 1.25). Accuracy relies on
the spectrum having decayed to ~tol at `kmax` (guaranteed by the kmax
estimate); the padding keeps the active modes away from the fine-grid
Nyquist, where the upsampfac≈1 interpolation kernel loses accuracy —
validated at ~5e-9 max relative deviation from the standard path (pad 1.25;
pad 1.0 degrades the gradient to ~2e-6). Transient construction memory peaks
at approximately 2× the final cached footprint, as the phi-corrected padded
arrays, the bfft workspace, and (if `compute_grad = true`) all four grids
coexist before the coefficient arrays are released (~16 GB at production scale).

Window-derivative mode (`window_derivatives = true`, opt-in). One real scalar fine grid serves the potential, the
gradient and the Hessian.

- **Construction.** The potential coefficients are divided by the Fourier factors of this mode's window, padded to
  `window_sigma * nmodes` per axis, and inverse-FFT'd once. Only the real part is kept, because the gathering weights
  are real. No derivative coefficient arrays or grids are built, and `compute_pot`/`compute_grad` are irrelevant:
  everything is served.
- **The window.** The ES window, tapered to vanish smoothly at its support ends, `exp(β(q - 1)) q^6` with
  `q = sqrt(1 - z²)`. It is evaluated analytically together with its first two derivatives. Its width comes from
  `window_tol` (default `tol * 1e-4`) and `window_sigma` (default 1.5).
- **Evaluation.** Per batch, with ξ_d = n_f,d (Δk_d (y_d - c_d)/2π + 1/2), t_d = ℓ_d - ξ_d and
  α_d = n_f,d Δk_d/2π, one gathering gives
  ∂_a φ = -C α_a Σ g w'(t_a) Π w, ∂_aa φ = C α_a² Σ g w''(t_a) Π w and ∂_ab φ = C α_a α_b Σ g w'(t_a) w'(t_b) Π w.
- **Accuracy.** On smooth fields it matches the spectral route at equal `tol`. A fitted (FINUFFT-style
  piecewise-polynomial) window stalled the Hessian at 1e-10 to 1e-9, and the untapered ES fails at stencil
  transitions. Modes at the edge of the mode box are about 100× less accurate than low ones; `kmax` keeps the
  spectrum decayed there.
"""
struct PrecomputedVolumeField{T <: AbstractFloat}
    sources::Matrix{T}
    charges::Vector{T}
    geom::NearFieldGeometry{T}
    nmodes::NTuple{3, Int}
    kmax::T
    tol::T
    prefactor::T
    # Caps the internal FINUFFT transforms (construction's type-1 and each
    # evaluation's type-2) at construction time, mirroring TKM3D.ltkm3dc's
    # _ltkm_default_nthreads() default of 16: FINUFFT's default FFTW planner
    # picks a pathological high-thread plan for these mode grids otherwise
    # (TKM3D continuous.jl:197-203). Stored on the struct because evaluation
    # (not just construction) issues FINUFFT calls.
    nthreads::Int
    coeff::Union{Nothing, Array{Complex{T}, 3}}
    grad_coeff::Union{Nothing, Array{Complex{T}, 4}}
    # cache_fft mode: kernel-corrected fine-grid values for interp-only type-2
    nfdim::Union{Nothing, NTuple{3, Int}}
    pot_grid::Union{Nothing, Array{Complex{T}, 3}}
    grad_grid::Union{Nothing, Array{Complex{T}, 4}}
    # frozen interpolation kernel parameters (set at construction in cache_fft
    # mode; Nothing otherwise): nspread and Horner coefficients derived from
    # nfdim and tol at construction time, stored so evaluation never re-derives
    # them and cannot silently diverge if TKM3D's size-dependent heuristics change.
    interp_nspread::Union{Nothing, Int}
    interp_hc::Union{Nothing, Matrix{Float64}}
    # window-derivative mode: the real scalar fine grid, the window's spread width and Horner coefficients, and
    # α_d = n_f,d Δk_d / 2π (d ξ_d / d y_d)
    wd_grid::Union{Nothing, Array{T, 3}}
    wd_nspread::Int
    wd_hc::Any                                   # the _FieldWindow (nothing otherwise)
    wd_alpha::NTuple{3, T}
end

function PrecomputedVolumeField(
    vs::VolumeSource{T, 3};
    tol::Real,
    kmax::Union{Nothing, Real} = nothing,
    c_pad::Real = 5.0,
    compute_pot::Bool = true,
    compute_grad::Bool = true,
    cache_fft::Bool = false,
    cache_fft_pad::Real = 1.25,
    window_derivatives::Bool = false,
    window_sigma::Real = 1.5,
    window_tol::Union{Nothing, Real} = nothing,
    # min(...) so a small Slurm allocation (4-8 CPUs) doesn't oversubscribe:
    # unlike ltkm3dc's unconditional _ltkm_default_nthreads(), this also caps at
    # the machine's CPU count. Sys.CPU_THREADS reports the machine total and can
    # over-report inside a cgroup-limited allocation, so TKM3D_FINUFFT_NTHREADS
    # (read by _ltkm_default_nthreads()) remains the precise control for batch
    # jobs; this is a floor safety net, not a substitute for setting it.
    nthreads::Integer = min(TKM3D._ltkm_default_nthreads(), Sys.CPU_THREADS),
) where {T <: AbstractFloat}
    (compute_pot || compute_grad) || throw(ArgumentError("at least one of compute_pot or compute_grad must be true"))
    window_derivatives && cache_fft && throw(ArgumentError("window_derivatives and cache_fft are exclusive"))
    window_sigma > 1 || throw(ArgumentError("window_sigma must be > 1"))
    cache_fft_pad >= 1 || throw(ArgumentError("cache_fft_pad must be >= 1"))
    tolT = T(tol)
    tolT > zero(T) || throw(ArgumentError("tol must be positive"))
    sources, charges = _volume_source_fmm_sources(vs)
    geom = near_field_geometry(vs; c_pad = c_pad)
    # k_max keeps the min-nearest-neighbour spacing: Section 3.3 leaves k_Nyq open
    # for a general lattice, so only h_n is pinned to Eq. (3.3)'s ||A_rho||_2.
    km = isnothing(kmax) ? T(_estimate_tkm3dc_kmax(_estimate_source_spacing(vs))) : T(kmax)
    km > zero(T) || throw(ArgumentError("kmax must be positive"))
    center = geom.center
    dks = geom.dks
    kx = TKM3D.centered_mode_axis(dks[1], km)
    ky = TKM3D.centered_mode_axis(dks[2], km)
    kz = TKM3D.centered_mode_axis(dks[3], km)
    nmodes = (length(kx), length(ky), length(kz))

    srcx = dks[1] .* (vec(view(sources, 1, :)) .- center[1])
    srcy = dks[2] .* (vec(view(sources, 2, :)) .- center[2])
    srcz = dks[3] .* (vec(view(sources, 3, :)) .- center[3])
    coeff0 = FINUFFT.nufft3d1(srcx, srcy, srcz, complex.(charges), -1, tolT, nmodes...;
        nthreads = nthreads)
    coeff = TKM3D._tkm_coeff3(coeff0)
    TKM3D._truncated_kernel_multiply!(coeff, kx, ky, kz, km, T(geom.L))
    prefactor = dks[1] * dks[2] * dks[3] / T(2π)^3
    if window_derivatives
        # one scalar grid: no derivative coefficients; the coefficients are dropped after the grid is built
        tw = window_tol === nothing ? tolT * T(1e-4) : T(window_tol)
        grid, ns, hc, nfdim_w = _field_window_grid(coeff, tw, T(window_sigma))
        alpha = ntuple(d -> T(nfdim_w[d] * dks[d] / (2π)), 3)
        coeff = nothing
        return PrecomputedVolumeField{T}(
            sources, charges, geom, nmodes, km, tolT, prefactor, Int(nthreads),
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            grid, ns, hc, alpha)
    end
    grad_coeff = compute_grad ? TKM3D._spectral_gradient_coeffs_3d(coeff, kx, ky, kz) : nothing
    nfdim = nothing; pot_grid = nothing; grad_grid = nothing
    interp_nspread = nothing; interp_hc = nothing
    if cache_fft
        nfdim, pot_grid, grad_grid, interp_nspread, interp_hc =
            _field_cache_fft_grids(coeff, grad_coeff, tolT, T(cache_fft_pad), compute_pot)
    end
    return PrecomputedVolumeField{T}(
        sources, charges, geom,
        nmodes, km, tolT, prefactor, Int(nthreads),
        (compute_pot && !cache_fft) ? coeff : nothing,
        cache_fft ? nothing : grad_coeff,
        nfdim, pot_grid, grad_grid,
        interp_nspread, interp_hc,
        nothing, 0, nothing, (zero(T), zero(T), zero(T)))
end

# --- cache_fft mode internals -----------------------------------------------
# Mirrors TKM3D.jl's discrete spread-only conventions (_ltkm3dd_eval_spreadonly,
# dir-2 leg): centered-mode coefficients are divided by the spreading-kernel
# Fourier factors phi (the deconvolution FINUFFT skips in spreadinterponly
# mode), placed in FFT ordering on the fine grid, and bfft'd (iflag = +1).
# A spreadinterponly type-2 exec on the result is then exactly equivalent to
# the standard type-2 on the original coefficients, provided the spectrum has
# decayed at the mode-box edge (see the struct docstring).
#
# Spread-interp helpers from the internal TKM3D submodule (src/tkm3d/) used here:
#   _ltkm3dd_spreadonly_upsampfac
#   _ltkm3dd_spreadonly_kernel_params
#   _ltkm3dd_spreadonly_horner_coeffs
#   _ltkm3dd_spreadonly_next235even
#   _ltkm3dd_spreadonly_onedim_fseries_kernel
#   _ltkm3dd_spreadonly_deconvolveshuffle3d_dir2
#   _ltkm3dd_spreadonly_evaluate_kernel_runtime
#   _finufft_type2_eval_3d
#   _spectral_gradient_coeffs_3d

function _field_cache_fft_grids(
    coeff::Array{Complex{T}, 3},
    grad_coeff::Union{Nothing, Array{Complex{T}, 4}},
    tol::T,
    pad::T,
    compute_pot::Bool,
) where {T <: AbstractFloat}
    nmodes = size(coeff)
    # Derive kernel parameters from nmodes (the construction-time fine-grid
    # size) rather than nfdim so that the stored params match the grids.
    sigma = TKM3D._ltkm3dd_spreadonly_upsampfac(nmodes, Float64(tol))
    params = TKM3D._ltkm3dd_spreadonly_kernel_params(Float64(tol), sigma; kerformula = 1)
    hc = TKM3D._ltkm3dd_spreadonly_horner_coeffs(params)
    # padded relative to the TKM rule: keeps the active modes away from the
    # fine-grid Nyquist, where the upsampfac≈1 kernel correction degrades
    nfdim = ntuple(d -> TKM3D._ltkm3dd_spreadonly_next235even(
        max(ceil(Int, Float64(pad) * params.upsampfac * nmodes[d]), 2 * params.nspread)), 3)
    phi1 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[1], params, hc))
    phi2 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[2], params, hc))
    phi3 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[3], params, hc))
    pot_grid = nothing
    if compute_pot
        fw = TKM3D._ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(coeff, nfdim, phi1, phi2, phi3)
        FFTW.bfft!(fw)   # in-place: fw is already the final-size padded array
        pot_grid = fw
    end
    grad_grid = nothing
    if grad_coeff !== nothing
        grad_grid = Array{Complex{T}, 4}(undef, nfdim..., 3)
        for d in 1:3
            fw = TKM3D._ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(
                view(grad_coeff, :, :, :, d), nfdim, phi1, phi2, phi3)
            FFTW.bfft!(fw)   # in-place: reuse fw without a second allocation
            grad_grid[:, :, :, d] = fw
        end
    end
    # Return frozen kernel parameters so the struct can store them; evaluation
    # uses these rather than re-deriving from nfdim, preventing silent divergence
    # if TKM3D's size-dependent heuristics ever change.
    return nfdim, pot_grid, grad_grid, params.nspread, hc
end

# Interp-only type-2 on the stored fine grid; equivalent to
# TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, coeff) of the original
# coefficients.
#
# The interpolation is done natively (threaded Julia) rather than through a
# FINUFFT spreadinterponly plan: finufft 2.5.x's execute allocates and
# value-initializes its nf-sized internal workspace on EVERY exec even though
# spreadinterponly never touches it (finufft_core.cpp, fwBatch_), a measured
# ~6.4 ns/fine-grid-pt floor that would cost ~0.8 s per exec at production
# scale — more than the FFT this mode exists to avoid. The native interp is
# exactly FINUFFT's: same ES kernel piecewise polynomials (shared with the
# phi correction above), same fold u = frac(x/2π + 1/2)·nf — the half-period
# offset is compensated by the (-1)^k sign FINUFFT's kernel Fourier series
# bakes into phi (the `-exp` factor in _ltkm3dd_spreadonly_onedim_fseries_
# kernel). Validated against finufft's spreadinterponly exec to 1.4e-8.

# ES-kernel stencil for one coordinate: grid indices (1-based, periodic) and
# kernel weights for the ns fine-grid points supporting a target at x.
function _field_interp_stencil!(
    idx::Vector{Int},
    w::Vector{T},
    x::T,
    nf::Int,
    ns::Int,
    hc::Matrix{Float64},
) where {T <: AbstractFloat}
    u = mod(Float64(x) / (2π) + 0.5, 1.0) * nf   # FINUFFT fold: grid units in [0, nf)
    istart = ceil(Int, u - ns / 2)
    x1 = istart - u                              # in [-ns/2, -ns/2+1)
    @inbounds for l in 0:(ns - 1)
        idx[l + 1] = mod(istart + l, nf) + 1
        w[l + 1] = T(TKM3D._ltkm3dd_spreadonly_evaluate_kernel_runtime(x1 + l, hc, ns))
    end
    return nothing
end

function _field_interp3d!(
    out::AbstractVector{Complex{T}},
    grid::AbstractArray{Complex{T}, 3},
    txn::Vector{T},
    tyn::Vector{T},
    tzn::Vector{T},
    ns::Int,
    hc::Matrix{Float64},
) where {T <: AbstractFloat}
    nf1, nf2, nf3 = size(grid)
    n = length(txn)
    nch = max(1, min(Threads.nthreads(), n))
    Threads.@threads for c in 1:nch
        i1 = Vector{Int}(undef, ns); w1 = Vector{T}(undef, ns)
        i2 = Vector{Int}(undef, ns); w2 = Vector{T}(undef, ns)
        i3 = Vector{Int}(undef, ns); w3 = Vector{T}(undef, ns)
        @inbounds for j in (div((c - 1) * n, nch) + 1):div(c * n, nch)
            _field_interp_stencil!(i1, w1, txn[j], nf1, ns, hc)
            _field_interp_stencil!(i2, w2, tyn[j], nf2, ns, hc)
            _field_interp_stencil!(i3, w3, tzn[j], nf3, ns, hc)
            acc = zero(Complex{T})
            for l3 in 1:ns
                iz = i3[l3]
                accz = zero(Complex{T})
                for l2 in 1:ns
                    iy = i2[l2]
                    s = zero(Complex{T})
                    @simd for l1 in 1:ns
                        s += grid[i1[l1], iy, iz] * w1[l1]
                    end
                    accz += s * w2[l2]
                end
                acc += accz * w3[l3]
            end
            out[j] = acc
        end
    end
    return out
end

function _field_interponly_type2(
    f::PrecomputedVolumeField{T},
    txn::Vector{T},
    tyn::Vector{T},
    tzn::Vector{T},
    grid::Array{Complex{T}, N},
) where {T <: AbstractFloat, N}
    # Use kernel parameters frozen at construction time (stored in the struct)
    # rather than re-deriving from nfdim; this prevents silent divergence if
    # TKM3D's _ltkm3dd_spreadonly_upsampfac heuristics ever become size-dependent.
    ns = f.interp_nspread::Int
    hc = f.interp_hc::Matrix{Float64}
    ntrans = N == 3 ? 1 : size(grid, 4)
    out = Matrix{Complex{T}}(undef, length(txn), ntrans)
    for d in 1:ntrans
        g = N == 3 ? grid : view(grid, :, :, :, d)
        _field_interp3d!(view(out, :, d), g, txn, tyn, tzn, ns, hc)
    end
    return out
end

# --- window-derivative mode internals ----------------------------------------------------------------------
# The real scalar fine grid: the coefficients divided by the window's Fourier factors (the spread-only convention of
# the cached path, FINUFFT's fold and phase), zero-padded to sigma * nmodes per axis and bfft'd; the real part only
# (the gathering weights are real, and the field is the real part of the sum).
# The window of this mode: the ES window tapered to vanish smoothly at the ends of its support,
#     w(t) = exp(β (q - 1)) q^(2m),   q = sqrt(1 - z²),   z = t / (ns/2),
# evaluated analytically with its first two derivatives. A fitted window (FINUFFT's piecewise polynomials) loses
# about N² per derivative of the interpolant's accuracy (the Hessian stalled at 1e-10 to 1e-9), and the untapered
# ES has singular derivatives at the support ends, where gathering stencils change; with m = 3 the window and its
# first two derivatives go continuously to zero there. Width and shape follow FINUFFT's rules for the oversampling
# sigma (ns from the window tolerance, at most nsmax; β = 0.97 π ns (1 - 1/(2 sigma))).
struct _FieldWindow
    ns::Int
    beta::Float64
    m::Int
end

function _field_window_params(tolw::Real, sigma::Real; nsmax::Int = 24, taper::Int = 3)
    ns = clamp(ceil(Int, log(1 / max(Float64(tolw), eps())) / (π * sqrt(1 - 1 / sigma))), 2, nsmax)
    return _FieldWindow(ns, 0.97 * π * ns * (1 - 1 / (2 * sigma)), taper)
end

# w, dw/dt, d²w/dt² at t (grid units); zero outside (-ns/2, ns/2)
@inline function _field_window_eval(win::_FieldWindow, t::Real)
    h = win.ns / 2; z = t / h; r = 1 - z * z
    r <= 0 && return zero(t), zero(t), zero(t)
    q = sqrt(r); β = win.beta; m2 = 2 * win.m
    w = exp(β * (q - 1)) * q^m2
    dq = -z / q / h; d2q = -1 / (q * r) / h^2                        # dq/dt, d²q/dt²
    a = β + m2 / q                                                   # g = β(q - 1) + 2m log q: g' = a dq
    g1 = a * dq; g2 = a * d2q - m2 / r * dq^2                        # g'' = a d2q - (2m/q²) dq²
    return w, w * g1, w * (g2 + g1^2)
end

# Fourier factors of the window on a fine grid of nf points (FINUFFT's onedim_fseries_kernel: Gauss-Legendre
# quadrature of the window over its support, with the (-1)^k of the half-period fold)
function _field_window_fseries(nf::Int, win::_FieldWindow)
    J2 = win.ns / 2.0
    q = 2 * floor(Int, 2 + 3.0 * J2)                                 # twice FINUFFT's: the window is not a polynomial
    z, w = TKM3D._ltkm3dd_spreadonly_gaussquad(2 * q)
    f = Vector{Float64}(undef, q); a = Vector{ComplexF64}(undef, q)
    for n in 1:q
        zn = z[n] * J2
        f[n] = J2 * w[n] * first(_field_window_eval(win, zn))
        a[n] = -exp(2π * im * zn / nf)
    end
    aj = ones(ComplexF64, q); out = Vector{Float64}(undef, nf ÷ 2 + 1)
    for j in 0:(nf ÷ 2)
        x = 0.0
        for n in 1:q
            x += f[n] * 2.0 * real(aj[n])
        end
        out[j + 1] = x
        for n in 1:q
            aj[n] *= a[n]
        end
    end
    return out
end

function _field_window_grid(coeff::Array{Complex{T}, 3}, tolw::T, sigma::T; nsmax::Int = 24,
                            taper::Int = 3) where {T <: AbstractFloat}
    nmodes = size(coeff)
    win = _field_window_params(tolw, sigma; nsmax, taper)
    ns = win.ns
    nfdim = ntuple(d -> TKM3D._ltkm3dd_spreadonly_next235even(max(ceil(Int, Float64(sigma) * nmodes[d]), 2 * ns)), 3)
    phis = ntuple(d -> T.(_field_window_fseries(nfdim[d], win)), 3)
    fw = TKM3D._ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(coeff, nfdim, phis...)
    FFTW.bfft!(fw)
    grid = Array{T, 3}(undef, nfdim)
    Threads.@threads for i in eachindex(grid)
        @inbounds grid[i] = real(fw[i])
    end
    return grid, ns, win, nfdim
end

# the window and its first two derivatives at the ns stencil points t_l = x1 + l (l = 0..ns-1) of a target
@inline function _field_window_weights!(w0::AbstractVector{T}, w1::AbstractVector{T}, w2::AbstractVector{T},
                                        x1::Float64, win::_FieldWindow, ns::Int, order::Int) where {T}
    @inbounds for i in 1:ns
        a, b, c = _field_window_eval(win, x1 + (i - 1))
        w0[i] = T(a); w1[i] = T(b); w2[i] = T(c)
    end
    return nothing
end

# stencil of one coordinate: the first index and t_0 = x1 (FINUFFT's fold u = frac(x/2π + 1/2) nf)
@inline function _field_window_stencil(x::Float64, nf::Int, ns::Int)
    u = mod(x / (2π) + 0.5, 1.0) * nf
    istart = ceil(Int, u - ns / 2)
    return istart, istart - u
end

# Fused gathering of the potential (pot), gradient (grad, 3 × n) and Hessian (hess, 6 × n: xx, yy, zz, xy, xz, yz)
# at the scaled targets (x = Δk (y - c)) from the real grid; any of pot, grad, hess may be `nothing`. Physical
# scaling: C (the prefactor) and α (d ξ / d y).
function _field_window_gather!(pot, grad, hess, grid::Array{T, 3}, txn, tyn, tzn, ns::Int, hc::_FieldWindow,
                               α::NTuple{3, T}, C::T) where {T <: AbstractFloat}
    order = hess !== nothing ? 2 : grad !== nothing ? 1 : 0
    nf1, nf2, nf3 = size(grid)
    n = length(txn)
    nch = max(1, min(Threads.nthreads(), n))
    Threads.@threads for ch in 1:nch
        a0 = Vector{T}(undef, ns); a1 = similar(a0); a2 = similar(a0)
        b0 = similar(a0); b1 = similar(a0); b2 = similar(a0)
        c0 = similar(a0); c1 = similar(a0); c2 = similar(a0)
        i1 = Vector{Int}(undef, ns)
        @inbounds for j in (div((ch - 1) * n, nch) + 1):div(ch * n, nch)
            s1, x1 = _field_window_stencil(Float64(txn[j]), nf1, ns)
            s2, x2 = _field_window_stencil(Float64(tyn[j]), nf2, ns)
            s3, x3 = _field_window_stencil(Float64(tzn[j]), nf3, ns)
            _field_window_weights!(a0, a1, a2, x1, hc, ns, order)
            _field_window_weights!(b0, b1, b2, x2, hc, ns, order)
            _field_window_weights!(c0, c1, c2, x3, hc, ns, order)
            for l in 1:ns
                i1[l] = mod(s1 + l - 1, nf1) + 1
            end
            u000 = u100 = u010 = u001 = u200 = u020 = u002 = u110 = u101 = u011 = zero(T)
            for l3 in 1:ns
                iz = mod(s3 + l3 - 1, nf3) + 1
                t00 = t10 = t01 = t20 = t02 = t11 = zero(T)
                for l2 in 1:ns
                    iy = mod(s2 + l2 - 1, nf2) + 1
                    q0 = zero(T); q1 = zero(T); q2 = zero(T)
                    if order == 0
                        @simd for l1 in 1:ns
                            q0 = muladd(grid[i1[l1], iy, iz], a0[l1], q0)
                        end
                    elseif order == 1
                        @simd for l1 in 1:ns
                            g = grid[i1[l1], iy, iz]; q0 = muladd(g, a0[l1], q0); q1 = muladd(g, a1[l1], q1)
                        end
                    else
                        @simd for l1 in 1:ns
                            g = grid[i1[l1], iy, iz]
                            q0 = muladd(g, a0[l1], q0); q1 = muladd(g, a1[l1], q1); q2 = muladd(g, a2[l1], q2)
                        end
                    end
                    t00 = muladd(q0, b0[l2], t00)
                    if order >= 1
                        t10 = muladd(q1, b0[l2], t10); t01 = muladd(q0, b1[l2], t01)
                    end
                    if order == 2
                        t20 = muladd(q2, b0[l2], t20); t02 = muladd(q0, b2[l2], t02); t11 = muladd(q1, b1[l2], t11)
                    end
                end
                u000 = muladd(t00, c0[l3], u000)
                if order >= 1
                    u100 = muladd(t10, c0[l3], u100); u010 = muladd(t01, c0[l3], u010); u001 = muladd(t00, c1[l3], u001)
                end
                if order == 2
                    u200 = muladd(t20, c0[l3], u200); u020 = muladd(t02, c0[l3], u020); u002 = muladd(t00, c2[l3], u002)
                    u110 = muladd(t11, c0[l3], u110); u101 = muladd(t10, c1[l3], u101); u011 = muladd(t01, c1[l3], u011)
                end
            end
            pot === nothing || (pot[j] = C * u000)
            if grad !== nothing
                grad[1, j] = -C * α[1] * u100; grad[2, j] = -C * α[2] * u010; grad[3, j] = -C * α[3] * u001
            end
            if hess !== nothing
                hess[1, j] = C * α[1]^2 * u200; hess[2, j] = C * α[2]^2 * u020; hess[3, j] = C * α[3]^2 * u002
                hess[4, j] = C * α[1] * α[2] * u110; hess[5, j] = C * α[1] * α[3] * u101; hess[6, j] = C * α[2] * α[3] * u011
            end
        end
    end
    return nothing
end

"""
    volume_field_derivatives(field, targets; pot = true, grad = true, hess = true, far = nothing) -> (; pot, grad, hess)

The requested ones of the potential (vector), gradient (3 × n) and Hessian (6 × n: ∂xx, ∂yy, ∂zz, ∂xy, ∂xz, ∂yz) of
the field at `targets` (3 × n), the others `nothing`. With a window-derivative field (`window_derivatives = true`)
the in-box targets take one fused gathering from the scalar grid; otherwise each quantity is evaluated as by
[`volume_field_potential`](@ref), [`volume_field_gradient`](@ref) and [`volume_field_hessian`](@ref). Out-of-box
targets: point charges through `far` (default `FMM3DFarField(field.tol)`).
"""
function volume_field_derivatives(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real};
                                  pot::Bool = true, grad::Bool = true, hess::Bool = true, far = nothing) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    if f.wd_grid === nothing
        return (; pot = pot ? volume_field_potential(f, targets; far) : nothing,
                  grad = grad ? volume_field_gradient(f, targets; far) : nothing,
                  hess = hess ? volume_field_hessian(f, targets; far) : nothing)
    end
    trg = Matrix{T}(targets); n = size(trg, 2)
    P = pot ? Vector{T}(undef, n) : nothing; Gr = grad ? Matrix{T}(undef, 3, n) : nothing
    H = hess ? Matrix{T}(undef, 6, n) : nothing
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        nb = length(bidx)
        pb = pot ? Vector{T}(undef, nb) : nothing; gb = grad ? Matrix{T}(undef, 3, nb) : nothing
        hb = hess ? Matrix{T}(undef, 6, nb) : nothing
        _field_window_gather!(pb, gb, hb, f.wd_grid, txn, tyn, tzn, f.wd_nspread, f.wd_hc, f.wd_alpha, f.prefactor)
        pot && (P[bidx] .= pb); grad && (Gr[:, bidx] .= gb); hess && (H[:, bidx] .= hb)
    end
    if !isempty(oidx)
        fr = something(far, FMM3DFarField(f.tol)); To = trg[:, oidx]
        if fr isa FMM3DFarField                                     # one FMM for every requested quantity
            v = lfmm3d(fr.tol, Matrix{Float64}(f.sources); charges = Vector{Float64}(f.charges), targets = Matrix{Float64}(To),
                       pgt = hess ? 3 : grad ? 2 : 1)
            pot && (P[oidx] .= v.pottarg ./ (4π)); grad && (Gr[:, oidx] .= v.gradtarg ./ (4π))
            hess && (H[:, oidx] .= v.hesstarg ./ (4π))
        else
            pot && (P[oidx] .= point_charge_field(fr, f.sources, f.charges, To))
            grad && (Gr[:, oidx] .= point_charge_field(fr, f.sources, f.charges, To; grad = true))
            hess && (H[:, oidx] .= point_charge_field(fr, f.sources, f.charges, To; hess = true))
        end
    end
    return (; pot = P, grad = Gr, hess = H)
end

@inline in_field_box(f::PrecomputedVolumeField, targets::AbstractMatrix, i::Integer) =
    in_near_region(f.geom, targets, i)

function _field_scaled_targets(f::PrecomputedVolumeField{T}, targets, idxs) where {T}
    dks = f.geom.dks
    c = f.geom.center
    txn = T[dks[1] * (targets[1, i] - c[1]) for i in idxs]
    tyn = T[dks[2] * (targets[2, i] - c[2]) for i in idxs]
    tzn = T[dks[3] * (targets[3, i] - c[3]) for i in idxs]
    return txn, tyn, tzn
end


"""
    volume_field_potential(field, targets) -> Vector

Potential of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization. In-box targets via type-2 NUFFT, out-of-box via
the far-field evaluator `far` ([`point_charge_field`](@ref); default `FMM3DFarField(field.tol)`).
"""
function volume_field_potential(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}; far = nothing) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    f.wd_grid === nothing || return volume_field_derivatives(f, targets; pot = true, grad = false, hess = false, far).pot
    (f.coeff === nothing && f.pot_grid === nothing) &&
        throw(ArgumentError("field was built with compute_pot = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Vector{T}(undef, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = f.pot_grid !== nothing ?
            vec(_field_interponly_type2(f, txn, tyn, tzn, f.pot_grid)) :
            TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.coeff; nthreads = f.nthreads)
        for (k, i) in enumerate(bidx)
            out[i] = f.prefactor * real(vals[k])
        end
    end
    if !isempty(oidx)
        vals = point_charge_field(something(far, FMM3DFarField(f.tol)), f.sources, f.charges, trg[:, oidx])
        for (k, i) in enumerate(oidx)
            out[i] = vals[k]
        end
    end
    return out
end

"""
    volume_field_gradient(field, targets) -> 3 x n Matrix

Gradient of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization. In-box targets via type-2 NUFFT, out-of-box via
the far-field evaluator `far` ([`point_charge_field`](@ref); default `FMM3DFarField(field.tol)`).
"""
function volume_field_gradient(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}; far = nothing) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    f.wd_grid === nothing || return volume_field_derivatives(f, targets; pot = false, grad = true, hess = false, far).grad
    (f.grad_coeff === nothing && f.grad_grid === nothing) &&
        throw(ArgumentError("field was built with compute_grad = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Matrix{T}(undef, 3, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = f.grad_grid !== nothing ?
            _field_interponly_type2(f, txn, tyn, tzn, f.grad_grid) :
            TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.grad_coeff; nthreads = f.nthreads)
        for (k, i) in enumerate(bidx)
            out[1, i] = f.prefactor * real(vals[k, 1])
            out[2, i] = f.prefactor * real(vals[k, 2])
            out[3, i] = f.prefactor * real(vals[k, 3])
        end
    end
    if !isempty(oidx)
        vals = point_charge_field(something(far, FMM3DFarField(f.tol)), f.sources, f.charges, trg[:, oidx]; grad = true)
        for (k, i) in enumerate(oidx)
            out[1, i] = vals[1, k]; out[2, i] = vals[2, k]; out[3, i] = vals[3, k]
        end
    end
    return out
end

"""
    volume_field_hessian(field, targets) -> 6 x n Matrix

Hessian of the precomputed field at `targets` (3 x n), free-space `1/(4π r)` normalization, rows ∂xx, ∂yy, ∂zz,
∂xy, ∂xz, ∂yz. In-box targets via a type-2 NUFFT of the spectral second derivatives (needs the potential
coefficients: `compute_pot = true`, `cache_fft = false`), out-of-box targets by point charges through the
far-field evaluator `far` ([`point_charge_field`](@ref); default `FMM3DFarField(field.tol)`).
"""
function volume_field_hessian(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}; far = nothing) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    f.wd_grid === nothing || return volume_field_derivatives(f, targets; pot = false, grad = false, hess = true, far).hess
    f.coeff === nothing && throw(ArgumentError("the Hessian needs the potential coefficients (compute_pot = true, cache_fft = false)"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Matrix{T}(undef, 6, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        dks = f.geom.dks
        ks = ntuple(d -> TKM3D.centered_mode_axis(dks[d], f.kmax), 3)
        hc = Array{Complex{T}, 4}(undef, size(f.coeff)..., 6)
        pairs = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
        Threads.@threads for iz in axes(f.coeff, 3)
            @inbounds for iy in axes(f.coeff, 2), ix in axes(f.coeff, 1)
                kk = (ks[1][ix], ks[2][iy], ks[3][iz]); c = f.coeff[ix, iy, iz]
                for (m, (a, b)) in enumerate(pairs)
                    hc[ix, iy, iz, m] = -(kk[a] * kk[b]) * c
                end
            end
        end
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, hc; nthreads = f.nthreads)
        for (k, i) in enumerate(bidx), m in 1:6
            out[m, i] = f.prefactor * real(vals[k, m])
        end
    end
    if !isempty(oidx)
        out[:, oidx] .= point_charge_field(something(far, FMM3DFarField(f.tol)), f.sources, f.charges, trg[:, oidx]; hess = true)
    end
    return out
end

"""
    _rhs_volume_targets_field(field, targets, normals, eps_src) -> Vector

Dielectric-interface RHS values `-(n · ∇φ) / eps_src` at `targets` from a
precomputed field (free-space normalization, matching the TKM branch of
`_rhs_volume_targets_hybrid`). Replaces the per-call KDTree classify +
lfmm3d/ltkm3dc pair with one field evaluation.
"""
function _rhs_volume_targets_field(
    field::PrecomputedVolumeField{T},
    targets::Matrix{T},
    normals::Matrix{T},
    eps_src::T;
    far = nothing,
) where {T}
    n = size(targets, 2)
    @assert size(normals, 2) == n
    @assert size(normals, 1) == 3
    grad = volume_field_gradient(field, targets; far)
    rhs_vals = Vector{T}(undef, n)
    @inbounds for i in 1:n
        rhs_vals[i] = _rhs_from_grad(view(normals, :, i), view(grad, :, i), eps_src, one(T))
    end
    return rhs_vals
end
