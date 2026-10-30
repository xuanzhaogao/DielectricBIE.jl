# Old entry points kept for scripts. The in-house near quadrature they called (upsampling, adaptive quadtrees,
# HCubature corrections) is gone: every one is now a thin wrapper over the FMM node rule and SCTL's near and self
# corrections (sctl_gl_corrections, SCTLPotential), so all of them need `using SCTL`. Keywords of the old methods
# (`correct_edges`, `adaptive_atol`, `adaptive_rtol`, `adaptive_n_GL`, `adaptive_max_depth`, `range_factor`) and
# `max_order` are accepted and ignored: SCTL corrects every panel, edges included, at the tolerance given.

"""
    lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; kw...) -> LinearMap

Kept for old scripts: the dielectric operator `τ I + Dᵀ` as a `LinearMap`,
[`dielectric_operator`](@ref)`(interface; far = FMM3DFarField(fmm_tol), near = SCTLNearCorrection(tol = up_tol))`.
`max_order` and the keywords of the former upsampling quadrature (`correct_edges`, `adaptive_*`) are ignored. Needs
`using SCTL`.
"""
function lhs_dielectric_box3d_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real,
                                              max_order::Integer; kw...)
    A = dielectric_operator(interface; far = FMM3DFarField(fmm_tol), near = SCTLNearCorrection(tol = up_tol))
    return LinearMap{Float64}(x -> A * x, size(A)...)
end

"""
    Lhs_dielectric_box3d_fmm3d_corrected

Kept for old scripts: the same function as [`lhs_dielectric_box3d_fmm3d_corrected`](@ref).
"""
const Lhs_dielectric_box3d_fmm3d_corrected = lhs_dielectric_box3d_fmm3d_corrected

# the FMM node rule `F` plus SCTL's corrections `C` of every panel, as an N × N LinearMap
function _fmm_plus_sctl(F, interface::DielectricInterface, tol::Real, kernel::Symbol)
    C = BlockCorrection(sctl_gl_corrections(interface, eachindex(interface.panels); tol = Float64(tol), kernel),
                        interface)
    n = num_points(interface)
    return LinearMap{Float64}(x -> F * x .+ C * x, n, n)
end

"""
    laplace3d_DT_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; kw...) -> LinearMap

Kept for old scripts: the adjoint double layer `Dᵀ` (`N × N`), `laplace3d_DT_fmm3d(interface, fmm_tol)`
plus [`sctl_gl_corrections`](@ref)`(interface, all panels; tol = up_tol)`. `max_order` and the keywords of the
former upsampling quadrature are ignored. Needs `using SCTL`.
"""
laplace3d_DT_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real, max_order::Integer; kw...) =
    _fmm_plus_sctl(laplace3d_DT_fmm3d(interface, Float64(fmm_tol)), interface, up_tol, :DT)

"""
    laplace3d_DT_fmm3d_corrected_hcubature(interface, fmm_tol, atol; kw...) -> LinearMap

Kept for old scripts: [`laplace3d_DT_fmm3d_corrected`](@ref) at SCTL tolerance `atol` (the HCubature corrections
it named are gone). Needs `using SCTL`.
"""
laplace3d_DT_fmm3d_corrected_hcubature(interface::DielectricInterface, fmm_tol::Real, atol::Real; kw...) =
    _fmm_plus_sctl(laplace3d_DT_fmm3d(interface, Float64(fmm_tol)), interface, atol, :DT)

"""
    laplace3d_D_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; kw...) -> LinearMap

Kept for old scripts: the double layer `D` (`N × N`, the sign of `laplace3d_D_fmm3d`), `laplace3d_D_fmm3d(interface,
fmm_tol)` plus [`sctl_gl_corrections`](@ref)`(interface, all panels; tol = up_tol, kernel = :D)`. `max_order` and
the keywords of the former upsampling quadrature are ignored. Needs `using SCTL`.
"""
laplace3d_D_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real, max_order::Integer; kw...) =
    _fmm_plus_sctl(laplace3d_D_fmm3d(interface, Float64(fmm_tol)), interface, up_tol, :D)

"""
    laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, fmm_tol, atol, range_factor; kw...) -> LinearMap

Kept for old scripts: the off-surface single-layer potential at `targets` (`3 × Nt`),
[`potential_operator`](@ref)`(SCTLPotential(tol = atol), interface, targets)`. `fmm_tol` (SCTL's FMM runs at
`atol`), `range_factor` and `correct_edges` are ignored. Needs `using SCTL`.
"""
laplace3d_pottrg_fmm3d_corrected_hcubature(interface::DielectricInterface, targets::AbstractMatrix, fmm_tol::Real,
                                           atol::Real, range_factor::Real; kw...) =
    potential_operator(SCTLPotential(tol = atol), interface, targets)

"""
    laplace3d_pottrg_near(interface, target::NTuple{3}, sol, atol; range_factor = 5.0) -> Float64

Kept for old scripts: the single-layer potential of the density `sol` at one point `target`, by
[`SCTLPotential`](@ref)`(tol = atol)` ([`potential_operator`](@ref)). `range_factor` is ignored. Needs `using SCTL`.
"""
laplace3d_pottrg_near(interface::DielectricInterface, target::NTuple{3, Real}, sol::AbstractVector, atol::Real;
                      range_factor::Real = 5.0) =
    (potential_operator(SCTLPotential(tol = atol), interface, reshape(Float64[target...], 3, 1)) * sol)[1]
