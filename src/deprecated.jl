# Old entry points kept for scripts. The in-house near quadrature they called (upsampling, adaptive quadtrees,
# HCubature corrections) is gone: every one is now a thin wrapper over the FMM node rule and SCTL's near and self
# corrections (sctl_gl_corrections, SCTLPotential), so all of them need `using SCTL`. Keywords of the old methods
# (`correct_edges`, `adaptive_atol`, `adaptive_rtol`, `adaptive_n_GL`, `adaptive_max_depth`, `range_factor`) and
# `max_order` are accepted and ignored: SCTL corrects every panel, edges included, at the tolerance given. Each
# wrapper gives a deprecation warning (with `--depwarn=yes`) naming its replacement. SCTLCurvedNearQuad, the former
# near quadrature of curved interfaces, is GCQNearQuad now (its GCQ panels' rows in Julia, SCTL's for the GL panels).

"""
    lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; kw...) -> LinearMap

Kept for old scripts: the dielectric operator `τ I + Dᵀ` as a `LinearMap`,
[`dielectric_operator`](@ref)`(interface; far = FMM3DFarField(fmm_tol), near = SCTLNearCorrection(tol = up_tol))`.
`max_order` and the keywords of the former upsampling quadrature (`correct_edges`, `adaptive_*`) are ignored. Needs
`using SCTL`.
"""
function lhs_dielectric_box3d_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real,
                                              max_order::Integer; kw...)
    Base.depwarn("`lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)` is deprecated, use " *
                 "`dielectric_operator(interface; far = FMM3DFarField(fmm_tol), " *
                 "near = SCTLNearCorrection(tol = up_tol))`", :lhs_dielectric_box3d_fmm3d_corrected)
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
function laplace3d_DT_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real, max_order::Integer;
                                      kw...)
    Base.depwarn("`laplace3d_DT_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)` is deprecated, use " *
                 "`laplace3d_DT_fmm3d(interface, fmm_tol)` plus `sctl_gl_corrections(interface, " *
                 "eachindex(interface.panels); tol = up_tol)`, or `dielectric_operator`", :laplace3d_DT_fmm3d_corrected)
    return _fmm_plus_sctl(laplace3d_DT_fmm3d(interface, Float64(fmm_tol)), interface, up_tol, :DT)
end

"""
    laplace3d_DT_fmm3d_corrected_hcubature(interface, fmm_tol, atol; kw...) -> LinearMap

Kept for old scripts: [`laplace3d_DT_fmm3d_corrected`](@ref) at SCTL tolerance `atol` (the HCubature corrections
it named are gone). Needs `using SCTL`.
"""
function laplace3d_DT_fmm3d_corrected_hcubature(interface::DielectricInterface, fmm_tol::Real, atol::Real; kw...)
    Base.depwarn("`laplace3d_DT_fmm3d_corrected_hcubature(interface, fmm_tol, atol)` is deprecated, use " *
                 "`laplace3d_DT_fmm3d(interface, fmm_tol)` plus `sctl_gl_corrections(interface, " *
                 "eachindex(interface.panels); tol = atol)`, or `dielectric_operator`",
                 :laplace3d_DT_fmm3d_corrected_hcubature)
    return _fmm_plus_sctl(laplace3d_DT_fmm3d(interface, Float64(fmm_tol)), interface, atol, :DT)
end

"""
    laplace3d_D_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; kw...) -> LinearMap

Kept for old scripts: the double layer `D` (`N × N`, the sign of `laplace3d_D_fmm3d`), `laplace3d_D_fmm3d(interface,
fmm_tol)` plus [`sctl_gl_corrections`](@ref)`(interface, all panels; tol = up_tol, kernel = :D)`. `max_order` and
the keywords of the former upsampling quadrature are ignored. Needs `using SCTL`.
"""
function laplace3d_D_fmm3d_corrected(interface::DielectricInterface, fmm_tol::Real, up_tol::Real, max_order::Integer;
                                     kw...)
    Base.depwarn("`laplace3d_D_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)` is deprecated, use " *
                 "`laplace3d_D_fmm3d(interface, fmm_tol)` plus `sctl_gl_corrections(interface, " *
                 "eachindex(interface.panels); tol = up_tol, kernel = :D)`", :laplace3d_D_fmm3d_corrected)
    return _fmm_plus_sctl(laplace3d_D_fmm3d(interface, Float64(fmm_tol)), interface, up_tol, :D)
end

"""
    laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, fmm_tol, atol, range_factor; kw...) -> LinearMap

Kept for old scripts: the off-surface single-layer potential at `targets` (`3 × Nt`),
[`potential_operator`](@ref)`(SCTLPotential(tol = atol), interface, targets)`. `fmm_tol` (SCTL's FMM runs at
`atol`), `range_factor` and `correct_edges` are ignored. Needs `using SCTL`.
"""
function laplace3d_pottrg_fmm3d_corrected_hcubature(interface::DielectricInterface, targets::AbstractMatrix,
                                                    fmm_tol::Real, atol::Real, range_factor::Real; kw...)
    Base.depwarn("`laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, fmm_tol, atol, range_factor)` is " *
                 "deprecated, use `potential_operator(SCTLPotential(tol = atol), interface, targets)`",
                 :laplace3d_pottrg_fmm3d_corrected_hcubature)
    return potential_operator(SCTLPotential(tol = atol), interface, targets)
end

"""
    laplace3d_pottrg_near(interface, target::NTuple{3}, sol, atol; range_factor = 5.0) -> Float64

Kept for old scripts: the single-layer potential of the density `sol` at one point `target`, by
[`SCTLPotential`](@ref)`(tol = atol)` ([`potential_operator`](@ref)). `range_factor` is ignored. Needs `using SCTL`.
Each call builds SCTL's operator for its one target: for many targets, build one
`potential_operator(SCTLPotential(tol = atol), interface, targets)` (`3 × Nt`) and apply it.
"""
function laplace3d_pottrg_near(interface::DielectricInterface, target::NTuple{3, Real}, sol::AbstractVector, atol::Real;
                               range_factor::Real = 5.0)
    Base.depwarn("`laplace3d_pottrg_near(interface, target, sol, atol)` is deprecated, use " *
                 "`potential_operator(SCTLPotential(tol = atol), interface, targets) * sol` (one operator for " *
                 "all the targets)", :laplace3d_pottrg_near)
    return (potential_operator(SCTLPotential(tol = atol), interface, reshape(Float64[target...], 3, 1)) * sol)[1]
end

"""
    SCTLCurvedNearQuad(; tol = 1e-10, order = nothing, η = 1.5, ε_rel = 1e-9) -> GCQNearQuad

Kept for old scripts: [`GCQNearQuad`](@ref)`(quad_tol = tol)`, the near quadrature of `CurvedPanel` interfaces (their
default, with `quad_tol = 1e-12`): the GCQ edge panels' rows in Julia ([`gcq_curved_row`](@ref)), SCTL's
[`sctl_gl_corrections`](@ref) at `tol` for the Gauss-Legendre × Gauss-Legendre panels. `order`, `η` and `ε_rel`
(the proxy elements, near radius and principal value of the former SCTL quadrature of the GCQ panels) are ignored.
"""
function SCTLCurvedNearQuad(; tol::Real = 1e-10, order = nothing, η::Real = 1.5, ε_rel::Real = 1e-9)
    Base.depwarn("`SCTLCurvedNearQuad(; tol)` is deprecated, use `GCQNearQuad(quad_tol = tol)` (`order`, `η` and " *
                 "`ε_rel` are ignored)", :SCTLCurvedNearQuad)
    return GCQNearQuad(quad_tol = tol)
end
