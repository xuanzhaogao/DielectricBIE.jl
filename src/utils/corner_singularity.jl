# Predict corner singularity powers for 2D dielectric wedge
# using idea as in van Bladel's EM book, Sec 4.13.
# Barnett 11/6/25
#
# The "interior" material with relative permittivity epsilon
# occupies angle alpha. Exterior epsilon=1. Matching is
#     epsilon phi_n^- = phi_n^+
# where + denotes exterior and - interior side.
#
# For our SLP representation, the density power is the jump in normal
# derivative.
function theta_shooting_even(theta::T, eps::T, g::T) where T
	return sin(g * theta / 2) * cos(g * (π - theta / 2)) + cos(g * theta / 2) * sin(g * (π - theta/2)) / eps
end

function theta_shooting_odd(theta::T, eps::T, g::T) where T
    return cos(g * theta / 2) * sin(g * (π - theta / 2)) + sin(g * theta / 2) * cos(g * (π - theta / 2)) / eps
end

# Predict corner singularity power sequence for multijunction 2D dielectric
# generalized wedge, using idea as in van Bladel's EM book, Sec 4.13.
# Barnett 11/7/25, generalizing cornervanbladel2d.jl to use 2x2 det of ODE
# transmission matrix (or one element of such matrix when PEC).
#
# There are nm>=2 materials including vacuum (nm=2 is a plain diel corner).
# The last material is optionally PEC (eps=Inf).
# The relative permittivities are length-nm vector e, and angles a length nm-1
# vector (the last angle defined implicitly).
# Matching at each junction is
#     eps_j phi_n^- = eps_{j+1} phi_n^+
# where -(+) denotes  theta just below (above).
# For our SLP representation, the density power is the jump in normal
# derivative.
"""
	theta_ODE_det(a,e,g)

	return determinant for theta ODE transmission for multi-junction dielectric
	relative premittivity vector `e` and angles `a` (length one less than `e`,
	since the last angle is known), and trial power `g`
	(so u''+g^2u = 0 is the theta ODE in each material).
	Zero is returned iff `g` is a (nonlin) eigenvalue of the periodic ODE on
	[0,2pi].
	The last `e` value (only) may be `Inf`, in which case Dirichlet BCs are used
	on its angle endpoints (and the incoming u' to final u value map used).
	Note: Parity issues avoided due to solving the 2x2 transmission matrix
	on the entire [0,2pi] domain. (All parities are combined.)
"""
function theta_ODE_det(a::AbstractVector, e::AbstractVector, g)
    M = Matrix(1.0I, 2, 2)     # initial I transm matrix (at theta=0^+)
    aa = [a; 2pi - sum(a)]       # add the last angle
    @assert aa[end] >= 0 "angles ($a) sum to bigger than 2pi!"
    nm = length(e)           # no. materials
    @assert length(aa) == nm "input vectors wrong lengths!"
    nexte = circshift(e, -1)   # list of j+1'th epsilons
    for (j, e) in enumerate(e)
        if !isinf(e)          # if last material conductor (u=0) skip it
            c = cos(g * aa[j])
            s = sin(g * aa[j])      # where nonlin in g enters
            # update transm matrix by propagate by angle a then scale u'...
            soverg = g == 0.0 ? aa[j] : s / g       # handle g=0 (u=at+b)
            M = [1.0 0; 0 e/nexte[j]] * [c soverg; -g*s c] * M
        end
    end
    return isinf(e[end]) ? M[1, 2] : det(M - I)   # use u' -> u map if Dir BCs 
end

# `g -> theta_ODE_det(a, e, g)` for Float64 g, as `_junction_powers` evaluates it at ~2e5 trial powers per junction.
# The sector openings and permittivity ratios are formed once, and the 2×2 products and det(M - I) are written out in
# scalars with the floating-point operations of the Matrix code, so every value is bitwise that of `theta_ODE_det`
# (test/utils/edge_gcq.jl). `det` of a 2×2 Matrix is an LAPACK LU factorization (getrf), whose calls serialize
# across threads: 16 threads evaluating `theta_ODE_det` ran slower than one.
struct ThetaODEDet
    aa::Vector{Float64}     # sector openings, the last one 2π - sum(a)
    ratio::Vector{Float64}  # e_j / e_{j+1} (cyclic)
    skip::Vector{Bool}      # conducting sectors (u = 0 there)
    pec::Bool               # last sector conducting: return M[1, 2] (the u' -> u map)
end

function ThetaODEDet(a::AbstractVector, e::AbstractVector)
    aa = [a; 2pi - sum(a)]
    @assert aa[end] >= 0 "angles ($a) sum to bigger than 2pi!"
    @assert length(aa) == length(e) "input vectors wrong lengths!"
    nexte = circshift(e, -1)
    return ThetaODEDet(Float64.(aa), Float64[e[j] / nexte[j] for j in eachindex(e)], isinf.(e), isinf(e[end]))
end

function (d::ThetaODEDet)(g::Float64)
    M11, M21, M12, M22 = 1.0, 0.0, 0.0, 1.0
    @inbounds for j in eachindex(d.aa, d.ratio, d.skip)
        d.skip[j] && continue
        c = cos(g * d.aa[j])
        s = sin(g * d.aa[j])
        soverg = g == 0.0 ? d.aa[j] : s / g
        r = d.ratio[j]
        T11, T21, T12, T22 = c, r * (-g * s), soverg, r * c         # [1 0; 0 r] * [c soverg; -g*s c]
        M11, M21, M12, M22 = T11 * M11 + T12 * M21, T21 * M11 + T22 * M21,
                             T11 * M12 + T12 * M22, T21 * M12 + T22 * M22
    end
    d.pec && return M12
    return _det2_lu(M11 - 1.0, M21, M12, M22 - 1.0)
end

# det([a b; c d]) as `det` computes it for a Float64 Matrix: the diagonal product for a triangular matrix, otherwise
# LU with partial pivoting as getrf does it (the multiplier is the other row times the pivot's reciprocal), det =
# ±pivot * u22, and 0 for an exactly singular factor. (Subnormal pivots, which the transfer matrices never have, are
# not reproduced.)
function _det2_lu(a::Float64, c::Float64, b::Float64, d::Float64)
    (c == 0 || b == 0) && return a * d
    swap = abs(c) > abs(a)
    p, q, u, v = swap ? (c, a, d, b) : (a, c, b, d)
    u22 = v - q * (1 / p) * u
    u22 == 0 && return 0.0
    return swap ? -(p * u22) : p * u22
end
