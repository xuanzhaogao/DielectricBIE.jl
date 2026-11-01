# The sinh-mapped polar (Duffy) rule: quadrature of f = g / r on a rectangle of a panel's parameter square, with g smooth
# and r the distance to a target point c of the rectangle (the integrable 1/r of a target on its own curved panel).
# The rectangle is cut into the four triangles from c to its sides. In a triangle, the polar substitution
# z = c + ρ (side point) turns 1/r into a smooth integrand in ρ, but along the side the integrand still has a peak of
# width s (the distance of c from the side line) when c is close to the side: the sinh map t = t* + s sinh(τ) around the
# foot t* of c on the side resolves it with `nside` Gauss-Legendre points in τ. s comes from the Lagrange identity
# (no cancellation), so the rule stays accurate to rounding for targets arbitrarily close to a side (measured to 1e-15
# in a unit square) and fails only for a G that is not positive definite.

const DUFFY_NRHO = 16      # points in ρ, along each ray from the target to a side
const DUFFY_NSIDE = 32     # points along each side, in the sinh-mapped variable

# Gauss-Legendre on [0, 1]; the default sizes are tabulated (a read-only Dict, safe from threads)
_gl01(n) = ((t, w) = gausslegendre(n); ((1 .+ t) ./ 2, w ./ 2))
const _DUFFY_GL = Dict(n => _gl01(n) for n in (DUFFY_NRHO, DUFFY_NSIDE))
_duffy_gl(n) = get(() -> _gl01(n), _DUFFY_GL, n)

# the nodes of the side p0 + t e, t ∈ [0, 1] (p0 relative to the target, e the side vector, det = |p0 × e| > 0), in the
# metric G = (E, F, Gg): t = t* + s sinh(τ) with t* the foot of the target on the side line and s its distance from it,
# both in G (in units of the side parameter t); the weights carry the Jacobian of the substitution. With A = |e|²_G,
# B = ⟨p0, e⟩_G and C = |p0|²_G, s² = (C - B²/A)/A, which is (E Gg - F²) det² / A² by the Lagrange identity: the
# subtraction C - B²/A loses all digits as s → 0 (it threw for s ≲ 3e-9), the product does not
function _duffy_side(p0, e, G, det, τ, wτ)
    E, F, Gg = G
    A = E * e[1]^2 + 2F * e[1] * e[2] + Gg * e[2]^2
    B = E * p0[1] * e[1] + F * (p0[1] * e[2] + p0[2] * e[1]) + Gg * p0[2] * e[2]
    tstar = -B / A
    s = sqrt(E * Gg - F^2) * det / A
    a, b = asinh(-tstar / s), asinh((1 - tstar) / s)
    t = @. tstar + s * sinh(a + (b - a) * τ)
    w = @. wτ * (b - a) * s * cosh(a + (b - a) * τ)
    return t, w
end

"""
    duffy_rule(c, rect, G; nρ = DUFFY_NRHO, nside = DUFFY_NSIDE) -> (du, dv, w)

Quadrature on the parameter rectangle `rect = (ua, ub, va, vb)` for integrands `g(u, v) / r`, with `g` smooth and `r` the
distance to the target `c = (u0, v0)` in the first fundamental form `G = (E, F, Gg)` at `c`. The nodes are the offsets
`(du, dv)` from `c`, so `Σ w[q] f(u0 + du[q], v0 + dv[q]) ≈ ∫_rect f du dv` (the parameter measure, the Jacobian of the
polar substitution included in `w`). `c` lies in `rect`; a side through `c` contributes nothing (its triangle is
degenerate). `nρ` points in the radial direction and `nside` along each side, the sinh map around the foot of `c` on the
side keeping the rule accurate to rounding as `c` approaches a side (measured down to 1e-15 from it in a unit square).
`G` must be positive definite, else an `ArgumentError` is thrown.
"""
function duffy_rule(c::NTuple{2, Float64}, rect::NTuple{4, Float64}, G::NTuple{3, Float64};
                    nρ::Int = DUFFY_NRHO, nside::Int = DUFFY_NSIDE)
    ua, ub, va, vb = rect
    E, F, Gg = G
    (E > 0 && E * Gg > F^2) || throw(ArgumentError("duffy_rule: the first fundamental form G must be positive definite"))
    ρ, wρ = _duffy_gl(nρ)
    τ, wτ = _duffy_gl(nside)
    corners = ((ua, va), (ub, va), (ub, vb), (ua, vb))
    du = Float64[]; dv = Float64[]; w = Float64[]
    sizehint!(du, 4 * nρ * nside); sizehint!(dv, 4 * nρ * nside); sizehint!(w, 4 * nρ * nside)
    for i in 1:4
        p = corners[i]; q = corners[mod1(i + 1, 4)]
        p0 = (p[1] - c[1], p[2] - c[2]); e = (q[1] - p[1], q[2] - p[2])
        det = abs(p0[1] * e[2] - p0[2] * e[1])           # twice the triangle's area, the Jacobian of (ρ, t) ↦ c + ρ (p0 + t e)
        det == 0 && continue
        t, wt = _duffy_side(p0, e, G, det, τ, wτ)
        for (tj, wj) in zip(t, wt), (ρk, wk) in zip(ρ, wρ)
            dzu = p0[1] + tj * e[1]; dzv = p0[2] + tj * e[2]
            push!(du, ρk * dzu); push!(dv, ρk * dzv); push!(w, wk * wj * ρk * det)
        end
    end
    return du, dv, w
end
