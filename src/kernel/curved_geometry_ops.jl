# Geometry of curved GCQ panels in pure Julia: differences y(b) - y(a) of panel points that stay accurate however close
# the points are (the tangents integrated along the parameter segment), the point of a panel closest to a target, its
# tangents and diameter. Used by the near and self rows of `CurvedPanel`s (gcq_curved_near.jl).

_curved_diam(P, ua, ub, va, vb) = (c = [collect(curved_lpoint(P, u, v)) for (u, v) in
                                         ((ua, va), (ub, va), (ub, vb), (ua, vb))];
                                   maximum(norm(c[i] - c[j]) for i in 1:4 for j in 1:4))

_dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

_tangents(P, u, v) = P.dmap === nothing ? curved_frame(P, u, v)[1:2] : P.dmap(u, v)

# the parameters (u, v) of the point of panel P closest to x (panel frame), and the distance: Gauss-Newton from the
# nearest node, clamped to the parameter square (stopped where a map is singular, e.g. a polar chart's pole at a corner)
function _curved_closest(P, x)
    nv = length(P.rv.x)
    i = argmin(k -> sum(abs2, x .- P.lpoints[k]), eachindex(P.lpoints))
    u, v = P.ru.x[(i - 1) ÷ nv + 1], P.rv.x[(i - 1) % nv + 1]
    for _ in 1:50
        yu, yv = _tangents(P, u, v)
        d = x .- curved_lpoint(P, u, v)
        a, b, c = _dot3(yu, yu), _dot3(yu, yv), _dot3(yv, yv)
        gu, gv = _dot3(yu, d), _dot3(yv, d)
        det = a * c - b^2
        un = clamp(u + (c * gu - b * gv) / det, 0.0, 1.0); vn = clamp(v + (a * gv - b * gu) / det, 0.0, 1.0)
        (isfinite(un) && isfinite(vn) && all(isfinite, _tangents(P, un, vn)[1])) || break
        (un, vn) == (u, v) && break
        u, v = un, vn
    end
    return u, v, sqrt(sum(abs2, x .- curved_lpoint(P, u, v)))
end

# y(b) - y(a) for parameter points a, b of panel P: ∫ of the tangents along the straight segment, so accurate
# relative to its own size however small. The map is analytic over the parameter square: Gauss with n points is
# exact to rounding for a segment of parameter length λ once λ^(2n) ≲ 1e-16 (2 points below 1e-4, 16 up to 1)
const _DIFF_GL = Dict(n => gausslegendre(n) for n in (2, 4, 8, 16))
_curved_diff(P, a, b) = _curved_step(P, a, (b[1] - a[1], b[2] - a[2]))

# y(a + Δ) - y(a), the same integral for the parameter step Δ as given (a quadrature point's offset from a reference
# point, which (a + Δ) - a would round to the spacing of the numbers at a)
function _curved_step(P, a, Δ)
    du, dv = Δ
    s1 = s2 = s3 = 0.0
    (du == 0 && dv == 0) && return (s1, s2, s3)
    λ = hypot(du, dv)
    t, w = _DIFF_GL[λ < 1e-4 ? 2 : λ < 1e-2 ? 4 : λ < 0.1 ? 8 : 16]
    for (ti, wi) in zip(t, w)
        σ = (1 + ti) / 2
        yu, yv = _tangents(P, a[1] + σ * du, a[2] + σ * dv)
        s1 += wi / 2 * (yu[1] * du + yv[1] * dv); s2 += wi / 2 * (yu[2] * du + yv[2] * dv); s3 += wi / 2 * (yu[3] * du + yv[3] * dv)
    end
    return (s1, s2, s3)
end
