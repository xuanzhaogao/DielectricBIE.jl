# Geometry of curved GCQ panels in pure Julia: differences y(b) - y(a) of panel points that stay accurate however close
# the points are (the tangents integrated along the parameter segment, or a Chebyshev series along a u-line), the point
# of a panel closest to a target, and the scales of a row. Used by the near and self quadrature of `CurvedPanel`s.

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

# along the line u of panel P, as Chebyshev series in t = 2v - 1: the antiderivative (in v) of the tangent y_v
# (columns 1-3) and the Jacobian J (column 4); an analytic map over a patch: 32 terms reach the rounding level
const _CHEB_N = 32
function _curved_vgeom(P, u)
    N = _CHEB_N
    t = [cos(π * (j + 0.5) / N) for j in 0:N-1]
    F = zeros(N, 4)
    for (j, tj) in enumerate(t)
        _, yv, _, J = curved_frame(P, u, (1 + tj) / 2)
        F[j, 1:3] .= yv; F[j, 4] = J
    end
    T = [cos(k * π * (j + 0.5) / N) for k in 0:N-1, j in 0:N-1]
    c = (2 / N) .* (T * F); c[1, :] ./= 2                     # f = Σ_k c[k+1] T_k
    G = zeros(N + 1, 4)
    G[1:N, 4] .= c[:, 4]
    # ∫ f dv = ½ ∫ f dt: coefficients (c_{k-1} - c_{k+1}) / (2k) (2c_0 for k = 1), halved
    for col in 1:3, k in 1:N
        cm = k == 1 ? 2 * c[1, col] : c[k, col]; cp = k + 1 <= N - 1 ? c[k + 2, col] : 0.0
        G[k + 1, col] = (cm - cp) / (4k)
    end
    return G
end

# F(t0) - F(t) of the antiderivative series (columns 1-3 of G) as (t0 - t) Σ_k G_k (T_k(t0) - T_k(t)) / (t0 - t),
# the divided differences by Δ_{k+1} = 2 t0 Δ_k + 2 T_k(t) - Δ_{k-1}: no cancellation for t near t0; and J(t)
function _cheb_diff(G, t0, t)
    h = t0 - t
    Δm, Δ = 0.0, 1.0                                          # Δ_0, Δ_1
    Tm, T = 1.0, t                                            # T_0(t), T_1(t)
    s1 = G[2, 1]; s2 = G[2, 2]; s3 = G[2, 3]
    J = G[1, 4] + G[2, 4] * t
    @inbounds for k in 2:size(G, 1)-1
        Δm, Δ = Δ, 2t0 * Δ + 2T - Δm                          # Δ_k
        Tm, T = T, 2t * T - Tm                                # T_k(t)
        s1 += G[k + 1, 1] * Δ; s2 += G[k + 1, 2] * Δ; s3 += G[k + 1, 3] * Δ
        J += G[k + 1, 4] * T
    end
    return (h * s1, h * s2, h * s3, J)
end

# the first m segments of a GCQ rule [0, s] as one rule: Gauss-Legendre nodes on [0, s] and the moments
# C[q, :] = Σ_fine fw ℓ_q(fx) P[fine, :] of the Lagrange basis at them (∫ σ g ≈ Σ_q g(u_q) C[q, :]·σ for g smooth on
# [0, s], i.e. from a target at least s away)
const _CURVED_GROUP_NODES = 20
function _curved_group_rule(ru, m)
    nz = length(ru.z0); s = ru.segs[m][2]
    z, w = gausslegendre(_CURVED_GROUP_NODES)
    L = _tensor_lagrange_mat(z, gl_barycentric_weights(z, w), 2 .* vec(ru.fx[:, 1:m]) ./ s .- 1)
    C = L * (vec(ru.fw[:, 1:m]) .* ru.P[1:(m * nz), :])
    return s .* (1 .+ z) ./ 2, C
end

# Σ_j |node-rule entry| of the row: the row's scale
function _curved_node_scale(P, x, n)
    s = 0.0
    for (y, w) in zip(P.lpoints, P.weights)
        d = x .- y; r2 = _dot3(d, d)
        r2 == 0 && continue
        s += abs(w * _dot3(n, d) / (4π * r2 * sqrt(r2)))
    end
    return s
end
