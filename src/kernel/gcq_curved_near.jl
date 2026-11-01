# Near rows of curved GCQ panels (`CurvedPanel`s with a GCQ rule in u, the edge at u = 0): the product integration of
# tensor_near_row (laplace3d_tensor_near.jl) on the exact map y(u, v) of the panel, in pure Julia.
#
# The quadtree over (u, v) and its leaf rules are tensor_near_row's own: a GCQ direction's root is the cluster of all
# fine segments (_tensor_dir_split peels its last segment off), and _tensor_leaf_map! gives a leaf's points and its map
# to the panel's node basis (the clusters' moment rules, the segments' own nodes, 12-point Gauss-Legendre, Gauss-Jacobi
# for the weight u^pw on a leaf at the edge), here in parameter measure (L = 1) with the Jacobian J = |y_u × y_v| taken
# into the kernel at every leaf point. Sizes and distances are physical: a leaf's sides are its parameter lengths times
# |y_u|, |y_v| at its point p̂ nearest the target's closest point (u*, v*) of the panel, and its distance from the target
# is that of p̂. One departure: a leaf is never the whole weighted edge segment [0, sb] (u^pw, pw ≠ 0) of a direction,
# which _tensor_leaf_map! would integrate with the segment's own rule. That rule's product weights for u^pw are
# interpolatory, exact to degree nz - 1 only, and leave up to 2.3e-7 of max|row| for targets within a few segment widths
# of the edge (lens, across the rim, 6.9e-8 from the panel); the segment is split instead, into a Gauss-Jacobi and a
# Gauss-Legendre half, and the rows agree with η = 6 to rounding.
#
# x - y is not formed by subtracting positions near the target. w = x - y(u*, v*) is measured from the panel's nearest
# node y_i (x - y_i, plus y_i - y(u*, v*) from the tangents), and a point at the offset (ou, ov) from (u*, v*) has
# x - y = w - (y(u* + ou, v* + ov) - y(u*, v*)), the tangents integrated over the parameter step (ou, ov) as given.
# Leaves at least GCQ_CURVED_PLAIN × the panel's diameter from the target take the plain difference x - y(u, v), which
# is cheaper and rounds below 1e-13 of |x - y| there. Dᵀ rows of targets on neighbouring panels of the same smooth
# surface, where n·(x - y) ~ |x - y|², have a floor whatever this cutoff: the mesh data's rounding (~1e-16 × the
# diameter, in anchors and node positions) amplified by the near-tangent n·(x - y), measured at 1.7e-11 (lens, nlev 4)
# and 7.1e-11 (nlev 5) of max|row| against a 128-bit-geometry reference (a cutoff of 1e-2 is no better).
#
# Targets on the panel (`on`, the target's parameters (u0, v0); for example a node of P in the operator's self block)
# take the weakly singular integral in two parts. Around the target, the Duffy region R: its fine u-segment [sa, sb]
# (physical width h) × a v-interval of physical length h centred on it, clipped to [0, 1], integrated by the
# sinh-mapped polar rule (duffy_rule, kernel/duffy.jl). The density is smooth there (the segment's Lagrange basis
# times u^pw, the edge singularity at least a segment away), and a wider region is worse (2h: 4.4e-11, 3h: 9.8e-11 of
# max|row| in spike curved3d/53). The rest of the panel, [0, 1]² minus R, goes through the quadtree above, from root
# boxes that tile it, with the target as its own reference point. Dᵀ needs n·(x - y) ~ κ|x - y|²/2 there, which the
# path integral of x - y gives only to ~1e-16 |x - y| (the innermost rows floor at 1e-12 … 1e-11) and plain
# differences of positions only to ~1e-16 × the diameter: in R and at leaf points within GCQ_CURVED_HESS × the
# diameter of the target it comes from the map's second derivatives instead (_curved_nd_hess). That cutoff is 1e-1:
# at 1e-2, leaf points between 1e-2 and ~1e-1 × the diameter (past GCQ_CURVED_PLAIN) took n·(x - y) from plain
# differences, which left 8.2e-13 of max|row| on the lens's innermost Dᵀ self rows and 4.6e-13 … 6.6e-13 on the split
# sphere's (4.9e-14 and 3.3e-15 at 1e-1, for 11-36 % more time per Dᵀ row). A target in the weighted edge segment
# [0, 2^-K] throws: the density's u^pw is singular in R.

const GCQ_CURVED_PLAIN = 1e-3   # leaves this far from the target (× the panel's diameter): plain differences
const GCQ_CURVED_ON = 1e-13     # a target this close (× the diameter) to the panel's interior lies on it
const GCQ_CURVED_HESS = 1e-1    # on-panel targets: n·(x - y) by _curved_nd_hess this close (× the diameter)

# whether the part (kseg, a, b) of one direction of a quadtree node is the whole weighted edge segment, which
# _tensor_leaf_map! would integrate with the segment's own rule (its condition, for the segment at the edge)
_gcq_whole_edge_seg(r::TensorRule1D, kseg, c, a, b) =
    kseg >= 1 && r.pw != 0 && r.segs[kseg][1] == 0 && a == r.segs[kseg][1] - c && b == r.segs[kseg][2] - c &&
    length(r.z0) >= TENSOR_NEAR_M

_cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
_norm3(a) = sqrt(_dot3(a, a))

# n·(y(c + Δ) - y(c)) for n normal to P at c, from the second derivatives of the map along the parameter segment:
# Taylor's remainder ∫₀¹ (1 - σ) n·y''(σ) dσ of y(σ) = y(c + σΔ), whose first-order term n·y'(0) vanishes. It is
# accurate relative to itself (~κ|Δy|²/2), where differences of positions or of the tangents' path integral are
# accurate only relative to |Δy|. 8-point Gauss-Legendre in σ; y'' from ForwardDiff of P.dmap, or of lmap twice
const _HESS_GL = _gl01(8)
function _curved_nd_hess(P, c::NTuple{2, Float64}, Δ::NTuple{2, Float64}, n)
    σs, ws = _HESS_GL
    s = 0.0
    for q in eachindex(σs)
        s += ws[q] * (1 - σs[q]) * _curved_nd2(P, c, Δ, n, σs[q])
    end
    return s
end
# n·y''(σ) for y(σ) = y(c + σΔ)
function _curved_nd2(P, c, Δ, n, σ)
    if P.dmap === nothing
        ny = t -> _dot3(n, P.lmap(c[1] + t * Δ[1], c[2] + t * Δ[2]))
        return ForwardDiff.derivative(s -> ForwardDiff.derivative(ny, s), σ)
    end
    return ForwardDiff.derivative(σ) do t
        yu, yv = P.dmap(c[1] + t * Δ[1], c[2] + t * Δ[2])
        _dot3(n, yu) * Δ[1] + _dot3(n, yv) * Δ[2]
    end
end

"""
    gcq_curved_row(kind, P, x; n = nothing, on = nothing, η = 1.5, maxdepth = 120) -> Vector{Float64}

Row r over the nodes of the `CurvedPanel` P (P's node order, (i_u, j_v) -> (i_u - 1) n_v + j_v) with
Σ_j r_j σ_j ≈ ∫_P K(x, y) σ(y) dA(y) for σ in P's interpolation space (`tensor_rule_interp`). `kind = :dt` is
K = n·(x - y) / (4π |x - y|³), with `n` the target normal; `kind = :sl` is K = 1 / (4π |x - y|). `x` is the target in
P's anchor frame (x - P.anchor, formed by the caller without cancellation).

A target off the panel: product integration on tensor_near_row's quadtree on the exact map, leaves split while their
distance from the target is at most η × their larger side, up to `maxdepth` levels. A target within 1e-13 × P's
diameter of P's interior throws an `ArgumentError`.

A target on the panel: `on = (u0, v0)`, its parameters, with `x = curved_lpoint(P, u0, v0)` (for P's own node (a, b),
`(P.ru.x[a], P.rv.x[b])` and `P.lpoints[(a - 1) n_v + b]`) and, for `:dt`, `n` P's normal there. The sinh-mapped polar
rule on the Duffy region around the target (its fine u-segment × a v-interval as long physically, clipped to [0, 1])
and the quadtree on the rest of the panel. A target in the weighted edge segment [0, 2^-K] (`ru.pw ≠ 0`) throws an
`ArgumentError`, and so does a v-rule of more than one segment.
"""
function gcq_curved_row(kind::Symbol, P::CurvedPanel, x::NTuple{3, Float64}; n = nothing, on = nothing,
                        η::Real = 1.5, maxdepth::Int = 120)
    kind === :dt || kind === :sl || throw(ArgumentError("gcq_curved_row: kind must be :dt or :sl, got :$kind"))
    kind === :dt && n === nothing && throw(ArgumentError("gcq_curved_row: kind = :dt needs the target normal `n`"))
    nt = kind === :dt ? NTuple{3, Float64}(n) : nothing
    on === nothing || return _gcq_curved_row_on(Val(kind), nt, P, x, NTuple{2, Float64}(on), Float64(η), maxdepth)
    if kind === :dt
        return _gcq_curved_row(d -> (r2 = _dot3(d, d); _dot3(nt, d) / (4π * r2 * sqrt(r2))), P, x, Float64(η), maxdepth)
    end
    return _gcq_curved_row(d -> 1 / (4π * sqrt(_dot3(d, d))), P, x, Float64(η), maxdepth)
end

# the rules' near tables, and the buffers of a row (M zeroed: nbu × nbv)
function _gcq_curved_setup(P::CurvedPanel)
    ru, rv = P.ru, P.rv
    tu_, tv_ = _tensor_near_tables(ru), _tensor_near_tables(rv)
    (isempty(tu_.clu) && length(ru.segs) > 1 || isempty(tv_.clu) && length(rv.segs) > 1) &&
        throw(ArgumentError("gcq_curved_row: the fine segments of a rule must tile [0, 1] from 0"))
    nbu, nbv = size(ru.P, 2), size(rv.P, 2)
    ws = _TensorRowWS(max(length(ru.z0), length(rv.z0)), max(nbu, nbv))
    M = view(ws.M, 1:nbu, 1:nbv); fill!(M, 0.0)
    return tu_, tv_, ws, M
end

function _gcq_curved_row(kern::F, P::CurvedPanel, x, η, maxdepth) where F
    ru, rv = P.ru, P.rv
    tu_, tv_, ws, M = _gcq_curved_setup(P)
    # the reference point: the closest point (u*, v*) of P, and w = x - y(u*, v*) from P's nearest node
    nvn = length(rv.x)
    i = argmin(k -> sum(abs2, x .- P.lpoints[k]), eachindex(P.lpoints))
    us, vs, dist = _curved_closest(P, x)
    diam = _curved_diam(P, 0.0, 1.0, 0.0, 1.0)
    dist <= GCQ_CURVED_ON * diam && 0 < us < 1 && 0 < vs < 1 &&
        throw(ArgumentError("gcq_curved_row: the target lies on the panel; pass its parameters as `on`"))
    w = (x .- P.lpoints[i]) .+ _curved_diff(P, (us, vs), (ru.x[(i - 1) ÷ nvn + 1], rv.x[(i - 1) % nvn + 1]))
    # leaves are carried as offsets from (u*, v*), as tensor_near_row's from the target's projection
    root(r, tb, c) = isempty(tb.clu) ? (1, -c, 1 - c) : (-length(r.segs), -c, 1 - c)
    ku0, a0u, b0u = root(ru, tu_, us); kv0, a0v, b0v = root(rv, tv_, vs)
    push!(ws.stack, (ku0, a0u, b0u, kv0, a0v, b0v, 0))
    _gcq_curved_tree!(M, (d, ou, ov) -> kern(d), P, tu_, tv_, ws, x, (us, vs), w, GCQ_CURVED_PLAIN * diam, η, maxdepth)
    return vec(permutedims(M))
end

# M += the quadtree's integrals from the root boxes on ws.stack: offsets (ou, ov) from the reference point c = (u*, v*),
# w = x - y(c); kern(x - y, ou, ov) (the offsets for the on-panel kernels' second derivatives)
function _gcq_curved_tree!(M, kern::F, P::CurvedPanel, tu_, tv_, ws, x, c, w, plain, η, maxdepth) where F
    ru, rv = P.ru, P.rv
    us, vs = c
    Kb = ws.K; Tb = view(ws.T, 1:size(M, 1), :)
    stack = ws.stack
    while !isempty(stack)
        ku, ua, ub, kv, va, vb, depth = pop!(stack)
        pu = clamp(0.0, ua, ub); pv = clamp(0.0, va, vb)              # p̂
        yu, yv = _tangents(P, us + pu, vs + pv)
        du = (ub - ua) * _norm3(yu); dv = (vb - va) * _norm3(yv)
        d = _norm3(w .- _curved_step(P, (us, vs), (pu, pv)))
        # split while near, and the whole weighted edge segment always (in its direction; see above)
        eu = _gcq_whole_edge_seg(ru, ku, us, ua, ub); ev = _gcq_whole_edge_seg(rv, kv, vs, va, vb)
        near = d <= η * max(du, dv)
        if (near || eu || ev) && depth < maxdepth
            su, sv = eu, ev
            if near
                su = du > dv / 2 && η * du >= d
                sv = dv > du / 2 && η * dv >= d
                !su && !sv && (du >= dv ? (su = true) : (sv = true))
            end
            uch = su ? _tensor_dir_split(ru, ku, us, ua, ub) : ((ku, ua, ub),)
            vch = sv ? _tensor_dir_split(rv, kv, vs, va, vb) : ((kv, va, vb),)
            for (k1, a1, b1) in uch, (k2, a2, b2) in vch
                push!(stack, (k1, a1, b1, k2, a2, b2, depth + 1))
            end
            continue
        end
        ptu, shu, Gu, fu = _tensor_leaf_map!(ws.u, ru, tu_, 1.0, ku, us, ua, ub)
        ptv, shv, Gv, fv = _tensor_leaf_map!(ws.v, rv, tv_, 1.0, kv, vs, va, vb)
        nu, nv = length(ptu), length(ptv)
        K = view(Kb, 1:nu, 1:nv); T1 = view(Tb, :, 1:nv)
        far = d >= plain
        @inbounds for b in 1:nv, a in 1:nu
            ou = ptu[a] - shu; ov = ptv[b] - shv
            u = us + ou; v = vs + ov
            tu, tv = _tangents(P, u, v)
            sep = far ? x .- curved_lpoint(P, u, v) : w .- _curved_step(P, (us, vs), (ou, ov))
            K[a, b] = kern(sep, ou, ov) * _norm3(_cross3(tu, tv))
        end
        mul!(T1, Gu, K)
        mul!(M, T1, Gv', fu * fv, true)
    end
    return M
end

# the kernels of a target on the panel at c with normal n, at x - y = d, y = y(c + (ou, ov)): Dᵀ with n·(x - y) from
# the second derivatives (hess) or from d, and the single layer
_gcq_on_kern(::Val{:dt}, P, c, n, d, ou, ov, hess) =
    (r2 = _dot3(d, d); (hess ? -_curved_nd_hess(P, c, (ou, ov), n) : _dot3(n, d)) / (4π * r2 * sqrt(r2)))
_gcq_on_kern(::Val{:sl}, P, c, n, d, ou, ov, hess) = 1 / (4π * sqrt(_dot3(d, d)))

function _gcq_curved_row_on(K::Val, n, P::CurvedPanel, x, c, η, maxdepth)
    ru, rv = P.ru, P.rv
    u0, v0 = c
    (0 <= u0 <= 1 && 0 <= v0 <= 1) || throw(ArgumentError("gcq_curved_row: `on` must lie in [0, 1]², got $c"))
    (length(rv.segs) == 1 && rv.pw == 0) ||
        throw(ArgumentError("gcq_curved_row: targets on the panel need a one-segment unweighted rule in v"))
    # the target's fine u-segment k (taken from `on`, never from x: a target at u0 = 0 is in the edge segment)
    k = clamp(searchsortedlast(ru.segs, (u0, Inf)), 1, length(ru.segs))
    sa, sb = ru.segs[k]
    sa == 0 && ru.pw != 0 &&
        throw(ArgumentError("gcq_curved_row: target within the edge segment [0, 2^-K] of a GCQ panel"))
    diam = _curved_diam(P, 0.0, 1.0, 0.0, 1.0)
    _norm3(x .- curved_lpoint(P, u0, v0)) <= GCQ_CURVED_ON * diam ||
        throw(ArgumentError("gcq_curved_row: x is not the point of the panel at `on`"))
    tu_, tv_, ws, M = _gcq_curved_setup(P)
    # the Duffy region R = [sa, sb] × [va, vb], the first fundamental form G at the target
    yu, yv = _tangents(P, u0, v0)
    G = (_dot3(yu, yu), _dot3(yu, yv), _dot3(yv, yv))
    Δv = (sb - sa) * sqrt(G[1] / G[3])
    va = max(0.0, v0 - Δv / 2); vb = min(1.0, v0 + Δv / 2)
    _gcq_duffy!(M, (d, ou, ov) -> _gcq_on_kern(K, P, c, n, d, ou, ov, true), P, tu_, tv_, k, c, (sa, sb, va, vb), G)
    # the rest, from root boxes tiling [0, 1]² \ R (offsets from the target): the cluster of segments 1..k-1, segment k
    # below and above R in v, and the segments k+1.. one by one
    stack = ws.stack
    k > 1 && push!(stack, (k - 1 == 1 ? 1 : -(k - 1), -u0, sa - u0, 1, -v0, 1 - v0, 0))
    va > 0 && push!(stack, (k, sa - u0, sb - u0, 1, -v0, va - v0, 0))
    vb < 1 && push!(stack, (k, sa - u0, sb - u0, 1, vb - v0, 1 - v0, 0))
    for kk in k+1:length(ru.segs)
        push!(stack, (kk, ru.segs[kk][1] - u0, ru.segs[kk][2] - u0, 1, -v0, 1 - v0, 0))
    end
    hc2 = (GCQ_CURVED_HESS * diam)^2
    _gcq_curved_tree!(M, (d, ou, ov) -> _gcq_on_kern(K, P, c, n, d, ou, ov, _dot3(d, d) < hc2), P, tu_, tv_, ws, x, c,
                      (0.0, 0.0, 0.0), GCQ_CURVED_PLAIN * diam, η, maxdepth)
    return vec(permutedims(M))
end

# M += ∫_R K φ_ab dA on the Duffy region R = (sa, sb, va, vb) of fine u-segment k around the target c, by duffy_rule
# in the first fundamental form G at c: φ_ab is segment k's Lagrange basis through P (times u^pw) in u, as
# tensor_rule_interp, times v's Lagrange basis; x - y from the tangents (_curved_step), kern(x - y, ou, ov)
function _gcq_duffy!(M, kern::F, P::CurvedPanel, tu_, tv_, k, c, R, G) where F
    ru, rv = P.ru, P.rv
    du, dv, w = duffy_rule(c, R, G)
    sa, sb = ru.segs[k]
    tq = similar(w); τu = similar(w); τv = similar(w)
    @inbounds for q in eachindex(w)
        ou, ov = du[q], dv[q]
        u = c[1] + ou; v = c[2] + ov
        tu, tv = _tangents(P, u, v)
        sep = .-_curved_step(P, c, (ou, ov))
        tq[q] = w[q] * kern(sep, ou, ov) * _norm3(_cross3(tu, tv)) * (ru.pw == 0 ? 1.0 : u^ru.pw)
        τu[q] = 2 * (u - sa) / (sb - sa) - 1; τv[q] = 2 * v - 1
    end
    Gu = tu_.pk[k] * (_tensor_lagrange_mat(ru.z0, ru.bw, τu) .* tq')          # nbu × points, the weights included
    Ev = tv_.pk[1] * _tensor_lagrange_mat(rv.z0, rv.bw, τv)                   # nbv × points
    mul!(M, Gu, Ev', 1.0, 1.0)
    return M
end
