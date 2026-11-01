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

const GCQ_CURVED_PLAIN = 1e-3   # leaves this far from the target (× the panel's diameter): plain differences
const GCQ_CURVED_ON = 1e-13     # a target this close (× the diameter) to the panel's interior lies on it

# whether the part (kseg, a, b) of one direction of a quadtree node is the whole weighted edge segment, which
# _tensor_leaf_map! would integrate with the segment's own rule (its condition, for the segment at the edge)
_gcq_whole_edge_seg(r::TensorRule1D, kseg, c, a, b) =
    kseg >= 1 && r.pw != 0 && r.segs[kseg][1] == 0 && a == r.segs[kseg][1] - c && b == r.segs[kseg][2] - c &&
    length(r.z0) >= TENSOR_NEAR_M

_cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
_norm3(a) = sqrt(_dot3(a, a))

"""
    gcq_curved_row(kind, P, x; n = nothing, on = nothing, η = 1.5, maxdepth = 120) -> Vector{Float64}

Row r over the nodes of the `CurvedPanel` P (P's node order, (i_u, j_v) -> (i_u - 1) n_v + j_v) with
Σ_j r_j σ_j ≈ ∫_P K(x, y) σ(y) dA(y) for σ in P's interpolation space (`tensor_rule_interp`). `kind = :dt` is
K = n·(x - y) / (4π |x - y|³), with `n` the target normal; `kind = :sl` is K = 1 / (4π |x - y|). `x` is the target in
P's anchor frame (x - P.anchor, formed by the caller without cancellation), off the panel: a target within
1e-13 × P's diameter of P's interior throws an `ArgumentError`.

Product integration on tensor_near_row's quadtree on the exact map: leaves are split while their distance from the
target is at most η × their larger side, up to `maxdepth` levels.
"""
function gcq_curved_row(kind::Symbol, P::CurvedPanel, x::NTuple{3, Float64}; n = nothing, on = nothing,
                        η::Real = 1.5, maxdepth::Int = 120)
    on === nothing || throw(ArgumentError("gcq_curved_row: targets on the panel (`on`) are not supported yet"))
    if kind === :dt
        n === nothing && throw(ArgumentError("gcq_curved_row: kind = :dt needs the target normal `n`"))
        nt = NTuple{3, Float64}(n)
        return _gcq_curved_row(d -> (r2 = _dot3(d, d); _dot3(nt, d) / (4π * r2 * sqrt(r2))), P, x, Float64(η), maxdepth)
    elseif kind === :sl
        return _gcq_curved_row(d -> 1 / (4π * sqrt(_dot3(d, d))), P, x, Float64(η), maxdepth)
    end
    throw(ArgumentError("gcq_curved_row: kind must be :dt or :sl, got :$kind"))
end

function _gcq_curved_row(kern::F, P::CurvedPanel, x, η, maxdepth) where F
    ru, rv = P.ru, P.rv
    tu_, tv_ = _tensor_near_tables(ru), _tensor_near_tables(rv)
    (isempty(tu_.clu) && length(ru.segs) > 1 || isempty(tv_.clu) && length(rv.segs) > 1) &&
        throw(ArgumentError("gcq_curved_row: the fine segments of a rule must tile [0, 1] from 0"))
    # the reference point: the closest point (u*, v*) of P, and w = x - y(u*, v*) from P's nearest node
    nvn = length(rv.x)
    i = argmin(k -> sum(abs2, x .- P.lpoints[k]), eachindex(P.lpoints))
    us, vs, dist = _curved_closest(P, x)
    diam = _curved_diam(P, 0.0, 1.0, 0.0, 1.0)
    dist <= GCQ_CURVED_ON * diam && 0 < us < 1 && 0 < vs < 1 &&
        throw(ArgumentError("gcq_curved_row: the target lies on the panel; pass its parameters as `on`"))
    w = (x .- P.lpoints[i]) .+ _curved_diff(P, (us, vs), (ru.x[(i - 1) ÷ nvn + 1], rv.x[(i - 1) % nvn + 1]))
    plain = GCQ_CURVED_PLAIN * diam
    nbu, nbv = size(ru.P, 2), size(rv.P, 2)
    ws = _TensorRowWS(max(length(ru.z0), length(rv.z0)), max(nbu, nbv))
    M = view(ws.M, 1:nbu, 1:nbv); fill!(M, 0.0)
    Kb = ws.K; Tb = view(ws.T, 1:nbu, :)
    # leaves are carried as offsets from (u*, v*), as tensor_near_row's from the target's projection
    root(r, tb, c) = isempty(tb.clu) ? (1, -c, 1 - c) : (-length(r.segs), -c, 1 - c)
    ku0, a0u, b0u = root(ru, tu_, us); kv0, a0v, b0v = root(rv, tv_, vs)
    stack = ws.stack; push!(stack, (ku0, a0u, b0u, kv0, a0v, b0v, 0))
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
            K[a, b] = kern(sep) * _norm3(_cross3(tu, tv))
        end
        mul!(T1, Gu, K)
        mul!(M, T1, Gv', fu * fv, true)
    end
    return vec(permutedims(M))
end
