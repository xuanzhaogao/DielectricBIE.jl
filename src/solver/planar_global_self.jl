# Experimental: self rows of the single layer at a point of a flat GCQ panel with the density from the global GCQ
# interpolant (global_interp!, GCQGlobalInterp) throughout the row (plan docs/plans/2026-11-05-global-gcq-local-duffy.md).
#
# The piecewise basis of planar_near.jl (Lagrange in each fine segment through P) is one polynomial only inside a fine
# cell, so its Duffy region stays in the node's cell, and the rest of the cell (strips) and the panel outside it are
# refined by the walk down to the node's distance from the cell's sides. The global interpolant is one formula on the
# whole of (0, 1] in a GCQ direction, so the Duffy region R may cross fine boundaries:
#     row = ∫_R (duffy_rule, the global density) + ∫_{Ω \ R} (the walk of planar_near.jl, global leaf maps),
# Ω \ R walked from the whole panel with R left out (planar_near.jl's _flat_complement_boxes).
# - R (_global_self_region): the old region R0 (_flat_self_regions) and the square Q of half-side λ ℓ around the node
#   in the panel's metric at the node (ℓ the cell's shorter physical side), their bounding rectangle, kept in the 3 × 3
#   fine cells around the node's, off a GCQ edge (u ≥ u0 / 4: the density's singular end stays outside R) and in the
#   panel. λ = 0 is R0; a node deep inside R0 keeps R0; one beside a cell side gets R across it.
# - Duffy (_global_duffy!): duffy_rule's points and weights (target-relative offsets, _flat_step distances), the
#   density by the global formula, contracted modally in a GCQ direction: Σ_q W_q φ_i(u_q) ψ_j(v_q) =
#   diag(x^-pw) Hn⁻ᵀ Vᵀ [Σ_q f(u_q)ᵀ u_q^pw W_q ψ_j(v_q)], in blocks of points.
# - The complement (_flat_complement_boxes, from the whole panel with R left out; study 75): walk boxes (a cluster
#   [0, s_k] or a part of one fine segment per direction), walked by _flat_walk! with _GlobalLeaf maps in the GCQ
#   directions: a cluster [0, s_k] (or a part [0, b] of the weighted segment) by the rule's GCQ nodes scaled to it,
#   a whole segment by its 14 Gauss-Legendre points, a part of one by 12-point Gauss-Legendre, all with the density
#   from the global formula (full density values, plain weights: the weight s^pw is in the density). A GL direction's
#   interpolant is global already (one segment), and its leaf maps are planar_near.jl's.
# The moment rules of the endpoint clusters integrate the piecewise basis and are not used: endpoint_rule = :gcq.
# backend = :piecewise runs the same partition with planar_near.jl's density (R = R0 only), the adapter's check.

# one GCQ direction's leaf maps under the global backend (dispatch of _flat_leaf_map! on ex); buffers per row
struct _GlobalLeaf
    r::TensorRule1D
    ws::GlobalInterpWorkspace
    clu::Vector{Tuple{Vector{Float64}, Matrix{Float64}}}      # cluster k: points s_k x, map (n × n)
    seg::Vector{Tuple{Vector{Float64}, Matrix{Float64}}}      # whole segment k: offsets from its start, map (n × nz)
    pts::Vector{Float64}
    off::Vector{Float64}
    E::Matrix{Float64}
    G::Matrix{Float64}
    nmaps::Base.RefValue{Int}
end

# (points, map) of the rule r on [a, b] at the points a + off (offsets), weights w (full density): G = (E .* w)ᵀ
function _global_map(r::TensorRule1D, ws, pts, w)
    E = global_interp!(zeros(length(pts), length(r.x)), r, pts, ws)
    return Matrix(transpose(E .* w))
end

# the cluster and whole-segment maps of a rule, built once per rule
const _GLOBAL_LEAF_MAPS = IdDict{TensorRule1D, Tuple{Vector{Tuple{Vector{Float64}, Matrix{Float64}}},
                                                       Vector{Tuple{Vector{Float64}, Matrix{Float64}}}}}()
const _GLOBAL_LEAF_MAPS_LOCK = ReentrantLock()
_global_leaf_maps(r::TensorRule1D) = lock(_GLOBAL_LEAF_MAPS_LOCK) do
    get!(_GLOBAL_LEAF_MAPS, r) do
        ws = GlobalInterpWorkspace(r)
        clu = map(r.segs) do (_, sb)
            xs = sb .* r.x
            all(>=(floatmin(Float64)), xs) || throw(ArgumentError("global leaf maps: the cluster [0, $sb] puts nodes " *
                                                                   "below the normal Float64 range"))
            (xs, _global_map(r, ws, xs, sb .* r.w))
        end
        seg = map(r.segs) do (sa, sb)
            o = (sb - sa) / 2 .* (1 .+ r.z0)
            (o, sa == 0 && r.pw < 0 ? zeros(length(r.x), 0) : _global_map(r, ws, sa .+ o, (sb - sa) / 2 .* r.wz))
        end
        return clu, seg
    end
end

function _GlobalLeaf(r::TensorRule1D)
    clu, seg = _global_leaf_maps(r)
    m = max(length(r.x), length(r.z0), TENSOR_NEAR_M)
    return _GlobalLeaf(r, _global_ws(r, m, :leaf), clu, seg, zeros(m), zeros(m), zeros(m, length(r.x)),
                       zeros(length(r.x), m), Ref(0))
end

# a GlobalInterpWorkspace of the rule r with nmax points for the use `tag`, reused within the current task (task-local
# storage: never shared between threads). Uses whose buffers are live at the same time need distinct tags (the two
# directions of a Duffy contraction, which may share one rule object on a GCQ x GCQ panel)
_global_ws(r::TensorRule1D, nmax::Int, tag::Symbol) =
    get!(() -> GlobalInterpWorkspace(r, nmax), task_local_storage(), (:_global_ws, objectid(r.P), nmax, tag))::GlobalInterpWorkspace

# one direction of a leaf under the global backend: (lower end c, points as offsets from c, map n × points)
function _flat_leaf_map!(buf, r::TensorRule1D, tb::_TensorNearTables, ex::_GlobalLeaf, k, a, b)
    ex.nmaps[] += 1
    k < 0 && return 0.0, ex.clu[-k]...
    sa, sb = r.segs[k]
    if a == 0                                                 # at the edge (the weighted segment): scaled GCQ on [0, b]
        b == sb && return 0.0, ex.clu[k]...
        n = length(r.x)
        pts = view(ex.pts, 1:n); pts .= b .* r.x
        all(>=(floatmin(Float64)), pts) ||
            throw(ArgumentError("global leaf map: [0, $b] puts GCQ nodes below the normal Float64 range"))
        E = view(ex.E, 1:n, :); G = view(ex.G, :, 1:n)
        global_interp!(E, r, pts, ex.ws)
        @inbounds for l in 1:n, j in axes(G, 1)
            G[j, l] = E[l, j] * b * r.w[l]
        end
        return 0.0, pts, G
    end
    a == sa && b == sb && length(r.z0) >= TENSOR_NEAR_M && return sa, ex.seg[k]...
    zm, wm = TENSOR_NEAR_GL; M = TENSOR_NEAR_M
    off = view(ex.off, 1:M); pts = view(ex.pts, 1:M)
    @inbounds for q in 1:M
        off[q] = (b - a) * (1 + zm[q]) / 2; pts[q] = a + off[q]
    end
    E = view(ex.E, 1:M, :); G = view(ex.G, :, 1:M)
    global_interp!(E, r, pts, ex.ws)
    @inbounds for q in 1:M, j in axes(G, 1)
        G[j, q] = E[q, j] * (b - a) / 2 * wm[q]
    end
    return a, off, G
end

"""
    _global_self_region(g, ru, rv, k, l, u0, v0, λ) -> (R, R0, G)

The Duffy rectangle R = (ua, ub, va, vb) of the global backend at (u0, v0) in the fine cell (k, l): the bounding
rectangle of R0 (`_flat_self_regions`) and Q = [u0 ∓ λℓ/√G11] × [v0 ∓ λℓ/√G22], ℓ the cell's shorter side in the
panel's metric G at the node, kept in the 3 × 3 fine cells around (k, l), in the panel, and in a GCQ direction at
u ≥ u0/4 (v ≥ v0/4): the density's endpoint s = 0 is not smooth. R0 is never shrunk; λ = 0 gives R0.
"""
function _global_self_region(g::_FlatGeom, ru::TensorRule1D, rv::TensorRule1D, k, l, u0, v0, λ)
    λ >= 0 || throw(ArgumentError("_global_self_region: λ must be ≥ 0, got $λ"))
    R0, _, G = _flat_self_regions(g, ru, rv, k, l, u0, v0)
    λ == 0 && return R0, R0, G
    lu, lv = sqrt(G[1]), sqrt(G[3])
    (sa, sb), (ta, tb) = ru.segs[k], rv.segs[l]
    ell = min((sb - sa) * lu, (tb - ta) * lv)
    hu, hv = λ * ell / lu, λ * ell / lv
    function side(r, kk, c, h, a0, b0)
        lo = r.segs[max(kk - 1, 1)][1]; hi = r.segs[min(kk + 1, length(r.segs))][2]
        is_gcq(r) && (lo = max(lo, c / 4))
        return max(min(a0, c - h), lo, 0.0), min(max(b0, c + h), hi, 1.0)
    end
    ua, ub = side(ru, k, u0, hu, R0[1], R0[2])
    va, vb = side(rv, l, v0, hv, R0[3], R0[4])
    R = (min(ua, R0[1]), max(ub, R0[2]), min(va, R0[3]), max(vb, R0[4]))
    return R, R0, G
end

# the complement boxes: planar_near.jl's _flat_complement_boxes

# points along a ray of the Duffy rule: as planar_near.jl's (degree nz_u + nz_v - 2 plus 10)
_global_duffy_nrho(P) = (nρ = cld(length(P.ru.z0) + length(P.rv.z0) + 9, 2); max(DUFFY_NRHO, nρ + isodd(nρ)))

"Staged counters of planar_global_self_row (seconds, points, leaves); `stats = _GlobalSelfStats()` to collect them."
Base.@kwdef mutable struct _GlobalSelfStats
    t_region::Float64 = 0.0
    t_walk::Float64 = 0.0
    t_quad::Float64 = 0.0
    t_interp::Float64 = 0.0
    t_contract::Float64 = 0.0
    n_duffy::Int = 0
    n_leafmaps::Int = 0
end

"""
    _global_duffy!(row, P, g, wsu, wsv, u0, v0, D, G; nρ, nside, stats = nothing, nblock = 512)

row (nu × nv) += the single layer at (u0, v0) on the rectangle D = (ua, ub, va, vb) of P against the global basis
(φ_i(u) ψ_j(v), full densities): duffy_rule in the metric G, the density by global_interp!, the contraction modal in
the first GCQ direction. `wsu`, `wsv` are GlobalInterpWorkspaces (nothing for a GL direction). D must not touch a GCQ
edge (the density's singular end): an ArgumentError.
"""
function _global_duffy!(row::AbstractMatrix, P::CurvedPanel, g::_FlatGeom, wsu, wsv, u0, v0, D, G;
                        nρ::Int = _global_duffy_nrho(P), nside::Int = DUFFY_NSIDE, stats = nothing, nblock::Int = 512)
    ru, rv = P.ru, P.rv
    is_gcq(ru) || is_gcq(rv) || throw(ArgumentError("_global_duffy!: the panel has no GCQ direction"))
    for (r, ws) in ((ru, wsu), (rv, wsv))
        is_gcq(r) && !(ws isa GlobalInterpWorkspace && ws.P === r.P) &&
            throw(ArgumentError("_global_duffy!: a GCQ direction needs a GlobalInterpWorkspace of its own rule"))
    end
    is_gcq(ru) && is_gcq(rv) && wsu === wsv &&
        throw(ArgumentError("_global_duffy!: the two directions need distinct workspaces (both are live at once)"))
    (is_gcq(ru) && D[1] <= 0 || is_gcq(rv) && D[3] <= 0) &&
        throw(ArgumentError("_global_duffy!: the Duffy rectangle $D touches a GCQ edge (the density's singular end)"))
    t0 = time_ns()
    du, dv, w = duffy_rule((u0, v0), D, G; nρ, nside)
    t1 = time_ns()
    # the modal direction m (a GCQ one) and the other o
    mu = is_gcq(ru)
    rm, ro = mu ? (ru, rv) : (rv, ru); wsm, wso = mu ? (wsu, wsv) : (wsv, wsu)
    gm = wsm.g; nm, no = length(rm.x), length(ro.x)
    Y = zeros(length(gm.q), no)
    tm = zeros(nblock); to = zeros(nblock); wb = zeros(nblock)
    Eo = zeros(nblock, no); Wb = zeros(nblock, no)
    FT = wsm.FT; size(FT, 2) >= nblock || (FT = zeros(length(gm.q), nblock))
    ti = 0; tc = 0
    for c0 in 1:nblock:length(w)
        idx = c0:min(c0 + nblock - 1, length(w)); nb = length(idx)
        @inbounds for (i, q) in enumerate(idx)
            u = u0 + du[q]; v = v0 + dv[q]
            J = _norm3(_cross3(g.a .+ v .* g.c, g.b .+ u .* g.c))
            wb[i] = w[q] * J / (4π * _norm3(_flat_step(g, u0, v0, du[q], dv[q])))
            tm[i], to[i] = mu ? (u, v) : (v, u)
        end
        s0 = time_ns()
        tmb = view(tm, 1:nb)
        for t in tmb
            _global_interp_check(gm, t)
        end
        F = view(FT, :, 1:nb)
        _gcq_powers_t!(F, gm, tmb)
        Ev = view(Eo, 1:nb, :)
        global_interp!(Ev, ro, view(to, 1:nb), wso)
        s1 = time_ns()
        @inbounds for i in 1:nb
            wb[i] *= gm.pw == 0 ? 1.0 : tm[i]^gm.pw
        end
        @inbounds for j in 1:no, i in 1:nb
            Wb[i, j] = wb[i] * Ev[i, j]
        end
        mul!(Y, F, view(Wb, 1:nb, :), 1.0, 1.0)              # Σ_q f(t_q)ᵀ t_q^pw W_q ψ(o_q)
        s2 = time_ns()
        ti += s1 - s0; tc += s2 - s1
    end
    s0 = time_ns()
    Z = transpose(gm.V) * Y                                   # nm × no
    ldiv!(transpose(gm.lu), Z)
    Z .*= gm.xpw
    mu ? (row .+= Z) : (row .+= transpose(Z))
    tc += time_ns() - s0
    if stats !== nothing
        stats.t_quad += (t1 - t0) / 1e9; stats.t_interp += ti / 1e9; stats.t_contract += tc / 1e9
        stats.n_duffy += length(w)
    end
    return row
end

# E[a, :] = the piecewise basis of rule r at ts[a] (full density): Lagrange in the fine segment holding t (the upper one at
# a boundary) through P (the near tables' pk), times t^pw; ℓ a buffer of length nz (one small GEMV per point measured
# faster than gathering the points of each segment into a GEMM)
function _piecewise_basis!(E, r::TensorRule1D, tb::_TensorNearTables, ts, ℓ)
    ns = length(r.segs)
    @inbounds for (a, t) in enumerate(ts)
        k = clamp(searchsortedlast(r.segs, (t, Inf)), 1, ns)
        sa, sb = r.segs[k]
        τ = 2 * (t - sa) / (sb - sa) - 1
        j = findfirst(==(τ), r.z0)
        if j === nothing
            sm = 0.0
            for i in eachindex(ℓ)
                ℓ[i] = r.bw[i] / (τ - r.z0[i]); sm += ℓ[i]
            end
            ℓ ./= sm
        else
            fill!(ℓ, 0.0); ℓ[j] = 1.0
        end
        mul!(view(E, a, :), tb.pk[k], ℓ)
        r.pw == 0 || (view(E, a, :) .*= t^r.pw)
    end
    return E
end

"""
    _piecewise_duffy!(row, P, g, tu, tv, u0, v0, D, G; nρ, nside)

row (nu × nv) += the single layer at (u0, v0) on the rectangle D of P against the piecewise basis (planar_near.jl's
interpolant), D anywhere in the panel off a GCQ edge: each Duffy point takes the polynomial of the fine cell holding
it. Across fine boundaries this integrates the interpolant's jumps as they are: accurate only for rules whose cells
resolve the global interpolant, cardinal functions included. `edge_gcq_rule(...; ninterp = 28)` does so for a one-power
family (jumps ~1e-15); a 40-power family needs ~36. The default 14 points jump by up to ~5e-2 on cardinal functions
(study 62).
"""
function _piecewise_duffy!(row::AbstractMatrix, P::CurvedPanel, g::_FlatGeom, tu::_TensorNearTables, tv::_TensorNearTables,
                           u0, v0, D, G; nρ::Int = _global_duffy_nrho(P), nside::Int = DUFFY_NSIDE)
    ru, rv = P.ru, P.rv
    (is_gcq(ru) && D[1] <= 0 || is_gcq(rv) && D[3] <= 0) &&
        throw(ArgumentError("_piecewise_duffy!: the Duffy rectangle $D touches a GCQ edge (the density's singular end)"))
    du, dv, w = duffy_rule((u0, v0), D, G; nρ, nside)
    N = length(w)
    Eu = zeros(N, length(ru.x)); Ev = zeros(N, length(rv.x)); wt = similar(w)
    @inbounds for q in 1:N
        u = u0 + du[q]; v = v0 + dv[q]
        wt[q] = w[q] * _norm3(_cross3(g.a .+ v .* g.c, g.b .+ u .* g.c)) / (4π * _norm3(_flat_step(g, u0, v0, du[q], dv[q])))
    end
    _piecewise_basis!(Eu, ru, tu, u0 .+ du, zeros(length(ru.z0)))
    _piecewise_basis!(Ev, rv, tv, v0 .+ dv, zeros(length(rv.z0)))
    mul!(row, transpose(Eu), wt .* Ev, 1.0, 1.0)
    return row
end

"""
    planar_global_self_row(P, u0, v0; λ = 0.5, q = PlanarProductNearQuad(endpoint_rule = :gcq), nρ, nside,
                           backend = :global, stats = nothing) -> row

Experimental: the single-layer row ∫_P φ_j(y) / (4π |y(u0, v0) - y|) dA(y) of the flat GCQ panel P at its point
(u0, v0) (a node, or any point of a fine cell above the weighted segment at a GCQ edge), ordering (i_u - 1) n_v + j_v,
the density from the global GCQ interpolant (`backend = :global`, `q.endpoint_rule` must be `:gcq`), the Duffy region
enlarged by λ (`_global_self_region`) and the rest of the panel walked as planar_near.jl's boxes with η_leaf and
max_depth of `q`. `backend = :piecewise` is the same partition with planar_near.jl's piecewise density (at λ = 0 its
Duffy rule on the node's cell; for λ > 0 `_piecewise_duffy!` across fine cells, meaningful only for rules whose cells
resolve the global interpolant, see there).
`stats` (a `_GlobalSelfStats`) collects staged times and counts. See the header of planar_global_self.jl.
"""
function planar_global_self_row(P::CurvedPanel, u0::Real, v0::Real; λ::Real = 0.5,
                                q::PlanarProductNearQuad = PlanarProductNearQuad(endpoint_rule = :gcq),
                                nρ::Int = _global_duffy_nrho(P), nside::Int = DUFFY_NSIDE, backend::Symbol = :global,
                                stats = nothing)
    backend in (:global, :piecewise) ||
        throw(ArgumentError("planar_global_self_row: backend must be :global or :piecewise, got :$backend"))
    backend == :global && q.endpoint_rule != :gcq &&
        throw(ArgumentError("planar_global_self_row: the global backend needs endpoint_rule = :gcq (the moment rules " *
                            "integrate the piecewise basis)"))
    inside(r, t) = is_gcq(r) ? 0 < t <= 1 : 0 <= t <= 1
    inside(P.ru, u0) && inside(P.rv, v0) ||
        throw(ArgumentError("planar_global_self_row: the point ($u0, $v0) is not in the panel (or on a GCQ edge)"))
    ru, rv = P.ru, P.rv
    is_gcq(ru) || is_gcq(rv) ||
        throw(ArgumentError("planar_global_self_row: the panel has no GCQ direction (use the production self row)"))
    k = _pn_cell(ru, u0, 0, "u"); l = _pn_cell(rv, v0, 0, "v")
    for (r, c, d) in ((ru, k, "u"), (rv, l, "v"))
        is_gcq(r) && r.segs[c][1] == 0 &&
            throw(ArgumentError("planar_global_self_row: the point lies in the weighted segment [0, $(r.segs[c][2])] " *
                                "of its GCQ direction $d: no self row"))
    end
    g = _flat_geom(P)
    t0 = time_ns()
    R, R0, G = _global_self_region(g, ru, rv, k, l, u0, v0, λ)
    t1 = time_ns()
    tu, tv = _tensor_near_tables(ru), _tensor_near_tables(rv)
    nu, nv = length(ru.x), length(rv.x)
    T = _FlatTargets([Float64(u0)], [Float64(v0)], zeros(3, 1), nothing, [(0, 0)])
    buf(r) = (zeros(TENSOR_NEAR_M), zeros(TENSOR_NEAR_M), zeros(length(r.z0), TENSOR_NEAR_M), zeros(size(r.P, 2), TENSOR_NEAR_M))
    ws = (buf(ru), buf(rv))
    mom = _flat_cluster_rules(P, tu, tv, q.endpoint_rule)
    ex = backend == :global ? (is_gcq(ru) ? _GlobalLeaf(ru) : mom[1], is_gcq(rv) ? _GlobalLeaf(rv) : mom[2]) : mom
    Racc = zeros(nu, 1, nv)
    for box in _flat_complement_boxes(ru, rv, g, R)
        _flat_walk!(Racc, P, g, tu, tv, ex, ws, T, (box..., 0), [1], false, q)
    end
    t2 = time_ns()
    row = dropdims(Racc; dims = 2)
    if backend == :global
        _global_duffy!(row, P, g, is_gcq(ru) ? _global_ws(ru, 512, :u) : nothing, is_gcq(rv) ? _global_ws(rv, 512, :v) : nothing,
                       u0, v0, R, G; nρ, nside, stats)
    elseif λ == 0
        _flat_self_duffy!(Racc, P, g, tu, tv, 1, u0, v0, k, l, R, G)
        row = dropdims(Racc; dims = 2)
    else
        _piecewise_duffy!(row, P, g, tu, tv, u0, v0, R, G; nρ, nside)
    end
    if stats !== nothing
        stats.t_region += (t1 - t0) / 1e9; stats.t_walk += (t2 - t1) / 1e9
        stats.n_leafmaps += sum((e isa _GlobalLeaf ? e.nmaps[] : 0) for e in ex)
    end
    return vec(transpose(row))
end
