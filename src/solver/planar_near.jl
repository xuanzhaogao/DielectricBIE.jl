# Near corrections of D^T for flat CurvedPanels (planar polyhedra), GCQ in u, v or both, by product integration.
#
# On a flat panel D^T vanishes for every target in the panel's plane (its own nodes and those of coplanar panels),
# so there is no on-surface singular quadrature: those targets are skipped. A target off the plane that lies within
# η × the panel's diameter of one of its nodes gets the exact row
#     ∫_panel n_x·(x - y) / (4π |x - y|³) σ(y) dA(y),   σ in the panel's tensor interpolation space,
# as in GCQNearQuad for TensorPanels: every (fine u-segment, fine v-segment) pair is a leaf, split (along its
# longer side, or both when they are comparable) until the target is farther than η_leaf leaf diameters from it.
# A whole segment pair uses the segments' own nodes and weights (the weight s^pw and the Gauss-Jacobi end segment
# included), a sub-leaf 12-point Gauss-Legendre (Gauss-Jacobi for s^pw on a leaf touching s = 0) with the
# segment's Lagrange basis. The stored correction is that row minus the node rule (dt_node_rule_block).
#
# Cost: the leaf tree of a panel is walked once for all its near targets, each leaf taking the targets far enough
# from it as one batch (kernel matrix, then the two bases by BLAS). With `symmetry`, rows are also reused between
# congruent panels: a panel's class is its rules and its shape in its own frame (anchor, first side, normal), a row
# is keyed by its class and the target's position (to 1e-13 of the panel's diameter) and normal in that frame, and
# each distinct row is computed once (on a symmetric polyhedron, once per symmetry orbit). The node rule is always
# subtracted per actual pair, from the absolute coordinates, as the far field adds it.
# Coordinates are offsets from the source panel's anchor; target offsets come from the stored absolute anchors and
# local points, so a target a distance d from an edge it shares with the panel carries a relative error ~1e-16/d.
#
# The single layer (planar_slp_corrections, planar_slp_potential) uses the same rows without the in-plane skip, keyed by
# the target's position only. On-surface targets are allowed: the node a target lies on is left out of the node rule
# (r = 0 terms are dropped everywhere), and with `duffy` a target's own panel is integrated on the target's own
# (fine u-segment, fine v-segment) leaf by Duffy's rule on four triangles from the node (_pn_self_row), the rest of the
# panel by the leaf tree; without it the tree is refined around the target down to `max_depth`.

"""
    PlanarProductNearQuad(; η = 1.5, η_leaf = 1.0, max_depth = 80, symmetry = true, duffy = true, verbose = false)

Near corrections of Dᵀ (and of the single layer, `planar_slp_corrections`) for interfaces of flat `CurvedPanel`s
(planar polyhedra, `polyhedron_mesh`), with GCQ rules in u, v or both (GCQ × GCQ vertex corners), by product
integration; see the header of `planar_near.jl`. A target within `η` panel diameters of one of a panel's nodes (for
Dᵀ: off the panel's plane) gets the exact row; leaves are split until the target is `η_leaf` leaf diameters away (or
`max_depth`). `symmetry` reuses rows between congruent panels (exact to ~1e-13 of the panel size); `duffy` integrates
a single-layer row at a panel's own node by Duffy's rule on the node's own leaf; `verbose` prints the counts and
timings.
"""
Base.@kwdef struct PlanarProductNearQuad <: NearQuadrature
    η::Float64 = 1.5
    η_leaf::Float64 = 1.0
    max_depth::Int = 80
    symmetry::Bool = true
    duffy::Bool = true
    verbose::Bool = false
end

const _PN_M = 12
const _PN_GL = gausslegendre(_PN_M)

# fine-grid rows of segment k of rule r
_pn_seg(r, k) = (nz = length(r.z0); (k - 1) * nz .+ (1:nz))

# points, weights (s^pw included) and basis (points × rule nodes) of the leaf [a, b] of segment k
function _pn_leaf_rule(r, k, a, b, gj)
    sa, sb = r.segs[k]
    Pk = r.P[_pn_seg(r, k), :]
    a == sa && b == sb && length(r.z0) >= _PN_M && return r.fx[:, k], r.fw[:, k], Pk
    if a == 0 && r.pw != 0
        x, w = gj[r.pw]
        t = b .* (1 .+ x) ./ 2; wt = (b / 2)^(1 + r.pw) .* w
    else
        x, w = _PN_GL
        t = (a + b) / 2 .+ (b - a) / 2 .* x; wt = (b - a) / 2 .* w .* (r.pw == 0 ? 1.0 : t .^ r.pw)
    end
    L = _tensor_lagrange_mat(r.z0, r.bw, 2 .* (t .- sa) ./ (sb - sa) .- 1)
    return t, wt, L' * Pk
end

# the leaf rule, from the precomputed whole-segment rules `pre` (one vector per rule) when the leaf is a segment
_pn_rule(r, pre, k, a, b, gj) = (s = r.segs[k]; a == s[1] && b == s[2] ? pre[r][k] : _pn_leaf_rule(r, k, a, b, gj))

# centre and radius of the parameter rectangle [ua, ub] × [va, vb] of panel P (its offsets), and its corners
function _pn_ball(P, ua, ub, va, vb)
    c = [collect(curved_lpoint(P, u, v)) for (u, v) in ((ua, va), (ub, va), (ub, vb), (ua, vb))]
    m = sum(c) ./ 4
    return m, maximum(norm(x - m) for x in c), c
end

# Add the leaf [ua, ub] × [va, vb] (inside segments ku, lv) to the rows R (nu × targets × nv) of the targets `tix`
# (columns of Xl, Nl: positions in the panel's offsets, normals): the targets far from the leaf in one batch, the
# others in its children.
function _pn_walk!(R, P, Xl, Nl, tix, ku, ua, ub, lv, va, vb, depth, q, pre, gj)
    m, rad, c = _pn_ball(P, ua, ub, va, vb)
    lim2 = (rad * (1 + 2 * q.η_leaf))^2
    far = Int[]; near = Int[]
    for t in tix
        d2 = (Xl[1, t] - m[1])^2 + (Xl[2, t] - m[2])^2 + (Xl[3, t] - m[3])^2
        (d2 >= lim2 || depth >= q.max_depth) ? push!(far, t) : push!(near, t)
    end
    if !isempty(far)
        tu, wu, Bu = _pn_rule(P.ru, pre, ku, ua, ub, gj)
        tv, wv, Bv = _pn_rule(P.rv, pre, lv, va, vb, gj)
        mu, mv, nf = length(tu), length(tv), length(far)
        nu, nv = size(Bu, 2), size(Bv, 2)
        Y = [curved_lpoint(P, u, v) for u in tu, v in tv]
        JW = [curved_frame(P, u, v)[4] * a * b for (u, a) in zip(tu, wu), (v, b) in zip(tv, wv)]
        _pn_far!(R, Y, JW, Bu, Bv, Xl, Nl, far)
    end
    isempty(near) && return
    eu = norm(c[2] - c[1]) + norm(c[3] - c[4]); ev = norm(c[4] - c[1]) + norm(c[3] - c[2])
    um = (ua + ub) / 2; vm = (va + vb) / 2
    us = ev > 2eu ? ((ua, ub),) : ((ua, um), (um, ub))
    vs = eu > 2ev ? ((va, vb),) : ((va, vm), (vm, vb))
    for (a, b) in us, (s, t) in vs
        _pn_walk!(R, P, Xl, Nl, near, ku, a, b, lv, s, t, depth + 1, q, pre, gj)
    end
end

# the leaf's points Y, J × weights JW and bases Bu, Bv against the targets `far`, added to R (a function barrier:
# the leaf data come from untyped caches). The kernel is D^T, n_x·(x - y) / (4π |x - y|³), with the target normals
# Nl, or the single layer 1 / (4π |x - y|) with Nl = nothing.
function _pn_far!(R::Array{Float64, 3}, Y::Matrix{NTuple{3, Float64}}, JW::Matrix{Float64}, Bu::Matrix{Float64},
                  Bv::Matrix{Float64}, Xl::Matrix{Float64}, Nl::Union{Nothing, Matrix{Float64}}, far::Vector{Int})
    mu, mv = size(Y); nf = length(far); nu, nv = size(Bu, 2), size(Bv, 2)
    K = Array{Float64}(undef, mu, mv, nf)
    if Nl === nothing          # (a point on a target, possible for a target on the panel at max_depth, is skipped)
        @inbounds for (f, t) in enumerate(far)
            x1, x2, x3 = Xl[1, t], Xl[2, t], Xl[3, t]
            for j in 1:mv, i in 1:mu
                y = Y[i, j]
                r2 = (x1 - y[1])^2 + (x2 - y[2])^2 + (x3 - y[3])^2
                K[i, j, f] = ifelse(r2 > 0, JW[i, j] / (4π * sqrt(r2)), 0.0)
            end
        end
    else
        @inbounds for (f, t) in enumerate(far)
            x1, x2, x3 = Xl[1, t], Xl[2, t], Xl[3, t]; n1, n2, n3 = Nl[1, t], Nl[2, t], Nl[3, t]
            for j in 1:mv, i in 1:mu
                y = Y[i, j]
                d1 = x1 - y[1]; d2 = x2 - y[2]; d3 = x3 - y[3]; r2 = d1^2 + d2^2 + d3^2
                K[i, j, f] = JW[i, j] * (n1 * d1 + n2 * d2 + n3 * d3) / (4π * r2 * sqrt(r2))
            end
        end
    end
    A = Bu' * reshape(K, mu, mv * nf)                                          # nu × (mv nf)
    A = reshape(reshape(permutedims(reshape(A, nu, mv, nf), (1, 3, 2)), nu * nf, mv) * Bv, nu, nf, nv)
    @inbounds for (f, t) in enumerate(far), j in 1:nv, i in 1:nu
        R[i, t, j] += A[i, f, j]
    end
    return nothing
end

# exact rows (targets × panel nodes, node (i_u, j_v) at (i_u - 1) n_v + j_v) of panel P at targets Xl, Nl (D^T), or
# of the single layer with Nl = nothing
function _pn_rows(P, Xl, Nl, q, pre, gj)
    nu, nv, nt = length(P.ru.x), length(P.rv.x), size(Xl, 2)
    R = zeros(nu, nt, nv)
    tix = collect(1:nt)
    for (a, (ua, ub)) in enumerate(P.ru.segs), (b, (va, vb)) in enumerate(P.rv.segs)
        _pn_walk!(R, P, Xl, Nl, tix, a, ua, ub, b, va, vb, 0, q, pre, gj)
    end
    return reshape(permutedims(R, (2, 3, 1)), nt, nu * nv)
end

# the fine segment of rule r holding x (an interior point: a node)
_pn_segof(r, x) = clamp(searchsortedlast(first.(r.segs), x), 1, length(r.segs))

# values at the points t (in segment k) of the nu basis functions of rule r: the segment's Lagrange interpolant of the
# fine-grid rows (without the weight s^pw)
function _pn_basis(r, k, t)
    sa, sb = r.segs[k]
    return _tensor_lagrange_mat(r.z0, r.bw, 2 .* (t .- sa) ./ (sb - sa) .- 1)' * r.P[_pn_seg(r, k), :]
end

const _PN_DUFFY = gausslegendre(24)

# the single-layer row (as one row of _pn_rows) of the flat bilinear panel P at its own node (iu, jv). Every leaf but the
# node's own (fine u-segment, fine v-segment) leaf goes to the leaf tree; on that leaf, a box around the node of about
# equal physical sides (the leaf's shorter one) is integrated by Duffy's rule on the four triangles from the node to the
# box's sides, and the rest of the leaf (up to four strips) by the tree. On a triangle (node, A, B), y = node + ρ d(t),
# d(t) = A - node + t (B - A): the area element ρ |det| dρ dt cancels 1/|x - y| = 1/(ρ |D(ρ, t)|), with
# D = d_u x_u(u0, v0 + ρ d_v) + d_v x_v(u0, v0) the exact difference quotient of the bilinear map; 1/|D| is
# near-singular in t when the node is close to the side AB, so t is split geometrically around the closest point.
function _pn_self_row(P, iu, jv, q, pre, gj)
    P.dmap === nothing && throw(ArgumentError("_pn_self_row: the panel needs its exact tangents (dmap)"))
    ru, rv = P.ru, P.rv
    nu, nv = length(ru.x), length(rv.x)
    u0, v0 = ru.x[iu], rv.x[jv]
    k0, l0 = _pn_segof(ru, u0), _pn_segof(rv, v0)
    Xl = reshape(collect(P.lpoints[(iu - 1) * nv + jv]), 3, 1)
    R = zeros(nu, 1, nv)
    for (a, (ua, ub)) in enumerate(ru.segs), (b, (va, vb)) in enumerate(rv.segs)
        (a, b) == (k0, l0) || _pn_walk!(R, P, Xl, nothing, [1], a, ua, ub, b, va, vb, 0, q, pre, gj)
    end
    (ua, ub), (va, vb) = ru.segs[k0], rv.segs[l0]
    xu0, xv0 = (collect(t) for t in P.dmap(u0, v0))
    lu, lv = norm(xu0), norm(xv0)
    h = min((ub - ua) * lu, (vb - va) * lv)
    bu = (max(ua, u0 - h / lu), min(ub, u0 + h / lu)); bv = (max(va, v0 - h / lv), min(vb, v0 + h / lv))
    for (s, t, c, d) in ((ua, bu[1], va, vb), (bu[2], ub, va, vb), (bu[1], bu[2], va, bv[1]), (bu[1], bu[2], bv[2], vb))
        s < t && c < d && _pn_walk!(R, P, Xl, nothing, [1], k0, s, t, l0, c, d, 0, q, pre, gj)
    end
    # Duffy on the box
    xg, wg = _PN_DUFFY; xg = (1 .+ xg) ./ 2; wg = wg ./ 2
    us = Float64[]; vs = Float64[]; W = Float64[]
    cs = ((bu[1], bv[1]), (bu[2], bv[1]), (bu[2], bv[2]), (bu[1], bv[2]))
    for i in 1:4
        A, B = cs[i], cs[mod1(i + 1, 4)]
        e1 = (A[1] - u0, A[2] - v0); e2 = (B[1] - A[1], B[2] - A[2])
        dt = abs(e1[1] * (e1[2] + e2[2]) - e1[2] * (e1[1] + e2[1]))           # |det(A - node, B - node)|
        dt == 0 && continue
        # t-split: around the t of the closest point of the line AB to the node (at ρ → 0), geometrically
        p1 = e1[1] .* xu0 .+ e1[2] .* xv0; p2 = e2[1] .* xu0 .+ e2[2] .* xv0
        ts = clamp(-dot(p1, p2) / dot(p2, p2), 0.0, 1.0)
        δ = max(norm(p1 .+ ts .* p2) / norm(p2), 1e-300)
        br = [0.0, 1.0, ts]
        for side in (-1, 1)
            for j in 0:60
                x = ts + side * δ * 2.0^j
                0 < x < 1 || break
                push!(br, x)
            end
        end
        br = sort(unique(br))
        for (ta, tb) in zip(br[1:end-1], br[2:end]), (t, wt) in zip(ta .+ (tb - ta) .* xg, (tb - ta) .* wg)
            du = e1[1] + t * e2[1]; dv = e1[2] + t * e2[2]
            for (ρ, wρ) in zip(xg, wg)
                u = u0 + ρ * du; v = v0 + ρ * dv
                xu, _ = P.dmap(u0, v)
                D = norm(du .* collect(xu) .+ dv .* xv0)
                J = curved_frame(P, u, v)[4]
                push!(us, u); push!(vs, v)
                push!(W, wt * wρ * dt * J * (ru.pw == 0 ? 1.0 : u^ru.pw) * (rv.pw == 0 ? 1.0 : v^rv.pw) / (4π * D))
            end
        end
    end
    R[:, 1, :] .+= _pn_basis(ru, k0, us)' * (W .* _pn_basis(rv, l0, vs))
    return vec(permutedims(R, (2, 3, 1)))
end

# frame of a flat panel (columns e1 along its side v = 0, e2, e3 the normal) and its diameter
function _pn_frame(P)
    _, _, c = _pn_ball(P, 0.0, 1.0, 0.0, 1.0)
    e3 = collect(P.normals[1]); e1 = normalize(c[2] - c[1]); e2 = cross(e3, e1)
    return hcat(e1, e2, e3), maximum(norm(a - b) for a in c for b in c)
end

# whole-segment leaf rules of every rule of the panels `ps`, and Gauss-Jacobi rules for their weights
function _pn_setup(ps)
    gj = Dict(pw => gaussjacobi(_PN_M, 0.0, pw) for pw in unique(vcat([[P.ru.pw, P.rv.pw] for P in ps]...)) if pw != 0)
    pre = IdDict{TensorRule1D, Vector{Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}}}()
    for P in ps, r in (P.ru, P.rv)
        haskey(pre, r) || (pre[r] = [_pn_leaf_rule(r, k, s..., gj) for (k, s) in enumerate(r.segs)])
    end
    return pre, gj
end

near_corrections(q::PlanarProductNearQuad, interface::DielectricInterface) = _pn_corrections(q, interface, :dt)

"""
    planar_slp_corrections(interface; q = PlanarProductNearQuad()) -> SparseMatrixCSC

The near correction of the single layer at the nodes themselves (on-surface targets), as `near_corrections` builds it
for D^T: (product-integral row) - (node rule) for every target within `q.η` × a panel's diameter of one of its nodes,
the panel's own nodes and those in its plane included (a node's own panel by Duffy's rule on the node's own leaf with
`q.duffy`, else by the leaf tree refined around it down to `q.max_depth`), rows shared between congruent panels with
`q.symmetry`. The node rule skips the coincident node, as `_pn_direct_slp` does, so that
u = _pn_direct_slp(X, w σ, X) + C σ at the nodes X (`planar_slp_nodes`).
"""
planar_slp_corrections(interface::DielectricInterface; q::PlanarProductNearQuad = PlanarProductNearQuad()) =
    _pn_corrections(q, interface, :slp)

"""
    planar_slp_nodes(interface, σ; q = PlanarProductNearQuad()) -> Vector

The single-layer potential Sσ at the interface's own nodes: the node rule by direct summation without the coincident
node, plus `planar_slp_corrections`. With the weights w, ∫σ Sσ dA = dot(w .* σ, u).
"""
function planar_slp_nodes(interface::DielectricInterface, σ::AbstractVector; q::PlanarProductNearQuad = PlanarProductNearQuad())
    X = reduce(hcat, [collect(y) for P in interface.panels for y in P.points])
    return _pn_direct_slp(X, all_weights(interface) .* σ, X) .+ planar_slp_corrections(interface; q) * σ
end

# single-layer node rule w_j / (4π |x_t - y_j|) of panel P at the targets T (3 × m), the coincident node skipped
function _pn_slp_node_block(P, T::Matrix{Float64})
    B = zeros(size(T, 2), num_points(P))
    for (j, y) in enumerate(P.points), t in axes(T, 2)
        r2 = (T[1, t] - y[1])^2 + (T[2, t] - y[2])^2 + (T[3, t] - y[3])^2
        r2 > 0 && (B[t, j] = P.weights[j] / (4π * sqrt(r2)))
    end
    return B
end

# the sparse near correction of D^T (kernel = :dt: targets in a panel's plane skipped, where D^T vanishes; rows keyed by
# the target's position and normal) or of the single layer (kernel = :slp: every near target, keyed by position)
function _pn_corrections(q::PlanarProductNearQuad, interface::DielectricInterface, kernel::Symbol)
    kernel in (:dt, :slp) || throw(ArgumentError("kernel must be :dt or :slp"))
    ps = interface.panels
    all(p -> p isa CurvedPanel, ps) || throw(ArgumentError("PlanarProductNearQuad: CurvedPanels only"))
    np = length(ps); n = num_points(interface)
    offs = cumsum([0; [num_points(P) for P in ps]])
    owner = zeros(Int, n); loc = zeros(Int, n)
    for (k, P) in enumerate(ps), i in 1:num_points(P)
        owner[offs[k] + i] = k; loc[offs[k] + i] = i
    end
    tstart = time()
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
    tree = KDTree(X)
    pre, gj = _pn_setup(ps)
    frames = Vector{Matrix{Float64}}(undef, np); diams = zeros(np)
    for k in 1:np
        frames[k], diams[k] = _pn_frame(ps[k])
        maximum(norm(collect(v) - frames[k][:, 3]) for v in ps[k].normals) < 1e-12 || error("panel $k is not flat")
    end
    # near targets of every panel: off its plane, within η × diameter of one of its nodes
    tsl = Vector{Vector{Int}}(undef, np)
    Threads.@threads :dynamic for k in 1:np
        P = ps[k]; nP = frames[k][:, 3]; a0 = collect(P.anchor)
        ctr = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- ctr)) for y in P.points)
        lim2 = (q.η * diams[k])^2
        tsl[k] = sort(filter(inrange(tree, collect(ctr), rad + q.η * diams[k])) do t
            @inbounds x1, x2, x3 = X[1, t], X[2, t], X[3, t]
            @inbounds m1, m2, m3 = Nrm[1, t], Nrm[2, t], Nrm[3, t]
            off = nP[1] * (x1 - a0[1]) + nP[2] * (x2 - a0[2]) + nP[3] * (x3 - a0[3])
            c2 = (nP[2] * m3 - nP[3] * m2)^2 + (nP[3] * m1 - nP[1] * m3)^2 + (nP[1] * m2 - nP[2] * m1)^2
            kernel == :dt && abs(off) <= 1e-13 * diams[k] && c2 <= 1e-24 && return false
            any(y -> (x1 - y[1])^2 + (x2 - y[2])^2 + (x3 - y[3])^2 <= lim2, P.points)
        end)
    end
    t_list = time() - tstart
    # panel classes (rules and shape in the panel's frame) and row keys
    tgt(k, t) = collect((ps[owner[t]].anchor .- ps[k].anchor) .+ ps[owner[t]].lpoints[loc[t]])
    cls = Dict{Any, Int}()
    cid = map(1:np) do k
        P = ps[k]; F, d = frames[k], diams[k]
        shape = Tuple(round(Int, z / (1e-10 * d)) for (u, v) in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
                      for z in F' * collect(curved_lpoint(P, u, v)))
        q.symmetry ? get!(cls, (objectid(P.ru), objectid(P.rv), round(Int, d / 1e-10), shape), length(cls) + 1) : k
    end
    keys_ = Dict{Tuple{Int, NTuple{3, Int}, NTuple{3, Int}}, Int}()
    reps = Tuple{Int, Int}[]                     # (panel, target) that computes each distinct row
    rowid = Vector{Vector{Int}}(undef, np)
    for k in 1:np
        F, h = frames[k], 1e-13 * diams[k]
        rowid[k] = map(tsl[k]) do t
            xl = F' * tgt(k, t); nl = kernel == :dt ? F' * Nrm[:, t] : zeros(3)
            key = (cid[k], (round(Int, xl[1] / h), round(Int, xl[2] / h), round(Int, xl[3] / h)),
                   (round(Int, nl[1] * 1e9), round(Int, nl[2] * 1e9), round(Int, nl[3] * 1e9)))
            get!(() -> (push!(reps, (k, t)); length(reps)), keys_, key)
        end
    end
    t_keys = time() - tstart - t_list
    # the distinct rows, by the panel that represents them
    byp = Dict{Int, Vector{Int}}()
    for (i, (k, _)) in enumerate(reps)
        push!(get!(byp, k, Int[]), i)
    end
    rows = Vector{Vector{Float64}}(undef, length(reps))
    # chunks of at most 64 targets: the rows of a class gather on its first panels, so split them for the threads
    work = [(k, ids[c]) for (k, ids) in byp for c in Iterators.partition(eachindex(ids), 64)]
    Threads.@threads :dynamic for (k, ids) in work
        # with q.duffy, a single-layer row at a node of the panel itself by _pn_self_row
        own = kernel == :slp && q.duffy ? filter(i -> owner[reps[i][2]] == k, ids) : Int[]
        for i in own
            iu, jv = divrem(loc[reps[i][2]] - 1, length(ps[k].rv.x)) .+ 1
            rows[i] = _pn_self_row(ps[k], iu, jv, q, pre, gj)
        end
        rest = setdiff(ids, own)
        isempty(rest) && continue
        Xl = reduce(hcat, [tgt(k, reps[i][2]) for i in rest]); Nl = kernel == :dt ? Nrm[:, [reps[i][2] for i in rest]] : nothing
        Rk = _pn_rows(ps[k], Xl, Nl, q, pre, gj)
        for (r, i) in enumerate(rest)
            rows[i] = Rk[r, :]
        end
    end
    t_rows = time() - tstart - t_list - t_keys
    # corrections: exact rows minus the node rule of every actual pair, written in place into the CSC arrays (the columns
    # of panel k, its nodes, are consecutive and share the sorted rows tsl[k], so its block is one column-major range)
    colptr = ones(Int, n + 1)
    for k in 1:np, j in 1:num_points(ps[k])
        colptr[offs[k] + j + 1] = colptr[offs[k] + j] + length(tsl[k])
    end
    nnz_ = colptr[end] - 1
    rowval = Vector{Int}(undef, nnz_); nzval = Vector{Float64}(undef, nnz_)
    Threads.@threads :dynamic for k in 1:np
        ts = tsl[k]; m = length(ts); isempty(ts) && continue
        o = colptr[offs[k] + 1]; nk = num_points(ps[k])
        B = reshape(view(nzval, o:o + m * nk - 1), m, nk)
        for (r, i) in enumerate(rowid[k])
            B[r, :] .= rows[i]
        end
        B .-= kernel == :dt ? dt_node_rule_block(interface, k, X[:, ts], Nrm[:, ts]) : _pn_slp_node_block(ps[k], X[:, ts])
        for j in 1:nk
            copyto!(rowval, o + (j - 1) * m, ts, 1, m)
        end
    end
    C = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    q.verbose && @info "PlanarProductNearQuad" kernel near_rows = sum(length, tsl) distinct = length(reps) classes = (q.symmetry ? length(cls) : np) entries = nnz_ t_lists = t_list t_keys t_rows t_total = time() - tstart
    return C
end

# Σ_j c_j / (4π |t - x_j|) at every target t (threaded over targets), without the node a target lies on
function _pn_direct_slp(X::Matrix{Float64}, c::Vector{Float64}, T::Matrix{Float64})
    x1, x2, x3 = X[1, :], X[2, :], X[3, :]
    u = zeros(size(T, 2))
    Threads.@threads :dynamic for t in axes(T, 2)
        t1, t2, t3 = T[1, t], T[2, t], T[3, t]
        acc = 0.0
        @inbounds @simd for j in eachindex(c)
            r2 = (t1 - x1[j])^2 + (t2 - x2[j])^2 + (t3 - x3[j])^2
            acc += ifelse(r2 > 0, c[j] / sqrt(r2), 0.0)
        end
        u[t] = acc / (4π)
    end
    return u
end

"""
    planar_slp_potential(interface, σ, T; q = PlanarProductNearQuad(), far = :direct, fmm_tol = 1e-13, chunk = 4096) -> u

Single-layer potential u(x) = ∫ σ(y) / (4π |x - y|) dA(y) of the density σ (node values) on the flat CurvedPanels of
`interface`, at the targets `T` (3 × m, off the surface, or on it: e.g. the nodes, for ∫σ u dA; the node a target
lies on is left out of the node rule, and the target's own panel is integrated by the leaf tree, refined around the
target down to `q.max_depth`, where a leaf of ~2^-max_depth of the panel adds a negligible σ × its size; direct
summation only, as FMM3D does not drop the coincident node). The node rule everywhere, by direct summation
(`far = :direct`) or FMM3D (`far = :fmm`); for a target within `q.η` × a panel's diameter of one of its nodes, that
panel's node rule is replaced by its product integral (the leaf tree of `PlanarProductNearQuad`, with the
single-layer kernel), `chunk` targets at a time. The direct sum is the default: the near correction subtracts the
node terms exactly, so the node sum must be exact too, and a target δ from a node has a term ~ w σ / δ that FMM3D
evaluates only to ~1e-8 relative (a 1e-6 error at δ = 1e-8).
"""
function planar_slp_potential(interface::DielectricInterface, σ::AbstractVector, T::AbstractMatrix;
                              q::PlanarProductNearQuad = PlanarProductNearQuad(), far::Symbol = :direct,
                              fmm_tol::Real = 1e-13, chunk::Int = 4096)
    ps = interface.panels
    T = Matrix{Float64}(T)
    X, _ = _interface_targets_normals(interface); X = Matrix{Float64}(X)
    w = all_weights(interface)
    u = far == :fmm ? lfmm3d(fmm_tol, X; charges = w .* σ, targets = T, pgt = 1).pottarg ./ (4π) :
        far == :direct ? _pn_direct_slp(X, w .* σ, T) : throw(ArgumentError("far must be :direct or :fmm"))
    pre, gj = _pn_setup(ps)
    offs = cumsum([0; [num_points(P) for P in ps]])
    tree = KDTree(T)
    corr = Vector{Tuple{Vector{Int}, Vector{Float64}}}(undef, length(ps))
    Threads.@threads :dynamic for k in eachindex(ps)
        P = ps[k]; σk = σ[offs[k]+1:offs[k+1]]; wk = w[offs[k]+1:offs[k+1]]
        _, d = _pn_frame(P)
        ctr = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- ctr)) for y in P.points)
        lim2 = (q.η * d)^2
        ts = sort(filter(t -> any(y -> (T[1, t] - y[1])^2 + (T[2, t] - y[2])^2 + (T[3, t] - y[3])^2 <= lim2, P.points),
                         inrange(tree, collect(ctr), rad + q.η * d)))
        du = zeros(length(ts))
        for c in Iterators.partition(eachindex(ts), chunk)
            Xl = T[:, ts[c]] .- collect(P.anchor)
            du[c] .= _pn_rows(P, Xl, nothing, q, pre, gj) * σk
            for (i, t) in zip(c, ts[c]), (j, y) in enumerate(P.points)       # minus the node rule (FMM3D's part)
                r2 = (T[1, t] - y[1])^2 + (T[2, t] - y[2])^2 + (T[3, t] - y[3])^2
                r2 > 0 && (du[i] -= wk[j] * σk[j] / (4π * sqrt(r2)))
            end
        end
        corr[k] = (ts, du)
    end
    for (ts, du) in corr
        u[ts] .+= du
    end
    return u
end
