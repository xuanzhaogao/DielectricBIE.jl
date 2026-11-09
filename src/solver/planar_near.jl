# Near and self rows of flat panels: the CurvedPanels of planar polyhedra (polyhedron_mesh), bilinear maps with GCQ
# rules in u, v or both (edge panels GCQ × GL, vertex corners GCQ × GCQ, the rest GL × GL). PlanarProductNearQuad gives
# the near corrections of Dᵀ, planar_slp_corrections and planar_slp_potential those of the single layer.
#
# A row is ∫_P K(x - y) φ_j(y) dA(y) over the basis φ_j of P's interpolation space (one per node: in a GCQ direction the
# fine segments' Lagrange polynomials through the rule's matrix P, times s^pw), K = n_x·(x - y) / (4π |x - y|³) (Dᵀ)
# or 1 / (4π |x - y|) (the single layer). The near correction is the row minus the node rule (what the far field adds).
#
# Geometry. P is y(u, v) = o + u a + v b + u v c in its anchor's frame, so that
#     y(u + du, v + dv) - y(u, v) = du (a + (v + dv) c) + dv (b + u c)
# is accurate relative to its own size. A target is a point (ur, vr) of P's parameter plane and its offset
# w = x - y(ur, vr), and every separation is x - y = w - (y(u, v) - y(ur, vr)), never a difference of positions:
# - a node of P: (ur, vr) the node, w = 0;
# - a node of another panel Pt on a side line of P, through Pt's GCQ side u = 0 or v = 0 (an edge both panels touch):
#   (ur, vr) the node's foot on that line, w its offset from the foot by Pt's own step (u_t (a_t + v_t c_t)), so its
#   distance from the edge is exact however small; the foot's rounding only moves it along the edge;
# - any other target: (ur, vr) = (0, 0), w = x - y(0, 0).
# A leaf's points are offsets from its lower corner (cu, cv), and the parameter steps are (cu - ur) + offset, so a small
# leaf beside a target keeps its points exact relative to its size wherever it lies.
# Differences of Float64 positions err by ~1e-16 × the panel size in the distance of a node from an edge, which is down
# to ~1e-7 of the panel size beside a GCQ edge: Dᵀ row sums (σ = 1) beside the spike ridges of the small stellated
# dodecahedron were 1.3e-11 (GCQ tol 1e-12) and 3.7e-11 (tol 1e-14) off the closed form, and are 2e-12 (the check's own
# floor) this way.
#
# Near targets: adaptive refinement. A box of P's parameter square is one part per direction: a cluster of the fine
# segments 1..k (the interval [0, s_k] of a GCQ direction) or a part of one fine segment. From the whole square, a box
# is a leaf for the targets farther than (1 + 2 η_leaf) × its radius from its centre and splits for the others: a
# cluster into the cluster one segment shorter and its last segment, a part into halves, in both directions unless one
# side is more than twice the other. Leaf rules are the tensor near tables' (laplace3d_tensor_near.jl): a cluster's
# 18-point moment rule (the basis's moments exact), a segment's own nodes, 12-point Gauss-Legendre on a part of one
# (Gauss-Jacobi for the weight s^pw on a part touching s = 0); the whole weighted segment at the edge takes its moment
# rule (its own nodes integrate s^pw exactly but the rest only as an interpolant). The targets of a panel go down the
# tree together, and a leaf adds its kernel to the rows of its far targets by BLAS.
# With endpoint_rule = :gcq a cluster leaf [0, s_k] of a GCQ direction (and the weighted segment, k = 1) takes that
# direction's GCQ rule scaled to [0, s_k] instead (_flat_endpoint_gcq_map: s_k a power of 2, the scaled nodes exact;
# H[j, l] = s_k w_l φ_j(s_k x_l), the density weights once), each direction on its own: a leaf is GL × GL, scaled GCQ ×
# GL, GL × scaled GCQ or scaled GCQ × scaled GCQ, H_u (K J) H_vᵀ. A target's endpoint interval (per transverse box) is
# the first admissible cluster of its walk; the rest of [0, 1] its segments and their parts as before. The scaled rule
# integrates the rule's family (s^p times smooth factors) to its tolerance, not every density of the span (a cardinal
# function times the kernel is not in the family): on those, :gcq is no less accurate than the GCQ node rule itself is
# in the far field, and no better. It is the default (study 73: the PEC capacitance of the cube to ~3e-15 with it);
# :moment the alternative.
#
# Self rows (self = :block, the default): the single layer at a node (u0, v0) of P in its fine cell (k, l). One Duffy
# rule over the 3 × 3 fine cells around the node's (cells k - 1 .. k + 1 × l - 1 .. l + 1 in the panel, a GCQ
# direction's weighted cell at the edge left out, a GL direction one cell [0, 1]), the longer side (physically, at the
# node) cut to BLOCK_CLIP × the shorter around the node; the density at each Duffy point from the polynomial of the cell
# it falls in (through P, times s^pw). The rest of the panel by the walk, from the whole panel with D left out
# (_flat_complement_boxes, study 75: boxes disjoint from D are roots, overlapping ones are split at D's sides or their
# endpoint clusters peeled, so the cells at an edge stay clustered beyond D). A ray of the Duffy rule crosses cell boundaries, where the cells' polynomials
# jump by their interpolation error: for densities the fine grid resolves (the GCQ family to its grid_tol) the rows
# are as accurate as the own-cell rows below (study 73: σ = 1 to ~1e-14 of the closed form, smooth densities and the
# capacitance identical), but the rows of single cardinal functions err by up to ~1e-4 of the row (random node values).
#
# Self rows (self = :cell): the node's cell (its fine u-segment × fine v-segment) is left out
# of the tree (boxes inside it drop the node; a box overlapping it, however far, peels the cluster that holds the cell)
# and integrated in two parts. The Duffy region R: the cell's shorter side (physically, at the node) whole, along the
# longer an interval as long, centred on the node and clipped to the cell. The density is one polynomial there (times
# s^pw, the edge a segment away at least), and the sinh-mapped polar rule duffy_rule (kernel/duffy.jl) integrates it
# however close the node is to a side of R (a GCQ node is a 14-point Gauss-Legendre point of its fine segment, at least
# 0.69 % of the segment from its ends).
# The rest of the cell, at most two strips beside R, goes down the tree. So a self row is the disjoint sum
#     ∫_{Ω \ C} (the walk, under either endpoint rule) + ∫_{C \ R} (strips: no clusters) + ∫_R (Duffy),
# and the walk's leaves are each target's own: the peel of a cluster holding a node's cell applies to that node only.
# Owned nodes are the panel's own; a node in the weighted segment at an edge (none of a GCQ rule's) has no self row.

"""
    PlanarProductNearQuad(; η = 1.5, η_leaf = 1.0, max_depth = 80, symmetry = true, duffy = true,
                          endpoint_rule = :gcq, self = :block, verbose = false)

Near corrections of Dᵀ (and of the single layer, `planar_slp_corrections`) for interfaces of flat `CurvedPanel`s with
bilinear maps (planar polyhedra, `polyhedron_mesh`) and GCQ rules in u, v or both (GCQ × GCQ vertex corners), by
product integration; see the header of `planar_near.jl`. A target within `η` panel diameters of one of a panel's nodes
(for Dᵀ: off the panel's plane) gets the exact row; a box of the panel is a leaf once the target is `η_leaf` box
diameters from it, or after `max_depth` halvings. `symmetry` reuses rows between congruent panels (exact to ~1e-13 of
the panel size); `duffy` integrates a single-layer row at a panel's own node by the Duffy rule around it (else the
boxes around the node are halved `max_depth` times); `verbose` prints the counts and timings.

`endpoint_rule` integrates a leaf on an endpoint cluster [0, s_k] of a GCQ direction: `:gcq` (default) by that
direction's GCQ rule scaled to [0, s_k] (the cells at the edge as one smaller GCQ panel; GCQ × GL and GCQ × GCQ panels,
off-surface rows and the walk of self rows alike), accurate to the rule's tolerance for densities of its family (s^p
times smooth factors), and for other densities of the span at most the GCQ node rule's own error in the far field;
`:moment` by the cluster's 18-point rule with the basis's moments, exact for every density of the span.

`self` sets the single-layer row at a panel's own node (with `duffy`): `:block` (default) one Duffy rule over the
3 × 3 fine cells around the node, the density from each point's own cell polynomial, the rest of the panel by the walk;
accurate for densities the fine grid resolves (the solved densities), not for every single cardinal function (up to
~1e-4 of the row). `:cell` the Duffy rule in the node's own cell only, the rest of the cell and of the panel by the
walk, exact for every density of the span. See the header of `planar_near.jl`.
"""
Base.@kwdef struct PlanarProductNearQuad <: NearQuadrature
    η::Float64 = 1.5
    η_leaf::Float64 = 1.0
    max_depth::Int = 80
    symmetry::Bool = true
    duffy::Bool = true
    endpoint_rule::Symbol = :gcq
    self::Symbol = :block
    verbose::Bool = false
    function PlanarProductNearQuad(η, η_leaf, max_depth, symmetry, duffy, endpoint_rule, self, verbose)
        endpoint_rule in (:moment, :gcq) ||
            throw(ArgumentError("PlanarProductNearQuad: endpoint_rule must be :moment or :gcq, got :$endpoint_rule"))
        self in (:block, :cell) || throw(ArgumentError("PlanarProductNearQuad: self must be :block or :cell, got :$self"))
        return new(η, η_leaf, max_depth, symmetry, duffy, endpoint_rule, self, verbose)
    end
end

# the longest side of a :block self row's Duffy rectangle, in units of its shortest (physically, at the node)
const BLOCK_CLIP = 2.0

# P as y(u, v) = o + u a + v b + u v c (anchor frame), from its corners, and its diameter
struct _FlatGeom
    o::NTuple{3, Float64}
    a::NTuple{3, Float64}
    b::NTuple{3, Float64}
    c::NTuple{3, Float64}
    diam::Float64
end

# (an ArgumentError unless the map is bilinear, and a GCQ rule's fine segments tile [0, 1] from 0: its clusters)
function _flat_geom(P::CurvedPanel)
    y(u, v) = NTuple{3, Float64}(curved_lpoint(P, u, v))
    o = y(0.0, 0.0); a = y(1.0, 0.0) .- o; b = y(0.0, 1.0) .- o
    g = _FlatGeom(o, a, b, y(1.0, 1.0) .- o .- a .- b, _curved_diam(P, 0.0, 1.0, 0.0, 1.0))
    all(_norm3(y(u, v) .- o .- _flat_step(g, 0.0, 0.0, u, v)) <= 1e-12 * g.diam
        for (u, v) in ((0.3, 0.7), (0.8, 0.1), (0.5, 0.5))) ||
        throw(ArgumentError("PlanarProductNearQuad: the panel's map is not bilinear"))
    for r in (P.ru, P.rv)
        sg = r.segs
        length(sg) == 1 || sg[1][1] == 0 && sg[end][2] == 1 && all(sg[k][1] == sg[k - 1][2] for k in 2:length(sg)) ||
            throw(ArgumentError("PlanarProductNearQuad: the fine segments of a GCQ rule must tile [0, 1] from 0"))
    end
    return g
end

# y(u + du, v + dv) - y(u, v)
_flat_step(g::_FlatGeom, u, v, du, dv) = du .* (g.a .+ (v + dv) .* g.c) .+ dv .* (g.b .+ u .* g.c)

# the targets of a panel's rows: x = y(ur, vr) + w; nrm the normals (Dᵀ) or nothing (single layer); cell the fine
# (u, v) segments of a node of the panel whose row is a Duffy self row, (0, 0) for the others
struct _FlatTargets{N <: Union{Nothing, Matrix{Float64}}}
    ur::Vector{Float64}
    vr::Vector{Float64}
    w::Matrix{Float64}
    nrm::N
    cell::Vector{NTuple{2, Int}}
end

# The node (ut, vt) of panel Pt (geometry gt, anchor at `at` in P's frame) as a target of P (geometry g): (ur, vr, w).
# Through Pt's GCQ side u = 0 (or v = 0), the one nearer the node first, when that side lies on a side line of P
# (within `tol`); else from x = at + xl, xl its offset from Pt's anchor.
function _flat_target(g::_FlatGeom, Pt::CurvedPanel, gt::_FlatGeom, at, ut, vt, xl, tol)
    wu = _flat_step(gt, 0.0, vt, ut, 0.0)          # the node from its foot on Pt's side u = 0 (along gt.b)
    wv = _flat_step(gt, ut, 0.0, 0.0, vt)          # and on the side v = 0 (along gt.a)
    p0 = at .+ gt.o
    for (gcq, e, s, w) in (_norm3(wu) <= _norm3(wv) ? ((is_gcq(Pt.ru), gt.b, vt, wu), (is_gcq(Pt.rv), gt.a, ut, wv)) :
                                                       ((is_gcq(Pt.rv), gt.a, ut, wv), (is_gcq(Pt.ru), gt.b, vt, wu)))
        gcq || continue
        ref = _flat_on_side(g, p0, e, p0 .+ s .* e, tol)
        ref === nothing || return (ref..., w)
    end
    return (0.0, 0.0, (at .+ xl) .- g.o)
end

# the parameters on P (geometry g) of the point f of the line p0 + λ e, when that line is a side line of P: parallel to
# it (to 1e-10 relative) and through it (p0 within `tol`)
function _flat_on_side(g::_FlatGeom, p0, e, f, tol)
    for (side, q0, d) in ((1, g.o, g.b), (2, g.o, g.a), (3, g.o .+ g.a, g.b .+ g.c), (4, g.o .+ g.b, g.a .+ g.c))
        dd = _dot3(d, d)
        dd > 0 && _norm3(_cross3(e, d)) <= 1e-10 * _norm3(e) * sqrt(dd) &&
            _norm3(_cross3(p0 .- q0, d)) <= tol * sqrt(dd) || continue
        λ = _dot3(f .- q0, d) / dd
        return side == 1 ? (0.0, λ) : side == 2 ? (λ, 0.0) : side == 3 ? (1.0, λ) : (λ, 1.0)
    end
    return nothing
end

# the rows (targets × P's nodes, node (i_u, j_v) at (i_u - 1) n_v + j_v) of P (geometry g) at the targets T
function _flat_rows(P::CurvedPanel, g::_FlatGeom, T::_FlatTargets, q::PlanarProductNearQuad)
    ru, rv = P.ru, P.rv
    tu, tv = _tensor_near_tables(ru), _tensor_near_tables(rv)
    nu, nv, nt = size(ru.P, 2), size(rv.P, 2), length(T.ur)
    R = zeros(nu, nt, nv)
    buf(r) = (zeros(TENSOR_NEAR_M), zeros(TENSOR_NEAR_M), zeros(length(r.z0), TENSOR_NEAR_M), zeros(size(r.P, 2), TENSOR_NEAR_M))
    ws = (buf(ru), buf(rv))                                   # _tensor_leaf_map!'s buffers, one per direction
    root(r) = length(r.segs) == 1 ? (1, 0.0, 1.0) : (-length(r.segs), 0.0, 1.0)
    ex = _flat_cluster_rules(P, tu, tv, q.endpoint_rule)
    if q.self == :cell
        _flat_walk!(R, P, g, tu, tv, ex, ws, T, (root(ru)..., root(rv)..., 0), collect(1:nt), true, q)
        for t in 1:nt
            T.cell[t] == (0, 0) || _flat_self!(R, P, g, tu, tv, ex, ws, T, t, q)
        end
    else
        other = [t for t in 1:nt if T.cell[t] == (0, 0)]
        isempty(other) || _flat_walk!(R, P, g, tu, tv, ex, ws, T, (root(ru)..., root(rv)..., 0), other, false, q)
        for t in 1:nt
            T.cell[t] == (0, 0) || _flat_block_self!(R, P, g, tu, tv, ex, ws, T, t, q)
        end
    end
    return reshape(permutedims(R, (2, 3, 1)), nt, nu * nv)
end

# The leaf rules of the endpoint clusters [0, s_k] of P's two directions, as (points, maps) per cluster k: those of
# `rule`, the moment rules of the near tables or, per GCQ direction, the scaled GCQ rules (a GL direction has no
# clusters). A leaf takes each direction's rule on its own: GL × GL, scaled GCQ × GL, GL × scaled GCQ, or scaled GCQ ×
# scaled GCQ.
function _flat_cluster_rules(P::CurvedPanel, tu::_TensorNearTables, tv::_TensorNearTables, rule::Symbol)
    mom = ((tu.cx, tu.clu), (tv.cx, tv.clu))
    rule == :gcq || return mom
    return (is_gcq(P.ru) ? _flat_endpoint_gcq_rules(P.ru) : mom[1], is_gcq(P.rv) ? _flat_endpoint_gcq_rules(P.rv) : mom[2])
end

"""
    _flat_endpoint_gcq_map(r, a) -> (xs, H)

The GCQ rule r scaled to [0, a]: nodes xs = a r.x and the map H (basis × nodes) of r's parent basis φ_j (Lagrange in
each fine segment through r.P, times s^pw), H[j, l] = a r.w[l] φ_j(a r.x[l]). r.w are density weights (∫₀¹ σ ≈ Σ w σ(x)),
so the weights enter once and s^pw only through φ_j; ∫₀ᵃ K φ_j ≈ Σ_l H[j, l] K(xs_l). The scaled rule integrates the
family s^q to the rule's own accuracy, but φ_j(a s) is in the family only approximately (exactly for a ≤ the weighted
segment's end, where φ_j is s^pw times a polynomial).
"""
function _flat_endpoint_gcq_map(r::TensorRule1D, a::Real)
    is_gcq(r) || throw(ArgumentError("_flat_endpoint_gcq_map: not a GCQ rule"))
    0 < a <= 1 || throw(ArgumentError("_flat_endpoint_gcq_map: the cutoff a must be in (0, 1], got $a"))
    xs = a .* r.x; ws = a .* r.w
    all(>=(floatmin(Float64)), xs) ||
        throw(ArgumentError("_flat_endpoint_gcq_map: the cutoff $a puts nodes below the normal Float64 range"))
    H = transpose(tensor_rule_interp(r, xs)) .* transpose(ws)
    all(isfinite, H) || throw(ArgumentError("_flat_endpoint_gcq_map: the map at the cutoff $a is not finite"))
    return xs, Matrix(H)
end

# the scaled GCQ rules of a rule's endpoint clusters [0, s_k], s_k = r.segs[k][2] (k = 1..nsegs): built once per rule
const _FLAT_ENDPOINT_GCQ = IdDict{TensorRule1D, Tuple{Vector{Vector{Float64}}, Vector{Matrix{Float64}}}}()
const _FLAT_ENDPOINT_GCQ_LOCK = ReentrantLock()
_flat_endpoint_gcq_rules(r::TensorRule1D) = lock(_FLAT_ENDPOINT_GCQ_LOCK) do
    get!(_FLAT_ENDPOINT_GCQ, r) do
        maps = [_flat_endpoint_gcq_map(r, sb) for (_, sb) in r.segs]
        return first.(maps), last.(maps)
    end
end

# R += the box (ku, ua, ub) × (kv, va, vb) of P (a part per direction: a cluster -k, or a part of segment k; `depth`
# halvings deep) at the targets `tix`, a cluster's leaves by its rule in `ex`. With `drop`, a node of P leaves the boxes
# inside its own cell to its self row, and a box overlapping the cell, however far, peels the cluster that holds the
# cell (and only that: halving the box would leave its halves overlapping the cell, so that the boxes around a cell
# would double at every level). The peel is the held node's alone; the other targets split by the shape rule, so that
# every target's leaves are its own, whatever targets are walked with it.
function _flat_walk!(R, P, g, tu, tv, ex, ws, T, box, tix, drop, q)
    ku, ua, ub, kv, va, vb, depth = box
    um = (ua + ub) / 2; vm = (va + vb) / 2
    rad = maximum(_norm3(_flat_step(g, um, vm, s - um, t - vm)) for (s, t) in ((ua, va), (ub, va), (ub, vb), (ua, vb)))
    far = Int[]; near = Int[]; held = Int[]
    for t in tix
        k, l = T.cell[t]
        if drop && k > 0
            ku == k && kv == l && continue
            (ku == k || ku < 0 && -ku >= k) && (kv == l || kv < 0 && -kv >= l) && (push!(held, t); continue)
        end
        d = _norm3((T.w[1, t], T.w[2, t], T.w[3, t]) .- _flat_step(g, T.ur[t], T.vr[t], um - T.ur[t], vm - T.vr[t]))
        (d >= (1 + 2 * q.η_leaf) * rad || depth >= q.max_depth) ? push!(far, t) : push!(near, t)
    end
    isempty(far) || _flat_leaf!(R, P, g, tu, tv, ex, ws, T, ku, ua, ub, kv, va, vb, far)
    if !isempty(held)                                         # peel the clusters (a held box has one at least)
        uch = ku < 0 ? _tensor_dir_split(P.ru, ku, 0.0, ua, ub) : ((ku, ua, ub),)
        vch = kv < 0 ? _tensor_dir_split(P.rv, kv, 0.0, va, vb) : ((kv, va, vb),)
        for (k1, a1, b1) in uch, (k2, a2, b2) in vch
            _flat_walk!(R, P, g, tu, tv, ex, ws, T, (k1, a1, b1, k2, a2, b2, depth), held, drop, q)
        end
    end
    isempty(near) && return
    lu = _norm3(_flat_step(g, ua, vm, ub - ua, 0.0)); lv = _norm3(_flat_step(g, um, va, 0.0, vb - va))
    uch = lv > 2lu ? ((ku, ua, ub),) : _tensor_dir_split(P.ru, ku, 0.0, ua, ub)
    vch = lu > 2lv ? ((kv, va, vb),) : _tensor_dir_split(P.rv, kv, 0.0, va, vb)
    halved = length(uch) == 2 && ku > 0 || length(vch) == 2 && kv > 0
    for (k1, a1, b1) in uch, (k2, a2, b2) in vch
        _flat_walk!(R, P, g, tu, tv, ex, ws, T, (k1, a1, b1, k2, a2, b2, depth + halved), near, drop, q)
    end
end

# one direction of a leaf: its lower end c, its points as offsets from c, and the map G (basis × points, the weights
# included); a cluster [0, s_k], and the whole weighted segment at the edge (the cluster k = 1), by its rule in
# ex = (points, maps)
function _flat_leaf_map!(buf, r::TensorRule1D, tb::_TensorNearTables, ex, k, a, b)
    cx, clu = ex
    (k < 0 || r.pw != 0 && !isempty(clu) && r.segs[k][1] == 0 && a == 0 && b == r.segs[k][2]) &&
        return 0.0, cx[abs(k)], view(clu[abs(k)], 1:size(clu[abs(k)], 1), :)
    pts, sh, G, _ = _tensor_leaf_map!(buf, r, tb, 1.0, k, a, 0.0, b - a)
    return a, pts .- sh, G
end

# R += the leaf (ku, ua, ub) × (kv, va, vb) at its far targets
function _flat_leaf!(R, P, g, tu, tv, ex, ws, T, ku, ua, ub, kv, va, vb, far)
    cu, ou, Gu = _flat_leaf_map!(ws[1], P.ru, tu, ex[1], ku, ua, ub)
    cv, ov, Gv = _flat_leaf_map!(ws[2], P.rv, tv, ex[2], kv, va, vb)
    mu, mv, nf = length(ou), length(ov), length(far)
    nu, nv = size(Gu, 1), size(Gv, 1)
    yu = [g.a .+ (cv + o) .* g.c for o in ov]                 # the tangent y_u along each column of the leaf
    J = [_norm3(_cross3(yu[j], g.b .+ (cu + o) .* g.c)) for o in ou, j in 1:mv]
    K = Array{Float64}(undef, mu, mv, nf)
    _flat_kernel!(K, T.nrm, T, far, g, cu, ou, cv, ov, yu, J)
    A = Gu * reshape(K, mu, mv * nf)                          # nu × (mv nf)
    A = reshape(reshape(permutedims(reshape(A, nu, mv, nf), (1, 3, 2)), nu * nf, mv) * transpose(Gv), nu, nf, nv)
    @inbounds for (f, t) in enumerate(far), j in 1:nv, i in 1:nu
        R[i, t, j] += A[i, f, j]
    end
end

# K[i, j, f] = J × kernel at x - y = w - (y(cu + ou_i, cv + ov_j) - y(ur, vr)) for the target far[f]: the single layer
# (nrm = nothing; a target on a quadrature point, possible for a target on P after max_depth halvings, is skipped) or Dᵀ
function _flat_kernel!(K, nrm, T, far, g, cu, ou, cv, ov, yu, J)
    @inbounds for (f, t) in enumerate(far)
        ur, vr = T.ur[t], T.vr[t]
        yv = g.b .+ ur .* g.c
        n1, n2, n3 = nrm === nothing ? (0.0, 0.0, 0.0) : (nrm[1, t], nrm[2, t], nrm[3, t])
        bu = cu - ur; bv = cv - vr
        for j in eachindex(ov)
            dv = bv + ov[j]
            e1 = T.w[1, t] - dv * yv[1]; e2 = T.w[2, t] - dv * yv[2]; e3 = T.w[3, t] - dv * yv[3]
            a1, a2, a3 = yu[j]
            for i in eachindex(ou)
                du = bu + ou[i]
                d1 = e1 - du * a1; d2 = e2 - du * a2; d3 = e3 - du * a3
                r2 = d1 * d1 + d2 * d2 + d3 * d3
                K[i, j, f] = nrm === nothing ? ifelse(r2 > 0, J[i, j] / (4π * sqrt(r2)), 0.0) :
                                               J[i, j] * (n1 * d1 + n2 * d2 + n3 * d3) / (4π * r2 * sqrt(r2))
            end
        end
    end
end

# The Duffy region D = (ua, ub, va, vb) of the node (u0, v0) of P in its cell, the fine pair (k, l): the cell's shorter
# side (physically, at the node) whole and along the longer an interval as long, centred on the node and clipped to the
# cell; the strips (k, a1, b1, l, a2, b2) of the cell beside D (possibly empty); G the first fundamental form at the node
function _flat_self_regions(g::_FlatGeom, ru::TensorRule1D, rv::TensorRule1D, k, l, u0, v0)
    (sa, sb), (ta, tb) = ru.segs[k], rv.segs[l]
    yu = g.a .+ v0 .* g.c; yv = g.b .+ u0 .* g.c
    G = (_dot3(yu, yu), _dot3(yu, yv), _dot3(yv, yv))
    if (sb - sa)^2 * G[1] <= (tb - ta)^2 * G[3]               # the u side the shorter: D = [sa, sb] × v0 ± its half
        h = (sb - sa) * sqrt(G[1] / G[3]) / 2
        D = (sa, sb, max(ta, v0 - h), min(tb, v0 + h))
        return D, ((k, sa, sb, l, ta, D[3]), (k, sa, sb, l, D[4], tb)), G
    end
    h = (tb - ta) * sqrt(G[3] / G[1]) / 2
    D = (max(sa, u0 - h), min(sb, u0 + h), ta, tb)
    return D, ((k, sa, D[1], l, ta, tb), (k, D[2], sb, l, ta, tb)), G
end

# the self row of the target t, a node of P in the cell T.cell[t]: the strips beside the Duffy region by the walk
# (without drop), the Duffy region by duffy_rule
function _flat_self!(R, P, g, tu, tv, ex, ws, T, t, q)
    ru, rv = P.ru, P.rv
    k, l = T.cell[t]; u0, v0 = T.ur[t], T.vr[t]
    D, strips, G = _flat_self_regions(g, ru, rv, k, l, u0, v0)
    for (k1, a1, b1, k2, a2, b2) in strips
        a1 < b1 && a2 < b2 && _flat_walk!(R, P, g, tu, tv, ex, ws, T, (k1, a1, b1, k2, a2, b2, 0), [t], false, q)
    end
    _flat_self_duffy!(R, P, g, tu, tv, t, u0, v0, k, l, D, G)
end

# R[:, t, :] += the single layer at the node (u0, v0) on its Duffy region D ⊂ its cell (k, l), times the cell's basis
# (Lagrange in each segment through the rule, times s^pw)
function _flat_self_duffy!(R, P, g, tu, tv, t, u0, v0, k, l, D, G)
    ru, rv = P.ru, P.rv
    (sa, sb), (ta, tb) = ru.segs[k], rv.segs[l]
    nρ = cld(length(ru.z0) + length(rv.z0) + 9, 2)            # along a ray: degree nz_u + nz_v - 2, plus 10
    du, dv, w = duffy_rule((u0, v0), D, G; nρ = max(DUFFY_NRHO, nρ + isodd(nρ)))
    wt = similar(w); τu = similar(w); τv = similar(w)
    @inbounds for i in eachindex(w)
        u = u0 + du[i]; v = v0 + dv[i]
        J = _norm3(_cross3(g.a .+ v .* g.c, g.b .+ u .* g.c))
        wt[i] = w[i] * J / (4π * _norm3(_flat_step(g, u0, v0, du[i], dv[i]))) *
                (ru.pw == 0 ? 1.0 : u^ru.pw) * (rv.pw == 0 ? 1.0 : v^rv.pw)
        τu[i] = 2 * (u - sa) / (sb - sa) - 1; τv[i] = 2 * (v - ta) / (tb - ta) - 1
    end
    Bu = tu.pk[k] * (_tensor_lagrange_mat(ru.z0, ru.bw, τu) .* wt')
    Bv = tv.pk[l] * _tensor_lagrange_mat(rv.z0, rv.bw, τv)
    R[:, t, :] .+= Bu * Bv'
end

"""
    _flat_complement_boxes(ru, rv, g, R) -> [(ku, ua, ub, kv, va, vb)]

The root boxes of the walk on Ω \\ R, R = (ua, ub, va, vb) a rectangle of P's parameter square (geometry g), walked
from the whole panel (study 75): from the root box (the cluster [0, s_K] in a GCQ direction, [0, 1] in a GL one), a
box disjoint from R is a root, a box inside R is dropped, and a box overlapping R is split in a direction in which it
extends past R: a part of a fine segment cut at R's sides, else an endpoint cluster peeled (cluster k into cluster
k - 1 and segment k; the physically longer one when both directions are clusters). The cells at an edge so stay
clustered beyond R: on a corner node beside the vertex the leaves fall 2.4× against cutting Ω \\ R into four rectangles
at the fine grid, with the same rows on resolved densities (to 3e-15).
"""
function _flat_complement_boxes(ru::TensorRule1D, rv::TensorRule1D, g::_FlatGeom, R)
    out = Tuple{Int, Float64, Float64, Int, Float64, Float64}[]
    root(r) = length(r.segs) == 1 ? (1, 0.0, 1.0) : (-length(r.segs), 0.0, 1.0)
    _flat_exclude!(out, ru, rv, g, root(ru), root(rv), R)
    return out
end

# the part p = (k, a, b) of rule r (a cluster [0, s_{-k}] for k < 0) split against [lo, hi]: a cluster peeled (k into
# k - 1, as segment 1 itself when k - 1 = 1, and segment k), a part of a segment cut at lo and hi inside it
function _flat_exclude_split(r::TensorRule1D, p, lo, hi)
    k, a, b = p
    if k < 0
        kk = -k
        return ((kk - 1 == 1 ? 1 : -(kk - 1), 0.0, r.segs[kk - 1][2]), (kk, r.segs[kk][1], r.segs[kk][2]))
    end
    cuts = (a, (a < lo < b ? (lo,) : ())..., (a < hi < b ? (hi,) : ())..., b)
    return Tuple((k, cuts[i], cuts[i + 1]) for i in 1:length(cuts) - 1)
end

function _flat_exclude!(out, ru, rv, g, pu, pv, R)
    (_, ua, ub), (_, va, vb) = pu, pv
    if ub <= R[1] || ua >= R[2] || vb <= R[3] || va >= R[4]
        push!(out, (pu[1], ua, ub, pv[1], va, vb))
        return out
    end
    ua >= R[1] && ub <= R[2] && va >= R[3] && vb <= R[4] && return out
    ext_u = ua < R[1] || ub > R[2]; ext_v = va < R[3] || vb > R[4]
    cu = pu[1] < 0; cv = pv[1] < 0
    split_u = ext_u && !cu ? true : ext_v && !cv ? false :
              ext_u && ext_v ? (ub - ua) * _norm3(g.a) >= (vb - va) * _norm3(g.b) : ext_u
    if split_u
        for p in _flat_exclude_split(ru, pu, R[1], R[2])
            _flat_exclude!(out, ru, rv, g, p, pv, R)
        end
    else
        for p in _flat_exclude_split(rv, pv, R[3], R[4])
            _flat_exclude!(out, ru, rv, g, pu, p, R)
        end
    end
    return out
end

# the cells k - 1 .. k + 1 of a direction, as a parameter interval: a GL direction is one cell, [0, 1]; a GCQ
# direction's weighted cell at the edge (the density's factor s^pw is not polynomial there) is left out
function _flat_block_span(r::TensorRule1D, k)
    lo = max(k - 1, r.pw != 0 && r.segs[1][1] == 0 ? 2 : 1)
    hi = min(k + 1, length(r.segs))
    return r.segs[lo][1], r.segs[hi][2]
end

# The Duffy rectangle D = (ua, ub, va, vb) of a :block self row at the node (u0, v0) in the cell (k, l): the 3 × 3
# cells around it, the longer side (in the metric G at the node) cut to BLOCK_CLIP × the shorter around the node; and G
function _flat_block_region(g::_FlatGeom, ru::TensorRule1D, rv::TensorRule1D, k, l, u0, v0)
    yu = g.a .+ v0 .* g.c; yv = g.b .+ u0 .* g.c
    G = (_dot3(yu, yu), _dot3(yu, yv), _dot3(yv, yv))
    (ua, ub), (va, vb) = _flat_block_span(ru, k), _flat_block_span(rv, l)
    su, sv = sqrt(G[1]), sqrt(G[3])
    Lu, Lv = (ub - ua) * su, (vb - va) * sv
    if Lv > BLOCK_CLIP * Lu
        h = BLOCK_CLIP * Lu / (2sv); va, vb = max(va, v0 - h), min(vb, v0 + h)
    elseif Lu > BLOCK_CLIP * Lv
        h = BLOCK_CLIP * Lv / (2su); ua, ub = max(ua, u0 - h), min(ub, u0 + h)
    end
    return (ua, ub, va, vb), G
end

# E[a, :] = the basis of the rule r (Lagrange in each fine segment through P, times t^pw) at ts[a] (tensor_rule_interp
# without allocations per point)
function _flat_cell_basis!(E, r::TensorRule1D, ts)
    nz = length(r.z0); ℓ = zeros(nz); starts = first.(r.segs)
    @inbounds for (a, t) in enumerate(ts)
        k = clamp(searchsortedlast(starts, t), 1, length(r.segs)); sa, sb = r.segs[k]
        τ = 2 * (t - sa) / (sb - sa) - 1
        j = findfirst(==(τ), r.z0)
        if j === nothing
            s = 0.0
            for i in 1:nz
                ℓ[i] = r.bw[i] / (τ - r.z0[i]); s += ℓ[i]
            end
            ℓ ./= s
        else
            fill!(ℓ, 0.0); ℓ[j] = 1.0
        end
        f = r.pw == 0 ? 1.0 : t^r.pw
        o = (k - 1) * nz
        for jj in axes(E, 2)
            acc = 0.0
            for i in 1:nz
                acc += r.P[o + i, jj] * ℓ[i]
            end
            E[a, jj] = f * acc
        end
    end
    return E
end

# the :block self row of the target t, a node of P in the cell T.cell[t]: the walk on the complement of the Duffy
# rectangle D, and duffy_rule on D with the density of the cells its points fall in
function _flat_block_self!(R, P, g, tu, tv, ex, ws, T, t, q)
    ru, rv = P.ru, P.rv
    k, l = T.cell[t]; u0, v0 = T.ur[t], T.vr[t]
    D, G = _flat_block_region(g, ru, rv, k, l, u0, v0)
    for (k1, a1, b1, k2, a2, b2) in _flat_complement_boxes(ru, rv, g, D)
        _flat_walk!(R, P, g, tu, tv, ex, ws, T, (k1, a1, b1, k2, a2, b2, 0), [t], false, q)
    end
    nρ = cld(length(ru.z0) + length(rv.z0) + 9, 2)            # as _flat_self_duffy!
    du, dv, w = duffy_rule((u0, v0), D, G; nρ = max(DUFFY_NRHO, nρ + isodd(nρ)))
    us = u0 .+ du; vs = v0 .+ dv
    wt = similar(w)
    @inbounds for i in eachindex(w)
        J = _norm3(_cross3(g.a .+ vs[i] .* g.c, g.b .+ us[i] .* g.c))
        wt[i] = w[i] * J / (4π * _norm3(_flat_step(g, u0, v0, du[i], dv[i])))
    end
    Eu = _flat_cell_basis!(zeros(length(w), size(ru.P, 2)), ru, us)
    Ev = _flat_cell_basis!(zeros(length(w), size(rv.P, 2)), rv, vs)
    R[:, t, :] .+= transpose(Eu .* wt) * Ev
end

# frame of a flat panel: columns e1 along its side v = 0, e2, e3 the normal
function _pn_frame(P)
    e3 = collect(P.normals[1]); e1 = normalize(collect(curved_lpoint(P, 1.0, 0.0) .- curved_lpoint(P, 0.0, 0.0)))
    return hcat(e1, cross(e3, e1), e3)
end

# the fine segment of rule r holding the parameter x of a node of panel k (direction `dir`), for its self row: none in
# the weighted segment at the edge, where the density's factor s^pw is singular
function _pn_cell(r::TensorRule1D, x, k, dir)
    s = clamp(searchsortedlast(r.segs, (x, Inf)), 1, length(r.segs))
    r.pw != 0 && r.segs[s][1] == 0 &&
        throw(ArgumentError("PlanarProductNearQuad: a node of panel $k lies in the weighted segment [0, $(r.segs[s][2])] " *
                            "at the edge of its rule in $dir, where the density's factor $dir^pw is singular: no self row"))
    return s
end

# the targets `ts` (interface nodes: owner panel and local index) of panel k's rows; P's own nodes on the panel, with
# their cells (`cells`, (0, 0) for none) for Duffy self rows
function _pn_targets(ps, gs, k, ts, owner, loc, Nrm, cells, kernel)
    P = ps[k]; g = gs[k]; nt = length(ts)
    ur = zeros(nt); vr = zeros(nt); W = zeros(3, nt); cell = fill((0, 0), nt)
    tol = 1e-12 * (g.diam + _norm3(P.anchor))
    for (r, t) in enumerate(ts)
        Pt = ps[owner[t]]; nvt = length(Pt.rv.x); i = loc[t]
        ut, vt = Pt.ru.x[(i - 1) ÷ nvt + 1], Pt.rv.x[(i - 1) % nvt + 1]
        if owner[t] == k
            ur[r], vr[r] = ut, vt
            cell[r] = cells[t]
        else
            ur[r], vr[r], w = _flat_target(g, Pt, gs[owner[t]], Pt.anchor .- P.anchor, ut, vt, Pt.lpoints[i], tol)
            W[:, r] .= w
        end
    end
    return _FlatTargets(ur, vr, W, kernel == :dt ? Nrm[:, ts] : nothing, cell)
end

near_corrections(q::PlanarProductNearQuad, interface::DielectricInterface) = _pn_corrections(q, interface, :dt)
# dielectric_operator keeps the correction as per-panel blocks: assembled as blocks, without the sparse matrix
_dt_parts(far::FarField, near::PlanarProductNearQuad, interface::DielectricInterface) =
    (far_field_operator(far, interface), _pn_corrections(near, interface, :dt; blocks = true))

"""
    planar_slp_corrections(interface; q = PlanarProductNearQuad()) -> SparseMatrixCSC

The near correction of the single layer at the nodes themselves (on-surface targets), as `near_corrections` builds it
for D^T: (product-integral row) - (node rule) for every target within `q.η` × a panel's diameter of one of its nodes,
the panel's own nodes and those in its plane included (a node's own panel by the Duffy rule around the node with
`q.duffy`, else by boxes halved around it `q.max_depth` times), rows shared between congruent panels with
`q.symmetry`. The node rule skips the coincident node, as `_pn_direct_slp` does, so that
u = _pn_direct_slp(X, w σ, X) + C σ at the nodes X (`planar_slp_nodes`).
"""
planar_slp_corrections(interface::DielectricInterface; q::PlanarProductNearQuad = PlanarProductNearQuad()) =
    _pn_corrections(q, interface, :slp)

"""
    planar_slp_nodes(interface, σ; q = PlanarProductNearQuad(), far = :direct) -> Vector

The single-layer potential Sσ at the interface's own nodes: the node rule without the coincident node, plus
`planar_slp_corrections`. With the weights w, ∫σ Sσ dA = dot(w .* σ, u). The node rule by direct summation
(`far = :direct`) or by a far field such as [`SCTLFarField`](@ref) (`far = SCTLFarField(tol = 1e-12)`: PVFMM, which
needs the close node pairs to rounding: `SCTL.has_feature(:EXACT_NEAR)`; it skips the coincident node too).
"""
function planar_slp_nodes(interface::DielectricInterface, σ::AbstractVector; q::PlanarProductNearQuad = PlanarProductNearQuad(),
                          far = :direct)
    X = reduce(hcat, [collect(y) for P in interface.panels for y in P.points])
    u0 = far === :direct ? _pn_direct_slp(X, all_weights(interface) .* σ, X) :
         far isa FarField ? _pottrg_far(far, interface, X) * σ : throw(ArgumentError("far must be :direct or a FarField"))
    return u0 .+ _pn_corrections(q, interface, :slp; blocks = true) * σ
end

# B[r, j] -= the node rule from node j of a panel (its `points` and `weights`) at the node ts[r] (positions X, normals
# Nrm): the Dᵀ term of dt_node_rule_block (`dt`) or the single-layer term of _pn_slp_node_block, with their arithmetic,
# the coincident node skipped; in place, without the block of node-rule values
function _pn_sub_node_rule!(B::AbstractMatrix{Float64}, points::Vector{NTuple{3, Float64}}, weights::Vector{Float64},
                            ts::Vector{Int}, X::Matrix{Float64}, Nrm::Matrix{Float64}, dt::Bool)
    @inbounds for j in eachindex(points)
        y = points[j]; wj = weights[j]
        for (r, i) in enumerate(ts)
            d1, d2, d3 = X[1, i] - y[1], X[2, i] - y[2], X[3, i] - y[3]
            r2 = d1^2 + d2^2 + d3^2
            r2 == 0 && continue
            B[r, j] -= dt ? wj * (Nrm[1, i] * d1 + Nrm[2, i] * d2 + Nrm[3, i] * d3) / (4π * r2 * sqrt(r2)) :
                            wj / (4π * sqrt(r2))
        end
    end
    return B
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
# the target's position and normal) or of the single layer (kernel = :slp: every near target, keyed by position); with
# `blocks`, the same correction as a BlockCorrection, its per-panel blocks assembled directly (no sparse matrix, which
# would take twice the memory and a copy into blocks)
function _pn_corrections(q::PlanarProductNearQuad, interface::DielectricInterface, kernel::Symbol; blocks::Bool = false)
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
    X0, N0 = _interface_targets_normals(interface)
    X = Matrix{Float64}(X0); Nrm = Matrix{Float64}(N0)        # new names: the threads' closures capture X and Nrm,
    tree = KDTree(X)                                           # and a captured variable that is reassigned is boxed
    gs = _flat_geom.(ps); frames = _pn_frame.(ps); diams = [g.diam for g in gs]
    for k in 1:np
        maximum(norm(collect(v) - frames[k][:, 3]) for v in ps[k].normals) < 1e-12 || error("panel $k is not flat")
    end
    # the cells of the nodes for their Duffy self rows (single layer)
    cells = fill((0, 0), n)
    if kernel == :slp && q.duffy
        for (k, P) in enumerate(ps), i in 1:num_points(P)
            nv = length(P.rv.x)
            cells[offs[k] + i] = (_pn_cell(P.ru, P.ru.x[(i - 1) ÷ nv + 1], k, "u"), _pn_cell(P.rv, P.rv.x[(i - 1) % nv + 1], k, "v"))
        end
    end
    # near targets of every panel: off its plane, within η × diameter of one of its nodes. In its plane (Dᵀ) to the
    # rounding of absolute positions as well as of the panel's size: a node of a panel 3e-4 across at an O(1) position
    # with non-dyadic coordinates lies ~3e-17 off its own plane (1.1e-13 of the diameter), and kept, its own row hits
    # r = 0 (NaN; the octahedron at nlev = 12, study 73)
    # (the threads take the panels one at a time, the largest first: their costs differ by orders of magnitude)
    tsl = Vector{Vector{Int}}(undef, np)
    _tforeach(sortperm(diams; rev = true)) do k
        P = ps[k]; nP = frames[k][:, 3]; a0 = collect(P.anchor)
        ctr = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- ctr)) for y in P.points)
        lim2 = (q.η * diams[k])^2
        tsl[k] = sort(filter(inrange(tree, collect(ctr), rad + q.η * diams[k])) do t
            @inbounds x1, x2, x3 = X[1, t], X[2, t], X[3, t]
            @inbounds m1, m2, m3 = Nrm[1, t], Nrm[2, t], Nrm[3, t]
            off = nP[1] * (x1 - a0[1]) + nP[2] * (x2 - a0[2]) + nP[3] * (x3 - a0[3])
            c2 = (nP[2] * m3 - nP[3] * m2)^2 + (nP[3] * m1 - nP[1] * m3)^2 + (nP[1] * m2 - nP[2] * m1)^2
            kernel == :dt && abs(off) <= 1e-13 * (diams[k] + norm(a0)) && c2 <= 1e-24 && return false
            any(y -> (x1 - y[1])^2 + (x2 - y[2])^2 + (x3 - y[3])^2 <= lim2, P.points)
        end)
    end
    t_list = time() - tstart
    # panel classes (rules and shape in the panel's frame) and row keys
    cls = Dict{Any, Int}()
    cid = map(1:np) do k
        P = ps[k]; F, d = frames[k], diams[k]
        shape = Tuple(round(Int, z / (1e-10 * d)) for (u, v) in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
                      for z in F' * collect(curved_lpoint(P, u, v)))
        q.symmetry ? get!(cls, (objectid(P.ru), objectid(P.rv), round(Int, d / 1e-10), shape), length(cls) + 1) : k
    end
    # the key of every (panel, near target), in parallel: the target in the panel's frame in units of 1e-13 × its
    # diameter and, for Dᵀ, the target's normal in the frame in units of 1e-9 (F' x written out, no allocation)
    anchors = [P.anchor for P in ps]
    lpts = Vector{NTuple{3, Float64}}(undef, n)
    for (k, P) in enumerate(ps), i in 1:num_points(P)
        lpts[offs[k] + i] = P.lpoints[i]
    end
    rkeys = Vector{Vector{Tuple{NTuple{3, Int}, NTuple{3, Int}}}}(undef, np)
    _tforeach(sortperm(length.(tsl); rev = true)) do k
        F = frames[k]; h = 1e-13 * diams[k]; a = anchors[k]
        rkeys[k] = map(tsl[k]) do t
            b = anchors[owner[t]]; y = lpts[t]
            d1 = (b[1] - a[1]) + y[1]; d2 = (b[2] - a[2]) + y[2]; d3 = (b[3] - a[3]) + y[3]
            x1 = F[1, 1] * d1 + F[2, 1] * d2 + F[3, 1] * d3
            x2 = F[1, 2] * d1 + F[2, 2] * d2 + F[3, 2] * d3
            x3 = F[1, 3] * d1 + F[2, 3] * d2 + F[3, 3] * d3
            m1, m2, m3 = kernel == :dt ? (Nrm[1, t], Nrm[2, t], Nrm[3, t]) : (0.0, 0.0, 0.0)
            n1 = F[1, 1] * m1 + F[2, 1] * m2 + F[3, 1] * m3
            n2 = F[1, 2] * m1 + F[2, 2] * m2 + F[3, 2] * m3
            n3 = F[1, 3] * m1 + F[2, 3] * m2 + F[3, 3] * m3
            ((round(Int, x1 / h), round(Int, x2 / h), round(Int, x3 / h)),
             (round(Int, n1 * 1e9), round(Int, n2 * 1e9), round(Int, n3 * 1e9)))
        end
    end
    # the distinct rows: keys of different classes never agree, so each class on its own (in parallel), its panels in
    # order; the first (panel, target) with a key computes its row, the others reuse it
    members = Dict{Int, Vector{Int}}()
    for k in 1:np
        push!(get!(members, cid[k], Int[]), k)
    end
    groups = sort!(collect(values(members)); by = first)
    lid = Vector{Vector{Int}}(undef, np)
    greps = Vector{Vector{Tuple{Int, Int}}}(undef, length(groups))
    _tforeach(sortperm([sum(k -> length(tsl[k]), ks) for ks in groups]; rev = true)) do g
        seen = Dict{Tuple{NTuple{3, Int}, NTuple{3, Int}}, Int}(); r = Tuple{Int, Int}[]
        for k in groups[g]
            lid[k] = [get!(() -> (push!(r, (k, t)); length(r)), seen, key) for (t, key) in zip(tsl[k], rkeys[k])]
        end
        greps[g] = r
    end
    goff = cumsum([0; length.(greps)])
    reps = reduce(vcat, greps; init = Tuple{Int, Int}[])  # (panel, target) that computes each distinct row
    rowid = Vector{Vector{Int}}(undef, np)
    for (g, ks) in enumerate(groups), k in ks
        rowid[k] = lid[k] .+ goff[g]
    end
    t_keys = time() - tstart - t_list
    # the distinct rows, by the panel that represents them
    byp = Dict{Int, Vector{Int}}()
    for (i, (k, _)) in enumerate(reps)
        push!(get!(byp, k, Int[]), i)
    end
    rows = Vector{Vector{Float64}}(undef, length(reps))
    # chunks of at most 64 targets: the rows of a class gather on its first panels, so split them for the threads;
    # the chunks with the most self rows (a panel's own nodes, the costliest) first
    work = [(k, ids[c]) for (k, ids) in byp for c in Iterators.partition(eachindex(ids), 64)]
    nself = [count(i -> owner[reps[i][2]] == k, ids) for (k, ids) in work]
    _tforeach(sortperm(nself; rev = true)) do w
        k, ids = work[w]
        Rk = _flat_rows(ps[k], gs[k], _pn_targets(ps, gs, k, [reps[i][2] for i in ids], owner, loc, Nrm, cells, kernel), q)
        for (r, i) in enumerate(ids)
            rows[i] = Rk[r, :]
        end
    end
    t_rows = time() - tstart - t_list - t_keys
    # corrections: exact rows minus the node rule of every actual pair, panel k's block (its near targets tsl[k] × its
    # nodes) written in place: into the CSC arrays (the columns of panel k, its nodes, are consecutive and share the
    # sorted rows tsl[k], so its block is one column-major range), or as a matrix of its own (`blocks`)
    function fill_block!(B, k)
        for (r, i) in enumerate(rowid[k])
            B[r, :] .= rows[i]
        end
        return _pn_sub_node_rule!(B, ps[k].points, ps[k].weights, tsl[k], X, Nrm, kernel == :dt)
    end
    order = sortperm([length(tsl[k]) * num_points(ps[k]) for k in 1:np]; rev = true)
    if blocks
        # allocated here, then filled on the threads: hundreds of large allocations from all threads at once set off
        # one stop-the-world collection after another
        Bs = [Matrix{Float64}(undef, length(tsl[k]), num_points(ps[k])) for k in 1:np]
        _tforeach(order) do k
            fill_block!(Bs[k], k)
        end
        nnz_ = sum(length, Bs)
        C = _block_correction(n, n, tsl, panel_column_ranges(interface), Bs)
    else
        colptr = ones(Int, n + 1)
        for k in 1:np, j in 1:num_points(ps[k])
            colptr[offs[k] + j + 1] = colptr[offs[k] + j] + length(tsl[k])
        end
        nnz_ = colptr[end] - 1
        rowval = Vector{Int}(undef, nnz_); nzval = Vector{Float64}(undef, nnz_)
        _tforeach(order) do k
            ts = tsl[k]; m = length(ts); isempty(ts) && return
            o = colptr[offs[k] + 1]; nk = num_points(ps[k])
            fill_block!(reshape(view(nzval, o:o + m * nk - 1), m, nk), k)
            for j in 1:nk
                copyto!(rowval, o + (j - 1) * m, ts, 1, m)
            end
        end
        C = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    end
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
lies on is left out of the node rule, and the target's own panel is integrated by the adaptive refinement of
`PlanarProductNearQuad`, the boxes around the target halved `q.max_depth` times, where a box of ~2^-max_depth of the
panel adds a negligible σ × its size; direct
summation only, as FMM3D does not drop the coincident node). The node rule everywhere, by direct summation
(`far = :direct`) or FMM3D (`far = :fmm`); for a target within `q.η` × a panel's diameter of one of its nodes, that
panel's node rule is replaced by its product integral (the adaptive refinement of `PlanarProductNearQuad`, with
the single-layer kernel), `chunk` targets at a time. The direct sum is the default: the near correction subtracts the
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
    offs = cumsum([0; [num_points(P) for P in ps]])
    tree = KDTree(T)
    gs = _flat_geom.(ps)
    corr = Vector{Tuple{Vector{Int}, Vector{Float64}}}(undef, length(ps))
    Threads.@threads :dynamic for k in eachindex(ps)
        P = ps[k]; σk = σ[offs[k]+1:offs[k+1]]; wk = w[offs[k]+1:offs[k+1]]
        g = gs[k]; d = g.diam
        ctr = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- ctr)) for y in P.points)
        lim2 = (q.η * d)^2
        ts = sort(filter(t -> any(y -> (T[1, t] - y[1])^2 + (T[2, t] - y[2])^2 + (T[3, t] - y[3])^2 <= lim2, P.points),
                         inrange(tree, collect(ctr), rad + q.η * d)))
        du = zeros(length(ts))
        for c in Iterators.partition(eachindex(ts), chunk)
            m = length(c)
            Tg = _FlatTargets(zeros(m), zeros(m), T[:, ts[c]] .- collect(P.anchor) .- collect(g.o), nothing, fill((0, 0), m))
            du[c] .= _flat_rows(P, g, Tg, q) * σk
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
