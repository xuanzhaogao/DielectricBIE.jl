# SCTL.jl backends of the dielectric operator (`using SCTL`):
#   SCTLFarField            node-rule D^T by SCTL's ParticleFMM (PVFMM, OpenMP, no MPI), tree kept across applications;
#   SCTLFarField + SCTLNearQuad   the whole D^T by SCTL's BoundaryIntegralOp on a QuadElemList (FlatPanel interfaces);
#   sctl_gl_corrections     SCTL's near and Duffy self rows of GL×GL panels as a sparse correction (any far field): all
#                           of SCTLNearCorrection, the GL panels of GCQNearQuad and of SCTLCurvedNearQuad;
#   sctl_near_list          SCTL's classification as Julia data;
#   SCTLCurvedNearQuad      the GCQ edge panels of curved interfaces (proxy elements and thin rows);
#   SCTLPotential           the off-surface single-layer potential by SCTL's BoundaryIntegralOp.
module DielectricBIESCTLExt

using DielectricBIE
using SCTL
using LinearAlgebra
using SparseArrays

import DielectricBIE: far_field_operator, apply_far!, _dt_parts
using DielectricBIE: DielectricInterface, FlatPanel, SCTLFarField, SCTLNearQuad, num_points, all_weights,
                     _interface_targets_normals, _close_nodes, _close_direct!, _gl_diam

# PVFMM's gradients (Laplace3D_FxdU) lose about 1.5 digits relative to the requested accuracy: ask for 2 more
const _EXTRA_DIGITS = 2
_digits(tol::Real) = clamp(ceil(Int, -log10(tol)) + _EXTRA_DIGITS, 1, 15)

# ---- far field: ParticleFMM with Laplace3D_FxdU, sources = targets = the nodes ----

struct SCTLDTFar
    fmm::ParticleFMM{Laplace3D_FxdU}
    sources::Matrix{Float64}
    weights::Vector{Float64}
    norms::Matrix{Float64}
    close::Vector{Int}   # nodes summed directly instead of by PVFMM (SCTLFarField's `close`)
    F::Matrix{Float64}   # 1 × N density buffer
    G::Matrix{Float64}   # 3 × N gradient buffer
end

function far_field_operator(far::SCTLFarField, interface::DielectricInterface)
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X); n = size(X, 2)
    fmm = ParticleFMM(Laplace3D_FxdU(); digits = _digits(far.tol))
    set_sources!(fmm, X)
    set_targets!(fmm, X)
    return SCTLDTFar(fmm, X, Vector{Float64}(all_weights(interface)), Matrix{Float64}(Nrm), _close_nodes(X, far.close),
                     Matrix{Float64}(undef, 1, n), Matrix{Float64}(undef, 3, n))
end

# SCTL's Laplace3D_FxdU is ∇_x 1/(4π|x-y|) = -(x-y)/(4π r³) (zero at r = 0), so D^T = -n·U.
function apply_far!(Y::AbstractMatrix, f::SCTLDTFar, X::AbstractMatrix)
    n = length(f.weights)
    for k in axes(X, 2)
        @inbounds for i in 1:n
            f.F[1, i] = f.weights[i] * X[i, k]
        end
        f.F[1, f.close] .= 0
        set_density!(f.fmm, f.F)
        fmm_evaluate!(f.G, f.fmm)
        Threads.@threads for i in 1:n
            @inbounds Y[i, k] = -(f.norms[1, i] * f.G[1, i] + f.norms[2, i] * f.G[2, i] + f.norms[3, i] * f.G[3, i])
        end
    end
    return _close_direct!(Y, X, f.sources, f.norms, f.weights, f.close)
end

# ---- whole operator: BoundaryIntegralOp on QuadElemLists ----

struct SCTLDTOp
    op::BoundaryIntegralOp{Laplace3D_FxdU}
    perm::Vector{Int}      # SCTL node k is DielectricBIE node perm[k]
    x::Vector{Float64}
    y::Vector{Float64}
end

const _SCTL_ORDERS = (4, 8, 12, 16, 20)

# The FlatPanels of `interface` as SCTL QuadElemLists, one per quadrature order (ascending), with `perm[k]` the
# DielectricBIE node of the list's node k. The nodes are the same tensor Gauss-Legendre points in the same order
# (checked). `what` names the caller in errors; `orders` restricts the orders (SCTL's element quadratures).
function _sctl_groups(interface::DielectricInterface, what::AbstractString; orders = nothing)
    all(p -> p isa FlatPanel, interface.panels) ||
        throw(ArgumentError("$what needs a FlatPanel interface (use GCQNearQuad or another NearQuadrature " *
                            "for TensorPanel interfaces)"))
    offs = cumsum([0; [num_points(p) for p in interface.panels]])
    groups = NamedTuple{(:order, :ids, :elist, :perm), Tuple{Int, Vector{Int}, QuadElemList, Vector{Int}}}[]
    for q in sort!(unique(p.n_quad for p in interface.panels))
        orders === nothing || q in orders ||
            throw(ArgumentError("$what needs n_quad in $orders, the interface has n_quad = $q"))
        ids = findall(p -> p.n_quad == q, interface.panels)
        el = QuadElemList(q, length(ids)) do e, u, v
            a, b, c, d = interface.panels[ids[e]].corners
            cc = (a .+ b .+ c .+ d) ./ 4
            collect(cc .+ (b .- a) .* (u - 0.5) .+ (d .- a) .* (v - 0.5))
        end
        X = first(node_coord(el))
        perm = Int[]
        for (e, pid) in enumerate(ids)
            p = interface.panels[pid]
            for t in 1:q^2
                k = (e - 1) * q^2 + t
                maximum(abs, X[:, k] .- p.points[t]) <= 1e-12 * max(1.0, maximum(abs, p.points[t])) ||
                    error("panel $pid: nodes do not match SCTL's QuadElemList nodes (not a parallelogram panel?)")
                push!(perm, offs[pid] + t)
            end
        end
        push!(groups, (order = q, ids = ids, elist = el, perm = perm))
    end
    return groups
end

function _dt_parts(far::SCTLFarField, near::SCTLNearQuad, interface::DielectricInterface)
    tol = something(near.tol, far.tol)
    # the operator's FMM loses digits as ParticleFMM's does (6e-7 at accuracy 1e-8 with 288k nodes)
    op = BoundaryIntegralOp(Laplace3D_FxdU(); trg_normal_dot_prod = true, accuracy = tol / 10.0^_EXTRA_DIGITS)
    groups = _sctl_groups(interface, "SCTLNearQuad"; orders = _SCTL_ORDERS)
    # SCTL orders the nodes by list name, then element
    for g in groups
        add_elem_list!(op, g.elist, "q" * lpad(g.order, 2, '0'))
    end
    @assert elem_list_names(op) == ["q" * lpad(g.order, 2, '0') for g in groups]
    perm = reduce(vcat, [g.perm for g in groups])
    _, Nrm = _interface_targets_normals(interface)
    set_target_normal!(op, Nrm[:, perm])
    setup!(op)
    n = length(perm)
    return SCTLDTOp(op, perm, Vector{Float64}(undef, n), Vector{Float64}(undef, n)), nothing
end

# ---- SCTLNearCorrection: every panel (or the panels `sources`) by the GL routine below ----

DielectricBIE.near_corrections(q::DielectricBIE.SCTLNearCorrection, interface::DielectricInterface) =
    DielectricBIE.sctl_gl_corrections(interface, eachindex(interface.panels); tol = q.tol)
DielectricBIE.near_corrections(q::DielectricBIE.SCTLNearCorrection, interface::DielectricInterface,
                               sources::AbstractVector{<:Integer}) =
    DielectricBIE.sctl_gl_corrections(interface, sources; tol = q.tol)

# ---- SCTL elements of panels ----

# A panel's parametrization over [0,1]^2 and the parameters of its nodes in u and in v; nodes are ordered with v
# fastest (as QuadElemList's).
function _panel_param(P::FlatPanel)
    a, b, c, d = P.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    x = (P.gl_xs .+ 1) ./ 2
    return (u, v) -> collect(cc .+ (b .- a) .* (u - 0.5) .+ (d .- a) .* (v - 0.5)), x, x
end
_panel_param(P::DielectricBIE.TensorPanel) = (u, v) -> collect(P.anchor .+ DielectricBIE.tensor_lpoint(P, u, v)), P.ru.x, P.rv.x
_panel_param(P::DielectricBIE.CurvedPanel) = (u, v) -> collect(P.anchor .+ DielectricBIE.curved_lpoint(P, u, v)), P.ru.x, P.rv.x
_panel_param(P) = throw(ArgumentError("SCTL elements: no parametrization for panels of type $(typeof(P)); " *
                                      "define DielectricBIESCTLExt._panel_param for it"))

# ---- the GL routine: SCTL's near and Duffy self rows for GL×GL panels ----
#
# Every GL×GL source panel is one SCTL element over its parameter square, with the panel's map (its geometry is SCTL's
# order-p interpolant of it). When the panel's node count n per direction is an SCTL order (p = n) the element's nodes
# are the panel's; otherwise (flat panels) the element is a proxy of the next order and the density is interpolated to
# its nodes. SCTL's near list (radius sized for the panel's own n-point rule) and near_blocks give the accurate blocks;
# SCTL matches a target to an element node by exact equality, so the panel's own nodes are passed with the element
# nodes' coordinates (equal to rounding) and get the Duffy self rows. The correction is that block minus the node rule.

# the unit normal of node j of a panel
_gl_normal(P::Union{FlatPanel, DielectricBIE.TensorPanel}, j) = P.normal
_gl_normal(P::DielectricBIE.CurvedPanel, j) = P.normals[j]

# SCTL's near radius (QuadElemList::GetFarFieldNodes) of an element with nodes `xn` (in [0, 1]) in each direction
# and constant tangent lengths Lu, Lv, for the accuracy ellipse of the nu- and nv-point Gauss-Legendre rules at tol:
# node (i, j) gets max(d_u(xn[i]) Lu, d_v(xn[j]) Lv), d the parameter distance from the node to the ellipse of
# ρ = (64 / (15 tol))^(1 / 2n) (foci 0 and 1), tol clamped to 2^-53 as SCTL does. For proxies of affine panels: SCTL's
# own radius is that of the element's order, and asking far_field_nodes for the n-point rule's ρ through a smaller
# tol stops at the clamp.
function _sctl_radius(xn, nu::Integer, nv::Integer, Lu::Real, Lv::Real, tol::Real)
    function dist(n)
        t = max(tol, 2.0^-53)
        ρ = (64 / (15 * t))^(1 / (2n))
        a = (ρ - 1 / ρ) / 4; b = (ρ + 1 / ρ) / 4
        return map(xn) do x
            c = 4 * b * (x - 0.5)
            abs(c) <= 1 ? a * sqrt(1 + (a^2 / b^2 - 1) * c^2) : b - abs(x - 0.5)
        end
    end
    du = dist(nu); dv = dist(nv)
    return [max(du[i] * Lu, dv[j] * Lv) for i in eachindex(xn) for j in eachindex(xn)]
end

# the tangent lengths |∂X/∂u|, |∂X/∂v| of an affine panel's map (_panel_param)
_gl_tangents(P::FlatPanel) = (norm(collect(P.corners[2] .- P.corners[1])), norm(collect(P.corners[4] .- P.corners[1])))
_gl_tangents(P::DielectricBIE.TensorPanel) = (P.Lu * norm(collect(P.eu)), P.Lv * norm(collect(P.ev)))

# the SCTL elements of the panels `sources`, one QuadElemList per order (ascending): per element its panel `ids[e]`,
# whether it is the panel (`direct`) or a proxy (`interp`: panel nodes -> element nodes), the sign `sgn` of SCTL's
# normal (∂u × ∂v) against the panel's, and per node its coordinates (columns of X) and near radius
function _gl_elements(interface::DielectricInterface, sources, tol::Real, what::AbstractString)
    tol > 0 || throw(ArgumentError("$what: tol must be positive"))
    ps = interface.panels
    info = map(unique(sources)) do k
        P = ps[k]
        is_gl_panel(P) || throw(ArgumentError("$what: panel $k is not a GL×GL panel (it has a GCQ rule)"))
        check_gl_aspect(P, Int(k))
        f, xu, xv = _panel_param(P)
        nu, nv = length(xu), length(xv)
        max(nu, nv) <= last(_SCTL_ORDERS) ||
            throw(ArgumentError("$what: panel $k has $(max(nu, nv)) nodes per direction, above SCTL's largest order " *
                                "$(last(_SCTL_ORDERS))"))
        direct = nu == nv && nu in _SCTL_ORDERS
        P isa DielectricBIE.CurvedPanel && !direct &&
            throw(ArgumentError("$what: panel $k is a CurvedPanel with $nu × $nv nodes; it needs the same number in " *
                                "u and v, one of SCTL's orders $(_SCTL_ORDERS)"))
        p = direct ? nu : _SCTL_ORDERS[findfirst(>=(max(nu, nv)), _SCTL_ORDERS)]
        (k = Int(k), f = f, xu = xu, xv = xv, p = p, direct = direct)
    end
    groups = []
    for p in sort!(unique([i.p for i in info]))
        mine = filter(i -> i.p == p, info)
        el = QuadElemList(p, length(mine)) do e, u, v
            mine[e].f(u, v)
        end
        X, N = node_coord(el)
        rg(e) = (e - 1) * p^2 + 1:e * p^2
        # the near radius of the panel's own node rule (the far field): SCTL's for an element that is the panel; for a
        # proxy (order p above the panel's nu, nv nodes) SCTL's formula with the nu- and nv-point rules' ellipses
        xn = param_nodes(p)
        radius = zeros(size(X, 2))
        rs = any(i -> i.direct, mine) ? far_field_nodes(el; tol).radius : nothing
        for (e, i) in enumerate(mine)
            radius[rg(e)] .= i.direct ? rs[rg(e)] :
                             _sctl_radius(xn, length(i.xu), length(i.xv), _gl_tangents(ps[i.k])..., tol)
        end
        sgn = zeros(length(mine))
        interp = Vector{Union{Nothing, Matrix{Float64}}}(nothing, length(mine))
        for (e, i) in enumerate(mine)
            P = ps[i.k]
            if i.direct
                d = maximum(t -> norm(X[:, rg(e)[t]] .- collect(P.points[t])), 1:p^2)
                d <= 1e-13 * _gl_diam(P) ||
                    throw(ArgumentError("$what: panel $(i.k): its nodes are not those of SCTL's order-$p element " *
                                        "(off by $d): not Gauss-Legendre nodes?"))
            else
                interp[e] = kron(lagrange_interp_matrix(i.xu, xn), lagrange_interp_matrix(i.xv, xn))
            end
            # (a proxy is flat: its normal is the panel's at any node)
            sgn[e] = sign(sum(dot(N[:, rg(e)[t]], collect(_gl_normal(P, i.direct ? t : 1))) for t in 1:p^2))
        end
        push!(groups, (order = p, ids = [i.k for i in mine], direct = [i.direct for i in mine], elist = el, X = X,
                       radius = radius, sgn = sgn, interp = interp))
    end
    return groups
end

# SCTL's near list of a group's elements for targets Xt, keeping the pairs (e, t) with keep(e, t)
function _gl_near_list(g, Xt, keep = nothing)
    nl = near_list(g.X, g.radius, g.order^2, Xt)
    keep === nothing && return nl
    ptr = [1]; trg = Int[]
    for e in eachindex(g.ids)
        for t in near_targets(nl, e)
            keep(e, t) && push!(trg, t)
        end
        push!(ptr, length(trg) + 1)
    end
    return NearList(ptr, trg, nl.ntrg)
end

# node rules of panel P to targets Xt: the double layer w_j n_j·(y_j - x) / (4π r³) and the single layer
# w_j / (4π r), zero where x = y_j
function _d_node_rule_block(P, Xt::AbstractMatrix)
    B = zeros(size(Xt, 2), num_points(P))
    for j in 1:num_points(P)
        y = P.points[j]; n = _gl_normal(P, j); w = P.weights[j]
        for t in axes(Xt, 2)
            d1, d2, d3 = y[1] - Xt[1, t], y[2] - Xt[2, t], y[3] - Xt[3, t]
            r2 = d1^2 + d2^2 + d3^2
            r2 == 0 && continue
            B[t, j] = w * (n[1] * d1 + n[2] * d2 + n[3] * d3) / (4π * r2 * sqrt(r2))
        end
    end
    return B
end

function _s_node_rule_block(P, Xt::AbstractMatrix)
    B = zeros(size(Xt, 2), num_points(P))
    for j in 1:num_points(P)
        y = P.points[j]; w = P.weights[j]
        for t in axes(Xt, 2)
            r2 = (Xt[1, t] - y[1])^2 + (Xt[2, t] - y[2])^2 + (Xt[3, t] - y[3])^2
            r2 == 0 && continue
            B[t, j] = w / (4π * sqrt(r2))
        end
    end
    return B
end

# the accurate blocks of a group with DielectricBIE's sign (`sign_row(e, t)`: the sign of SCTL's row of target t),
# times the proxies' density maps, minus the node rule `rule(e, ts)`: per element its near targets and correction
function _gl_blocks(g, nl, blocks, sign_row, rule)
    out = Vector{Tuple{Vector{Int}, Matrix{Float64}}}(undef, length(g.ids))
    DielectricBIE._blas_single() do
        Threads.@threads :dynamic for e in eachindex(g.ids)
            ts = collect(near_targets(nl, e))
            A = blocks[e] .* [sign_row(e, t) for t in ts]
            g.interp[e] === nothing || (A = A * g.interp[e])
            out[e] = (ts, A .- rule(e, ts))
        end
    end
    return out
end

# the m × n sparse matrix with the column blocks `cols[k] => (rows, block)` of the panels, in column order
function _gl_assemble(blocks::Dict{Int, Tuple{Vector{Int}, Matrix{Float64}}}, cols, m::Int, n::Int)
    nz = sum(length(B) for (_, B) in values(blocks); init = 0)
    colptr = ones(Int, n + 1); rowval = Vector{Int}(undef, nz); nzval = Vector{Float64}(undef, nz)
    q = 0
    for (k, c) in enumerate(cols)
        ts, B = get(blocks, k, (Int[], zeros(0, length(c))))
        for (jc, j) in enumerate(c)
            rowval[q+1:q+length(ts)] .= ts; nzval[q+1:q+length(ts)] .= view(B, :, jc)
            q += length(ts)
            colptr[j + 1] = q + 1
        end
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function DielectricBIE.sctl_gl_corrections(interface::DielectricInterface, sources::AbstractVector{<:Integer};
                                           tol::Real, kernel::Symbol = :DT)
    kernel in (:DT, :D) || throw(ArgumentError("sctl_gl_corrections: kernel must be :DT or :D, got :$kernel"))
    ps = interface.panels
    n = num_points(interface)
    cols = DielectricBIE.panel_column_ranges(interface)
    groups = _gl_elements(interface, sources, tol, "sctl_gl_corrections")
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
    # the targets for SCTL: the nodes of the (non-proxy) source panels with their elements' exact coordinates
    Xs = copy(X)
    for g in groups, (e, k) in enumerate(g.ids)
        g.direct[e] && (Xs[:, cols[k]] .= g.X[:, (e - 1) * g.order^2 + 1:e * g.order^2])
    end
    out = Dict{Int, Tuple{Vector{Int}, Matrix{Float64}}}()
    quad = ElemQuadrature(kernel == :DT ? Laplace3D_FxdU() : Laplace3D_DxU())
    try
        for g in groups
            # flat panels: no rows for the targets in the panel's plane (with its normal, for Dᵀ), where the kernel
            # vanishes (their own nodes among them)
            plane = map(g.ids) do k
                P = ps[k]
                P isa Union{FlatPanel, DielectricBIE.TensorPanel} || return nothing
                (collect(P.normal), collect(P.points[1]), 1e-12 * _gl_diam(P))
            end
            nl = _gl_near_list(g, Xs, (e, t) -> begin
                plane[e] === nothing && return true
                pn, y1, h = plane[e]
                abs(pn[1] * (X[1, t] - y1[1]) + pn[2] * (X[2, t] - y1[2]) + pn[3] * (X[3, t] - y1[3])) <= h || return true
                kernel == :DT && abs(pn[1] * Nrm[1, t] + pn[2] * Nrm[2, t] + pn[3] * Nrm[3, t]) < 1 - 1e-12
            end)
            blocks = kernel == :DT ? near_blocks(quad, g.elist, nl, Xs; normals = Nrm, tol) :
                                     near_blocks(quad, g.elist, nl, Xs; tol)
            # SCTL's rows are -Dᵀ (Laplace3D_FxdU dotted with the target normal) and -D with the element's normal as
            # source normal; a self row (a panel's own node) is dotted with the element's normal too
            sign_row = kernel == :D ? (e, t) -> -g.sgn[e] :
                       (e, t) -> g.direct[e] && t in cols[g.ids[e]] ? -g.sgn[e] : -1.0
            rule = kernel == :DT ?
                (e, ts) -> DielectricBIE.dt_node_rule_block(interface, g.ids[e], X[:, ts], Nrm[:, ts]) :
                (e, ts) -> _d_node_rule_block(ps[g.ids[e]], X[:, ts])
            for (e, b) in enumerate(_gl_blocks(g, nl, blocks, sign_row, rule))
                out[g.ids[e]] = b
            end
        end
    finally
        close(quad)
        foreach(g -> close(g.elist), groups)
    end
    return _gl_assemble(out, cols, n, n)
end

function DielectricBIE.sctl_gl_potential_corrections(interface::DielectricInterface, sources::AbstractVector{<:Integer},
                                                     targets::AbstractMatrix; tol::Real)
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × Nt"))
    ps = interface.panels
    n = num_points(interface)
    cols = DielectricBIE.panel_column_ranges(interface)
    what = "sctl_gl_potential_corrections"
    groups = _gl_elements(interface, sources, tol, what)
    X = Matrix{Float64}(targets)
    nt = size(X, 2)
    # a target within 1e-13 of the panel's size of a node: the element node's exact coordinates (Duffy self row), or
    # an error on a proxy, whose nodes are not the panel's
    Xs = copy(X)
    if nt > 0
        tree = DielectricBIE.NearestNeighbors.KDTree(X)
        for g in groups, (e, k) in enumerate(g.ids)
            P = ps[k]; h = 1e-13 * _gl_diam(P)
            for (j, y) in enumerate(P.points), t in DielectricBIE.NearestNeighbors.inrange(tree, collect(y), h)
                g.direct[e] || throw(ArgumentError("$what: target $t lies on node $j of panel $k, a proxy element " *
                                                   "(its node count is not an SCTL order): SCTL has no self row there"))
                Xs[:, t] .= g.X[:, (e - 1) * g.order^2 + j]
            end
        end
    end
    out = Dict{Int, Tuple{Vector{Int}, Matrix{Float64}}}()
    quad = ElemQuadrature(Laplace3D_FxU())
    try
        for g in groups
            nl = _gl_near_list(g, Xs)
            blocks = near_blocks(quad, g.elist, nl, Xs; tol)
            for (e, b) in enumerate(_gl_blocks(g, nl, blocks, (e, t) -> 1.0,
                                               (e, ts) -> _s_node_rule_block(ps[g.ids[e]], X[:, ts])))
                out[g.ids[e]] = b
            end
        end
    finally
        close(quad)
        foreach(g -> close(g.elist), groups)
    end
    return _gl_assemble(out, cols, nt, n)
end

# ---- SCTL's near and on-surface quadrature for curved (GCQ) panels ----
#
# (SCTLCurvedNearQuad's GL×GL panels are the GL routine's above; this is the GCQ edge panels' quadrature.)
# The u-segments of a GCQ panel (the fine segments of its rule) next to an edge are far longer in v than wide in u:
# down to 2^-K of the panel against a v-length ~ the patch side. SCTL's near-singular rule is not made for such
# elements: on one row of a deep GCQ panel it is off by 1e-8 at aspect ratio 30 and has the wrong sign at 1e6. So:
#   - a segment of aspect ratio (v-length / width) up to _CURVED_ASPECT is one SCTL proxy element (split in u when it
#     is wider than long), built once per panel and evaluated for all its near targets at once;
#   - a thinner segment is integrated row by row: at least its own width h away from the target, by its fine Gauss
#     nodes in u (the rule's interpolation to them, the weight s^pw included) and adaptive quadrature in v, all such
#     segments in one vector integrand; closer, by square SCTL elements (side h) in a window of ±1.5h about the target
#     in v and outside the window as the far ones (1.5h or more away: Gauss in u still converges like 6^-2n). A node of
#     the panel is the centre of the middle square, where it gets the principal value.
# Near an edge two nodes of a panel 1e-7 apart have n·(x - y) ≈ |x - y|²/2 ~ 1e-14, below the rounding of positions
# (~2e-16): for the panel's own nodes x - y is never formed by subtracting positions. Across the edge it is a path
# integral of the tangent y_u (Gauss, once per target and u-line), along it the integral of a Chebyshev series of y_v
# (a divided difference of its antiderivative), and the SCTL elements of a window get coordinates relative to the
# target. Another panel's node is measured from its nearest node of this panel (the offset between the two is a plain
# difference, as exact as the panels' anchors), and the rest the same way.

# the proxy elements interpolate the curved geometry as well as the density: below order 16 the geometry error of a
# panel spanning ~45° of a sphere reaches 1e-4 (order 8), at 16 it is 1e-9
const _CURVED_MIN_ORDER = 16
# the window about the target: squares on each side of the middle one
const _CURVED_WINDOW = 1
# the largest aspect ratio of a proxy element (SCTL: ~1e-12 up to ~30, 1e-8 at 100)
const _CURVED_ASPECT = 4

_curved_diam(P, ua, ub, va, vb) = (c = [collect(DielectricBIE.curved_lpoint(P, u, v)) for (u, v) in
                                         ((ua, va), (ub, va), (ub, vb), (ua, vb))];
                                   maximum(norm(c[i] - c[j]) for i in 1:4 for j in 1:4))

_dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

_tangents(P, u, v) = P.dmap === nothing ? DielectricBIE.curved_frame(P, u, v)[1:2] : P.dmap(u, v)

# the parameters (u, v) of the point of panel P closest to x (panel frame), and the distance: Gauss-Newton from the
# nearest node, clamped to the parameter square (stopped where a map is singular, e.g. a polar chart's pole at a corner)
function _curved_closest(P, x)
    nv = length(P.rv.x)
    i = argmin(k -> sum(abs2, x .- P.lpoints[k]), eachindex(P.lpoints))
    u, v = P.ru.x[(i - 1) ÷ nv + 1], P.rv.x[(i - 1) % nv + 1]
    for _ in 1:50
        yu, yv = _tangents(P, u, v)
        d = x .- DielectricBIE.curved_lpoint(P, u, v)
        a, b, c = _dot3(yu, yu), _dot3(yu, yv), _dot3(yv, yv)
        gu, gv = _dot3(yu, d), _dot3(yv, d)
        det = a * c - b^2
        un = clamp(u + (c * gu - b * gv) / det, 0.0, 1.0); vn = clamp(v + (a * gv - b * gu) / det, 0.0, 1.0)
        (isfinite(un) && isfinite(vn) && all(isfinite, _tangents(P, un, vn)[1])) || break
        (un, vn) == (u, v) && break
        u, v = un, vn
    end
    return u, v, sqrt(sum(abs2, x .- DielectricBIE.curved_lpoint(P, u, v)))
end

# breakpoints of the window [v0 - (W + 1/2)Δ, v0 + (W + 1/2)Δ] ∩ [0, 1] into squares of side Δ, the middle one
# centred at v0 (pieces under Δ/4 at the ends merged into their neighbours); [0, 1] in ≈ 1/Δ equal pieces when the
# window would cover most of it
function _curved_window(v0, Δ)
    (2 * _CURVED_WINDOW + 1) * Δ >= 0.5 && return collect(range(0.0, 1.0, max(1, round(Int, 1 / Δ)) + 1))
    b = clamp.(v0 .+ ((-_CURVED_WINDOW - 0.5):(_CURVED_WINDOW + 0.5)) .* Δ, 0.0, 1.0)
    b = unique(b)
    length(b) > 2 && b[2] - b[1] < Δ / 4 && deleteat!(b, 2)
    length(b) > 2 && b[end] - b[end-1] < Δ / 4 && deleteat!(b, length(b) - 1)
    return b
end

# the scales of panel P (physical length per unit u and v: its mean chords across u and v, the measure of
# panel_aspect, so a singular point of the map, e.g. a polar chart's pole, does not matter) and which of its
# u-segments are thin
function _curved_split(P)
    su, sv = DielectricBIE._mean_chords((u, v) -> DielectricBIE.curved_lpoint(P, u, v))
    return [(sb - sa) * su * _CURVED_ASPECT < sv for (sa, sb) in P.ru.segs], su, sv
end

# the parameter rectangles of the proxy elements of the segments that are not thin
# (split in 2 × 2 until the element's interpolated geometry is within _CURVED_GEOM_TOL of its size: the principal
# value takes x ± εn with ε = ε_rel × size, which must lie on the two sides of the element)
function _curved_proxies(P, thin, su, sv, xn)
    rects = NTuple{4, Float64}[]
    for (k, (sa, sb)) in enumerate(P.ru.segs)
        thin[k] && continue
        nus = max(1, ceil(Int, (sb - sa) * su / (_CURVED_ASPECT * sv)))
        ub = range(sa, sb, nus + 1)
        for i in 1:nus
            _curved_refine!(rects, P, (ub[i], ub[i+1], 0.0, 1.0), xn, 0)
        end
    end
    return rects
end

const _CURVED_GEOM_TOL = 1e-12
function _curved_refine!(rects, P, r, xn, depth)
    if depth >= 8 || _curved_geom_error(P, r, xn) <= _CURVED_GEOM_TOL
        push!(rects, r)
    else
        ua, ub, va, vb = r; um = (ua + ub) / 2; vm = (va + vb) / 2
        for q in ((ua, um, va, vm), (um, ub, va, vm), (ua, um, vm, vb), (um, ub, vm, vb))
            _curved_refine!(rects, P, q, xn, depth + 1)
        end
    end
    return rects
end

# the largest distance between the order-p interpolant of the geometry on the parameter rectangle r (through the
# element's nodes) and the map, at points between the nodes, relative to the rectangle's diameter
function _curved_geom_error(P, r, xn)
    ua, ub, va, vb = r; p = length(xn)
    y(s, t) = collect(DielectricBIE.curved_lpoint(P, ua + (ub - ua) * s, va + (vb - va) * t))
    Y = [y(xn[i], xn[j]) for i in 1:p, j in 1:p]
    bw = [1 / prod(xn[j] - xn[k] for k in 1:p if k != j) for j in 1:p]
    ts = [(xn[i] + xn[i+1]) / 2 for i in 1:3:p-1]
    L = DielectricBIE._tensor_lagrange_mat(xn, bw, ts)
    err = 0.0
    for (a, s) in enumerate(ts), (b, t) in enumerate(ts)
        yi = sum(L[i, a] * L[j, b] .* Y[i, j] for i in 1:p, j in 1:p)
        err = max(err, norm(yi .- y(s, t)))
    end
    return err / _curved_diam(P, ua, ub, va, vb)
end

# the accurate D^T block of P's proxy elements (panel frame) for targets Xl with normals Nt; `own[r]` = (u, v) when
# target r is a node of P (principal value on its element)
function _curved_proxy_block(P, rects, Xl, Nt, own, p, xn, quad, lk; tol, ε_rel)
    acc = zeros(size(Xl, 2), num_points(P))
    el = lock(lk) do
        QuadElemList(p, length(rects)) do e, s, t
            ua, ub, va, vb = rects[e]
            collect(DielectricBIE.curved_lpoint(P, ua + (ub - ua) * s, va + (vb - va) * t))
        end
    end
    for (e, (ua, ub, va, vb)) in enumerate(rects)
        M = near_matrix(quad, el, e, Xl; normals = Nt, tol)              # SCTL sign: -D^T
        rows = [r for r in eachindex(own) if own[r] !== nothing && ua <= own[r][1] <= ub && va <= own[r][2] <= vb]
        if !isempty(rows)
            # principal value on the element: both sides at ε and 2ε, Richardson
            ε = ε_rel * _curved_diam(P, ua, ub, va, vb)
            Y = reduce(hcat, [Xl[:, r] .+ s * ε .* Nt[:, r] for r in rows for s in (1, -1, 2, -2)])
            N4 = reduce(hcat, [Nt[:, r] for r in rows for _ in 1:4])
            M4 = near_matrix(quad, el, e, Y; normals = N4, tol)
            for (i, r) in enumerate(rows)
                b = 4 * (i - 1)
                M[r, :] .= M4[b + 1, :] .+ M4[b + 2, :] .- (M4[b + 3, :] .+ M4[b + 4, :]) ./ 2
            end
        end
        acc .-= M * kron(DielectricBIE.tensor_rule_interp(P.ru, ua .+ (ub - ua) .* xn),
                         DielectricBIE.tensor_rule_interp(P.rv, va .+ (vb - va) .* xn))
    end
    lock(() -> close(el), lk)
    return acc
end

# the row of the thin segments of P for a target x with normal n (panel frame); `own` = (u, v) when the target is a
# node of P; su, sv: P's scales; p, xn: the order and nodes of the window elements, lk: the lock for creating them,
# cache: the panel's merged-segment rules and geometry along u-lines
function _curved_thin_row(P, x, n, own, thin, su0, sv0, p, xn, quad, lk, cache; tol, ε_rel)
    ru, rv = P.ru, P.rv
    nu, nv, nz = length(ru.x), length(rv.x), length(ru.z0)
    row = zeros(nu * nv)
    hmax = maximum((sb - sa) for (k, (sa, sb)) in enumerate(ru.segs) if thin[k]) * su0
    # where the target is: for another panel's node, the nearest node settles that it is more than the widest thin
    # segment away (the closest point to within a node gap of it); otherwise Gauss-Newton
    far = false
    if own !== nothing
        u0, v0, d = own[1], own[2], 0.0
        ub0, vb0, w = u0, v0, (0.0, 0.0, 0.0)                   # the base point: the node itself, x - y(base)
    else
        i = argmin(k -> sum(abs2, x .- P.lpoints[k]), eachindex(P.lpoints))
        ub0, vb0, w = P.ru.x[(i - 1) ÷ nv + 1], P.rv.x[(i - 1) % nv + 1], x .- P.lpoints[i]
        dn = sqrt(sum(abs2, x .- P.lpoints[i]))
        gap = maximum(diff([0.0; ru.x; 1.0])) * su0 + maximum(diff([0.0; rv.x; 1.0])) * sv0
        if dn - gap > 2hmax
            far = true
            u0, v0, d = P.ru.x[(i - 1) ÷ nv + 1], P.rv.x[(i - 1) % nv + 1], dn - gap   # d: a lower bound
        else
            u0, v0, d = _curved_closest(P, x)
        end
    end
    yu, yv = _tangents(P, clamp(u0, 1e-8, 1 - 1e-8), clamp(v0, 1e-8, 1 - 1e-8))   # (a corner may be singular)
    su, sv = norm(collect(yu)), norm(collect(yv))
    (isfinite(su) && su > 0 && isfinite(sv) && sv > 0) || ((su, sv) = (su0, sv0))   # at a singular point: the panel scales
    # thin segments: far ones by Gauss in u × adaptive v everywhere, near ones outside their windows
    seg = Vector{Tuple{Int, Float64, Float64}}()                # (segment, window lo, window hi); empty window = far
    rects = NTuple{4, Float64}[]                                # the square SCTL elements of all windows
    for (k, (sa, sb)) in enumerate(ru.segs)
        thin[k] || continue
        h = (sb - sa) * su
        if far || hypot(max(sa - u0, u0 - sb, 0.0) * su, d) >= h
            push!(seg, (k, 1.0, 0.0))
        else
            sa == 0 && ru.pw != 0 && _strip_error()           # the weighted segment s^pw: never near a node
            Δ = h / sv
            vb = _curved_window(v0, Δ)
            nus = max(1, round(Int, Δ))                        # wider than the panel is long: split in u too
            ub = range(sa, sb, nus + 1)
            append!(rects, [(ub[i], ub[i+1], vb[j], vb[j+1]) for i in 1:nus for j in 1:length(vb)-1])
            push!(seg, (k, vb[1], vb[end]))
        end
    end
    if !isempty(rects)
        # coordinates relative to the target (it sits at the origin): y - x = (y - y(base)) - w
        XE = Array{Float64}(undef, 3, p, p, length(rects))
        for (e, (ua_, ub_, va_, vb_)) in enumerate(rects), i in 1:p, j in 1:p
            XE[:, j, i, e] .= _curved_diff(P, (ub0, vb0), (ua_ + (ub_ - ua_) * xn[i], va_ + (vb_ - va_) * xn[j])) .- w
        end
        el = lock(() -> QuadElemList(p, XE), lk)
        X1 = zeros(3, 1); N1 = reshape(collect(n), 3, 1)
        for (e, (ua_, ub_, va_, vb_)) in enumerate(rects)
            Is = kron(DielectricBIE.tensor_rule_interp(ru, ua_ .+ (ub_ - ua_) .* xn),
                      DielectricBIE.tensor_rule_interp(rv, va_ .+ (vb_ - va_) .* xn))
            if own !== nothing && ua_ <= own[1] <= ub_ && va_ <= own[2] <= vb_
                # principal value on the element: both sides at ε and 2ε, Richardson
                ε = ε_rel * _curved_diam(P, ua_, ub_, va_, vb_)
                Y = reduce(hcat, [X1[:, 1] .+ s * ε .* N1[:, 1] for s in (1, -1, 2, -2)])
                M4 = near_matrix(quad, el, e, Y; normals = repeat(N1, 1, 4), tol)
                M = M4[1:1, :] .+ M4[2:2, :] .- (M4[3:3, :] .+ M4[4:4, :]) ./ 2
            else
                M = near_matrix(quad, el, e, X1; normals = N1, tol)
            end
            row .-= vec(M * Is)                                       # SCTL sign: -D^T
        end
        lock(() -> close(el), lk)
    end
    # the leading thin segments below half the target's distance from u = 0 (GCQ): one rule from moments (they are
    # all at least their total width away), not their fine nodes
    m = 0
    if length(ru.segs) > 1
        Dt = far ? d : hypot(u0 * su, d)                   # (far: d bounds the distance to the whole panel)
        while m < length(ru.segs) && thin[m+1] && ru.segs[m+1][2] * su <= Dt / 2
            m += 1
        end
        m < 2 && (m = 0)
    end
    Ug, Cg = m == 0 ? (Float64[], zeros(0, nu)) : get!(() -> _curved_group_rule(ru, m), cache.groups, m)
    # Gauss in u × adaptive v, on the pieces of [0, 1] between v0 and the window ends
    brk = sort(unique(vcat(0.0, 1.0, clamp(v0, 0.0, 1.0), [s[2] for s in seg if s[2] <= s[3]], [s[3] for s in seg if s[2] <= s[3]])))
    ref = _curved_node_scale(P, x, n)                          # scale of the row, for the absolute tolerance
    tb = 2vb0 - 1
    for i in 1:length(brk)-1
        a, b = brk[i], brk[i+1]
        b > a || continue
        ks = [k for (k, lo, hi) in seg if k > m && !(lo <= a && b <= hi)]
        isempty(ks) && m == 0 && continue
        U = vcat(Ug, reduce(vcat, [ru.fx[:, k] for k in ks]; init = Float64[]))
        C = vcat(Cg, reduce(vcat, [ru.fw[:, k] .* ru.P[(k - 1) * nz .+ (1:nz), :] for k in ks]; init = zeros(0, nu)))
        G = [get!(() -> _curved_vgeom(P, u), cache.vgeom, u) for u in U]
        # x - y(u_q, v_b) = w + (y(base) - y(u_q, v_b)), the second across the u-lines (cached per line and base)
        A = [w .+ get!(() -> _curved_diff(P, (U[q], vb0), (ub0, vb0)), cache.adiff, (U[q], ub0, vb0)) for q in eachindex(U)]
        f = v -> begin
            g = zeros(nu)
            t = 2v - 1
            for q in eachindex(U)
                b1, b2, b3, J = _cheb_diff(G[q], tb, t)               # y(u_q, v_b) - y(u_q, v)
                d1, d2, d3 = A[q][1] + b1, A[q][2] + b2, A[q][3] + b3
                r2 = d1^2 + d2^2 + d3^2
                c = J * (n[1] * d1 + n[2] * d2 + n[3] * d3) / (4π * r2 * sqrt(r2))
                @inbounds for a_ in 1:nu
                    g[a_] += c * C[q, a_]
                end
            end
            kron(g, DielectricBIE._tensor_lagrange_mat(rv.z0, rv.bw, [t])[:, 1])
        end
        I, _ = DielectricBIE.HCubature.hquadrature(f, a, b; rtol = tol, atol = 1e-2 * tol * ref, maxevals = 20_000)
        row .+= I
    end
    return row
end

# y(b) - y(a) for parameter points a, b of panel P: ∫ of the tangents along the straight segment, so accurate
# relative to its own size however small. The map is analytic over the parameter square: Gauss with n points is
# exact to rounding for a segment of parameter length λ once λ^(2n) ≲ 1e-16 (2 points below 1e-4, 16 up to 1)
const _DIFF_GL = Dict(n => DielectricBIE.gausslegendre(n) for n in (2, 4, 8, 16))
function _curved_diff(P, a, b)
    du = b[1] - a[1]; dv = b[2] - a[2]
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
        _, yv, _, J = DielectricBIE.curved_frame(P, u, (1 + tj) / 2)
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
    z, w = DielectricBIE.gausslegendre(_CURVED_GROUP_NODES)
    L = DielectricBIE._tensor_lagrange_mat(z, DielectricBIE.gl_barycentric_weights(z, w), 2 .* vec(ru.fx[:, 1:m]) ./ s .- 1)
    C = L * (vec(ru.fw[:, 1:m]) .* ru.P[1:(m * nz), :])
    return s .* (1 .+ z) ./ 2, C
end

_strip_error() = error("SCTLCurvedNearQuad: a target lies within the width of the edge segment [0, 2^-K] of a GCQ panel")

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

function DielectricBIE.near_corrections(q::DielectricBIE.SCTLCurvedNearQuad, interface::DielectricInterface)
    ps = interface.panels
    all(p -> p isa DielectricBIE.CurvedPanel, ps) ||
        throw(ArgumentError("SCTLCurvedNearQuad: the interface must consist of CurvedPanels"))
    any(p -> DielectricBIE.is_gcq(p.rv), ps) && throw(ArgumentError("SCTLCurvedNearQuad: GCQ rules in v are not supported"))
    q.order === nothing || q.order in _SCTL_ORDERS || throw(ArgumentError("SCTLCurvedNearQuad: order must be one of $(_SCTL_ORDERS)"))
    # the GL×GL panels: SCTL's elements of their own order; the GCQ (edge) panels: proxies and thin rows
    gl = findall(is_gl_panel, ps); gcq = findall(!is_gl_panel, ps)
    Cgl = isempty(gl) ? nothing : DielectricBIE.sctl_gl_corrections(interface, gl; tol = q.tol)
    isempty(gcq) && return Cgl
    C = _curved_gcq_corrections(q, interface, gcq)
    return Cgl === nothing ? C : C + Cgl
end

# the columns of the GCQ panels `gcq` of SCTLCurvedNearQuad
function _curved_gcq_corrections(q::DielectricBIE.SCTLCurvedNearQuad, interface::DielectricInterface, gcq)
    ps = interface.panels
    n = num_points(interface)
    offs = cumsum([0; [num_points(P) for P in ps]])
    owner = zeros(Int, n); loc = zeros(Int, n)
    for (k, P) in enumerate(ps), i in 1:num_points(P)
        owner[offs[k] + i] = k; loc[offs[k] + i] = i
    end
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
    tree = DielectricBIE.NearestNeighbors.KDTree(X)
    quad = ElemQuadrature(Laplace3D_FxdU())
    lk = ReentrantLock()
    blocks = Vector{Tuple{Vector{Int}, Matrix{Float64}}}(undef, length(ps))
    Threads.@threads :dynamic for k in gcq
        P = ps[k]; nv = length(P.rv.x)
        nmax = max(length(P.ru.z0), length(P.rv.z0), _CURVED_MIN_ORDER)
        p = q.order === nothing ? _SCTL_ORDERS[findfirst(>=(nmax), _SCTL_ORDERS)] : q.order
        xn = param_nodes(p)
        # near targets: within η × diameter of a node of the panel (and the panel's own nodes)
        diam = _curved_diam(P, 0.0, 1.0, 0.0, 1.0)
        c = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- c)) for y in P.points)
        cand = DielectricBIE.NearestNeighbors.inrange(tree, collect(c), rad + q.η * diam)
        ts = sort(filter(t -> owner[t] == k ||
                              minimum(norm(X[:, t] .- collect(y)) for y in P.points) <= q.η * diam, cand))
        # the targets in the panel's frame, from their own frames, and the parameters of P's own nodes
        xs = [(ps[owner[t]].anchor .- P.anchor) .+ ps[owner[t]].lpoints[loc[t]] for t in ts]
        own = [owner[t] == k ? (P.ru.x[(loc[t] - 1) ÷ nv + 1], P.rv.x[(loc[t] - 1) % nv + 1]) : nothing for t in ts]
        thin, su, sv = _curved_split(P)
        rects = _curved_proxies(P, thin, su, sv, xn)
        acc = isempty(rects) ? zeros(length(ts), num_points(P)) :
              _curved_proxy_block(P, rects, reduce(hcat, [collect(x) for x in xs]), Nrm[:, ts], own, p, xn, quad, lk;
                                  tol = q.tol, ε_rel = q.ε_rel)
        if any(thin)
            # the rows in chunks run as tasks, each with its own cache: an edge panel has a large share of all
            # targets near it, and its rows would otherwise keep one thread busy while the others idle
            @sync for ch in Iterators.partition(eachindex(ts), cld(length(ts), Threads.nthreads()))
                Threads.@spawn begin
                    cache = (groups = Dict{Int, Tuple{Vector{Float64}, Matrix{Float64}}}(),   # merged-segment rules,
                             vgeom = Dict{Float64, Matrix{Float64}}(),                       # geometry along u-lines,
                             adiff = Dict{NTuple{3, Float64}, NTuple{3, Float64}}())          # differences across them
                    for r in ch
                        t = ts[r]
                        acc[r, :] .+= _curved_thin_row(P, xs[r], (Nrm[1, t], Nrm[2, t], Nrm[3, t]), own[r], thin, su, sv,
                                                       p, xn, quad, lk, cache; tol = q.tol, ε_rel = q.ε_rel)
                    end
                end
            end
        end
        blocks[k] = (ts, acc .- DielectricBIE.dt_node_rule_block(interface, k, X[:, ts], Nrm[:, ts]))
    end
    close(quad)
    I = Int[]; J = Int[]; V = Float64[]
    for k in gcq
        ts, B = blocks[k]
        for j in axes(B, 2), (r, t) in enumerate(ts)
            push!(I, t); push!(J, offs[k] + j); push!(V, B[r, j])
        end
    end
    return sparse(I, J, V, n, n)
end

# ---- the surface corrections of cut densities: double and single layers of smooth densities, SCTL elements ----

# Gauss-Legendre elements on the rectangles (normal +e_k: the parametrization's d/du x d/dv), the nodes, normals and
# weights of their node rule, and SCTL's near corrections (accurate minus node rule) of the double layer (source
# normal) and the single layer at the targets
function DielectricBIE.gcq_surface_corrections(rects, targets::AbstractMatrix; order::Int = 8, tol::Real = 1e-10)
    # tangent pairs with t1 x t2 = +e_k
    tg(k) = k == 1 ? (2, 3) : k == 2 ? (3, 1) : (1, 2)
    el = QuadElemList(order, length(rects)) do e, u, v
        k, c, lo, hi, _ = rects[e]
        t1, t2 = DielectricBIE._surf_tang(k)                 # lo, hi are given in increasing axis order
        a1, a2 = tg(k)
        x = zeros(3); x[k] = c
        val = Dict(t1 => (lo[1], hi[1]), t2 => (lo[2], hi[2]))
        x[a1] = val[a1][1] + u * (val[a1][2] - val[a1][1]); x[a2] = val[a2][1] + v * (val[a2][2] - val[a2][1])
        x
    end
    ff = far_field_nodes(el; tol = tol)
    Y = Matrix{Float64}(targets)
    CD, _ = near_correction(Laplace3D_DxU(), el; targets = Y, tol = tol)
    CS, _ = near_correction(Laplace3D_FxU(), el; targets = Y, tol = tol)
    coef = repeat([r[5] for r in rects], inner = order^2)
    return DielectricBIE.GCQSurface(Matrix{Float64}(ff.X), Matrix{Float64}(ff.normal), Vector{Float64}(ff.weights), coef,
                                    sparse(CD), sparse(CS))
end

# ---- off-surface potential: SCTL's BoundaryIntegralOp for the single layer, targets set ----

function DielectricBIE.potential_operator(ev::DielectricBIE.SCTLPotential, interface::DielectricInterface,
                                          targets::AbstractMatrix)
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × Nt"))
    ps = interface.panels
    all(is_gl_panel, ps) ||
        throw(ArgumentError("SCTLPotential needs Gauss-Legendre × Gauss-Legendre panels (is_gl_panel: FlatPanels, " *
                            "GL TensorPanels and CurvedPanels); GCQ edge and corner panels are not"))
    # one SCTL element list per proxy order: the smallest order at least the panel's nodes per direction. The density
    # is interpolated from the panel's nodes to the proxy's (exact for its polynomial), so any n_quad works.
    offs = cumsum([0; [num_points(P) for P in ps]])
    params = [_panel_param(P) for P in ps]
    nper = [max(length(xu), length(xv)) for (_, xu, xv) in params]
    pord = map(nper) do q
        i = findfirst(>=(q), _SCTL_ORDERS)
        i === nothing && throw(ArgumentError("SCTLPotential: panels with $q nodes per direction exceed SCTL's orders"))
        _SCTL_ORDERS[i]
    end
    op = BoundaryIntegralOp(Laplace3D_FxU(); accuracy = ev.tol / 10.0^_EXTRA_DIGITS)
    I = Int[]; J = Int[]; V = Float64[]
    row = 0
    for p in sort!(unique(pord))
        ids = findall(==(p), pord)
        el = QuadElemList(p, length(ids)) do e, u, v
            params[ids[e]][1](u, v)
        end
        xn = param_nodes(p)
        for k in ids
            _, xu, xv = params[k]
            M = kron(lagrange_interp_matrix(xu, xn), lagrange_interp_matrix(xv, xn))   # p^2 × (panel nodes)
            for c in axes(M, 2), r in axes(M, 1)
                v = M[r, c]
                abs(v) > 1e-15 || continue                  # identity rows when the nodes coincide
                push!(I, row + r); push!(J, offs[k] + c); push!(V, v)
            end
            row += p^2
        end
        add_elem_list!(op, el, "q" * lpad(p, 2, '0'))
    end
    Pm = sparse(I, J, V, row, offs[end])                    # interface nodes -> SCTL proxy nodes
    nt = size(targets, 2)
    nt == 0 || set_target_coord!(op, Matrix{Float64}(targets))
    nt == 0 || setup!(op)
    apply = σ -> nt == 0 ? Float64[] : compute_potential(op, Pm * σ)
    return DielectricBIE.LinearMap{Float64}(apply, nt, offs[end])
end

# point charges -> targets (volume sources: right-hand sides, the far part of PrecomputedVolumeField)
function DielectricBIE.point_charge_field(far::SCTLFarField, sources::AbstractMatrix, charges::AbstractVector,
                                          targets::AbstractMatrix; grad::Bool = false, hess::Bool = false)
    (grad && hess) && throw(ArgumentError("ask for the gradient or the Hessian"))
    size(targets, 2) == 0 && return hess ? zeros(6, 0) : grad ? zeros(3, 0) : zeros(0)
    if hess                                    # Laplace3D_Fxd2U: row-major 3x3 per target -> xx, yy, zz, xy, xz, yz
        d = clamp(ceil(Int, -log10(far.tol)) + 5, 1, 15)
        U = fmm_eval(Laplace3D_Fxd2U(), Matrix{Float64}(sources), Vector{Float64}(charges), Matrix{Float64}(targets); digits = d)
        return U[[1, 5, 9, 2, 3, 6], :]
    end
    # PVFMM needs about 5 digits more than requested for a max error (relative to the largest value) below tol on
    # point charges, measured with 1e5 random points: gradients at digits 10 -> 2.5e-6, 12 -> 1.5e-8, 15 -> 7.8e-11;
    # potentials at 12 -> 4e-10 to 3.5e-9, 15 -> 8e-13
    d = clamp(ceil(Int, -log10(far.tol)) + 5, 1, 15)
    U = fmm_eval(grad ? Laplace3D_FxdU() : Laplace3D_FxU(), Matrix{Float64}(sources), Vector{Float64}(charges),
                 Matrix{Float64}(targets); digits = d)
    return grad ? U : vec(U)
end

# single- and double-layer potentials of nd densities on the same sources (layer_field): Laplace3D_FxU for the charges,
# Laplace3D_DxU (source normals) for the dipoles, one tree per kernel, nd evaluations on it
function DielectricBIE.layer_field(far::SCTLFarField, sources::AbstractMatrix, normals::AbstractMatrix,
                                   charges::AbstractMatrix, dipoles::AbstractMatrix, targets::AbstractMatrix)
    nd = size(charges, 1)
    size(targets, 2) == 0 && return zeros(nd, 0)
    d = clamp(ceil(Int, -log10(far.tol)) + 5, 1, 15)
    Xs = Matrix{Float64}(sources); Xn = Matrix{Float64}(normals); Xt = Matrix{Float64}(targets)
    U = zeros(nd, size(Xt, 2))
    for f in 1:nd
        U[f, :] .= vec(fmm_eval(Laplace3D_FxU(), Xs, Vector{Float64}(charges[f, :]), Xt; digits = d)) .+
                   vec(fmm_eval(Laplace3D_DxU(), Xs, Xn, Matrix{Float64}(reshape(dipoles[f, :], 1, :)), Xt; digits = d))
    end
    return U
end

# node-rule single-layer potential, nodes -> targets, by ParticleFMM (for GCQPotential(far = SCTLFarField()))
function DielectricBIE._pottrg_far(far::SCTLFarField, interface::DielectricInterface, targets::AbstractMatrix)
    X, _ = _interface_targets_normals(interface)
    w = Vector{Float64}(all_weights(interface))
    nt = size(targets, 2); n = length(w)
    fmm = ParticleFMM(Laplace3D_FxU(); digits = _digits(far.tol))
    set_sources!(fmm, X); set_targets!(fmm, Matrix{Float64}(targets))
    F = Matrix{Float64}(undef, 1, n); U = Matrix{Float64}(undef, 1, nt)
    return DielectricBIE.LinearMap{Float64}(nt, n) do σ
        F[1, :] .= w .* σ
        set_density!(fmm, F)
        vec(copy(fmm_evaluate!(U, fmm)))
    end
end

function DielectricBIE.sctl_near_list(interface::DielectricInterface; tol::Real = 1e-10, radius = nothing,
                                      targets = nothing)
    X, _ = _interface_targets_normals(interface)
    n = size(X, 2)
    r = if radius === nothing
        rr = Vector{Float64}(undef, n)
        for g in _sctl_groups(interface, "sctl_near_list (without `radius`)")
            rr[g.perm] .= far_field_nodes(g.elist; tol = tol).radius
        end
        rr
    elseif radius isa Real
        fill(Float64(radius), n)
    else
        length(radius) == n || throw(DimensionMismatch("need one radius per node ($n), got $(length(radius))"))
        Vector{Float64}(radius)
    end
    return near_list(X, r, [num_points(p) for p in interface.panels], targets === nothing ? X : targets)
end

function apply_far!(Y::AbstractMatrix, f::SCTLDTOp, X::AbstractMatrix)
    for k in axes(X, 2)
        @inbounds for (s, d) in enumerate(f.perm)
            f.x[s] = X[d, k]
        end
        compute_potential!(f.y, f.op, f.x)
        @inbounds for (s, d) in enumerate(f.perm)
            Y[d, k] = -f.y[s]
        end
    end
    return Y
end

# ---- precompile workload (see src/precompile.jl): the SCTL backends on small problems ----
# libsctljl needs AVX-512 (Illegal instruction otherwise): on other CPUs nothing is run at precompile time
using PrecompileTools: @setup_workload, @compile_workload
_avx512() = Sys.islinux() && occursin("avx512f", read("/proc/cpuinfo", String))
@setup_workload begin
    if _avx512()
        @compile_workload begin
            src = PointSource((0.1, 0.15, 0.2), 1.0)
            # FlatPanel box: the whole D^T by SCTL
            itf = DielectricBIE.single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 4.0, 1.0, Float64)
            dielectric_operator(itf; far = SCTLFarField(tol = 1e-6), near = SCTLNearQuad()) *
                rhs_dielectric_box3d(itf, src, 4.0)
            # GCQ box: SCTL's far field (the default with SCTL loaded) and the point-charge field
            gi = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 0, n_quad = 3,
                                             rule = edge_gcq_rule(10.0, 1.0; tol = 1e-3))
            solve_dielectric_box3d_gcq(dielectric_operator(gi), rhs_dielectric_box3d(gi, src, 10.0), gi; rtol = 1e-6)
            point_charge_field(SCTLFarField(), rand(3, 20), rand(20), rand(3, 5))
            # curved sphere with GCQ edge panels: SCTLCurvedNearQuad
            m = DielectricBIE._precompile_sphere_mesh()
            solve_dielectric_box3d_gcq(dielectric_operator(m.interface; near = SCTLCurvedNearQuad(tol = 1e-6)),
                                       rhs_dielectric_box3d(m.interface, PointSource((0.0, 0.0, 2.0), 1.0), 1.0),
                                       m.interface; rtol = 1e-6)
            # the campaign with SCTL's far field
            mktempdir() do dir
                DielectricBIE._precompile_four_index(DielectricBIE._precompile_campaign(dir), :sctl)
            end
        end
    end
end

end # module
