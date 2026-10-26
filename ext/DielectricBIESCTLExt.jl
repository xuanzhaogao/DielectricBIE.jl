# SCTL.jl backends of the dielectric operator (`using SCTL`):
#   SCTLFarField            node-rule D^T by SCTL's ParticleFMM (PVFMM, OpenMP, no MPI), tree kept across applications;
#   SCTLFarField + SCTLNearQuad   the whole D^T by SCTL's BoundaryIntegralOp on a QuadElemList (FlatPanel interfaces);
#   SCTLNearCorrection, sctl_near_list   SCTL's near correction / classification as Julia data (any far field);
#   SCTLProxyNearQuad       SCTL's quadrature for self-defined panels with polynomial densities (proxy elements), a
#                           fallback quadrature (e.g. GCQ) for the others;
#   SCTLPotential           the off-surface single-layer potential by SCTL's BoundaryIntegralOp.
module DielectricBIESCTLExt

using DielectricBIE
using SCTL
using LinearAlgebra
using SparseArrays

import DielectricBIE: far_field_operator, apply_far!, _dt_parts
using DielectricBIE: DielectricInterface, FlatPanel, SCTLFarField, SCTLNearQuad, num_points, all_weights,
                     _interface_targets_normals

# PVFMM's gradients (Laplace3D_FxdU) lose about 1.5 digits relative to the requested accuracy: ask for 2 more
const _EXTRA_DIGITS = 2
_digits(tol::Real) = clamp(ceil(Int, -log10(tol)) + _EXTRA_DIGITS, 1, 15)

# ---- far field: ParticleFMM with Laplace3D_FxdU, sources = targets = the nodes ----

struct SCTLDTFar
    fmm::ParticleFMM{Laplace3D_FxdU}
    weights::Vector{Float64}
    norms::Matrix{Float64}
    F::Matrix{Float64}   # 1 × N density buffer
    G::Matrix{Float64}   # 3 × N gradient buffer
end

function far_field_operator(far::SCTLFarField, interface::DielectricInterface)
    X, Nrm = _interface_targets_normals(interface)
    n = size(X, 2)
    fmm = ParticleFMM(Laplace3D_FxdU(); digits = _digits(far.tol))
    set_sources!(fmm, X)
    set_targets!(fmm, X)
    return SCTLDTFar(fmm, Vector{Float64}(all_weights(interface)), Matrix{Float64}(Nrm),
                     Matrix{Float64}(undef, 1, n), Matrix{Float64}(undef, 3, n))
end

# SCTL's Laplace3D_FxdU is ∇_x 1/(4π|x-y|) = -(x-y)/(4π r³) (zero at r = 0), so D^T = -n·U.
function apply_far!(Y::AbstractMatrix, f::SCTLDTFar, X::AbstractMatrix)
    n = length(f.weights)
    for k in axes(X, 2)
        @inbounds for i in 1:n
            f.F[1, i] = f.weights[i] * X[i, k]
        end
        set_density!(f.fmm, f.F)
        fmm_evaluate!(f.G, f.fmm)
        Threads.@threads for i in 1:n
            @inbounds Y[i, k] = -(f.norms[1, i] * f.G[1, i] + f.norms[2, i] * f.G[2, i] + f.norms[3, i] * f.G[3, i])
        end
    end
    return Y
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

# ---- near-field pieces: SCTL's classification and quadrature, assembled in Julia ----

# The interface nodes in DielectricBIE order with SCTL's coordinates (equal to rounding; exactly equal, a target on
# a node gets the node's self-interaction row), and the normals.
function _sctl_targets(interface::DielectricInterface, groups)
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X)
    for g in groups
        X[:, g.perm] .= first(node_coord(g.elist))
    end
    return X, Matrix{Float64}(Nrm)
end

function DielectricBIE.near_corrections(q::DielectricBIE.SCTLNearCorrection, interface::DielectricInterface)
    groups = _sctl_groups(interface, "SCTLNearCorrection"; orders = _SCTL_ORDERS)
    Xt, Nt = _sctl_targets(interface, groups)
    n = size(Xt, 2)
    I = Int[]; J = Int[]; V = Float64[]
    for g in groups
        # rows: all interface nodes; columns: the nodes of this order's panels. SCTL's Laplace3D_FxdU dotted with the
        # target normal is -D^T.
        C, _ = SCTL.near_correction(Laplace3D_FxdU(), g.elist; targets = Xt, normals = Nt, tol = q.tol)
        i, j, v = findnz(C)
        append!(I, i); append!(J, g.perm[j]); append!(V, .-v)
    end
    return sparse(I, J, V, n, n)
end

# ---- SCTL's quadrature for self-defined panels, through proxy elements ----

# A panel's parametrization over [0,1]^2 and the parameters of its nodes in u and in v; nodes are ordered with v
# fastest (as QuadElemList's).
function _panel_param(P::FlatPanel)
    a, b, c, d = P.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    x = (P.gl_xs .+ 1) ./ 2
    return (u, v) -> collect(cc .+ (b .- a) .* (u - 0.5) .+ (d .- a) .* (v - 0.5)), x, x
end
_panel_param(P::DielectricBIE.TensorPanel) = (u, v) -> collect(P.anchor .+ DielectricBIE.tensor_lpoint(P, u, v)), P.ru.x, P.rv.x
_panel_param(P) = throw(ArgumentError("SCTLProxyNearQuad: no parametrization for panels of type $(typeof(P)); " *
                                      "define DielectricBIESCTLExt._panel_param for it"))

function DielectricBIE.near_corrections(q::DielectricBIE.SCTLProxyNearQuad, interface::DielectricInterface)
    ps = interface.panels
    n = num_points(interface)
    offs = cumsum([0; [num_points(P) for P in ps]])
    poly = findall(q.polynomial, ps)
    other = setdiff(eachindex(ps), poly)
    I = Int[]; J = Int[]; V = Float64[]
    if !isempty(poly)
        params = [_panel_param(ps[k]) for k in poly]
        nmax = maximum(max(length(xu), length(xv)) for (_, xu, xv) in params)
        p = if q.order === nothing   # the smallest order at which the interpolation to the proxy is exact
            i = findfirst(>=(nmax), _SCTL_ORDERS)
            i === nothing && throw(ArgumentError("SCTLProxyNearQuad: panels with $nmax nodes per direction exceed SCTL's orders $(_SCTL_ORDERS)"))
            _SCTL_ORDERS[i]
        else
            q.order in _SCTL_ORDERS || throw(ArgumentError("SCTLProxyNearQuad: order must be one of $(_SCTL_ORDERS)"))
            q.order >= nmax || @warn "SCTLProxyNearQuad: proxy order $(q.order) below the panels' $nmax nodes per direction: the interpolation is not exact"
            q.order
        end
        el = QuadElemList(p, length(poly)) do e, u, v
            params[e][1](u, v)
        end
        xn = param_nodes(p)
        X, Nrm = _interface_targets_normals(interface)
        X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
        ff = far_field_nodes(el; tol = q.tol)
        nl = near_list(ff.X, ff.radius, p^2, X)
        quad = ElemQuadrature(Laplace3D_FxdU())
        acc = Vector{Matrix{Float64}}(undef, length(poly))
        plain = Vector{Matrix{Float64}}(undef, length(poly))
        Threads.@threads :dynamic for e in eachindex(poly)
            k = poly[e]; P = ps[k]
            _, xu, xv = params[e]
            ts = collect(near_targets(nl, e))
            B = DielectricBIE.dt_node_rule_block(interface, k, X[:, ts], Nrm[:, ts])
            plain[e] = copy(B)
            # D^T vanishes for targets in the panel's plane with the panel's normal (both rules give zero there)
            pn = collect(P.normal); y1 = collect(P.points[1])
            scale = maximum(norm(collect(c) .- y1) for c in P.corners)
            rows = findall(t -> !(abs(dot(pn, X[:, t] .- y1)) <= 1e-12 * scale && abs(dot(pn, Nrm[:, t])) >= 1 - 1e-12), ts)
            if !isempty(rows)
                Interp = kron(lagrange_interp_matrix(xu, xn), lagrange_interp_matrix(xv, xn))   # panel nodes -> proxy nodes
                M = near_matrix(quad, el, e, X[:, ts[rows]]; normals = Nrm[:, ts[rows]], tol = q.tol)  # SCTL sign: -D^T
                B[rows, :] = -M * Interp
            end
            acc[e] = B
        end
        close(quad)
        Cp = assemble_near_correction(nl, acc, [num_points(ps[k]) for k in poly]; subtract = plain)
        cols = reduce(vcat, [collect(offs[k]+1:offs[k+1]) for k in poly])
        i, j, v = findnz(Cp)
        append!(I, i); append!(J, cols[j]); append!(V, v)
    end
    if !isempty(other)
        i, j, v = findnz(DielectricBIE.near_corrections(q.fallback, interface, other))
        append!(I, i); append!(J, j); append!(V, v)
    end
    return sparse(I, J, V, n, n)
end

# ---- SCTL's near and on-surface quadrature for curved (GCQ) panels, through proxy elements per fine segment ----

# the proxy elements interpolate the curved geometry as well as the density: below order 16 the geometry error of a
# panel spanning ~45° of a sphere reaches 1e-4 (order 8), at 16 it is 1e-9
const _CURVED_MIN_ORDER = 16

# parameter rectangles (ua, ub, va, vb) of a panel's proxy elements: its u-segments but the GCQ segment touching the
# edge (index k0, summed directly), times its v-segments
function _curved_subelements(P)
    ru, rv = P.ru, P.rv
    DielectricBIE.is_gcq(rv) && throw(ArgumentError("SCTLCurvedNearQuad: GCQ rules in v are not supported"))
    k0 = DielectricBIE.is_gcq(ru) ? findfirst(sg -> sg[1] == 0, ru.segs) : nothing
    subs = [(ua, ub, va, vb) for (k, (ua, ub)) in enumerate(ru.segs) if k != k0 for (va, vb) in rv.segs]
    return subs, k0
end

_curved_diam(P, ua, ub, va, vb) = (c = [collect(DielectricBIE.curved_lpoint(P, u, v)) for (u, v) in
                                         ((ua, va), (ub, va), (ub, vb), (ua, vb))];
                                   maximum(norm(c[i] - c[j]) for i in 1:4 for j in 1:4))

# D^T from the edge strip [0, 2^-K] × [0, 1] of a GCQ panel to targets Xl (panel frame). In u the strip is far
# narrower than any target's distance: the Gauss-Jacobi points and weights of the rule's first segment (the weight
# s^pw included), v = σ / s^pw from the rule's interpolation matrix. In v it is as long as the panel, and a target
# near the edge sees a peak of the width of its distance: adaptive quadrature against the Lagrange basis of the
# v-nodes.
function _curved_strip_block(P, k0, Xl, Nt; tol)
    ru, rv = P.ru, P.rv
    nz = length(ru.z0); nv = length(rv.x)
    V = ru.P[(k0 - 1) * nz .+ (1:nz), :]
    B = zeros(size(Xl, 2), num_points(P))
    for i in 1:nz
        u = ru.fx[i, k0]
        for t in axes(Xl, 2)
            x = (Xl[1, t], Xl[2, t], Xl[3, t]); n = (Nt[1, t], Nt[2, t], Nt[3, t])
            f = v -> begin
                y = DielectricBIE.curved_lpoint(P, u, v)
                d = x .- y; r = sqrt(sum(abs2, d))
                (DielectricBIE.curved_frame(P, u, v)[4] * sum(n .* d) / (4π * r^3)) .*
                    DielectricBIE._tensor_lagrange_mat(rv.z0, rv.bw, [2v - 1])[:, 1]
            end
            Iv, _ = DielectricBIE.HCubature.hquadrature(f, 0.0, 1.0; rtol = tol, atol = 0.0, initdiv = nv)
            for j in 1:nv, a in axes(V, 2)
                B[t, (a - 1) * nv + j] += ru.fw[i, k0] * Iv[j] * V[i, a]
            end
        end
    end
    return B
end

function DielectricBIE.near_corrections(q::DielectricBIE.SCTLCurvedNearQuad, interface::DielectricInterface)
    ps = interface.panels
    all(p -> p isa DielectricBIE.CurvedPanel, ps) ||
        throw(ArgumentError("SCTLCurvedNearQuad: the interface must consist of CurvedPanels"))
    n = num_points(interface)
    offs = cumsum([0; [num_points(P) for P in ps]])
    owner = zeros(Int, n); loc = zeros(Int, n)
    for (k, P) in enumerate(ps), i in 1:num_points(P)
        owner[offs[k] + i] = k; loc[offs[k] + i] = i
    end
    X, Nrm = _interface_targets_normals(interface)
    X = Matrix{Float64}(X); Nrm = Matrix{Float64}(Nrm)
    tree = DielectricBIE.NearestNeighbors.KDTree(X)
    # proxy elements in each panel's frame (built serially: they are SCTL objects)
    prox = map(ps) do P
        subs, k0 = _curved_subelements(P)
        nmax = max(length(P.ru.z0), length(P.rv.z0), _CURVED_MIN_ORDER)
        p = q.order === nothing ? _SCTL_ORDERS[findfirst(>=(nmax), _SCTL_ORDERS)] : q.order
        q.order === nothing || p in _SCTL_ORDERS || throw(ArgumentError("SCTLCurvedNearQuad: order must be one of $(_SCTL_ORDERS)"))
        xn = param_nodes(p)
        el = QuadElemList(p, length(subs)) do e, u, v
            ua, ub, va, vb = subs[e]
            collect(DielectricBIE.curved_lpoint(P, ua + (ub - ua) * u, va + (vb - va) * v))
        end
        Is = [kron(DielectricBIE.tensor_rule_interp(P.ru, ua .+ (ub - ua) .* xn),
                   DielectricBIE.tensor_rule_interp(P.rv, va .+ (vb - va) .* xn)) for (ua, ub, va, vb) in subs]
        (el = el, subs = subs, k0 = k0, Is = Is, diam = _curved_diam(P, 0.0, 1.0, 0.0, 1.0))
    end
    quad = ElemQuadrature(Laplace3D_FxdU())
    blocks = Vector{Tuple{Vector{Int}, Matrix{Float64}}}(undef, length(ps))
    Threads.@threads :dynamic for k in eachindex(ps)
        P = ps[k]; pr = prox[k]; nv = length(P.rv.x)
        # near targets: within η × diameter of a node of the panel (and the panel's own nodes)
        c = reduce(.+, P.points) ./ num_points(P)
        rad = maximum(norm(collect(y .- c)) for y in P.points)
        cand = DielectricBIE.NearestNeighbors.inrange(tree, collect(c), rad + q.η * pr.diam)
        ts = sort(filter(t -> owner[t] == k ||
                              minimum(norm(X[:, t] .- collect(y)) for y in P.points) <= q.η * pr.diam, cand))
        # target positions in the panel's frame, from the targets' own frames
        Xl = reduce(hcat, [collect((ps[owner[t]].anchor .- P.anchor) .+ ps[owner[t]].lpoints[loc[t]]) for t in ts])
        Nt = Nrm[:, ts]
        # the element of each of the panel's own nodes
        selfe = Dict{Int, Vector{Int}}()                      # element -> rows of targets on it
        for (r, t) in enumerate(ts)
            owner[t] == k || continue
            u = P.ru.x[(loc[t] - 1) ÷ nv + 1]; v = P.rv.x[(loc[t] - 1) % nv + 1]
            e = findfirst(s -> s[1] <= u <= s[2] && s[3] <= v <= s[4], pr.subs)
            e === nothing && error("SCTLCurvedNearQuad: node ($u, $v) of panel $k lies on no proxy element")
            push!(get!(selfe, e, Int[]), r)
        end
        acc = zeros(length(ts), num_points(P))
        for (e, (ua, ub, va, vb)) in enumerate(pr.subs)
            M = near_matrix(quad, pr.el, e, Xl; normals = Nt, tol = q.tol)       # SCTL sign: -D^T
            rows = get(selfe, e, Int[])
            if !isempty(rows)
                # principal value on the element: both sides at ε and 2ε, Richardson
                ε = q.ε_rel * _curved_diam(P, ua, ub, va, vb)
                Y = reduce(hcat, [Xl[:, r] .+ s * ε .* Nt[:, r] for r in rows for s in (1, -1, 2, -2)])
                N4 = reduce(hcat, [Nt[:, r] for r in rows for _ in 1:4])
                M4 = near_matrix(quad, pr.el, e, Y; normals = N4, tol = q.tol)
                for (i, r) in enumerate(rows)
                    b = 4 * (i - 1)
                    M[r, :] .= M4[b + 1, :] .+ M4[b + 2, :] .- (M4[b + 3, :] .+ M4[b + 4, :]) ./ 2
                end
            end
            acc .-= M * pr.Is[e]
        end
        pr.k0 === nothing || (acc .+= _curved_strip_block(P, pr.k0, Xl, Nt; tol = q.tol))
        blocks[k] = (ts, acc .- DielectricBIE.dt_node_rule_block(interface, k, X[:, ts], Nt))
    end
    close(quad)
    I = Int[]; J = Int[]; V = Float64[]
    for (k, (ts, B)) in enumerate(blocks), j in axes(B, 2), (r, t) in enumerate(ts)
        push!(I, t); push!(J, offs[k] + j); push!(V, B[r, j])
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
    all(DielectricBIE.default_polynomial_panel, ps) ||
        throw(ArgumentError("SCTLPotential needs panels with polynomial densities (FlatPanels, Gauss-Legendre " *
                            "TensorPanels); GCQ edge panels are not"))
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

end # module
