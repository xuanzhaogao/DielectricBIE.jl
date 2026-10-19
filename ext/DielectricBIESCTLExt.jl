# SCTL.jl backends of the dielectric operator (`using SCTL`):
#   SCTLFarField            node-rule D^T by SCTL's ParticleFMM (PVFMM, OpenMP, no MPI), tree kept across applications;
#   SCTLFarField + SCTLNearQuad   the whole D^T by SCTL's BoundaryIntegralOp on a QuadElemList (FlatPanel interfaces);
#   SCTLNearCorrection, sctl_near_list   SCTL's near correction / classification as Julia data (any far field).
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
