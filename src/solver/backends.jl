# Dielectric BIE operator assembled from interchangeable parts,
#
#     A σ = τ σ + D^T_nodes σ + C σ,
#
# with τ the per-panel contrast diagonal, D^T_nodes the adjoint double layer by the smooth node rule (quadrature
# points and weights of the interface) evaluated by a far-field backend (`FarField`), and C the sparse near
# correction (exact minus node rule) of a near quadrature (`NearQuadrature`). The DielectricBIE sign convention is
# that of `laplace3d_DT`: D^T[i, j] = w_j n_i·(x_i - y_j) / (4π |x_i - y_j|³).

# ---------------------------------------------------------------------------------------------------------------
# far field
# ---------------------------------------------------------------------------------------------------------------

"""
    FarField

Backend that evaluates the node-rule adjoint double layer `D^T_nodes X` (for a vector or an `N × K` block `X`).
Available: [`FMM3DFarField`](@ref) (default) and [`SCTLFarField`](@ref) (needs `using SCTL`). A backend
implements `far_field_operator(far, interface)`, returning an object `F` with
`apply_far!(Y, F, X)` (`Y = D^T_nodes X`, both `N × K`).
"""
abstract type FarField end

"""
    FMM3DFarField(tol; close = 0.0)

Far field by FMM3D's `lfmm3d` at tolerance `tol`; a block of `K` densities is one vectorized FMM (`nd = K`).
FMM3D's direct sum between two points a distance r apart carries a relative error ~1e-17/r (its tree shifts and
scales the coordinates), far above rounding for the node pairs a GCQ rule puts on both sides of an edge (r ~ 1e-13:
1e-4), where the near correction must cancel that term to many digits. With `close` > 0 the nodes that have another
node closer than `close` are no FMM sources: their contributions are summed directly, with the same arithmetic as the
node-rule term the near corrections subtract (`dt_node_rule_block`), so the cancellation is exact up to rounding.
"""
struct FMM3DFarField <: FarField
    tol::Float64
    close::Float64
end
FMM3DFarField(tol::Real; close::Real = 0.0) = FMM3DFarField(Float64(tol), Float64(close))

"""
    SCTLFarField(; tol = 1e-10, close = nothing)

Far field by SCTL.jl's `ParticleFMM` (PVFMM, OpenMP, no MPI) at relative accuracy `tol`, keeping the octree across
applications; with fewer than 40000 nodes it is an exact direct sum. PVFMM's gradients lose about 1.5 digits, so it
runs with `-log10(tol) + 2` digits (at most 15; PVFMM reaches about 1e-11 at best). Needs `using SCTL` (the
`DielectricBIESCTLExt` extension). A block of `K` densities is `K` evaluations on the same tree.

Like FMM3D (see [`FMM3DFarField`](@ref)), PVFMM's sum between two points a distance r apart is off by ~1e-16/r
relative (3e-5 at r = 1e-11), so the nodes with a neighbour closer than `close` are summed directly instead, with
the arithmetic of the node-rule term the near corrections subtract. `close = nothing` (default): 1e-4 times the
diameter of the node set; `close = 0`: everything through PVFMM.
"""
struct SCTLFarField <: FarField
    tol::Float64
    close::Union{Nothing, Float64}
end
SCTLFarField(; tol::Real = 1e-10, close::Union{Nothing, Real} = nothing) =
    SCTLFarField(Float64(tol), close === nothing ? nothing : Float64(close))
SCTLFarField(tol::Real) = SCTLFarField(; tol)

"""
    point_charge_field(far::FarField, sources, charges, targets; grad = false, hess = false)

Free-space Laplace field of point charges, kernel `1/(4π r)`: the potential `Σ_s q_s / (4π |x_t - x_s|)` at the
targets (a vector), its gradient (`grad = true`, `3 × Nt`) or its Hessian (`hess = true`, `6 × Nt`, rows ∂xx, ∂yy,
∂zz, ∂xy, ∂xz, ∂yz). `sources` and `targets` are `3 × N` matrices. Used
for volume sources (their quadrature points and masses) away from the source support: right-hand sides and
the far part of `PrecomputedVolumeField`. `FMM3DFarField` uses `lfmm3d`; `SCTLFarField` SCTL's `ParticleFMM`
(after `using SCTL`), whose cost is set mostly by the points actually present rather than by a fixed source tree.
"""
function point_charge_field(far::FMM3DFarField, sources::AbstractMatrix, charges::AbstractVector,
                            targets::AbstractMatrix; grad::Bool = false, hess::Bool = false)
    (grad && hess) && throw(ArgumentError("ask for the gradient or the Hessian"))
    size(targets, 2) == 0 && return hess ? zeros(6, 0) : grad ? zeros(3, 0) : zeros(0)
    v = lfmm3d(far.tol, Matrix{Float64}(sources); charges = Vector{Float64}(charges),
               targets = Matrix{Float64}(targets), pgt = hess ? 3 : grad ? 2 : 1)
    return hess ? v.hesstarg ./ (4π) : grad ? v.gradtarg ./ (4π) : v.pottarg ./ (4π)   # FMM3D's kernel is 1/r
end
point_charge_field(far::SCTLFarField, sources, charges, targets; grad::Bool = false, hess::Bool = false) =
    error("SCTLFarField needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    layer_field(far::FarField, sources, normals, charges, dipoles, targets) -> nd × Nt

Free-space Laplace potentials of `nd` densities of point charges and normal dipoles on the same sources:
row f is `Σ_s charges[f, s] / (4π r) + dipoles[f, s] n_s · (x_t - x_s) / (4π r³)`, `r = |x_t - x_s|` (the node rule of
single and double layers with source normals `normals`, 3 × Ns). Coincident points are skipped.
"""
function layer_field(far::FMM3DFarField, sources::AbstractMatrix, normals::AbstractMatrix, charges::AbstractMatrix,
                     dipoles::AbstractMatrix, targets::AbstractMatrix)
    nd, ns = size(charges)
    size(targets, 2) == 0 && return zeros(nd, 0)
    dip = zeros(nd, 3, ns)
    for i in 1:ns, f in 1:nd, d in 1:3
        dip[f, d, i] = dipoles[f, i] * normals[d, i]
    end
    v = lfmm3d(far.tol, Matrix{Float64}(sources); charges = Matrix{Float64}(charges), dipvecs = dip,
               targets = Matrix{Float64}(targets), pgt = 1, nd = nd)
    return reshape(v.pottarg, nd, :) ./ (4π)                  # FMM3D: q/r and d·(t - s)/r³
end
layer_field(far::SCTLFarField, sources, normals, charges, dipoles, targets) =
    error("SCTLFarField needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    far_field_operator(far::FarField, interface) -> F

Prepare the far-field evaluation of `D^T_nodes` on the nodes of `interface`; apply it with `apply_far!(Y, F, X)`.
"""
function far_field_operator end

"""
    apply_far!(Y, F, X) -> Y

`Y = D^T_nodes X` for `N × K` blocks (`F` from [`far_field_operator`](@ref)).
"""
function apply_far! end

far_field_operator(far::SCTLFarField, interface) =
    error("SCTLFarField needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

struct FMM3DDTFar
    sources::Matrix{Float64}
    weights::Vector{Float64}
    norms::Matrix{Float64}
    tol::Float64
    close::Vector{Int}          # nodes summed directly instead of by the FMM (FMM3DFarField's `close`)
end

function far_field_operator(far::FMM3DFarField, interface::DielectricInterface)
    sources, norms = _interface_targets_normals(interface)
    S = Matrix{Float64}(sources)
    return FMM3DDTFar(S, Vector{Float64}(all_weights(interface)), Matrix{Float64}(norms), far.tol,
                      _close_nodes(S, far.close))
end

# the nodes (columns of X) with another node closer than `close` (none for close <= 0); `nothing`: 1e-4 times the
# diameter of the node set
function _close_nodes(X::AbstractMatrix, close)
    c = close === nothing ? 1e-4 * norm(vec(maximum(X, dims = 2) .- minimum(X, dims = 2))) : close
    c > 0 || return Int[]
    return [i for (i, nb) in enumerate(inrange(KDTree(X), X, c)) if length(nb) > 1]
end

# Y += the node-rule D^T of the `close` nodes as sources, at every node (the arithmetic of dt_node_rule_block)
function _close_direct!(Y::AbstractMatrix, X::AbstractMatrix, sources::AbstractMatrix, norms::AbstractMatrix,
                        weights::AbstractVector, close::AbstractVector{Int})
    isempty(close) && return Y
    Threads.@threads for i in axes(sources, 2)
        @inbounds begin
            x1 = sources[1, i]; x2 = sources[2, i]; x3 = sources[3, i]
            n1 = norms[1, i]; n2 = norms[2, i]; n3 = norms[3, i]
            for j in close
                d1, d2, d3 = x1 - sources[1, j], x2 - sources[2, j], x3 - sources[3, j]
                r2 = d1^2 + d2^2 + d3^2
                r2 == 0 && continue
                b = weights[j] * (n1 * d1 + n2 * d2 + n3 * d3) / (4π * r2 * sqrt(r2))
                for k in axes(X, 2)
                    Y[i, k] += b * X[j, k]
                end
            end
        end
    end
    return Y
end

function apply_far!(Y::AbstractMatrix, F::FMM3DDTFar, X::AbstractMatrix)
    n = length(F.weights); K = size(X, 2)
    charges = Matrix{Float64}(undef, K, n)
    Threads.@threads for i in 1:n
        @inbounds for k in 1:K
            charges[k, i] = F.weights[i] * X[i, k]
        end
    end
    charges[:, F.close] .= 0
    vals = lfmm3d(F.tol, F.sources, charges = charges, pg = 2, nd = K)
    grad = reshape(vals.grad, K, 3, n)    # (K, 3, n); also for nd == 1
    Threads.@threads for i in 1:n
        @inbounds begin
            n1 = F.norms[1, i]; n2 = F.norms[2, i]; n3 = F.norms[3, i]
            for k in 1:K
                Y[i, k] = -(n1 * grad[k, 1, i] + n2 * grad[k, 2, i] + n3 * grad[k, 3, i]) / (4π)
            end
        end
    end
    return _close_direct!(Y, X, F.sources, F.norms, F.weights, F.close)
end

# ---------------------------------------------------------------------------------------------------------------
# near quadrature
# ---------------------------------------------------------------------------------------------------------------

"""
    NearQuadrature

Near-field quadrature of the dielectric operator: [`near_corrections`](@ref)`(q, interface)` returns the sparse
`N × N` correction `C` = (accurate `D^T`) - (node-rule `D^T`), so that `D^T_nodes + C` is the corrected operator.

Built in:
- [`SCTLNearCorrection`](@ref): SCTL's near and singular quadrature of Gauss-Legendre × Gauss-Legendre panels
  ([`sctl_gl_corrections`](@ref)) as a sparse correction, with any far field (`FlatPanel` interfaces, default;
  `using SCTL`);
- [`GCQNearQuad`](@ref): the product-integration rules of the GCQ edge and corner panels, SCTL's for the
  Gauss-Legendre ones (`TensorPanel` and `CurvedPanel` interfaces, default);
- [`SCTLNearQuad`](@ref): SCTL's singular quadrature of `QuadElemList` as its whole operator (needs `SCTLFarField`
  and `using SCTL`);
- [`CorrectionMatrix`](@ref) (a precomputed `C`) and [`FunctionNearQuad`](@ref) (`C = f(interface)`).

A self-defined quadrature is a subtype with a `near_corrections` method:

```julia
struct MyQuad <: DielectricBIE.NearQuadrature
    tol::Float64
end
DielectricBIE.near_corrections(q::MyQuad, interface) = my_sparse_correction(interface, q.tol)
```

and is used as `dielectric_operator(interface; near = MyQuad(1e-10))`. Rows of `C` are targets, columns sources,
in the node order of `interface`; entries follow `laplace3d_DT`'s sign convention.
"""
abstract type NearQuadrature end

"""
    near_corrections(q::NearQuadrature, interface) -> SparseMatrixCSC
    near_corrections(q::NearQuadrature, interface, sources) -> SparseMatrixCSC

The sparse near correction `C` of `q` (see [`NearQuadrature`](@ref)). With `sources` (panel indices) only the
columns of those source panels, the others zero: what a combination of quadratures asks for. The default computes
all columns and drops the others; a quadrature that can do less work implements the three-argument method (as
[`GCQNearQuad`](@ref) and [`SCTLNearCorrection`](@ref) do).
"""
function near_corrections end

near_corrections(q::NearQuadrature, interface::DielectricInterface, sources::AbstractVector{<:Integer}) =
    _keep_source_columns(near_corrections(q, interface), interface, sources)

# zero all columns but those of the panels `sources`
function _keep_source_columns(C::SparseMatrixCSC, interface::DielectricInterface, sources)
    offs = cumsum([0; [num_points(p) for p in interface.panels]])
    keep = falses(size(C, 2))
    for k in sources
        keep[offs[k]+1:offs[k+1]] .= true
    end
    I, J, V = findnz(C)
    m = keep[J]
    return sparse(I[m], J[m], V[m], size(C)...)
end

"""
    GCQNearQuad(; quad_tol = 1e-12, η_gcq = 2.0, reuse = true, gcq_far = :max)

Near corrections of GCQ interfaces, `TensorPanel` and `CurvedPanel` ones: DielectricBIE's product integration for
the GCQ edge and corner panels, in Julia, and SCTL's quadrature ([`sctl_gl_corrections`](@ref) at `quad_tol`) for the
Gauss-Legendre × Gauss-Legendre panels ([`is_gl_panel`](@ref)), which needs `using SCTL` when the interface has any.
`η_gcq`, `reuse` and `gcq_far` concern the GCQ panels.

- `TensorPanel`s: [`laplace3d_DT_tensor_corrections`](@ref). A pair is near when `tensor_isfar` rejects it at
  `quad_tol`; `gcq_far = :direction` measures the near distance of edge panels against their GCQ direction instead
  of their larger side; `reuse` computes equal rows (the same relative position) once.
- `CurvedPanel`s (GCQ rule in u, the edge at u = 0): the rows of [`gcq_curved_row`](@ref) on the panel's exact map
  minus the node rule. A target is near a GCQ panel when it lies within `η_gcq` × L of one of its nodes, L the
  panel's larger mean chord (across u or across v; with `gcq_far = :direction` the one across u), and the panel's
  own nodes always are: they get its on-surface rows. `quad_tol` goes to the GL×GL panels only, and `reuse` is
  ignored. The GL×GL panels are SCTL elements of their own order, whose geometry is SCTL's interpolant of the map
  (see [`sctl_gl_corrections`](@ref)).
"""
Base.@kwdef struct GCQNearQuad <: NearQuadrature
    quad_tol::Float64 = 1e-12
    η_gcq::Float64 = 2.0
    reuse::Bool = true
    gcq_far::Symbol = :max
end

near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:TensorPanel}) =
    SparseMatrixCSC{Float64, Int}(first(laplace3d_DT_tensor_corrections(interface;
        reuse = q.reuse, quad_tol = q.quad_tol, η_gcq = q.η_gcq, gcq_far = q.gcq_far)))
near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:TensorPanel}, sources::AbstractVector{<:Integer}) =
    SparseMatrixCSC{Float64, Int}(first(laplace3d_DT_tensor_corrections(interface;
        reuse = q.reuse, quad_tol = q.quad_tol, η_gcq = q.η_gcq, gcq_far = q.gcq_far, sources = sources)))

near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:CurvedPanel}) =
    near_corrections(q, interface, eachindex(interface.panels))

function near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:CurvedPanel},
                          sources::AbstractVector{<:Integer})
    ps = interface.panels
    q.gcq_far in (:max, :direction) || throw(ArgumentError("gcq_far must be :max or :direction, got :$(q.gcq_far)"))
    gcq = filter(k -> !is_gl_panel(ps[k]), unique(sources))
    any(k -> is_gcq(ps[k].rv), gcq) &&
        throw(ArgumentError("GCQNearQuad: a CurvedPanel's GCQ rule must be in u (GCQ rules in v are not supported)"))
    # the GL×GL source panels: SCTL (first, so that a missing SCTL fails before the GCQ rows are computed)
    gl = intersect(findall(is_gl_panel, ps), sources)
    Cgl = isempty(gl) ? nothing : sctl_gl_corrections(interface, gl; tol = q.quad_tol)
    isempty(gcq) && return Cgl === nothing ? spzeros(num_points(interface), num_points(interface)) : Cgl
    C = _gcq_curved_corrections(interface, gcq; η_gcq = q.η_gcq, gcq_far = q.gcq_far)
    return Cgl === nothing ? C : _add_corrections(C, Cgl)
end

# The columns of the curved GCQ panels `gcq` (panel indices) of GCQNearQuad: per panel P, gcq_curved_row's Dᵀ rows
# minus the node rule for its near targets, those within η_gcq × L of one of its nodes (L its larger mean chord, or the
# one across u with gcq_far = :direction) and its own nodes. Its own nodes are targets on the panel (`on`, their
# parameters); another panel's node is taken into P's frame from its own, (anchor_t - anchor_P) + lpoint_t, never by
# subtracting absolute positions. Threaded over the panels, and the rows of one panel over tasks: an edge panel has a
# large share of all targets near it, and its own rows next to the edge cost the most, so the rows are dealt to the
# tasks in turn rather than in blocks.
function _gcq_curved_corrections(interface::DielectricInterface{<:CurvedPanel}, gcq; η_gcq::Real, gcq_far::Symbol)
    ps = interface.panels
    n = num_points(interface)
    offs = cumsum([0; num_points.(ps)])
    owner = zeros(Int, n); loc = zeros(Int, n)
    for (k, P) in enumerate(ps), i in 1:num_points(P)
        owner[offs[k] + i] = k; loc[offs[k] + i] = i
    end
    X0, N0 = _interface_targets_normals(interface)
    X = Matrix{Float64}(X0); Nrm = Matrix{Float64}(N0)      # (new names: closures capture X and Nrm, unboxed)
    tree = KDTree(X)
    foreach(k -> (_tensor_near_tables(ps[k].ru); _tensor_near_tables(ps[k].rv)), gcq)
    blocks = Vector{Tuple{Vector{Int}, Matrix{Float64}}}(undef, length(ps))
    _blas_single() do
    Threads.@threads :dynamic for k in gcq
        P = ps[k]; nv = length(P.rv.x)
        su, sv = _mean_chords((u, v) -> curved_lpoint(P, u, v))
        R = η_gcq * (gcq_far === :max ? max(su, sv) : su)
        # the candidates within R of the ball around P's nodes, then the distance to P's nearest node
        Y = reduce(hcat, [collect(y) for y in P.points])
        c = vec(sum(Y; dims = 2)) ./ size(Y, 2)
        cand = inrange(tree, c, maximum(norm(Y[:, j] .- c) for j in axes(Y, 2)) + R)
        _, dmin = isempty(cand) ? (Int[], Float64[]) : nn(KDTree(Y), X[:, cand])
        ts = sort!([t for (i, t) in enumerate(cand) if owner[t] == k || dmin[i] <= R])
        rows = Matrix{Float64}(undef, length(ts), num_points(P))
        nch = min(Threads.nthreads(), length(ts))
        @sync for ch in 1:nch
            Threads.@spawn for r in ch:nch:length(ts)
                t = ts[r]; j = loc[t]; nt = (Nrm[1, t], Nrm[2, t], Nrm[3, t])
                if owner[t] == k                                    # P's own node (a, b)
                    a, b = (j - 1) ÷ nv + 1, (j - 1) % nv + 1
                    rows[r, :] .= gcq_curved_row(:dt, P, P.lpoints[j]; n = nt, on = (P.ru.x[a], P.rv.x[b]))
                else
                    Pt = ps[owner[t]]
                    rows[r, :] .= gcq_curved_row(:dt, P, (Pt.anchor .- P.anchor) .+ Pt.lpoints[j]; n = nt)
                end
            end
        end
        blocks[k] = (ts, rows .- dt_node_rule_block(interface, k, X[:, ts], Nrm[:, ts]))
    end
    end
    I = Int[]; J = Int[]; V = Float64[]
    for k in gcq
        ts, B = blocks[k]
        for j in axes(B, 2), (r, t) in enumerate(ts)
            push!(I, t); push!(J, offs[k] + j); push!(V, B[r, j])
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    SCTLNearQuad(; tol = nothing)

SCTL's singular and near-singular quadrature (`QuadElemList`, `BoundaryIntegralOp`) for `FlatPanel` interfaces with
`n_quad` in 4, 8, 12, 16, 20. It replaces the whole `D^T_nodes + C` by SCTL's operator, so it only combines with
[`SCTLFarField`](@ref); `tol` (default: the far-field tolerance) is the target accuracy (SCTL's operator gets
`tol / 100`, as its FMM loses about two digits). Needs `using SCTL`.
"""
Base.@kwdef struct SCTLNearQuad <: NearQuadrature
    tol::Union{Nothing, Float64} = nothing
end

near_corrections(::SCTLNearQuad, interface) =
    error("SCTLNearQuad is SCTL's whole operator; use it with far = SCTLFarField() (and `using SCTL`)")

"""
    SCTLNearCorrection(; tol = 1e-10)

SCTL's near correction as a sparse matrix, for interfaces of Gauss-Legendre × Gauss-Legendre panels
([`is_gl_panel`](@ref): `FlatPanel`s of any `n_quad` up to 20, GL `TensorPanel`s, GL `CurvedPanel`s with 4, 8, 12, 16
or 20 nodes per direction): [`sctl_gl_corrections`](@ref)`(interface, panels; tol)`, SCTL's near test and singular /
near-singular quadrature, each panel one SCTL element (a proxy element of the next SCTL order when its node count is
not one), minus the node rule. A GCQ panel throws an `ArgumentError`. The default for `FlatPanel` interfaces; unlike
[`SCTLNearQuad`](@ref) it combines with any far field (`FMM3DFarField` too). Needs `using SCTL`.
"""
Base.@kwdef struct SCTLNearCorrection <: NearQuadrature
    tol::Float64 = 1e-10
end

near_corrections(::SCTLNearCorrection, interface) =
    error("SCTLNearCorrection needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    sctl_near_list(interface; tol = 1e-10, radius = nothing, targets = nothing) -> SCTL.NearList

SCTL's near classification of the panels of `interface`: target `t` is near panel `k` if it lies within the radius
of one of the panel's nodes (SCTL's octree search). `SCTL.near_targets(nl, k)` lists them; targets default to the
interface nodes (indices in node order).

`radius` is one radius per node (a vector), or one for all (a number), so any panel type works (e.g. `TensorPanel`).
Without it, `FlatPanel` interfaces get SCTL's radius at accuracy `tol`: the distance at which the panel's
Gauss-Legendre node rule reaches `tol` (Bernstein ellipse). Needs `using SCTL`.

With [`dt_node_rule_block`](@ref) and `SCTL.assemble_near_correction` this is the scaffolding of a near quadrature
written in Julia (see the guide).
"""
function sctl_near_list end

"""
    dt_node_rule_block(interface, k, targets) -> Matrix
    dt_node_rule_block(interface, k, Xt, Nt) -> Matrix

The node-rule `Dᵀ` from the nodes of panel `k` to targets (`length(targets) × num_points(panel k)`):
`w_j n_t·(x_t - y_j) / (4π |x_t - y_j|³)`, zero where `x_t = y_j`. `targets` are interface node indices (their
positions and normals), or give positions `Xt` and normals `Nt` (`3 × m`). This is what the far field contributes
for these pairs; a near correction is (accurate block) − (this block).
"""
function dt_node_rule_block(interface::DielectricInterface, k::Integer, Xt::AbstractMatrix, Nt::AbstractMatrix)
    p = interface.panels[k]
    size(Xt, 1) == 3 && size(Nt) == size(Xt) || throw(DimensionMismatch("Xt and Nt must be 3 × m"))
    B = zeros(size(Xt, 2), num_points(p))
    for j in 1:num_points(p), t in axes(Xt, 2)
        y = p.points[j]
        d1, d2, d3 = Xt[1, t] - y[1], Xt[2, t] - y[2], Xt[3, t] - y[3]
        r2 = d1^2 + d2^2 + d3^2
        r2 == 0 && continue
        B[t, j] = p.weights[j] * (Nt[1, t] * d1 + Nt[2, t] * d2 + Nt[3, t] * d3) / (4π * r2 * sqrt(r2))
    end
    return B
end

function dt_node_rule_block(interface::DielectricInterface, k::Integer, targets::AbstractVector{<:Integer})
    X, Nrm = _interface_targets_normals(interface)
    return dt_node_rule_block(interface, k, X[:, targets], Nrm[:, targets])
end

"""
    sctl_gl_corrections(interface, sources; tol, kernel = :DT) -> SparseMatrixCSC{Float64, Int}

SCTL's near correction of the GL×GL panels `sources` (panel indices, [`is_gl_panel`](@ref)): the `N × N` matrix,
rows = interface nodes, nonzero only in the columns of `sources`, of (SCTL's accurate block) − (DielectricBIE's node
rule) for every target near a source panel. `kernel = :DT` is `Dᵀ` (`w_j n_i·(x_i − y_j) / (4π r³)`, the node rule
[`dt_node_rule_block`](@ref)); `kernel = :D` the double layer with the sign of `laplace3d_D_fmm3d`
(`w_j n_j·(y_j − x_i) / (4π r³)`). Added to the node rule (the far field) it gives the accurate operator.

Each source panel is one SCTL element (`QuadElemList`) over its parameter square, with the panel's map:
- order `n`, the panel's node count per direction, when `n` is one of SCTL's orders 4, 8, 12, 16, 20 (the same in u
  and v): its nodes are the panel's (checked to 1e-13 of the panel's size). An interface node of such a panel is
  passed to SCTL with the element node's exact coordinates, so that it gets SCTL's Duffy self row;
- otherwise (`FlatPanel`, `TensorPanel`) a proxy element of the next SCTL order, the density interpolated from the
  panel's nodes (exact: it is a polynomial of lower degree). A `CurvedPanel` must have an SCTL order.
The element's geometry is SCTL's order-`p` interpolant of the map, which bounds the accuracy on curved panels (unit
sphere, 40 panels: `Dᵀ1` off by 2e-4 at `n = 8`, 5e-9 at 16, 9e-12 at 20). A target is near a panel within the
near radius at `tol` of the panel's own node rule, which the far field sums: SCTL's (`far_field_nodes`) when the
element is the panel; for a proxy, SCTL's formula (the accuracy ellipse of the panel's `n`-point rule, at the proxy's
nodes) evaluated in Julia. On a flat panel, `Dᵀ` vanishes for targets in its plane with its normal and `D` for all
targets in its plane: those rows are zero.

Throws an `ArgumentError` for a non-GL panel in `sources`, a GL panel above [`GL_MAX_ASPECT`](@ref)
([`check_gl_aspect`](@ref)), a `CurvedPanel` whose node counts differ in u and v or are not an SCTL order (or whose
nodes are not SCTL's), and more than 20 nodes per direction. Needs `using SCTL`.
"""
sctl_gl_corrections(interface, sources; kwargs...) =
    error("sctl_gl_corrections needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    sctl_gl_potential_corrections(interface, sources, targets; tol) -> SparseMatrixCSC{Float64, Int}

SCTL's near correction of the single-layer potential `∫ σ(y) / (4π |x − y|) dS_y` of the GL×GL panels `sources` at
the points `targets` (`3 × Nt`): the `Nt × N` matrix of (SCTL's accurate block) − (the node rule
`w_j / (4π |x − y_j|)`, zero where `x = y_j`), nonzero only in the columns of `sources`. Elements, proxies and near
radii as for [`sctl_gl_corrections`](@ref); a target within 1e-13 of a panel's size of a node of a non-proxied panel
is passed with the node's exact SCTL coordinates and gets SCTL's Duffy self row. A target on a node of a proxied
panel throws an `ArgumentError` (SCTL has no self row there), as do the cases listed for `sctl_gl_corrections`.
Needs `using SCTL`.
"""
sctl_gl_potential_corrections(interface, sources, targets; kwargs...) =
    error("sctl_gl_potential_corrections needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    CorrectionMatrix(C)

A precomputed near correction `C` (sparse `N × N`), e.g. from a quadrature computed elsewhere.
"""
struct CorrectionMatrix <: NearQuadrature
    C::SparseMatrixCSC{Float64, Int}
end
CorrectionMatrix(C::AbstractMatrix) = CorrectionMatrix(SparseMatrixCSC{Float64, Int}(C))

function near_corrections(q::CorrectionMatrix, interface::DielectricInterface)
    n = num_points(interface)
    size(q.C) == (n, n) || throw(DimensionMismatch("correction matrix is $(size(q.C)), interface has $n nodes"))
    return q.C
end

"""
    FunctionNearQuad(f)

Near correction computed by `f(interface)` (returning a sparse `N × N` matrix).
"""
struct FunctionNearQuad{F} <: NearQuadrature
    f::F
end

near_corrections(q::FunctionNearQuad, interface::DielectricInterface) =
    near_corrections(CorrectionMatrix(q.f(interface)), interface)

# `near` may also be given as a matrix or a function
_near_quadrature(q::NearQuadrature) = q
_near_quadrature(C::AbstractMatrix) = CorrectionMatrix(C)
_near_quadrature(f::Function) = FunctionNearQuad(f)

"""
    default_near_quadrature(interface) -> NearQuadrature

[`SCTLNearCorrection`](@ref)`()` for `FlatPanel` interfaces, [`GCQNearQuad`](@ref)`()` for `TensorPanel` and
`CurvedPanel` ones: the GCQ panels' rows in Julia, the Gauss-Legendre × Gauss-Legendre panels' corrections SCTL's
([`sctl_gl_corrections`](@ref), at `quad_tol = 1e-12` under `GCQNearQuad()`), which need `using SCTL`.
"""
default_near_quadrature(::DielectricInterface{<:FlatPanel}) = SCTLNearCorrection()
default_near_quadrature(::DielectricInterface{<:TensorPanel}) = GCQNearQuad()
default_near_quadrature(::DielectricInterface{<:CurvedPanel}) = GCQNearQuad()

"""
    default_far_field() -> FarField

`SCTLFarField()` (PVFMM, tol 1e-10, close node pairs summed directly) when SCTL is loaded (`using SCTL`), else
`FMM3DFarField(1e-9)`.
"""
default_far_field() =
    Base.get_extension(@__MODULE__, :DielectricBIESCTLExt) === nothing ? FMM3DFarField(1e-9) : SCTLFarField()

# The two operator parts (far-field evaluator, correction or `nothing`); a backend that computes the whole D^T
# (SCTL's operator for SCTLNearQuad) overrides this for its (far, near) pair.
_dt_parts(far::FarField, near::NearQuadrature, interface) =
    (far_field_operator(far, interface), near_corrections(near, interface))

# ---------------------------------------------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------------------------------------------

"""
    DielectricOperator

`A = τ I + D^T_nodes + C` of [`dielectric_operator`](@ref); `A * x`, `A * X`, `mul!(y, A, x)` and `mul!(Y, A, X)`
work for a vector or an `N × K` block, so it serves `Krylov.gmres` and `Krylov.block_gmres` (batch mode).
"""
struct DielectricOperator{F}
    far::F
    corrections::Union{Nothing, BlockCorrection{Float64}}   # per-panel dense blocks
    diag::Vector{Float64}
    n::Int
end

"""
    dielectric_operator(interface; far = default_far_field(), near = default_near_quadrature(interface))

The dielectric BIE operator `τ I + D^T` of `interface` from a far-field backend ([`FarField`](@ref)) and a near
quadrature ([`NearQuadrature`](@ref), or a sparse correction matrix, or a function `interface -> C`). Examples:

```julia
A = dielectric_operator(interface)                                     # default far field and near quadrature
A = dielectric_operator(interface; far = FMM3DFarField(1e-12; close = 1e-4))    # FMM3D
A = dielectric_operator(interface; far = SCTLFarField(), near = SCTLNearQuad())  # all SCTL (FlatPanel)
A = dielectric_operator(gcq_interface; far = SCTLFarField(), near = GCQNearQuad(quad_tol = 1e-12))
A = dielectric_operator(gcq_interface; near = MyQuad(...))             # a self-defined quadrature
```
"""
function dielectric_operator(interface::DielectricInterface;
                             far::FarField = default_far_field(), near = default_near_quadrature(interface))
    F, C = _dt_parts(far, _near_quadrature(near), interface)
    C === nothing || (C = BlockCorrection(C, interface))
    diag = reduce(vcat, [fill(dielectric_tau(interface.eps_in[i], interface.eps_out[i]), num_points(p))
                         for (i, p) in enumerate(interface.panels)])
    return DielectricOperator(F, C, diag, length(diag))
end

Base.size(op::DielectricOperator) = (op.n, op.n)
Base.size(op::DielectricOperator, d::Integer) = d <= 2 ? op.n : 1
Base.eltype(::DielectricOperator) = Float64

function LinearAlgebra.mul!(Y::AbstractMatrix, op::DielectricOperator, X::AbstractMatrix)
    size(X, 1) == op.n && size(Y) == size(X) || throw(DimensionMismatch())
    apply_far!(Y, op.far, X)
    op.corrections === nothing || mul!(Y, op.corrections, X, true, true)
    Y .+= op.diag .* X
    return Y
end

LinearAlgebra.mul!(y::AbstractVector, op::DielectricOperator, x::AbstractVector) =
    (mul!(reshape(y, op.n, 1), op, reshape(x, op.n, 1)); y)
Base.:*(op::DielectricOperator, X::AbstractMatrix) = mul!(Matrix{Float64}(undef, op.n, size(X, 2)), op, X)
Base.:*(op::DielectricOperator, x::AbstractVector) = mul!(Vector{Float64}(undef, op.n), op, x)

# ---------------------------------------------------------------------------------------------------------------
# off-surface potential
# ---------------------------------------------------------------------------------------------------------------

"""
    PotentialEvaluator

How the single-layer potential `u(x) = ∫ G(x - y) σ(y) dS_y` of a solved density is evaluated at targets off (or
near) the interface: [`potential_operator`](@ref)`(ev, interface, targets)` returns a `LinearMap` `P` (`Nt × N`),
applied to one density at a time (`P * σ`). Available: [`SCTLPotential`](@ref) (the default of
`evaluate_batch_potential`, `four_index_matrix` and the campaign's eval phase) for Gauss-Legendre × Gauss-Legendre
interfaces, and [`GCQPotential`](@ref) for `TensorPanel` (GCQ) interfaces. `SCTLPotential` needs `using SCTL`, and so
does `GCQPotential` on an interface with GL×GL panels.
"""
abstract type PotentialEvaluator end

"""
    SCTLPotential(; tol = 1e-10)

SCTL's `BoundaryIntegralOp` for the Laplace single layer with the targets set: SCTL's near test (per-node Bernstein
radius at `tol`), its near-singular rule (each target's closest point on the panel, dyadic refinement toward it) and
PVFMM for the far field, the octree kept for all densities. Accurate to about `tol` at any distance from the surface,
except for rounding: the near correction cancels the node rule's term of the nearest node, which for a target at
distance `d` from a node costs about `1e-16 × (node spacing) / d` (1e-9 at `d = 5e-9`). Gauss-Legendre ×
Gauss-Legendre panels ([`is_gl_panel`](@ref): `FlatPanel`s of any `n_quad` ≤ 20, GL `TensorPanel`s and
`CurvedPanel`s): each is a proxy SCTL element of order 4, 8, 12, 16 or 20 with the panel's geometry, the density
interpolated to its nodes (exact); an interface with GCQ edge or corner panels is an `ArgumentError` (use
[`GCQPotential`](@ref)). On the surface: a target within max(1e-13 × the panel's size, 2^-48 × the extent of nodes
and targets) of a node of a panel with 4, 8, 12, 16 or 20 nodes per direction (its own element) is moved onto the
element's node, where SCTL's row is the Duffy self row; elsewhere, the proxies' nodes included, the near-singular rule
(the single layer is continuous). The default evaluator of `evaluate_batch_potential`, `four_index_matrix` (both with
`tol = lhs_tol`) and the campaign's eval phase. Needs `using SCTL`.
"""
Base.@kwdef struct SCTLPotential <: PotentialEvaluator
    tol::Float64 = 1e-10
end

"""
    potential_operator(ev::PotentialEvaluator, interface, targets) -> P

The off-surface single-layer potential operator, a `LinearMap` of size `size(targets, 2) × num_points(interface)`, for
densities in the interface's node order: `P * σ` (for a block apply it per column; `P * Σ` is a lazy map).
"""
potential_operator(::SCTLPotential, interface, targets) =
    error("SCTLPotential needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    GCQPotential(; quad_tol = 1e-12, η_gcq = 2.0, gcq_far = :max, far = FMM3DFarField(1e-12))

Off-surface single-layer potential for `TensorPanel` (GCQ) interfaces, accurate near edges and corners: the node
rule over all nodes by the far-field backend `far` ([`FMM3DFarField`](@ref) or [`SCTLFarField`](@ref)), plus near
corrections. The Gauss-Legendre × Gauss-Legendre panels ([`is_gl_panel`](@ref)): SCTL's,
[`sctl_gl_potential_corrections`](@ref)`(interface, gl, targets; tol = quad_tol)` (needs `using SCTL`). The GCQ edge
and corner panels: for every (panel, target) pair that `tensor_isfar` rejects at `quad_tol`, the product-integration
row of `tensor_near_row` minus the node rule; candidate pairs are found with a KD tree, `gcq_far` as for
[`GCQNearQuad`](@ref). A target within max(1e-13 × a GL panel's size, 2^-48 × the extent of nodes and targets) of one
of its nodes is moved onto the node first, so that the far field and SCTL's correction skip the same pair: a target on
a node gets SCTL's Duffy self row (on a node of a GL panel whose `n_quad` is not an SCTL order, an `ArgumentError`).
The same quadrature as [`potential_tensor`](@ref), which sums directly over all pairs.
"""
Base.@kwdef struct GCQPotential{F <: FarField} <: PotentialEvaluator
    quad_tol::Float64 = 1e-12
    η_gcq::Float64 = 2.0
    gcq_far::Symbol = :max
    far::F = FMM3DFarField(1e-12)
end

# node-rule single-layer potential from the interface nodes to the targets (Nt × N LinearMap)
_pottrg_far(far::FMM3DFarField, interface::DielectricInterface, targets::AbstractMatrix) =
    laplace3d_pottrg_fmm3d(interface, Matrix{Float64}(targets), far.tol)
_pottrg_far(far::SCTLFarField, interface, targets) =
    error("SCTLFarField needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

function potential_operator(ev::GCQPotential, interface::DielectricInterface{TensorPanel{T}, T},
                            targets::AbstractMatrix) where T
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × Nt"))
    panels = interface.panels
    gl = findall(is_gl_panel, panels)
    Xt = _snap_to_gl_nodes(panels, gl, targets)
    nt = size(Xt, 2)
    offs = cumsum([0; num_points.(panels)])
    # the GL×GL panels: SCTL (first, so that a missing SCTL fails before the GCQ rows are computed)
    Cgl = isempty(gl) ? nothing : sctl_gl_potential_corrections(interface, gl, Xt; tol = ev.quad_tol)
    # the GCQ panels: product integration
    gcq = findall(!is_gl_panel, panels)
    G(d) = 1 / (4π * sqrt(d[1]^2 + d[2]^2 + d[3]^2))
    tree = KDTree(Xt)
    rows = [Int[] for _ in panels]; cols = [Int[] for _ in panels]; vals = [T[] for _ in panels]
    foreach(k -> (_tensor_near_tables(panels[k].ru); _tensor_near_tables(panels[k].rv)), gcq)
    _blas_single() do
    Threads.@threads :dynamic for k in gcq
        ps = panels[k]
        ws = _TensorRowWS([ps])
        L = max(ps.Lu, ps.Lv)
        cc = ps.anchor .+ ps.o .+ (ps.Lu / 2) .* ps.eu .+ (ps.Lv / 2) .* ps.ev
        half_diag = hypot(ps.Lu, ps.Lv) / 2
        # every target that tensor_isfar can reject: within η L of a GCQ panel, or inside the Bernstein ellipse of
        # the Gauss-Legendre directions at quad_tol (ρ = quad_tol^(-1/2n), semi-axes A⋆, B⋆ in units of L/2)
        n = min(length(ps.ru.x), length(ps.rv.x))
        ρs = ev.quad_tol^(-1 / (2n))
        As = (ρs + 1 / ρs) / 2; Bs = (ρs - 1 / ρs) / 2
        R = 1.05 * max(ev.η_gcq * L + half_diag, L / 2 * sqrt(As^2 + (1 + Bs)^2 + Bs^2))
        for t in inrange(tree, collect(cc), R)
            x = (Xt[1, t], Xt[2, t], Xt[3, t])
            xl = x .- ps.anchor
            tensor_isfar(ps, xl; quad_tol = ev.quad_tol, η_gcq = ev.η_gcq, gcq_far = ev.gcq_far) && continue
            row = tensor_near_row(G, ps, xl; ws)
            for (j, y) in enumerate(ps.points)
                push!(rows[k], t); push!(cols[k], offs[k] + j)
                push!(vals[k], row[j] - G(x .- y) * ps.weights[j])
            end
        end
    end
    end
    Cgcq = sparse(reduce(vcat, rows), reduce(vcat, cols), reduce(vcat, vals), nt, offs[end])
    C = Cgl === nothing ? Cgcq : _add_corrections(Cgcq, Cgl)     # (not reassigned: the closure below captures it)
    F = _pottrg_far(ev.far, interface, Xt)
    return LinearMap{Float64}(σ -> F * σ .+ C * σ, nt, offs[end])
end
