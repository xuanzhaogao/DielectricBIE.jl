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
    FMM3DFarField(tol)

Far field by FMM3D's `lfmm3d` at tolerance `tol`; a block of `K` densities is one vectorized FMM (`nd = K`).
"""
struct FMM3DFarField <: FarField
    tol::Float64
end

"""
    SCTLFarField(; tol = 1e-10)

Far field by SCTL.jl's `ParticleFMM` (PVFMM, OpenMP, no MPI) at relative accuracy `tol`, keeping the octree across
applications; with fewer than 40000 nodes it is an exact direct sum. PVFMM's gradients lose about 1.5 digits, so it
runs with `-log10(tol) + 2` digits (at most 15; PVFMM reaches about 1e-11 at best). Needs `using SCTL` (the
`DielectricBIESCTLExt` extension). A block of `K` densities is `K` evaluations on the same tree.
"""
struct SCTLFarField <: FarField
    tol::Float64
end
SCTLFarField(; tol::Real = 1e-10) = SCTLFarField(Float64(tol))

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
end

function far_field_operator(far::FMM3DFarField, interface::DielectricInterface)
    sources, norms = _interface_targets_normals(interface)
    return FMM3DDTFar(Matrix{Float64}(sources), Vector{Float64}(all_weights(interface)),
                      Matrix{Float64}(norms), far.tol)
end

function apply_far!(Y::AbstractMatrix, F::FMM3DDTFar, X::AbstractMatrix)
    n = length(F.weights); K = size(X, 2)
    charges = Matrix{Float64}(undef, K, n)
    Threads.@threads for i in 1:n
        @inbounds for k in 1:K
            charges[k, i] = F.weights[i] * X[i, k]
        end
    end
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
    return Y
end

# ---------------------------------------------------------------------------------------------------------------
# near quadrature
# ---------------------------------------------------------------------------------------------------------------

"""
    NearQuadrature

Near-field quadrature of the dielectric operator: [`near_corrections`](@ref)`(q, interface)` returns the sparse
`N × N` correction `C` = (accurate `D^T`) - (node-rule `D^T`), so that `D^T_nodes + C` is the corrected operator.

Built in:
- [`UpsampledNearQuad`](@ref): upsampling plus adaptive quadtrees for `FlatPanel` interfaces (the default there);
- [`GCQNearQuad`](@ref): the product-integration rules of the GCQ edge panels (`TensorPanel` interfaces, default);
- [`SCTLNearQuad`](@ref): SCTL's singular quadrature of `QuadElemList` (needs `SCTLFarField` and `using SCTL`);
- [`SCTLNearCorrection`](@ref): the same quadrature as a sparse correction, with any far field (`using SCTL`);
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
columns of those source panels, the others zero: what a combination of quadratures (e.g.
[`SCTLProxyNearQuad`](@ref)'s fallback) asks for. The default computes all columns and drops the others; a
quadrature that can do less work implements the three-argument method (as [`GCQNearQuad`](@ref) does).
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
    UpsampledNearQuad(; up_tol = 1e-9, max_order = 8, correct_edges = true, adaptive_atol = up_tol,
                      adaptive_rtol = sqrt(eps()), adaptive_n_GL = 0, adaptive_max_depth = 20)

Near corrections of `laplace3d_DT_fmm3d_corrected` for `FlatPanel` interfaces: Bernstein-radius neighbor
lists, upsampled quadrature for near pairs and adaptive quadtrees for touching pairs.
"""
Base.@kwdef struct UpsampledNearQuad <: NearQuadrature
    up_tol::Float64 = 1e-9
    max_order::Int = 8
    correct_edges::Bool = true
    adaptive_atol::Float64 = up_tol
    adaptive_rtol::Float64 = sqrt(eps(Float64))
    adaptive_n_GL::Int = 0
    adaptive_max_depth::Int = 20
end

function near_corrections(q::UpsampledNearQuad, interface::DielectricInterface)
    cfg = AdaptiveConfig(q.adaptive_atol, q.adaptive_rtol, q.adaptive_n_GL, q.adaptive_max_depth)
    (; upsample, adaptive) = build_neighbor_list(interface, q.max_order, q.up_tol;
        correct_edges = q.correct_edges, adaptive_cfg = cfg)
    return SparseMatrixCSC{Float64, Int}(laplace3d_DT_corrections(interface, upsample, adaptive))
end

"""
    GCQNearQuad(; quad_tol = 1e-12, η_gcq = 2.0, reuse = true)

Near corrections of `TensorPanel` (GCQ) interfaces, from [`laplace3d_DT_tensor_corrections`](@ref).
"""
Base.@kwdef struct GCQNearQuad <: NearQuadrature
    quad_tol::Float64 = 1e-12
    η_gcq::Float64 = 2.0
    reuse::Bool = true
end

near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:TensorPanel}) =
    SparseMatrixCSC{Float64, Int}(first(laplace3d_DT_tensor_corrections(interface;
        reuse = q.reuse, quad_tol = q.quad_tol, η_gcq = q.η_gcq)))
near_corrections(q::GCQNearQuad, interface::DielectricInterface{<:TensorPanel}, sources::AbstractVector{<:Integer}) =
    SparseMatrixCSC{Float64, Int}(first(laplace3d_DT_tensor_corrections(interface;
        reuse = q.reuse, quad_tol = q.quad_tol, η_gcq = q.η_gcq, sources = sources)))

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

SCTL's near correction as a sparse matrix: SCTL's near test (which targets are near which panel) and the singular /
near-singular quadrature of its `QuadElemList`, minus the node rule (SCTL.jl's `near_correction`). For `FlatPanel`
interfaces with `n_quad` in 4, 8, 12, 16, 20; unlike [`SCTLNearQuad`](@ref) it combines with any far field
(`FMM3DFarField` too). Needs `using SCTL`.
"""
Base.@kwdef struct SCTLNearCorrection <: NearQuadrature
    tol::Float64 = 1e-10
end

near_corrections(::SCTLNearCorrection, interface) =
    error("SCTLNearCorrection needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

"""
    SCTLProxyNearQuad(; tol = 1e-12, fallback = GCQNearQuad(), polynomial = default_polynomial_panel, order = nothing)

SCTL's near quadrature for any panels whose density is a polynomial, through proxy elements, and `fallback` for the
others. For each panel `k` with `polynomial(panel)` true:

1. a proxy `QuadElemList` element of order `order` (default: the smallest of 4, 8, 12, 16, 20 at least the panels'
   node counts per direction) with the panel's geometry;
2. SCTL's classification (near radius at `tol`) of all interface nodes against it;
3. SCTL's near-singular rule for the near targets (off the panel's plane; `Dᵀ` vanishes in it), times the
   interpolation from the panel's nodes to the proxy's, which is exact for the panel's polynomial density;
4. minus the node rule ([`dt_node_rule_block`](@ref)).

The other panels' columns come from `near_corrections(fallback, interface, sources)`, so only those are computed.
The defaults suit GCQ boxes: SCTL for the Gauss-Legendre panels (`kind == :gl`), the GCQ rules for the edge and
corner panels, whose densities are singular. For `FlatPanel` interfaces every panel is polynomial and the fallback
is unused. Combines with any far field. Needs `using SCTL`.
"""
Base.@kwdef struct SCTLProxyNearQuad{Q, F} <: NearQuadrature
    tol::Float64 = 1e-12
    fallback::Q = GCQNearQuad()
    polynomial::F = default_polynomial_panel
    order::Union{Nothing, Int} = nothing
end

"""
    default_polynomial_panel(panel) -> Bool

Whether a panel's density is a tensor polynomial through its nodes (so SCTL's quadrature applies): every `FlatPanel`,
and `TensorPanel`s with Gauss-Legendre rules in both directions (`kind == :gl`).
"""
default_polynomial_panel(::FlatPanel) = true
default_polynomial_panel(p::TensorPanel) = p.kind == :gl

near_corrections(::SCTLProxyNearQuad, interface) =
    error("SCTLProxyNearQuad needs the SCTL.jl package: run `using SCTL` (it loads DielectricBIESCTLExt)")

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

[`UpsampledNearQuad`](@ref)`()` for `FlatPanel` interfaces, [`GCQNearQuad`](@ref)`()` for `TensorPanel` ones.
"""
default_near_quadrature(::DielectricInterface{<:FlatPanel}) = UpsampledNearQuad()
default_near_quadrature(::DielectricInterface{<:TensorPanel}) = GCQNearQuad()

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
    dielectric_operator(interface; far = FMM3DFarField(1e-9), near = default_near_quadrature(interface))

The dielectric BIE operator `τ I + D^T` of `interface` from a far-field backend ([`FarField`](@ref)) and a near
quadrature ([`NearQuadrature`](@ref), or a sparse correction matrix, or a function `interface -> C`). Examples:

```julia
A = dielectric_operator(interface)                                     # FMM3D + DielectricBIE's quadrature
A = dielectric_operator(interface; far = SCTLFarField(tol = 1e-10))    # SCTL's FMM (using SCTL)
A = dielectric_operator(interface; far = SCTLFarField(), near = SCTLNearQuad())  # all SCTL (FlatPanel)
A = dielectric_operator(gcq_interface; far = SCTLFarField(), near = GCQNearQuad(quad_tol = 1e-12))
A = dielectric_operator(gcq_interface; near = MyQuad(...))             # a self-defined quadrature
```
"""
function dielectric_operator(interface::DielectricInterface;
                             far::FarField = FMM3DFarField(1e-9), near = default_near_quadrature(interface))
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
