# Tensor-product panels with a separate 1D rule in each direction.
#
# A `TensorRule1D` is a rule on [0,1] with interpolation data: Gauss-Legendre
# (one segment, identity interpolation) or a weighted GCQ edge rule (many fine
# segments, interpolation matrix, singular weight x^pw). A `TensorPanel` is the
# rectangle y(u,v) = anchor + o + u Lu eu + v Lv ev with rule `ru` in u and `rv`
# in v; edge panels use the GCQ rule across the edge (u, with the edge at u = 0)
# and Gauss-Legendre along it, corner panels may use GCQ in both directions.
#
# The panel stores its geometry relative to an exact anchor point (a box
# vertex): GCQ nodes lie within 1e-6 of an edge and their fine grids within
# 1e-16, below the resolution of absolute coordinates. Absolute `points` and
# `weights` are also stored, so the panel works with every routine that
# iterates `eachpoint` (FMM, right-hand sides, fluxes, potentials).

"""
    TensorRule1D

1D rule on [0,1]: nodes/weights `x, w`; fine segments `segs` with `z0, wz`
(Gauss-Legendre on [-1,1]) and barycentric weights `bw`; `P` maps node values
to fine-node values of v = f / x^pw; `fx, fw` (nz × nsegs) are each segment's
nodes and weights on [0,1], the weight x^pw included.
"""
struct TensorRule1D
    x::Vector{Float64}
    w::Vector{Float64}
    segs::Vector{Tuple{Float64, Float64}}
    z0::Vector{Float64}
    wz::Vector{Float64}
    bw::Vector{Float64}
    P::Matrix{Float64}
    pw::Float64
    fx::Matrix{Float64}
    fw::Matrix{Float64}
end

function TensorRule1D(x, w, segs, z0, wz, P, pw)
    fx = reduce(hcat, [sa .+ (sb - sa) / 2 .* (1 .+ z0) for (sa, sb) in segs])
    fw = reduce(hcat, [(sb - sa) / 2 .* wz for (sa, sb) in segs])
    pw != 0 && (fw .*= fx .^ pw)
    return TensorRule1D(x, w, segs, z0, wz, gl_barycentric_weights(z0, wz), P, pw, fx, fw)
end

"n-point Gauss-Legendre as a `TensorRule1D`."
function gl_tensor_rule(n::Int)
    t, wt = gausslegendre(n)
    return TensorRule1D((1 .+ t) ./ 2, wt ./ 2, [(0.0, 1.0)], t, wt, Matrix(1.0I, n, n), 0.0)
end

"Weighted GCQ edge rule as a `TensorRule1D`."
gcq_tensor_rule(r::EdgeGCQRule) = TensorRule1D(r.x, r.w, r.segs, r.z0, r.wz, r.P, r.pw)

is_gcq(r::TensorRule1D) = length(r.segs) > 1

"""
    tensor_rule_interp(r, ts)

(length(ts) × n) matrix mapping node values of a function in the rule's span to
its values at the points `ts` ∈ (0,1): Lagrange interpolation on the fine
segment containing t, through the interpolation matrix `P`, times t^pw.
"""
function tensor_rule_interp(r::TensorRule1D, ts)
    E = zeros(length(ts), length(r.x))
    for (a, t) in enumerate(ts)
        k = clamp(searchsortedlast([sa for (sa, _) in r.segs], t), 1, length(r.segs))
        sa, sb = r.segs[k]
        τ = 2 * (t - sa) / (sb - sa) - 1
        j = findfirst(==(τ), r.z0)
        ℓ = j === nothing ? (c = r.bw ./ (τ .- r.z0); c ./ sum(c)) :
                            [i == j ? 1.0 : 0.0 for i in eachindex(r.z0)]
        nz = length(r.z0)
        E[a, :] .= (r.P[(k - 1) * nz .+ (1:nz), :]' * ℓ) .* (r.pw == 0 ? 1.0 : t^r.pw)
    end
    return E
end

"""
    TensorPanel{T}

Flat rectangular panel with tensor-product rules (see the file header). `kind`
is :gl, :edge (GCQ × GL) or :corner. Nodes are ordered (i_u, j_v) ->
(i-1) n_v + j.
"""
struct TensorPanel{T} <: AbstractPanel
    normal::NTuple{3, T}
    corners::Vector{NTuple{3, T}}
    is_edge::Bool
    kind::Symbol
    anchor::NTuple{3, T}
    o::NTuple{3, T}
    eu::NTuple{3, T}
    ev::NTuple{3, T}
    Lu::T
    Lv::T
    ru::TensorRule1D
    rv::TensorRule1D
    plane::Int
    points::Vector{NTuple{3, T}}
    weights::Vector{T}
end

tensor_lpoint(p::TensorPanel, u, v) = p.o .+ (u * p.Lu) .* p.eu .+ (v * p.Lv) .* p.ev

function TensorPanel(normal, anchor, o, eu, ev, Lu, Lv, ru::TensorRule1D, rv::TensorRule1D,
                     kind::Symbol, plane::Int)
    lp(u, v) = o .+ (u * Lu) .* eu .+ (v * Lv) .* ev
    corners = [anchor .+ lp(u, v) for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1))]
    points = [anchor .+ lp(u, v) for u in ru.x for v in rv.x]
    weights = [Lu * Lv * wu * wv for wu in ru.w for wv in rv.w]
    return TensorPanel{Float64}(normal, corners, kind != :gl, kind, anchor, o, eu, ev, Lu, Lv,
                                ru, rv, plane, points, weights)
end

Base.show(io::IO, p::TensorPanel) =
    print(io, "TensorPanel ($(p.kind)) with $(length(p.ru.x))×$(length(p.rv.x)) points")

num_points(p::TensorPanel) = length(p.points)
tensor_local_points(p::TensorPanel) = [tensor_lpoint(p, u, v) for u in p.ru.x for v in p.rv.x]

eachpoint(p::TensorPanel{T}) where T =
    (FlatPanelPointInfo{T, 3}(p.points[i], p.normal, p.weights[i], i) for i in eachindex(p.points))
