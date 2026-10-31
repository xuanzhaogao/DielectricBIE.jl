# Curved tensor-product panels.
#
# A `CurvedPanel` is the image of [0,1]^2 under a smooth map y(u,v) = anchor + lmap(u,v), with a
# `TensorRule1D` in each direction, so the GCQ edge rules of `TensorPanel` carry over unchanged
# (GCQ in u with the edge at u = 0, Gauss-Legendre in v). The map is stored relative to an anchor
# point on the edge: GCQ nodes lie 1e-6 … 1e-12 from the edge, and `lmap` must give their offsets
# without cancellation (the mesher computes it so). Normals and Jacobians come from the tangents y_u, y_v:
# from `dmap` when given (the mesher passes the patch map's derivatives by the chain rule), else from
# ForwardDiff of `lmap`; the normal is ±(y_u × y_v) / |y_u × y_v|, the sign set by `flip`.

"""
    CurvedPanel(anchor, lmap, ru, rv; kind = :gl, flip = false, dmap = nothing)

Panel y(u,v) = anchor + lmap(u,v), (u,v) ∈ [0,1]², with rule `ru` in u and `rv` in v
(`TensorRule1D`; `kind = :edge` for a GCQ rule in u with the edge at u = 0). `lmap` returns a
3-tuple; `dmap(u, v)` returns the tangents (y_u, y_v), and without it `lmap` is differentiated by
ForwardDiff. Stores per-node local and absolute points,
unit normals (along y_u × y_v, negated with `flip`) and weights J ru.w rv.w, J = |y_u × y_v|.
Nodes are ordered (i_u, j_v) -> (i-1) n_v + j, as for `TensorPanel`.
"""
struct CurvedPanel{F, D} <: AbstractPanel
    anchor::NTuple{3, Float64}
    lmap::F
    dmap::D
    ru::TensorRule1D
    rv::TensorRule1D
    kind::Symbol
    flip::Bool
    lpoints::Vector{NTuple{3, Float64}}
    points::Vector{NTuple{3, Float64}}
    normals::Vector{NTuple{3, Float64}}
    weights::Vector{Float64}
end

"Offset from the anchor of the point (u, v) of a `CurvedPanel`."
curved_lpoint(p::CurvedPanel, u, v) = NTuple{3, Float64}(p.lmap(u, v))

_frame_from_tangents(Xu, Xv, flip) = begin
    c = cross(collect(Xu), collect(Xv)); J = norm(c)
    (Tuple(Xu), Tuple(Xv), Tuple((flip ? -c : c) ./ J), J)
end

_curved_frame(lmap, flip, u, v) =
    _frame_from_tangents(ForwardDiff.derivative(t -> collect(lmap(t, v)), u),
                         ForwardDiff.derivative(t -> collect(lmap(u, t)), v), flip)

_curved_frame(lmap, dmap, flip, u, v) =
    dmap === nothing ? _curved_frame(lmap, flip, u, v) : _frame_from_tangents(dmap(u, v)..., flip)

"""
    curved_frame(p, u, v) -> (y_u, y_v, n, J)

Tangents, unit normal and Jacobian J = |y_u × y_v| of a `CurvedPanel` at (u, v).
"""
curved_frame(p::CurvedPanel, u, v) = _curved_frame(p.lmap, p.dmap, p.flip, u, v)

function CurvedPanel(anchor, lmap, ru::TensorRule1D, rv::TensorRule1D; kind::Symbol = :gl, flip::Bool = false,
                     dmap = nothing)
    a = NTuple{3, Float64}(anchor)
    lpoints = NTuple{3, Float64}[]; normals = NTuple{3, Float64}[]; weights = Float64[]
    for (u, wu) in zip(ru.x, ru.w), (v, wv) in zip(rv.x, rv.w)
        push!(lpoints, NTuple{3, Float64}(lmap(u, v)))
        _, _, n, J = _curved_frame(lmap, dmap, flip, u, v)
        push!(normals, n); push!(weights, J * wu * wv)
    end
    points = [a .+ l for l in lpoints]
    return CurvedPanel(a, lmap, dmap, ru, rv, kind, flip, lpoints, points, normals, weights)
end

Base.show(io::IO, p::CurvedPanel) =
    print(io, "CurvedPanel ($(p.kind)) with $(length(p.ru.x))×$(length(p.rv.x)) points")

num_points(p::CurvedPanel) = length(p.points)

eachpoint(p::CurvedPanel) =
    (FlatPanelPointInfo{Float64, 3}(p.points[i], p.normals[i], p.weights[i], i) for i in eachindex(p.points))
