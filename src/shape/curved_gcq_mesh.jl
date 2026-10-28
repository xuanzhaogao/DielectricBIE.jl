# Meshing a `CurvedGeometry` with `CurvedPanel`s, graded toward its edges.
#
# Each patch's parameter square is split into a tensor grid: dyadically toward every side on a graded curve
# (an edge, by default every curve that is not a smooth seam), uniformly otherwise. A panel touching a graded
# curve is re-parametrized so that the curve is at its u = 0 and gets the weighted GCQ rule of that curve's
# junction across it (Gauss-Legendre along it). Panel points are stored as offsets from an anchor on the
# curve: along the curve by a plain difference (the same for every node on a line across it), across it as
# a path integral of the map's derivative (16-point Gauss-Legendre), so a map written in plain absolute
# coordinates still gives the offsets of nodes 1e-12 from the curve to full relative precision. Tangents come
# from the map's derivatives by the chain rule.

const _OFFSET_GL = gausslegendre(16)

# map(b) - map(a) along the straight parameter segment a -> b, as ∫ d/ds map(a + s (b - a)) ds
function _segment_offset(f, a, b)
    t, w = _OFFSET_GL
    du = b[1] - a[1]; dv = b[2] - a[2]
    acc = zero(du + dv) .* (1, 1, 1)
    (iszero(du) && iszero(dv)) && return acc
    for (ti, wi) in zip(t, w)
        s = (1 + ti) / 2
        g = ForwardDiff.derivative(τ -> collect(f(a[1] + τ * du, a[2] + τ * dv)), s)
        acc = acc .+ (wi / 2) .* (g[1], g[2], g[3])
    end
    return acc
end

# breakpoints of [0, 1]: dyadic toward a graded end (the edge interval [0, 2^-nlev]), uniform without one. With
# across = a > 1 a graded direction is first split into a equal panels and the end ones graded (nlev = 0: uniform)
function _graded_breaks(lo::Bool, hi::Bool, nlev::Int, m::Int, across::Int = 1)
    (lo || hi) || return collect(range(0.0, 1.0, length = m + 1))
    if across == 1
        lo && hi && return vcat(0.0, [2.0^-k for k in nlev:-1:2], 0.5, [1 - 2.0^-k for k in 2:nlev], 1.0)
        lo && return vcat(0.0, [2.0^-k for k in nlev:-1:1], 1.0)
        return 1 .- reverse(_graded_breaks(true, false, nlev, m))
    end
    h = 1 / across
    b = collect(range(0.0, 1.0, length = across + 1))
    lo && (b = vcat(b, [h * 2.0^-k for k in 1:nlev]))
    hi && (b = vcat(b, [1 - h * 2.0^-k for k in 1:nlev]))
    return sort(unique(b))
end

"""
    CurvedMesh

Result of `curved_gcq_mesh`: the `interface` (`CurvedPanel`s, permittivities from the patches), the
`geometry`, which curves are `graded` and their density `powers`, the node `ranges` of every patch, and for
every panel its patch (`panel_patch`) and its parameter map (u, v) -> (U, V) into the patch (`panel_param`).
"""
struct CurvedMesh
    interface::DielectricInterface{CurvedPanel, Float64}
    geometry::CurvedGeometry
    graded::Vector{Bool}
    powers::Vector{Vector{Float64}}
    ranges::Vector{UnitRange{Int}}
    panel_patch::Vector{Int}
    panel_param::Vector{Any}
end

Base.show(io::IO, m::CurvedMesh) =
    print(io, "CurvedMesh with $(length(m.interface.panels)) panels, $(num_points(m.interface)) nodes, ",
          "$(count(m.graded)) graded curves")

num_points(m::CurvedMesh) = num_points(m.interface)

"""
    curved_gcq_mesh(geometry; nlev = 4, n_quad = 10, m = 2, across = 1, tol = 1e-10, qmax = nothing, gmax = 20.0,
                    edge = :gcq, edge_rule = nothing, grade = c -> !c.smooth) -> CurvedMesh

Mesh the patches of a `CurvedGeometry`. A curve `c` (a `CurveJunction`) is graded when `grade(c)`: the
patch sides on it get `nlev` dyadic levels, and the panels touching it the weighted GCQ rule of its junction
across it (`edge_gcq_rule(powers; tol, qmax)`, powers up to `gmax` from the sector angles and
permittivities) and `n_quad`-point Gauss-Legendre along it; `edge = :gl` puts Gauss-Legendre there too (an
h-graded reference). Directions without a graded side get `m` equal panels; across a graded curve the patch is
first split into `across` equal panels, the `nlev` levels then grade the one at the curve (`nlev = 0`, `across = a`:
uniform panels of width 1/a, the GCQ rule on the one at the curve, for h-refinement studies). `edge_rule` replaces the
junction's own rule on every graded curve: an `EdgeGCQRule`, or a function of the curve's `CurveJunction` returning one
(needed for edges of varying angle, e.g. `universal_gcq_rule(pmin)` with pmin the smallest power along the edge). Vertices,
where graded curves meet, are not supported yet: graded curves on two adjacent sides of a patch throw an `ArgumentError`.
"""
function curved_gcq_mesh(g::CurvedGeometry; nlev::Int = 4, n_quad::Int = 10, m::Int = 2, across::Int = 1, tol::Real = 1e-10,
                         qmax::Union{Nothing, Real} = nothing, gmax::Real = 20.0, edge::Symbol = :gcq,
                         edge_rule = nothing, grade = c -> !c.smooth)
    edge in (:gcq, :gl) || throw(ArgumentError("edge must be :gcq or :gl"))
    across >= 1 || throw(ArgumentError("across must be at least 1"))
    graded = Bool[grade(c) for c in g.curves]
    powers = [graded[i] ? _junction_density_powers(c.angles, c.eps; gmax) : Float64[] for (i, c) in enumerate(g.curves)]
    rgl = gl_tensor_rule(n_quad)
    rules = Dict{Int, TensorRule1D}()
    given = edge_rule isa EdgeGCQRule ? gcq_tensor_rule(edge_rule) : nothing
    rule(i) = get!(rules, i) do
        edge == :gl ? rgl : given !== nothing ? given : edge_rule !== nothing ? gcq_tensor_rule(edge_rule(g.curves[i])) :
            gcq_tensor_rule(qmax === nothing ? edge_gcq_rule(powers[i]; tol) : edge_gcq_rule(powers[i]; tol, qmax))
    end
    panels = CurvedPanel[]; ein = Float64[]; eout = Float64[]; ranges = UnitRange{Int}[]
    ppatch = Int[]; pparam = Any[]
    n0 = 0
    for (k, P) in enumerate(g.patches)
        gs = [graded[g.side_curve[s, k]] for s in 1:4]
        (gs[1] || gs[2]) && (gs[3] || gs[4]) &&
            throw(ArgumentError("patch $k has graded curves on adjacent sides (a vertex): not supported yet"))
        Ub = _graded_breaks(gs[1], gs[2], nlev, m, across); Vb = _graded_breaks(gs[3], gs[4], nlev, m, across)
        f = P.map
        for (U0, U1) in zip(Ub[1:end-1], Ub[2:end]), (V0, V1) in zip(Vb[1:end-1], Vb[2:end])
            dU = U1 - U0; dV = V1 - V0
            # the side on a graded curve this panel touches (if any) goes to the panel's u = 0
            s = gs[1] && U0 == 0 ? 1 : gs[2] && U1 == 1 ? 2 : gs[3] && V0 == 0 ? 3 : gs[4] && V1 == 1 ? 4 : 0
            pa, A, along_v, sgn = s == 1 ? (((u, v) -> (U0 + u * dU, V0 + v * dV)), (0.0, V0), true, 1) :
                                  s == 2 ? (((u, v) -> (1 - u * dU, V0 + v * dV)), (1.0, V0), true, -1) :
                                  s == 3 ? (((u, v) -> (U0 + v * dU, u * dV)), (U0, 0.0), false, -1) :
                                  s == 4 ? (((u, v) -> (U0 + v * dU, 1 - u * dV)), (U0, 1.0), false, 1) :
                                           (((u, v) -> (U0 + u * dU, V0 + v * dV)), (U0, V0), true, 1)
            anchor = NTuple{3, Float64}(f(A...))
            lmap = s == 0 ? ((u, v) -> f(pa(u, v)...) .- anchor) : (u, v) -> begin
                U, V = pa(u, v)
                C = along_v ? (A[1], V) : (U, A[2])           # along the curve, then across it
                (f(C...) .- anchor) .+ _segment_offset(f, C, (U, V))
            end
            # tangents of the panel from the patch map's (x_U, x_V), by the chain rule through pa
            cu, cv = s == 1 || s == 0 ? ((dU, 0.0), (0.0, dV)) : s == 2 ? ((-dU, 0.0), (0.0, dV)) :
                     s == 3 ? ((0.0, dV), (dU, 0.0)) : ((0.0, -dV), (dU, 0.0))
            dmap = (u, v) -> begin
                U, V = pa(u, v)
                XU = ForwardDiff.derivative(t -> collect(f(t, V)), U); XV = ForwardDiff.derivative(t -> collect(f(U, t)), V)
                (Tuple(cu[1] .* XU .+ cu[2] .* XV), Tuple(cv[1] .* XU .+ cv[2] .* XV))
            end
            ru = s == 0 ? rgl : rule(g.side_curve[s, k])
            kind = s != 0 && edge == :gcq ? :edge : :gl
            push!(panels, CurvedPanel(anchor, lmap, ru, rgl; kind, flip = P.flip ⊻ (sgn < 0), dmap))
            push!(ein, P.eps_in); push!(eout, P.eps_out); push!(ppatch, k); push!(pparam, pa)
        end
        n1 = sum(num_points, panels; init = 0)
        push!(ranges, n0+1:n1); n0 = n1
    end
    itf = DielectricInterface{CurvedPanel, Float64}(panels, ein, eout)
    return CurvedMesh(itf, g, graded, powers, ranges, ppatch, pparam)
end
