# Several dielectric (or perfectly conducting) boxes, each meshed separately with
# `single_dielectric_box3d_gcq`, solved together with one charge constraint per body.
#
# Near-null modes. As a body's contrast grows, τ -> -1/2 and τ I + Dᵀ has a near-zero eigenvalue
# τ + 1/2 = 1/(1 - ε_in/ε_out) per closed body, whose eigenvector is the body's equilibrium charge
# (a charged conductor). A discretization error δ then enters the total charge of each body
# amplified by ~ ε_in/ε_out, and at ε_in = Inf the mode is a null vector. Gauss's law fixes each
# body's total charge, and imposing it (`solve_dielectric_box3d_gcq(...; constraints)`) removes the
# mode: the error stays at δ for every contrast, including perfect conductors.

"""
    join_bodies(parts) -> (interface, bodies)

Concatenate the interfaces `parts` (one per closed body, e.g. from
`single_dielectric_box3d_gcq`) into one `DielectricInterface`, and return with it the index
range of each body's quadrature nodes in the joined unknown vector.
"""
function join_bodies(parts::AbstractVector{<:DielectricInterface})
    interface = DielectricInterface(reduce(vcat, [p.panels for p in parts]), reduce(vcat, [p.eps_in for p in parts]),
                                    reduce(vcat, [p.eps_out for p in parts]))
    offs = cumsum([0; [num_points(p) for p in parts]])
    return interface, [offs[k]+1:offs[k+1] for k in eachindex(parts)]
end

# axis-aligned bounding box of a body's nodes (a box body is its own bounding box)
function _body_bounds(part::DielectricInterface)
    pts = reduce(vcat, [p.points for p in part.panels])
    return ntuple(k -> minimum(x[k] for x in pts), 3), ntuple(k -> maximum(x[k] for x in pts), 3)
end

# 1 inside, 0 outside, error on the surface (relative tolerance on the box size)
function _box_side(part::DielectricInterface, x; rtol = 1e-10)
    lo, hi = _body_bounds(part)
    h = rtol * maximum(hi .- lo)
    all(lo[k] + h < x[k] < hi[k] - h for k in 1:3) && return 1
    any(x[k] < lo[k] - h || x[k] > hi[k] + h for k in 1:3) && return 0
    throw(ArgumentError("source point $x lies on the surface of a body"))
end

"""
    source_permittivity(parts, x)

Permittivity at the point `x`: `eps_in` of the box body containing it, else the common exterior
permittivity `eps_out`. Throws if `x` lies on a body's surface.
"""
function source_permittivity(parts::AbstractVector{<:DielectricInterface}, x)
    for p in parts
        _box_side(p, x) == 1 && return p.eps_in[1]
    end
    return parts[1].eps_out[1]
end

"""
    box_charge_constraints(parts, sources; bodies = nothing)

Total-charge constraints `[(range_c, Q_c), ...]` for `solve_dielectric_box3d_gcq(...; constraints)`,
one per box body in `parts` (as returned by `join_bodies`). By Gauss's law the induced charge on
the surface of body c is

    Q_c = Σ_{sources q inside c} q (1/ε_out - 1/ε_in(c)),

and 0 for a body that contains no source (with a perfect conductor, ε_in = Inf, a source inside is
not allowed). `sources` is a `PointSource` or a vector of them.
"""
function box_charge_constraints(parts::AbstractVector{<:DielectricInterface}, sources; bodies = nothing)
    srcs = sources isa PointSource ? [sources] : collect(sources)
    bodies === nothing && (bodies = join_bodies(parts)[2])
    cons = Tuple{UnitRange{Int}, Float64}[]
    for (c, p) in enumerate(parts)
        ε_in, ε_out = p.eps_in[1], p.eps_out[1]
        Q = 0.0
        for s in srcs
            _box_side(p, s.point) == 1 || continue
            isinf(ε_in) && throw(ArgumentError("a source inside a perfect conductor is not allowed"))
            Q += s.charge * (1 / ε_out - 1 / ε_in)
        end
        push!(cons, (bodies[c], Q))
    end
    return cons
end

"""
    solve_dielectric_boxes_gcq(parts, sources; deflate = true, fmm_tol = 1e-13, quad_tol = 1e-12,
                               η_gcq = 2.0, gcq_far = :max, rtol = 1e-13, itmax = 1000)

Solve the interface equation for several box bodies (`parts`, one interface per body, e.g. from
`single_dielectric_box3d_gcq`; `eps_in = Inf` is a perfect conductor) excited by point charges
`sources` (a `PointSource` or a vector of them; each charge is scaled by the permittivity of the
region containing it). Assembles the FMM operator with GCQ near corrections, the right-hand side,
and, with `deflate = true` (the default), one total-charge constraint per body from Gauss's law
(`box_charge_constraints`), then runs GMRES on the W^{1/2}-scaled system.

Returns a named tuple `(σ, interface, bodies, constraints, stats)`. Use `deflate = false` for
the plain system; it is accurate only at moderate contrast, since its error in each body's total
charge grows like the contrast (see `solve_dielectric_box3d_gcq`).

`far` and `near` select the operator backends (see [`dielectric_operator`](@ref)): e.g.
`far = SCTLFarField(tol = 1e-12)` for SCTL's FMM (after `using SCTL`), or `near = q` for a
self-defined [`NearQuadrature`](@ref) `q`. Unset, they are `FMM3DFarField(fmm_tol)` and
`GCQNearQuad(; quad_tol, η_gcq, gcq_far)`.
"""
function solve_dielectric_boxes_gcq(parts::AbstractVector{<:DielectricInterface}, sources; deflate::Bool = true, kw...)
    srcs = sources isa PointSource ? [sources] : collect(sources)
    interface, bodies = join_bodies(parts)
    constraints = deflate ? box_charge_constraints(parts, srcs; bodies) : nothing
    return _solve_boxes_gcq(interface, bodies, srcs, x -> source_permittivity(parts, x), constraints; kw...)
end

"""
    solve_dielectric_boxes_gcq(mesh::MultiBoxGCQ, sources; deflate = true, fmm_tol = 1e-13,
                               quad_tol = 1e-12, η_gcq = 2.0, gcq_far = :max, rtol = 1e-13, itmax = 1000)

The same for a mesh from `multi_dielectric_box3d_gcq` (boxes that may share faces), with one
total-charge constraint per cluster of face-sharing boxes. `bodies` in the result are the
clusters' node ranges.
"""
function solve_dielectric_boxes_gcq(m::MultiBoxGCQ, sources; deflate::Bool = true, kw...)
    srcs = sources isa PointSource ? [sources] : collect(sources)
    constraints = deflate ? box_charge_constraints(m, srcs) : nothing
    return _solve_boxes_gcq(m.interface, m.bodies, srcs, x -> source_permittivity(m, x), constraints; kw...)
end

function _solve_boxes_gcq(interface, bodies, srcs, epsat, constraints; fmm_tol::Real = 1e-13,
                          quad_tol::Real = 1e-12, η_gcq::Real = 2.0, gcq_far::Symbol = :max, rtol::Real = 1e-13, itmax::Int = 1000,
                          far::Union{Nothing, FarField} = nothing, near = nothing)
    A = if far === nothing && near === nothing
        first(lhs_dielectric_box3d_gcq_fmm3d(interface, Float64(fmm_tol); quad_tol, η_gcq, gcq_far))
    else
        dielectric_operator(interface; far = something(far, FMM3DFarField(Float64(fmm_tol))),
            near = something(near, GCQNearQuad(; quad_tol = Float64(quad_tol), η_gcq = Float64(η_gcq), gcq_far)))
    end
    b = zeros(Float64, num_points(interface))
    for s in srcs
        b .+= rhs_dielectric_box3d(interface, s, Float64(epsat(s.point)))
    end
    σ, stats = solve_dielectric_box3d_gcq(A, b, interface; rtol, itmax, constraints)
    return (σ = σ, interface = interface, bodies = bodies, constraints = constraints, stats = stats)
end
