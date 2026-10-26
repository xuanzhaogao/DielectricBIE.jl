# Lens and split sphere with GCQ edge panels on curved charts.
#
# The edge is the circle of radius a in z = 0. The upper cap is the part above z = 0 of the
# sphere of radius R1 = a / sin θ1 centred at (0, 0, -R1 cos θ1); it meets the plane z = 0 at
# the angle θ1 ∈ (0, π) inside the upper body. The lower cap is its mirror image with θ2. With
# eps_upper == eps_lower the body is one lens (edge angle θ1 + θ2); otherwise the disk z = 0,
# r < a is an ε_lower | ε_upper interface and the edge is a triple junction with sectors
# θ1 (upper), θ2 (lower) and 2π - θ1 - θ2 (outside). θ1 = θ2 = π/2 is the sphere split at its
# equator.
#
# Charts follow the cube sphere: a top face and four side faces per cap (side k turned by kπ/2
# about z), the side faces running from the edge (U = 0) to the top face (U = 1); the disk is a
# central square and four rim charts. A side chart's point is a(cos ψ, sin ψ, 0) + offsets that
# are written without cancellation for small U (GCQ nodes lie 1e-6 … 1e-12 from the edge), and
# every panel of a side chart is anchored at the edge point of its first v-node line.

# quarter turn k (0..3) about z of the plane vector (x, y)
_qrot(k, x, y) = k == 0 ? (x, y) : k == 1 ? (-y, x) : k == 2 ? (-x, -y) : (y, -x)

# cap side chart k, absolute point (sgn = +1 upper, -1 lower); reference formula
function _cap_side_point(a, θ, k, U, V, sgn)
    y = 2V - 1; s = sqrt(1 + y^2)
    φ = (2θ / π) * atan(s, U)                     # polar angle on the cap sphere
    R = a / sin(θ); r = R * sin(φ)
    x1, y1 = _qrot(k, r / s, r * y / s)
    return (x1, y1, sgn * R * (cos(φ) - cos(θ)))
end

# offset of the cap side point (U, V) from the edge point (0, V0), cancellation-free in U
function _cap_side_offset(a, θ, k, U, V, V0, sgn)
    y = 2V - 1; s = sqrt(1 + y^2); y0 = 2V0 - 1; s0 = sqrt(1 + y0^2)
    Δ = (2θ / π) * atan(U, s)                     # θ minus the polar angle
    R = a / sin(θ); sh = sin(Δ / 2)
    dr = -2R * cos(θ - Δ / 2) * sh; dz = 2R * sin(θ - Δ / 2) * sh
    x1, y1 = _qrot(k, a * (1 / s - 1 / s0) + dr / s, a * (y / s - y0 / s0) + dr * y / s)
    return (x1, y1, sgn * dz)
end

# cap top chart, (xp, yp) ∈ [-1, 1]² (nodes are interior, but a panel may be anchored at the centre)
function _cap_top_point(a, θ, xp, yp, sgn)
    ρ = hypot(xp, yp); φ = (2θ / π) * atan(ρ)
    R = a / sin(θ); f = iszero(ρ) ? R * (2θ / π) : R * sin(φ) / ρ
    return (f * xp, f * yp, sgn * R * (cos(φ) - cos(θ)))
end

# disk rim chart k: from the rim (U = 0) to the edge of the central square [-a/2, a/2]² (U = 1)
function _disk_side_point(a, k, U, V)
    y = 2V - 1; s = sqrt(1 + y^2)
    x1, y1 = _qrot(k, (1 - U) * a / s + U * a / 2, (1 - U) * a * y / s + U * a * y / 2)
    return (x1, y1, zero(x1))
end

function _disk_side_offset(a, k, U, V, V0)
    y = 2V - 1; s = sqrt(1 + y^2); y0 = 2V0 - 1; s0 = sqrt(1 + y0^2)
    x1, y1 = _qrot(k, a * (1 / s - 1 / s0) + U * (a / 2 - a / s), a * (y / s - y0 / s0) + U * (a * y / 2 - a * y / s))
    return (x1, y1, zero(x1))
end

"""
    LensGCQ

Result of `lens_gcq`: the `interface` (`CurvedPanel`s), the geometry `a, θ1, θ2`, the
permittivities, the density `powers` at the edge and the node `ranges` of `:upper`, `:lower`
and (two permittivities) `:disk`.
"""
struct LensGCQ
    interface::DielectricInterface{CurvedPanel, Float64}
    a::Float64
    θ1::Float64
    θ2::Float64
    eps_upper::Float64
    eps_lower::Float64
    eps_out::Float64
    powers::Vector{Float64}
    ranges::Dict{Symbol, UnitRange{Int}}
end

Base.show(io::IO, m::LensGCQ) =
    print(io, "LensGCQ (θ1 = $(m.θ1), θ2 = $(m.θ2), ε = $(m.eps_upper) | $(m.eps_lower) in $(m.eps_out)), ",
          "$(length(m.interface.panels)) panels")

num_points(m::LensGCQ) = num_points(m.interface)

# panel with the normal turned to agree with `ref(point)` at its first node
function _oriented_panel(anchor, lmap, ru, rv, kind, ref)
    p = CurvedPanel(anchor, lmap, ru, rv; kind)
    return dot(p.normals[1], ref(p.points[1])) > 0 ? p : CurvedPanel(anchor, lmap, ru, rv; kind, flip = true)
end

"""
    lens_gcq(a, θ1, θ2, eps_upper, eps_lower, eps_out; nlev = 4, n_quad = 10, nv = 2, m = 2,
             tol = 1e-10, qmax = nothing, gmax = 20.0, edge = :gcq) -> LensGCQ

Lens bounded by two spherical caps over the circle of radius `a` in z = 0, meeting the plane
at the angles `θ1` (upper) and `θ2` (lower), meshed with `CurvedPanel`s (see the file
header). With `eps_upper == eps_lower` it is one body with an edge of angle θ1 + θ2; otherwise
the disk z = 0 separates the two parts (normal +z: `eps_in = eps_lower`, `eps_out = eps_upper`)
and the edge is a triple junction. The cap normals point out of their part.

Each side chart is graded toward the edge: GL panels on [2^-k, 2^-(k-1)], k = 1..`nlev`, and on
[0, 2^-nlev] an edge panel with the weighted GCQ rule of the junction in u
(`edge_gcq_rule(powers; tol, qmax)`) and `n_quad`-point Gauss-Legendre in v; `nv` panels along
each side chart, `m` × `m` panels on the top charts and the disk centre. `edge = :gl` puts
Gauss-Legendre on the edge panels too (an h-graded reference).
"""
function lens_gcq(a::Real, θ1::Real, θ2::Real, eps_upper::Real, eps_lower::Real, eps_out::Real;
                  nlev::Int = 4, n_quad::Int = 10, nv::Int = 2, m::Int = 2, tol::Real = 1e-10,
                  qmax::Union{Nothing, Real} = nothing, gmax::Real = 20.0, edge::Symbol = :gcq)
    0 < θ1 < π && 0 < θ2 < π || throw(ArgumentError("cap angles must lie in (0, π)"))
    edge in (:gcq, :gl) || throw(ArgumentError("edge must be :gcq or :gl"))
    a, θ1, θ2 = Float64(a), Float64(θ1), Float64(θ2)
    eu, el, eo = Float64(eps_upper), Float64(eps_lower), Float64(eps_out)
    split = eu != el
    powers = split ? _junction_density_powers([2π - θ1 - θ2, θ1, θ2], [eo, eu, el]; gmax) :
                     edge_density_powers(eu, eo; angle = θ1 + θ2, gmax)
    rgl = gl_tensor_rule(n_quad)
    redge = edge == :gcq ? gcq_tensor_rule(qmax === nothing ? edge_gcq_rule(powers; tol) :
                                                              edge_gcq_rule(powers; tol, qmax)) : rgl
    kedge = edge == :gcq ? :edge : :gl
    us = vcat([(0.0, 2.0^-nlev)], [(2.0^-k, 2.0^(-k + 1)) for k in nlev:-1:1])
    vs = [((j - 1) / nv, j / nv) for j in 1:nv]
    ts = [(-1 + 2 * (i - 1) / m, -1 + 2 * i / m) for i in 1:m]

    panels = CurvedPanel[]; ein = Float64[]; eout = Float64[]; ranges = Dict{Symbol, UnitRange{Int}}()
    function add!(name, ps, e_in, e_out)
        n0 = sum(num_points, panels; init = 0)
        append!(panels, ps); append!(ein, fill(e_in, length(ps))); append!(eout, fill(e_out, length(ps)))
        ranges[name] = n0+1:n0+sum(num_points, ps)
    end
    for (name, θ, sgn, e_in) in ((:upper, θ1, 1, eu), (:lower, θ2, -1, el))
        c = (0.0, 0.0, -sgn * a / sin(θ) * cos(θ))
        ref = x -> x .- c
        ps = CurvedPanel[]
        for k in 0:3, (V0, V1) in vs, (U0, U1) in us
            anchor = _cap_side_point(a, θ, k, 0.0, V0, sgn)
            lmap = (u, v) -> _cap_side_offset(a, θ, k, U0 + u * (U1 - U0), V0 + v * (V1 - V0), V0, sgn)
            push!(ps, _oriented_panel(anchor, lmap, U0 == 0 ? redge : rgl, rgl, U0 == 0 ? kedge : :gl, ref))
        end
        for (x0, x1) in ts, (y0, y1) in ts
            anchor = _cap_top_point(a, θ, x0, y0, sgn)
            lmap = (u, v) -> _cap_top_point(a, θ, x0 + u * (x1 - x0), y0 + v * (y1 - y0), sgn) .- anchor
            push!(ps, _oriented_panel(anchor, lmap, rgl, rgl, :gl, ref))
        end
        add!(name, ps, e_in, eo)
    end
    if split
        ref = x -> (0.0, 0.0, 1.0)
        ps = CurvedPanel[]
        for k in 0:3, (V0, V1) in vs, (U0, U1) in us
            anchor = _disk_side_point(a, k, 0.0, V0)
            lmap = (u, v) -> _disk_side_offset(a, k, U0 + u * (U1 - U0), V0 + v * (V1 - V0), V0)
            push!(ps, _oriented_panel(anchor, lmap, U0 == 0 ? redge : rgl, rgl, U0 == 0 ? kedge : :gl, ref))
        end
        L = a / m
        for i in 1:m, j in 1:m
            anchor = (-a / 2 + (i - 1) * L, -a / 2 + (j - 1) * L, 0.0)
            push!(ps, _oriented_panel(anchor, (u, v) -> (u * L, v * L, zero(u * v)), rgl, rgl, :gl, ref))
        end
        add!(:disk, ps, el, eu)
    end
    itf = DielectricInterface{CurvedPanel, Float64}(panels, ein, eout)
    return LensGCQ(itf, a, θ1, θ2, eu, el, eo, powers, ranges)
end
