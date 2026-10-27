# Curved geometries: parametric surface patches, the curves they share, and the junction at each curve.
#
# A body (or several, with interfaces between them) is a list of `SurfacePatch`es: smooth maps of the
# parameter square [0,1]², each with the permittivity on either side. `CurvedGeometry` finds which patch
# sides coincide (every side must be shared: the surfaces are closed) and, for each shared curve, the
# junction around it (after chunkie's chunkgraph, lifted to 3D): in the plane normal to the curve, every
# patch leaves the curve along its inward tangent; sorted by angle about the curve tangent they bound the
# sectors, and each sector's permittivity is read from its two bounding patches, which must agree. A curve
# met by two patches at the angle π is a smooth seam (two charts of one smooth interface); every other curve
# is an edge, whose density powers follow from the sector angles and permittivities.
#
# Sides are numbered 1: u = 0, 2: u = 1, 3: v = 0, 4: v = 1.

"""
    SurfacePatch(map; eps_in, eps_out, flip = false)

Smooth surface patch x = map(u, v), (u, v) ∈ [0,1]² (a 3-tuple; differentiable by ForwardDiff, and generic
in the number type for high-precision checks). Its unit normal is (x_u × x_v)/|x_u × x_v|, negated with
`flip`; `eps_out` is the permittivity on the normal side, `eps_in` on the other (`Inf`: perfect conductor).
"""
struct SurfacePatch{F}
    map::F
    eps_in::Float64
    eps_out::Float64
    flip::Bool
end

SurfacePatch(map; eps_in::Real, eps_out::Real, flip::Bool = false) =
    SurfacePatch(map, Float64(eps_in), Float64(eps_out), flip)

"""
    patch_frame(p::SurfacePatch, u, v) -> (x_u, x_v, n, J)

Tangents, unit normal (with the patch's orientation) and Jacobian J = |x_u × x_v| at (u, v).
"""
patch_frame(p::SurfacePatch, u, v) = _curved_frame(p.map, p.flip, u, v)

# parameters of the point t ∈ [0, 1] of side s, and the derivative into the patch across it
_side_uv(s, t) = s == 1 ? (0.0, t) : s == 2 ? (1.0, t) : s == 3 ? (t, 0.0) : (t, 1.0)
_side_point(p, s, t) = NTuple{3, Float64}(p.map(_side_uv(s, t)...))
function _inward_tangent(p, s, t)
    xu, xv, _, _ = patch_frame(p, _side_uv(s, t)...)
    return s == 1 ? xu : s == 2 ? .-xu : s == 3 ? xv : .-xv
end

# the parameter of the point of side s of p closest to x (coarse samples, then golden section)
function _closest_on_side(p, s, x)
    d(t) = sum(abs2, _side_point(p, s, t) .- x)
    ts = range(0, 1, length = 65)
    i = argmin(d.(ts))
    a, b = ts[max(i - 1, 1)], ts[min(i + 1, end)]
    g = (sqrt(5) - 1) / 2
    for _ in 1:80
        c = b - g * (b - a); e = a + g * (b - a)
        d(c) < d(e) ? (b = e) : (a = c)
    end
    t = (a + b) / 2
    return t, sqrt(d(t))
end

"""
    CurveJunction

A curve shared by patch sides: `sides` (patch, side) in the angular order about the curve tangent at
`midpoint`; `angles[k]` the sector between sides k and k+1 and `eps[k]` its permittivity; `smooth` when
two patches meet at the angle π (a seam inside one smooth interface).
"""
struct CurveJunction
    sides::Vector{Tuple{Int, Int}}
    angles::Vector{Float64}
    eps::Vector{Float64}
    smooth::Bool
    midpoint::NTuple{3, Float64}
end

"""
    CurvedGeometry(patches; tol = 1e-9)

The patches, the curves they share (`curves::Vector{CurveJunction}`) and the curve of every patch side
(`side_curve[s, k]`). Two sides are one curve when their end points coincide and each passes through the
other's middle point, to `tol` times the size of the geometry. Every side must be shared, the
permittivities around a curve must agree, and the sector angles must be the same along it (checked at
three points); otherwise an `ArgumentError` is thrown.
"""
struct CurvedGeometry
    patches::Vector{SurfacePatch}
    curves::Vector{CurveJunction}
    side_curve::Matrix{Int}
end

function CurvedGeometry(patches::AbstractVector{<:SurfacePatch}; tol::Real = 1e-9)
    ps = Vector{SurfacePatch}(patches)
    np = length(ps)
    ends = [(_side_point(p, s, 0.0), _side_point(p, s, 0.5), _side_point(p, s, 1.0)) for s in 1:4, p in ps]
    allx = [x for e in ends for x in e]
    scale = maximum(norm(collect(x .- y)) for x in allx for y in allx[1:min(end, 50)])
    δ = tol * scale
    near(x, y) = norm(collect(x .- y)) <= δ
    same(s, k, s2, k2) = begin
        a0, am, a1 = ends[s, k]; b0, bm, b1 = ends[s2, k2]
        ((near(a0, b0) && near(a1, b1)) || (near(a0, b1) && near(a1, b0))) &&
            _closest_on_side(ps[k2], s2, am)[2] <= δ && _closest_on_side(ps[k], s, bm)[2] <= δ
    end
    side_curve = zeros(Int, 4, np)
    groups = Vector{Vector{Tuple{Int, Int}}}()
    for k in 1:np, s in 1:4
        side_curve[s, k] == 0 || continue
        grp = [(k, s)]
        for k2 in k:np, s2 in 1:4
            (k2, s2) == (k, s) || side_curve[s2, k2] != 0 || !same(s, k, s2, k2) || push!(grp, (k2, s2))
        end
        length(grp) >= 2 || throw(ArgumentError("side $s of patch $k is shared with no other patch: the surfaces must be closed"))
        push!(groups, grp)
        for (kk, ss) in grp
            side_curve[ss, kk] = length(groups)
        end
    end
    curves = [_curve_junction(ps, grp) for grp in groups]
    return CurvedGeometry(ps, curves, side_curve)
end

Base.show(io::IO, g::CurvedGeometry) =
    print(io, "CurvedGeometry with $(length(g.patches)) patches, $(length(g.curves)) curves ",
          "($(count(c -> !c.smooth, g.curves)) edges)")

# sector angles and permittivities about the curve at the point t0 of the first side
function _junction_at(ps, grp, t0)
    k1, s1 = grp[1]
    x0 = _side_point(ps[k1], s1, t0)
    xu, xv, _, _ = patch_frame(ps[k1], _side_uv(s1, t0)...)
    T = collect(s1 <= 2 ? xv : xu); T ./= norm(T)
    dirs = Vector{Vector{Float64}}(); nrms = Vector{Vector{Float64}}()
    for (k, s) in grp
        t = k == k1 && s == s1 ? t0 : _closest_on_side(ps[k], s, x0)[1]
        d = collect(_inward_tangent(ps[k], s, t)); d .-= dot(d, T) .* T; d ./= norm(d)
        push!(dirs, d); push!(nrms, collect(patch_frame(ps[k], _side_uv(s, t)...)[3]))
    end
    e1 = dirs[1]; e2 = cross(T, e1)
    φ = [mod(atan(dot(d, e2), dot(d, e1)), 2π) for d in dirs]
    o = sortperm(φ)
    K = length(grp)
    angles = [k < K ? φ[o[k + 1]] - φ[o[k]] : 2π - φ[o[K]] + φ[o[1]] for k in 1:K]
    eps = map(1:K) do k
        a = o[k]; b = o[mod1(k + 1, K)]
        # the sector lies counterclockwise of side a and clockwise of side b (about T)
        ea = dot(nrms[a], cross(T, dirs[a])) > 0 ? ps[grp[a][1]].eps_out : ps[grp[a][1]].eps_in
        eb = dot(nrms[b], -cross(T, dirs[b])) > 0 ? ps[grp[b][1]].eps_out : ps[grp[b][1]].eps_in
        ea == eb || throw(ArgumentError("the permittivities around the curve through $x0 disagree: " *
                                        "patch $(grp[a][1]) says $ea, patch $(grp[b][1]) says $eb"))
        ea
    end
    return grp[o], angles, eps, x0
end

function _curve_junction(ps, grp)
    sides, angles, eps, x0 = _junction_at(ps, grp, 0.5)
    for t in (0.2, 0.8)
        _, a, _, x = _junction_at(ps, grp, t)
        maximum(abs.(sort(a) .- sort(angles))) <= 1e-8 ||
            throw(ArgumentError("the sector angles vary along the curve through $x0 (at $x: $(sort(a)) vs " *
                                "$(sort(angles))); edges of varying angle are not supported"))
    end
    smooth = length(sides) == 2 && all(a -> abs(a - π) <= 1e-9, angles)
    return CurveJunction(sides, angles, eps, smooth, x0)
end
