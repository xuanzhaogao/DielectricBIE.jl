# Lens / split sphere as user-side SurfacePatches (test helper; the package has no built-in shapes).
#
# Edge: the circle of radius a in z = 0. The upper cap is the part above z = 0 of the sphere of radius
# a / sin θ1 centred at (0, 0, -a cot θ1), meeting the plane at the angle θ1 inside the upper body; the
# lower cap mirrors it with θ2. Cube-sphere charts: four side patches per cap (side k turned by kπ/2
# about z, the rim at u = 0) and a top patch; with two permittivities the disk z = 0 (four rim patches
# and a central square, normal +z) separates the two parts. Written generically so that BigFloat works.
using DielectricBIE

_qrot(k, x, y) = k == 0 ? (x, y) : k == 1 ? (-y, x) : k == 2 ? (-x, -y) : (y, -x)

function cap_side_point(a, θ, k, U, V, sgn)
    y = 2V - 1; s = sqrt(1 + y^2)
    φ = (2θ / π) * atan(s, U)                     # polar angle on the cap sphere
    R = a / sin(θ); r = R * sin(φ)
    x1, y1 = _qrot(k, r / s, r * y / s)
    return (x1, y1, sgn * R * (cos(φ) - cos(θ)))
end

function cap_top_point(a, θ, U, V, sgn)
    xp = 2U - 1; yp = 2V - 1
    ρ = hypot(xp, yp); φ = (2θ / π) * atan(ρ)
    R = a / sin(θ); f = iszero(ρ) ? R * (2θ / π) : R * sin(φ) / ρ
    return (f * xp, f * yp, sgn * R * (cos(φ) - cos(θ)))
end

function disk_side_point(a, k, U, V)
    y = 2V - 1; s = sqrt(1 + y^2)
    x1, y1 = _qrot(k, (1 - U) * a / s + U * a / 2, (1 - U) * a * y / s + U * a * y / 2)
    return (x1, y1, zero(x1))
end

disk_centre_point(a, U, V) = (a * (U - 0.5), a * (V - 0.5), zero(U * V))

# orient a patch so that its normal agrees with ref(point) at an interior point (not the centre: the top
# chart is singular there)
function oriented_patch(f, ref; eps_in, eps_out)
    p = SurfacePatch(f; eps_in, eps_out)
    _, _, n = DielectricBIE.patch_frame(p, 0.3, 0.6)
    return sum(n .* ref(f(0.3, 0.6))) > 0 ? p : SurfacePatch(f; eps_in, eps_out, flip = true)
end

"Patches of the lens (eps_upper == eps_lower) or of the split lens (a disk interface between the parts)."
function lens_patches(a, θ1, θ2, eps_upper, eps_lower, eps_out)
    ps = SurfacePatch[]
    for (θ, sgn, e) in ((θ1, 1, eps_upper), (θ2, -1, eps_lower))
        c = (0.0, 0.0, -sgn * a / sin(θ) * cos(θ))
        ref = x -> x .- c
        for k in 0:3
            push!(ps, oriented_patch((u, v) -> cap_side_point(a, θ, k, u, v, sgn), ref; eps_in = e, eps_out))
        end
        push!(ps, oriented_patch((u, v) -> cap_top_point(a, θ, u, v, sgn), ref; eps_in = e, eps_out))
    end
    if eps_upper != eps_lower
        ref = x -> (0.0, 0.0, 1.0)
        for k in 0:3
            push!(ps, oriented_patch((u, v) -> disk_side_point(a, k, u, v), ref; eps_in = eps_lower, eps_out = eps_upper))
        end
        push!(ps, oriented_patch((u, v) -> disk_centre_point(a, u, v), ref; eps_in = eps_lower, eps_out = eps_upper))
    end
    return ps
end

"The unit cube [0, 1]³ as six flat patches (outward normals)."
function cube_patches(eps_in, eps_out)
    faces = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
             ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
             ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    ref = x -> x .- (0.5, 0.5, 0.5)
    return [oriented_patch((u, v) -> o .+ u .* e1 .+ v .* e2, ref; eps_in, eps_out) for (o, e1, e2) in faces]
end
