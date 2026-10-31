# The intersection of two solid cylinders of radius r, axes along x at height z1 and along y at height z2
# (0 < z2 - z1 < 2r), as SurfacePatches. Its boundary is Γ1 (on the x-cylinder, inside the y-cylinder: the
# top of the x-cylinder) and Γ2 (on the y-cylinder, inside the x-cylinder: its bottom), two disks glued
# along one smooth closed edge on which the interior angle varies: α = π - acos((z - z1)(z - z2)/r²).
#
# Edge: subtracting the cylinder equations gives z = zc + (x² - y²)/(2Δ) (Δ = z2 - z1, zc = (z1 + z2)/2),
# then in polar coordinates x = R cos θ, y = R sin θ, R² = Q(θ) = 2c / (1 + sqrt(1 + 2c cos²2θ / Δ²)),
# c = 2r² - Δ²/2, and z = zc + Q cos 2θ / (2Δ): smooth and regular in θ.
# Faces: each cylinder is developed isometrically onto the plane, (x, s) -> (x, r sin(s/r), z1 + r cos(s/r))
# for Γ1 and (y, s) -> (r sin(s/r), y, z2 - r cos(s/r)) for Γ2, where the face is a smooth convex-like domain
# bounded by the edge γ(θ). Each face: four rim charts (θ ∈ [θk, θk + π/2], θk = (2k - 1)π/4, edge at U = 0,
# blended linearly to a side of an inner rectangle at U = 1) and the rectangle. Generic in the number type.
using DielectricBIE

function _cyl_edge(r, z1, z2, θ)
    Δ = z2 - z1; c = 2r^2 - Δ^2 / 2
    Q = 2c / (1 + sqrt(1 + 2c * cos(2θ)^2 / Δ^2))
    R = sqrt(Q)
    return (R * cos(θ), R * sin(θ), (z1 + z2) / 2 + Q * cos(2θ) / (2Δ))
end

# the edge point θ in the developed coordinates of face f (1: (x, s) on the x-cylinder, 2: (y, s) on the y-cylinder)
function _cyl_dev_edge(r, z1, z2, f, θ)
    X, Y, Z = _cyl_edge(r, z1, z2, θ)
    return f == 1 ? (X, r * atan(Y, Z - z1)) : (Y, r * atan(X, z2 - Z))
end

_cyl_lift(r, z1, z2, f, p, s) = f == 1 ? (p, r * sin(s / r), z1 + r * cos(s / r)) : (r * sin(s / r), p, z2 - r * cos(s / r))

"""
    cylinders_patches(r, z1, z2, eps_in, eps_out; λ = 0.5)

The 10 patches (5 per face) of the intersection of the cylinders y² + (z - z1)² ≤ r² and x² + (z - z2)² ≤ r².
The inner rectangle of each face spans λ times the face's extent in each developed direction.
"""
function cylinders_patches(r, z1, z2, eps_in, eps_out; λ = 0.5)
    0 < z2 - z1 < 2r || throw(ArgumentError("need 0 < z2 - z1 < 2r"))
    ps = SurfacePatch[]
    for f in (1, 2)
        ex = abs(_cyl_dev_edge(r, z1, z2, f, f == 1 ? 0.0 : π / 2)[1])   # half-widths of the developed face
        es = abs(_cyl_dev_edge(r, z1, z2, f, f == 1 ? π / 2 : 0.0)[2])
        corner(k) = (g = _cyl_dev_edge(r, z1, z2, f, (2k + 1) * π / 4); (λ * ex * sign(g[1]), λ * es * sign(g[2])))
        ref = f == 1 ? (x -> (zero(x[1]), x[2], x[3] - z1)) : (x -> (x[1], zero(x[2]), x[3] - z2))
        for k in 0:3
            ca = corner(k); cb = corner(k + 1)
            fk = (U, V) -> begin
                θ = (2k + 1 + 2V) * π / 4
                g = _cyl_dev_edge(r, z1, z2, f, θ)
                p = (1 - U) * g[1] + U * ((1 - V) * ca[1] + V * cb[1])
                s = (1 - U) * g[2] + U * ((1 - V) * ca[2] + V * cb[2])
                _cyl_lift(r, z1, z2, f, p, s)
            end
            push!(ps, _cyl_oriented(fk, ref, eps_in, eps_out))
        end
        c0 = corner(2); c2 = corner(0)
        fc = (U, V) -> _cyl_lift(r, z1, z2, f, c0[1] + U * (c2[1] - c0[1]), c0[2] + V * (c2[2] - c0[2]))
        push!(ps, _cyl_oriented(fc, ref, eps_in, eps_out))
    end
    return ps
end

function _cyl_oriented(f, ref, eps_in, eps_out)
    p = SurfacePatch(f; eps_in, eps_out)
    _, _, n = DielectricBIE.patch_frame(p, 0.3, 0.6)
    return sum(n .* ref(f(0.3, 0.6))) > 0 ? p : SurfacePatch(f; eps_in, eps_out, flip = true)
end
