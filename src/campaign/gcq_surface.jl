# GCQ campaign mode: the incident field of a density cut by any box faces, by surface corrections of its smooth field.
#
# With a = 1/eps per region, phi0 = N[rho], psi = grad phi0 and H = grad psi (TKM of the full, smooth density), the
# incident field g = grad N[a rho] at a point y inside an interface face (every GCQ node) is
#
#     g(y) = (a_A + a_B)/2 psi(y) + sum_m (a_A - a_B)_m (D_m[psi](y) - S_m[s](y)),   s = H n + rho n = H n - tr(H) n
#
# over every distinct interface rectangle m (normal n_m = +e_k from A to B), with D the double layer (source normal;
# zero on the target's own face and on coplanar ones) and S the single layer. This is Green's identity for each region,
# grad N[chi rho] = chi psi + D[psi] - S[H n + rho n], with the two copies of each shared face merged. Derivation and
# benchmarks: fmm_experiment/cut_orbital/rhs_surface_correction.md.
#
# The integrands vary on the scale of the density, finer than the interface mesh of the solve resolves, so they live on
# their own source surface: SCTL Gauss-Legendre elements on the interface rectangles, graded toward the cut densities.
# psi and H are evaluated at its nodes (TKM near the density, point charges elsewhere); the node rule of both layers
# is one FMM (three densities, charges and dipoles), and SCTL's singular and near-singular quadrature corrects the
# near pairs at the targets (the interface nodes). The corrections depend on the geometry only: built once.

"""
    GCQSurface

The source surface of the surface corrections: nodes `X`, unit normals `Nrm` (+e_k of their rectangle), weights `W`
and the jump `coef` = a_minus − a_plus of a = 1/ε across their rectangle; near corrections (accurate minus node rule,
unit coefficient) of the double layer `CD` and the single layer `CS` from the nodes to the targets (targets × nodes),
from SCTL ([`gcq_surface_corrections`](@ref)).
"""
struct GCQSurface
    X::Matrix{Float64}
    Nrm::Matrix{Float64}
    W::Vector{Float64}
    coef::Vector{Float64}
    CD::SparseMatrixCSC{Float64, Int}
    CS::SparseMatrixCSC{Float64, Int}
end

"""
    gcq_surface_corrections(rects, targets; order = 8, tol = 1e-10) -> GCQSurface

SCTL Gauss-Legendre elements of `order`² nodes on the rectangles `rects` (`(k, c, lo, hi, coef)`: the plane
x[k] = c, `lo`..`hi` in the two other axes in increasing order, normal +e_k, coefficient `coef`), and the near
corrections of the Laplace double layer (source normal) and single layer from them to `targets` (3 × Nt) at
accuracy `tol`. Needs `using SCTL`.
"""
function gcq_surface_corrections end

_surf_tang(k) = k == 1 ? (2, 3) : k == 2 ? (1, 3) : (1, 2)
_box_lohi(b) = (ntuple(d -> b.center[d] - (b.Lx, b.Ly, b.Lz)[d] / 2, 3), ntuple(d -> b.center[d] + (b.Lx, b.Ly, b.Lz)[d] / 2, 3))

# a = 1/ε at a point off the interfaces (the exterior: 1/eps_out)
function _surf_a(boxes, epses, eps_out, x)
    for (b, e) in zip(boxes, epses)
        lo, hi = _box_lohi(b)
        all(lo[d] < x[d] < hi[d] for d in 1:3) && return 1 / e
    end
    return 1 / eps_out
end

# every distinct interface rectangle (the cells of the grid of all face coordinates, as multi_dielectric_box3d_gcq):
# (k, c, lo, hi, a_minus - a_plus), normal +e_k
function _gcq_interface_rects(boxes, epses, eps_out)
    lh = _box_lohi.(boxes)
    # the face coordinates per axis, merged within rounding (center ± L/2 of neighbouring boxes)
    function merged(v)
        v = sort(v); out = [v[1]]
        for x in v[2:end]
            abs(x - out[end]) <= 1e-12 * max(1.0, abs(x)) || push!(out, x)
        end
        return out
    end
    X = [merged(vcat([l[1][d] for l in lh], [l[2][d] for l in lh])) for d in 1:3]
    R = Tuple{Int, Float64, NTuple{2, Float64}, NTuple{2, Float64}, Float64}[]
    for k in 1:3
        t1, t2 = _surf_tang(k)
        for i in eachindex(X[k]), j in 1:length(X[t1])-1, l in 1:length(X[t2])-1
            mid = zeros(3); mid[t1] = (X[t1][j] + X[t1][j + 1]) / 2; mid[t2] = (X[t2][l] + X[t2][l + 1]) / 2
            del = 1e-9 * max(1.0, abs(X[k][i]))
            mid[k] = X[k][i] - del; am = _surf_a(boxes, epses, eps_out, mid)
            mid[k] = X[k][i] + del; ap = _surf_a(boxes, epses, eps_out, mid)
            am == ap || push!(R, (k, X[k][i], (X[t1][j], X[t2][l]), (X[t1][j + 1], X[t2][l + 1]), am - ap))
        end
    end
    return R
end

# the rectangles split until each is at most κ (d + L0) in size, d its distance to the nearest box (lo, hi) of env
function _gcq_surface_rects(rects, env; κ::Real = 0.25, L0::Real)
    out = similar(rects, 0)
    for (k, c, lo0, hi0, coef) in rects
        t1, t2 = _surf_tang(k)
        stack = [(lo0, hi0)]
        while !isempty(stack)
            lo, hi = pop!(stack)
            sz = max(hi[1] - lo[1], hi[2] - lo[2])
            d = minimum(env; init = Inf) do (blo, bhi)
                dk = max(blo[k] - c, 0.0, c - bhi[k])
                d1 = max(blo[t1] - hi[1], 0.0, lo[1] - bhi[t1]); d2 = max(blo[t2] - hi[2], 0.0, lo[2] - bhi[t2])
                sqrt(dk^2 + d1^2 + d2^2)
            end
            if sz > κ * (d + L0)
                m1 = (lo[1] + hi[1]) / 2; m2 = (lo[2] + hi[2]) / 2
                push!(stack, (lo, (m1, m2)), ((m1, lo[2]), (hi[1], m2)), ((lo[1], m2), (m1, hi[2])), ((m1, m2), hi))
            else
                push!(out, (k, c, lo, hi, coef))
            end
        end
    end
    return out
end

# targets on a face snapped onto the plane of the face's rectangles: a target at a face, along its normal axis k within
# rounding of a rectangle plane x[k] = c (the mesh and the rectangles compute face coordinates separately), gets
# x[k] = c exactly, so that the face's own double layer is its on-plane value (zero), not a one-sided limit
function _gcq_surface_snap(rects, Y::AbstractMatrix, nrm::AbstractMatrix)
    planes = [sort!(unique([r[2] for r in rects if r[1] == k])) for k in 1:3]
    Z = Matrix{Float64}(Y)
    for j in axes(Z, 2)
        k = findfirst(d -> abs(abs(nrm[d, j]) - 1) < 1e-12, 1:3)
        k === nothing && continue
        for c in planes[k]
            abs(Z[k, j] - c) <= 1e-12 * max(1.0, abs(c)) && (Z[k, j] = c; break)
        end
    end
    return Z
end

# the trace coefficient (a_A + a_B)/2 at the interface nodes (the average of a across each node's face)
_gcq_surface_trace(interface::DielectricInterface) =
    reduce(vcat, [fill((1 / interface.eps_in[i] + 1 / interface.eps_out[i]) / 2, num_points(p)) for (i, p) in enumerate(interface.panels)])

# s = H n - tr(H) n at points with normals nrm, from the Hessian rows (xx, yy, zz, xy, xz, yz)
function _gcq_surface_sdensity(Hs::AbstractMatrix, nrm::AbstractMatrix)
    Sd = zeros(3, size(Hs, 2))
    for i in axes(Hs, 2)
        h = view(Hs, :, i); n1, n2, n3 = nrm[1, i], nrm[2, i], nrm[3, i]; tr = h[1] + h[2] + h[3]
        Sd[1, i] = h[1] * n1 + h[4] * n2 + h[5] * n3 - tr * n1
        Sd[2, i] = h[4] * n1 + h[2] * n2 + h[6] * n3 - tr * n2
        Sd[3, i] = h[5] * n1 + h[6] * n2 + h[3] * n3 - tr * n3
    end
    return Sd
end

"""
    _gcq_surface_layers(S, Y, Ψs, Hs; tol = 1e-12) -> 3 × Nt

sum_m coef_m (D_m[psi] - S_m[s]) at the targets `Y` (those of `S`'s corrections), for psi (`Ψs`, 3 × nodes) and its
Hessian (`Hs`, 6 × nodes) at the source nodes: the node rule by one FMM3D call (three densities, charges and dipoles
along the normals; FMM3D's q/r and d·(t - s)/r³) plus SCTL's near corrections.
"""
function _gcq_surface_layers(S::GCQSurface, Y::AbstractMatrix, Ψs::AbstractMatrix, Hs::AbstractMatrix; tol = 1e-12)
    Sd = _gcq_surface_sdensity(Hs, S.Nrm)
    ns = size(S.X, 2); chg = zeros(3, ns); dip = zeros(3, 3, ns)
    for i in 1:ns
        cw = S.coef[i] * S.W[i]
        for f in 1:3
            chg[f, i] = -cw * Sd[f, i]
            for d in 1:3
                dip[f, d, i] = cw * Ψs[f, i] * S.Nrm[d, i]
            end
        end
    end
    v = lfmm3d(tol, S.X; charges = chg, dipvecs = dip, targets = Matrix{Float64}(Y), pgt = 1, nd = 3)
    return v.pottarg ./ (4π) .+ Matrix((S.CD * (S.coef .* Ψs'))') .- Matrix((S.CS * (S.coef .* Sd'))')
end
