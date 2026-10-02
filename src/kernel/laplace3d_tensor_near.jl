# Near-singular D^T corrections for `TensorPanel` interfaces.
#
# D^T vanishes between coplanar points, so on box geometries only panels on
# different planes interact. For a target near a source panel, the exact row
#     ∫_panel K(x - y) σ(y) dA(y),   σ in the panel's tensor interpolation space,
# is computed by product integration: every (fine u-segment, fine v-segment)
# pair is subdivided (quadtree) until each leaf is farther than η·(its size)
# from the target, and leaves use the segments' own nodes or 12-point
# Gauss-Legendre (Gauss-Jacobi for the weight x^pw on a leaf touching an edge)
# with the segments' Lagrange bases. The correction stored is
#     exact row - plain row (absolute coordinates, exactly what FMM adds),
# so that   FMM(D^T) + C   is the corrected operator.
#
# The entries are invariant under scaling, rotations and reflections of the
# whole configuration, so rows are keyed by the target position and normal in
# the source panel's normalized frame and computed once per distinct key; on
# vertex-graded box meshes only a few percent of the near rows are distinct.

const TENSOR_NEAR_M = 12
const TENSOR_NEAR_GL = gausslegendre(TENSOR_NEAR_M)
const TENSOR_GJ_CACHE = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
const TENSOR_GJ_LOCK = ReentrantLock()
_tensor_gj(pw) = lock(() -> get!(() -> gaussjacobi(TENSOR_NEAR_M, 0.0, pw), TENSOR_GJ_CACHE, pw),
                      TENSOR_GJ_LOCK)

# Lagrange basis at points t (columns); exact-node test only, so points close to
# a node are still interpolated (barycentric_row! snaps within isapprox)
function _tensor_lagrange_mat(z, bw, t)
    L = zeros(length(z), length(t))
    for (a, ta) in enumerate(t)
        j = findfirst(==(ta), z)
        if j !== nothing
            L[j, a] = 1.0
            continue
        end
        c = bw ./ (ta .- z)
        L[:, a] .= c ./ sum(c)
    end
    return L
end

# distance from local point xl to the sub-rectangle [ua,ub]×[va,vb] of p
function _tensor_rect_dist(p::TensorPanel, xl, ua = 0.0, ub = 1.0, va = 0.0, vb = 1.0)
    r = xl .- p.o
    u = dot(r, p.eu) / p.Lu; v = dot(r, p.ev) / p.Lv
    uc = clamp(u, ua, ub); vc = clamp(v, va, vb)
    return norm(r .- (uc * p.Lu) .* p.eu .- (vc * p.Lv) .* p.ev)
end

# points, weights (with x^pw and length L) and basis for [a,b] ⊂ segment k
function _tensor_leaf_rule(r::TensorRule1D, L, k, a, b)
    sa, sb = r.segs[k]
    wt(u) = r.pw == 0 ? 1.0 : u^r.pw
    if a == sa && b == sb && length(r.z0) >= TENSOR_NEAR_M
        return r.fx[:, k], L .* r.fw[:, k], nothing
    elseif r.pw != 0 && a == 0
        zj, wj = _tensor_gj(r.pw)
        pts = b .* (1 .+ zj) ./ 2
        wts = (b / 2)^(r.pw + 1) .* wj .* L
    else
        zm, wm = TENSOR_NEAR_GL
        pts = a .+ (b - a) .* (1 .+ zm) ./ 2
        wts = (b - a) / 2 .* wm .* L .* wt.(pts)
    end
    return pts, wts, _tensor_lagrange_mat(r.z0, r.bw, 2 .* (pts .- sa) ./ (sb - sa) .- 1)
end

"""
    tensor_near_row(kern, p, xl; η=1.5, maxdepth=80)

Row r with Σ_k r_k σ_k ≈ ∫_p kern(x - y) σ(y) dA(y) for σ in the panel's tensor
interpolation space; `xl` is the target in p's anchor frame (x - p.anchor).
"""
function tensor_near_row(kern, p::TensorPanel, xl; η = 1.5, maxdepth = 80)
    ru, rv = p.ru, p.rv
    nzu, nzv = length(ru.z0), length(rv.z0)
    rowfine = zeros(length(ru.segs) * nzu, length(rv.segs) * nzv)
    direct = nzu >= TENSOR_NEAR_M && nzv >= TENSOR_NEAR_M
    stack = NTuple{5, Float64}[]
    for (ku, (sau, sbu)) in enumerate(ru.segs), (kv, (sav, sbv)) in enumerate(rv.segs)
        iu = (ku - 1) * nzu .+ (1:nzu); iv = (kv - 1) * nzv .+ (1:nzv)
        if direct && _tensor_rect_dist(p, xl, sau, sbu, sav, sbv) >
                     η * max((sbu - sau) * p.Lu, (sbv - sav) * p.Lv)
            @inbounds for b in 1:nzv, a in 1:nzu
                y = tensor_lpoint(p, ru.fx[a, ku], rv.fx[b, kv])
                rowfine[iu[a], iv[b]] += kern(xl .- y) * (p.Lu * ru.fw[a, ku]) * (p.Lv * rv.fw[b, kv])
            end
            continue
        end
        empty!(stack); push!(stack, (sau, sbu, sav, sbv, 0.0))
        while !isempty(stack)
            ua, ub, va, vb, depth = pop!(stack)
            du = (ub - ua) * p.Lu; dv = (vb - va) * p.Lv
            d = _tensor_rect_dist(p, xl, ua, ub, va, vb)
            if d <= η * max(du, dv) && depth < maxdepth
                su = du > dv / 2 && η * du >= d
                sv = dv > du / 2 && η * dv >= d
                !su && !sv && (du >= dv ? (su = true) : (sv = true))
                us = su ? ((ua, (ua + ub) / 2), ((ua + ub) / 2, ub)) : ((ua, ub),)
                vs = sv ? ((va, (va + vb) / 2), ((va + vb) / 2, vb)) : ((va, vb),)
                for (a1, b1) in us, (a2, b2) in vs
                    push!(stack, (a1, b1, a2, b2, depth + 1))
                end
                continue
            end
            pu, wu, Bu = _tensor_leaf_rule(ru, p.Lu, ku, ua, ub)
            pv, wv, Bv = _tensor_leaf_rule(rv, p.Lv, kv, va, vb)
            K = [kern(xl .- tensor_lpoint(p, pu[a], pv[b])) * wu[a] * wv[b]
                 for a in eachindex(pu), b in eachindex(pv)]
            C = Bu === nothing ? K : Bu * K
            C = Bv === nothing ? C : C * Bv'
            @views rowfine[iu, iv] .+= C
        end
    end
    M = ru.P' * rowfine * rv.P
    return vec(permutedims(M))
end

# Bernstein-ellipse parameter of a target at reference coordinate t ∈ R and
# scaled perpendicular distance δ from the segment [-1, 1]
function _bernstein_rho(t, δ)
    z = complex(t, δ)
    w = abs(z + sqrt(z - 1) * sqrt(z + 1))
    return max(w, 1 / w)
end

"""
    tensor_isfar(p, xl; quad_tol=1e-12, η_gcq=2.0)

Whether the panel's own rule may be used for target `xl` (in p's anchor frame).
A Gauss-Legendre direction with n points has error ≈ ρ^{-2n}, with ρ the
Bernstein-ellipse parameter of the target for that direction (its distance from
the panel in the other two directions counted as the ellipse's minor axis); the
sum over the GL directions must be below `quad_tol`. A GCQ direction (span of
polynomial degree ≈ 20 times the edge powers) requires a distance of at least
`η_gcq` panel sizes.
"""
function tensor_isfar(p::TensorPanel, xl; quad_tol = 1e-12, η_gcq = 2.0)
    if (is_gcq(p.ru) || is_gcq(p.rv)) && _tensor_rect_dist(p, xl) <= η_gcq * max(p.Lu, p.Lv)
        return false
    end
    r = xl .- p.o
    u = dot(r, p.eu) / p.Lu; v = dot(r, p.ev) / p.Lv; h = abs(dot(r, p.normal))
    du = max(0.0, -u, u - 1) * p.Lu; dv = max(0.0, -v, v - 1) * p.Lv
    est = 0.0
    is_gcq(p.ru) || (est += _bernstein_rho(2u - 1, 2 * hypot(h, dv) / p.Lu)^(-2 * length(p.ru.x)))
    is_gcq(p.rv) || (est += _bernstein_rho(2v - 1, 2 * hypot(h, du) / p.Lv)^(-2 * length(p.rv.x)))
    return est < quad_tol
end

function _tensor_near_key(ps::TensorPanel, xl, nx)
    r = xl .- ps.o
    q(t) = round(t, sigdigits = 11) + 0.0
    return (objectid(ps.ru), objectid(ps.rv), q(ps.Lv / ps.Lu),
            q(dot(r, ps.eu) / ps.Lu), q(dot(r, ps.ev) / ps.Lu), q(dot(r, ps.normal) / ps.Lu),
            q(dot(nx, ps.eu)), q(dot(nx, ps.ev)), q(dot(nx, ps.normal)))
end

# ∂_{n_x} G(x,y) for d = x - y, G = 1/(4π|d|)  (the negative of laplace3d_grad(y, x, n))
_tensor_dnG(nx, d) = -(nx[1] * d[1] + nx[2] * d[2] + nx[3] * d[3]) / (4π * sqrt(d[1]^2 + d[2]^2 + d[3]^2)^3)

"""
    laplace3d_DT_tensor_corrections(interface; reuse=true, quad_tol=1e-12, η_gcq=2.0)

Sparse correction C such that `laplace3d_DT_fmm3d(interface, tol) + C` is the
near-corrected D^T operator in DielectricBIE's sign convention
(entries laplace3d_grad(y_j, x_i, n_i) w_j). Pairs that `tensor_isfar` rejects at
`quad_tol` get the product-integration row. Returns (C, stats).
"""
function laplace3d_DT_tensor_corrections(interface::DielectricInterface{TensorPanel{T}, T};
                                         reuse::Bool = true, quad_tol = 1e-12, η_gcq = 2.0) where T
    panels = interface.panels
    counts = num_points.(panels)
    offs = cumsum([0; counts])
    rng = [offs[k]+1:offs[k+1] for k in eachindex(panels)]
    Yl = [tensor_local_points(p) for p in panels]
    near = [Tuple{Int, Int, NTuple{3, T}, NTuple{3, T}}[] for _ in panels]
    Threads.@threads :dynamic for it in eachindex(panels)
        pt = panels[it]
        for (i, xo) in zip(rng[it], Yl[it])
            for (js, ps) in enumerate(panels)
                ps.plane == pt.plane && continue
                xl = (pt.anchor .- ps.anchor) .+ xo
                tensor_isfar(ps, xl; quad_tol, η_gcq) || push!(near[it], (i, js, xl, pt.normal))
            end
        end
    end
    pairs = reduce(vcat, near)
    if reuse
        keys = Vector{Any}(undef, length(pairs))
        Threads.@threads for k in eachindex(pairs)
            _, js, xl, nx = pairs[k]
            keys[k] = _tensor_near_key(panels[js], xl, nx)
        end
        first_of = Dict{Any, Int}()
        uid = [get!(first_of, keys[k], length(first_of) + 1) for k in eachindex(pairs)]
        reps = zeros(Int, length(first_of))
        for k in eachindex(pairs)
            reps[uid[k]] == 0 && (reps[uid[k]] = k)
        end
    else
        uid = collect(eachindex(pairs)); reps = collect(eachindex(pairs))
    end
    rows = Vector{Vector{T}}(undef, length(reps))
    Threads.@threads :dynamic for u in eachindex(reps)
        _, js, xl, nx = pairs[reps[u]]
        rows[u] = tensor_near_row(d -> _tensor_dnG(nx, d), panels[js], xl)
    end
    # correction = exact (BI sign: -row) - plain, plain in absolute coordinates as FMM
    nnz_total = sum(k -> counts[pairs[k][2]], eachindex(pairs); init = 0)
    Is = Vector{Int}(undef, nnz_total); Js = Vector{Int}(undef, nnz_total); Vs = Vector{T}(undef, nnz_total)
    starts = cumsum([0; [counts[pairs[k][2]] for k in eachindex(pairs)]])
    X = reduce(vcat, [p.points for p in panels])
    Threads.@threads for k in eachindex(pairs)
        i, js, _, _ = pairs[k]
        ps = panels[js]
        xi = X[i]
        ni = pairs[k][4]
        row = rows[uid[k]]
        for (jj, j) in enumerate(rng[js])
            idx = starts[k] + jj
            plain = laplace3d_grad(ps.points[jj], xi, ni) * ps.weights[jj]
            Is[idx] = i; Js[idx] = j; Vs[idx] = -row[jj] - plain
        end
    end
    N = offs[end]
    return sparse(Is, Js, Vs, N, N), (pairs = length(pairs), unique = length(reps))
end
