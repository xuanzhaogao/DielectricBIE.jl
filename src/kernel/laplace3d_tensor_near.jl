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

# points (as offsets from c), weights (with x^pw and length L) and basis for the leaf
# [c + a, c + b] ⊂ segment k. Leaves are carried as offsets from the target's
# projection c onto the panel, so that leaves far smaller than the floating-point
# spacing at c (a target within ~1e-17 of an edge, at an O(1) position along it)
# still have distinct nodes; absolute positions are used only for the basis and the
# weight x^pw, where rounding is harmless.
function _tensor_leaf_rule(r::TensorRule1D, L, k, c, a, b)
    sa, sb = r.segs[k]
    wt(u) = r.pw == 0 ? 1.0 : u^r.pw
    if a == sa - c && b == sb - c && length(r.z0) >= TENSOR_NEAR_M
        return r.fx[:, k] .- c, L .* r.fw[:, k], nothing
    elseif r.pw != 0 && a == -c
        zj, wj = _tensor_gj(r.pw)
        off = a .+ (b - a) .* (1 .+ zj) ./ 2              # = c + off ∈ [0, c + b]
        wts = ((c + b) / 2)^(r.pw + 1) .* wj .* L
    else
        zm, wm = TENSOR_NEAR_GL
        off = a .+ (b - a) .* (1 .+ zm) ./ 2
        wts = (b - a) / 2 .* wm .* L .* wt.(c .+ off)
    end
    return off, wts, _tensor_lagrange_mat(r.z0, r.bw, 2 .* ((c .+ off) .- sa) ./ (sb - sa) .- 1)
end

# distance from the target to the leaf [cu+ua, cu+ub] × [cv+va, cv+vb]; (tu, tv) are the
# target's offsets from (cu, cv) (zero when it projects inside the panel), h its height
_tensor_leaf_dist(p::TensorPanel, tu, tv, h, ua, ub, va, vb) =
    sqrt(((tu - clamp(tu, ua, ub)) * p.Lu)^2 + ((tv - clamp(tv, va, vb)) * p.Lv)^2 + h^2)

"""
    tensor_near_row(kern, p, xl; η=1.5, maxdepth=120)

Row r with Σ_k r_k σ_k ≈ ∫_p kern(x - y) σ(y) dA(y) for σ in the panel's tensor
interpolation space; `xl` is the target in p's anchor frame (x - p.anchor).
"""
function tensor_near_row(kern, p::TensorPanel, xl; η = 1.5, maxdepth = 120)
    ru, rv = p.ru, p.rv
    nzu, nzv = length(ru.z0), length(rv.z0)
    rowfine = zeros(length(ru.segs) * nzu, length(rv.segs) * nzv)
    direct = nzu >= TENSOR_NEAR_M && nzv >= TENSOR_NEAR_M
    r0 = xl .- p.o
    ut = dot(r0, p.eu) / p.Lu; vt = dot(r0, p.ev) / p.Lv; h = dot(r0, p.normal)
    cu = clamp(ut, 0.0, 1.0); cv = clamp(vt, 0.0, 1.0)
    tu = ut - cu; tv = vt - cv
    # x - y for y at offsets (ou, ov) from (cu, cv)
    sep(ou, ov) = ((tu - ou) * p.Lu) .* p.eu .+ ((tv - ov) * p.Lv) .* p.ev .+ h .* p.normal
    stack = NTuple{5, Float64}[]
    for (ku, (sau, sbu)) in enumerate(ru.segs), (kv, (sav, sbv)) in enumerate(rv.segs)
        iu = (ku - 1) * nzu .+ (1:nzu); iv = (kv - 1) * nzv .+ (1:nzv)
        a0u, b0u, a0v, b0v = sau - cu, sbu - cu, sav - cv, sbv - cv
        if direct && _tensor_leaf_dist(p, tu, tv, h, a0u, b0u, a0v, b0v) >
                     η * max((sbu - sau) * p.Lu, (sbv - sav) * p.Lv)
            @inbounds for b in 1:nzv, a in 1:nzu
                rowfine[iu[a], iv[b]] += kern(sep(ru.fx[a, ku] - cu, rv.fx[b, kv] - cv)) *
                                         (p.Lu * ru.fw[a, ku]) * (p.Lv * rv.fw[b, kv])
            end
            continue
        end
        empty!(stack); push!(stack, (a0u, b0u, a0v, b0v, 0.0))
        while !isempty(stack)
            ua, ub, va, vb, depth = pop!(stack)
            du = (ub - ua) * p.Lu; dv = (vb - va) * p.Lv
            d = _tensor_leaf_dist(p, tu, tv, h, ua, ub, va, vb)
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
            pu, wu, Bu = _tensor_leaf_rule(ru, p.Lu, ku, cu, ua, ub)
            pv, wv, Bv = _tensor_leaf_rule(rv, p.Lv, kv, cv, va, vb)
            K = [kern(sep(pu[a], pv[b])) * wu[a] * wv[b] for a in eachindex(pu), b in eachindex(pv)]
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
    laplace3d_DT_tensor_corrections(interface; reuse=true, quad_tol=1e-12, η_gcq=2.0, sources=nothing)

Sparse correction C such that `laplace3d_DT_fmm3d(interface, tol) + C` is the
near-corrected D^T operator in DielectricBIE's sign convention
(entries laplace3d_grad(y_j, x_i, n_i) w_j). Pairs that `tensor_isfar` rejects at
`quad_tol` get the product-integration row. With `sources` (panel indices) only the
columns of those source panels are computed, the others are zero. Returns (C, stats).
"""
function laplace3d_DT_tensor_corrections(interface::DielectricInterface{TensorPanel{T}, T};
                                         reuse::Bool = true, quad_tol = 1e-12, η_gcq = 2.0,
                                         sources = nothing) where T
    panels = interface.panels
    counts = num_points.(panels)
    offs = cumsum([0; counts])
    rng = [offs[k]+1:offs[k+1] for k in eachindex(panels)]
    Yl = [tensor_local_points(p) for p in panels]
    near = [Tuple{Int, Int, NTuple{3, T}, NTuple{3, T}}[] for _ in panels]
    is_src = sources === nothing ? trues(length(panels)) : (b = falses(length(panels)); b[collect(sources)] .= true; b)
    Threads.@threads :dynamic for it in eachindex(panels)
        pt = panels[it]
        for (i, xo) in zip(rng[it], Yl[it])
            for (js, ps) in enumerate(panels)
                is_src[js] || continue
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
    # FMM3D skips pairs closer than ≈ 2^-51 × (box size); the plain term must skip the
    # same pairs (GCQ rules for p1 → -1 put a node within ~1e-20 of the edge, which
    # can meet its mirror node across the edge, or coincide with it in absolute
    # coordinates). Pairs in the ambiguous band [2^-54, 2^-48] × box are counted.
    box = maximum(maximum(x[d] for x in X) - minimum(x[d] for x in X) for d in 1:3)
    rcut = 2.0^-54 * box
    ambiguous = Threads.Atomic{Int}(0)
    Threads.@threads for k in eachindex(pairs)
        i, js, _, _ = pairs[k]
        ps = panels[js]
        xi = X[i]
        ni = pairs[k][4]
        row = rows[uid[k]]
        for (jj, j) in enumerate(rng[js])
            idx = starts[k] + jj
            dr = norm(xi .- ps.points[jj])
            dr < 2.0^-48 * box && dr >= rcut && Threads.atomic_add!(ambiguous, 1)
            plain = dr < rcut ? zero(T) : laplace3d_grad(ps.points[jj], xi, ni) * ps.weights[jj]
            Is[idx] = i; Js[idx] = j; Vs[idx] = -row[jj] - plain
        end
    end
    ambiguous[] > 0 && @warn "$(ambiguous[]) near pairs closer than 2^-48 × box size: FMM3D may or may not skip them"
    N = offs[end]
    return sparse(Is, Js, Vs, N, N), (pairs = length(pairs), unique = length(reps))
end

"""
    potential_tensor(interface, σ, targets; quad_tol=1e-12, η_gcq=2.0)

Single-layer potential u(x) = ∫ σ(y) / (4π|x - y|) dA(y) at off-surface `targets`
(absolute coordinates). Panels that `tensor_isfar` accepts use their own rule; the
others the product-integration row of `tensor_near_row` (accurate for targets close
to edges and corners, down to the floating-point resolution of the target position).
"""
function potential_tensor(interface::DielectricInterface{TensorPanel{T}, T}, σ::AbstractVector,
                          targets::AbstractVector; quad_tol = 1e-12, η_gcq = 2.0) where T
    panels = interface.panels
    offs = cumsum([0; num_points.(panels)])
    G(d) = 1 / (4π * sqrt(d[1]^2 + d[2]^2 + d[3]^2))
    u = zeros(T, length(targets))
    Threads.@threads :dynamic for it in eachindex(targets)
        x = targets[it]
        acc = zero(T)
        for (k, ps) in enumerate(panels)
            σk = view(σ, offs[k]+1:offs[k+1])
            xl = x .- ps.anchor
            if tensor_isfar(ps, xl; quad_tol, η_gcq)
                for (j, y) in enumerate(ps.points)
                    acc += G(x .- y) * ps.weights[j] * σk[j]
                end
            else
                acc += dot(tensor_near_row(G, ps, xl), σk)
            end
        end
        u[it] = acc
    end
    return u
end
