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

# The product integration of tensor_near_row on the fine grid: every (fine u-segment, fine v-segment) pair on its own,
# accumulated at the fine nodes and projected with P at the end. The reference for tensor_near_row.
function _tensor_near_row_fine(kern::F, p::TensorPanel, xl; η = 1.5, maxdepth = 120) where F
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

# Tables of a rule for tensor_near_row, in its node basis (columns of P): for every segment k, `seg[k]` (nb × nz) maps
# values at the segment's fine nodes, times their weights, to the basis; `pk[k]` = P[rows of k, :]'. For a GCQ rule
# (dyadic segments from the edge), `clu[k]` (nb × Q) maps values at the Q Gauss-Legendre points `cx[k]` of
# [0, sb_k] (segments 1..k) to the basis: clu[k][j, q] = ∫_0^{sb_k} ℓ_q(s) s^pw v_j(s) ds, with ℓ_q the Lagrange
# polynomials of those points and v_j = P[:, j] on each segment, so that ∫ K(s) φ_j(s) ds ≈ Σ_q clu[k][j, q] K(cx_q)
# when K is smooth on [0, sb_k] (the target farther than η sb_k).
struct _TensorNearTables
    seg::Vector{Matrix{Float64}}
    pk::Vector{Matrix{Float64}}
    clu::Vector{Matrix{Float64}}
    cx::Vector{Vector{Float64}}
    gj::Tuple{Vector{Float64}, Vector{Float64}}     # TENSOR_NEAR_M-point Gauss-Jacobi for the weight x^pw
    fx::Vector{Vector{Float64}}                     # the segments' fine nodes on [0, 1]
end

const TENSOR_NEAR_Q = 18
# rule tables by objectid(r.P): read without a lock from an immutable snapshot, replaced (copy-on-write) under the
# lock when a new rule appears
const TENSOR_TABLES = Ref(Dict{UInt, Tuple{TensorRule1D, _TensorNearTables}}())
const TENSOR_TABLES_LOCK = ReentrantLock()

function _tensor_near_tables(r::TensorRule1D)
    key = objectid(r.P)
    e = get(TENSOR_TABLES[], key, nothing)
    e !== nothing && e[1] === r && return e[2]
    lock(TENSOR_TABLES_LOCK) do
        e = get(TENSOR_TABLES[], key, nothing)
        e !== nothing && e[1] === r && return e[2]
        t = _tensor_near_tables_build(r)
        d = copy(TENSOR_TABLES[]); d[key] = (r, t)
        TENSOR_TABLES[] = d
        return t
    end
end

function _tensor_near_tables_build(r::TensorRule1D)
    nz = length(r.z0); nb = size(r.P, 2); ns = length(r.segs)
    pk = [Matrix(r.P[(k-1)*nz+1:k*nz, :]') for k in 1:ns]
    seg = [pk[k] .* r.fw[:, k]' for k in 1:ns]
    clu = Matrix{Float64}[]; cx = Vector{Float64}[]
    contiguous = ns > 1 && r.segs[1][1] == 0 && all(r.segs[k][1] == r.segs[k-1][2] for k in 2:ns)
    if contiguous
        zq, _ = gausslegendre(TENSOR_NEAR_Q)
        bq = gl_barycentric_weights(zq, gausslegendre(TENSOR_NEAR_Q)[2])
        nq = 2 * (nz + TENSOR_NEAR_Q)
        tg, wg = gausslegendre(nq)
        tj, wj = r.pw == 0 ? (tg, wg) : gaussjacobi(nq, 0.0, r.pw)
        # per segment, the exact integrals ∫ s^pw v_j(s) g(s) ds as (points, weights, v at the points)
        quad = map(1:ns) do k
            sa, sb = r.segs[k]
            if sa == 0 && r.pw != 0
                y = sb .* (1 .+ tj) ./ 2; w = (sb / 2)^(r.pw + 1) .* wj
            else
                y = sa .+ (sb - sa) .* (1 .+ tg) ./ 2; w = (sb - sa) / 2 .* wg .* (r.pw == 0 ? 1.0 : y .^ r.pw)
            end
            V = _tensor_lagrange_mat(r.z0, r.bw, 2 .* (y .- sa) ./ (sb - sa) .- 1)' * r.P[(k-1)*nz+1:k*nz, :]
            (y, w, V)
        end
        for k in 1:ns
            sb = r.segs[k][2]
            push!(cx, sb .* (1 .+ zq) ./ 2)
            G = zeros(nb, TENSOR_NEAR_Q)
            for kk in 1:k
                y, w, V = quad[kk]
                G .+= V' * (w .* _tensor_lagrange_mat(zq, bq, 2 .* y ./ sb .- 1)')
            end
            push!(clu, G)
        end
    end
    gj = r.pw == 0 ? TENSOR_NEAR_GL : gaussjacobi(TENSOR_NEAR_M, 0.0, r.pw)
    return _TensorNearTables(seg, pk, clu, cx, (collect(gj[1]), collect(gj[2])), [r.fx[:, k] for k in 1:ns])
end

# one direction of a quadtree node of tensor_near_row: the cluster of segments 1..k (seg = -k, the interval
# [0, sb_k]) or the part [c + a, c + b] of segment k (seg = k); a, b offsets from the target's projection c.
# Returns the leaf's points (offsets from c: pts .- sh), its map G (nb × npoints) from point values to the basis,
# and the factor of G
function _tensor_leaf_map!(ws, r::TensorRule1D, tb::_TensorNearTables, L, kseg, c, a, b)
    nb = size(tb.pk[1], 1)
    kseg < 0 && return tb.cx[-kseg], c, view(tb.clu[-kseg], 1:nb, :), L
    sa, sb = r.segs[kseg]
    nz = length(r.z0)
    a == sa - c && b == sb - c && nz >= TENSOR_NEAR_M && return tb.fx[kseg], c, view(tb.seg[kseg], 1:nb, :), L
    # TENSOR_NEAR_M points on the leaf (as _tensor_leaf_rule), the segment's Lagrange basis at them times the weights
    off, wts, Bf, Gf = ws
    B = view(Bf, 1:nz, :); G = view(Gf, 1:nb, :)
    if r.pw != 0 && a == -c
        zj, wj = tb.gj
        sc = ((c + b) / 2)^(r.pw + 1) * L
        @inbounds for q in 1:TENSOR_NEAR_M
            off[q] = a + (b - a) * (1 + zj[q]) / 2; wts[q] = sc * wj[q]
        end
    else
        zm, wm = TENSOR_NEAR_GL
        @inbounds for q in 1:TENSOR_NEAR_M
            off[q] = a + (b - a) * (1 + zm[q]) / 2
            wts[q] = (b - a) / 2 * wm[q] * L * (r.pw == 0 ? 1.0 : (c + off[q])^r.pw)
        end
    end
    @inbounds for q in 1:TENSOR_NEAR_M
        t = 2 * ((c + off[q]) - sa) / (sb - sa) - 1
        j = findfirst(==(t), r.z0)
        if j !== nothing
            B[:, q] .= 0; B[j, q] = wts[q]
        else
            sm = 0.0
            for i in 1:nz
                B[i, q] = r.bw[i] / (t - r.z0[i]); sm += B[i, q]
            end
            for i in 1:nz
                B[i, q] *= wts[q] / sm
            end
        end
    end
    mul!(G, tb.pk[kseg], B)
    return off, 0.0, G, 1.0
end

# children of one direction of a node: a cluster splits into the cluster one segment shorter and its last segment,
# a part of a segment into halves
function _tensor_dir_split(r::TensorRule1D, kseg, c, a, b)
    if kseg < 0
        k = -kseg
        sa, sb = r.segs[k]
        return ((k - 1 == 1 ? 1 : -(k - 1), -c, sa - c), (k, sa - c, sb - c))
    end
    m = (a + b) / 2
    return ((kseg, a, m), (kseg, m, b))
end

"""
    tensor_near_row(kern, p, xl; η=1.5, maxdepth=120)

Row r with Σ_k r_k σ_k ≈ ∫_p kern(x - y) σ(y) dA(y) for σ in the panel's tensor
interpolation space; `xl` is the target in p's anchor frame (x - p.anchor).

Product integration on a quadtree over the panel, subdivided until each leaf is
farther than η·(its size) from the target. In a GCQ direction the root is the
cluster of all fine segments, which splits into the cluster one segment shorter and
its last segment, so that the dyadic segments near the edge stay one leaf while the
target is far from them. Leaves use the clusters' moment rules, the segments' own
nodes, or 12-point Gauss-Legendre (Gauss-Jacobi on a leaf touching the edge) with
the segments' Lagrange bases, and add directly into the panel's node basis.
"""
# buffers of tensor_near_row for rules with up to nz fine nodes per segment and nb basis functions
struct _TensorRowWS
    M::Matrix{Float64}
    K::Matrix{Float64}
    T::Matrix{Float64}
    u::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}}
    v::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}}
    stack::Vector{Tuple{Int, Float64, Float64, Int, Float64, Float64, Int}}
end

function _TensorRowWS(nz::Int, nb::Int)
    q = max(TENSOR_NEAR_M, TENSOR_NEAR_Q, nz)
    dir() = (zeros(TENSOR_NEAR_M), zeros(TENSOR_NEAR_M), zeros(nz, TENSOR_NEAR_M), zeros(nb, TENSOR_NEAR_M))
    return _TensorRowWS(zeros(nb, nb), zeros(q, q), zeros(nb, q), dir(), dir(), Tuple{Int, Float64, Float64, Int, Float64, Float64, Int}[])
end

# a workspace for every panel of `panels`
_TensorRowWS(panels::AbstractVector{<:TensorPanel}) =
    _TensorRowWS(maximum(p -> max(length(p.ru.z0), length(p.rv.z0)), panels),
                 maximum(p -> max(size(p.ru.P, 2), size(p.rv.P, 2)), panels))

function tensor_near_row(kern::F, p::TensorPanel, xl; η = 1.5, maxdepth = 120, ws::_TensorRowWS = _TensorRowWS([p])) where F
    ru, rv = p.ru, p.rv
    tu_, tv_ = _tensor_near_tables(ru), _tensor_near_tables(rv)
    (isempty(tu_.clu) && length(ru.segs) > 1 || isempty(tv_.clu) && length(rv.segs) > 1) &&
        return _tensor_near_row_fine(kern, p, xl; η, maxdepth)
    r0 = xl .- p.o
    ut = dot(r0, p.eu) / p.Lu; vt = dot(r0, p.ev) / p.Lv; h = dot(r0, p.normal)
    cu = clamp(ut, 0.0, 1.0); cv = clamp(vt, 0.0, 1.0)
    tu = ut - cu; tv = vt - cv
    sep(ou, ov) = ((tu - ou) * p.Lu) .* p.eu .+ ((tv - ov) * p.Lv) .* p.ev .+ h .* p.normal
    nbu, nbv = size(ru.P, 2), size(rv.P, 2)
    M = view(ws.M, 1:nbu, 1:nbv); fill!(M, 0.0)
    Kb = ws.K; Tb = view(ws.T, 1:nbu, :); wsu = ws.u; wsv = ws.v
    root(r, tb, c) = isempty(tb.clu) ? (1, -c, 1 - c) : (-length(r.segs), -c, 1 - c)
    ku0, a0u, b0u = root(ru, tu_, cu); kv0, a0v, b0v = root(rv, tv_, cv)
    stack = ws.stack; empty!(stack); push!(stack, (ku0, a0u, b0u, kv0, a0v, b0v, 0))
    while !isempty(stack)
        ku, ua, ub, kv, va, vb, depth = pop!(stack)
        du = (ub - ua) * p.Lu; dv = (vb - va) * p.Lv
        d = _tensor_leaf_dist(p, tu, tv, h, ua, ub, va, vb)
        if d <= η * max(du, dv) && depth < maxdepth
            su = du > dv / 2 && η * du >= d
            sv = dv > du / 2 && η * dv >= d
            !su && !sv && (du >= dv ? (su = true) : (sv = true))
            us = su ? _tensor_dir_split(ru, ku, cu, ua, ub) : ((ku, ua, ub),)
            vs = sv ? _tensor_dir_split(rv, kv, cv, va, vb) : ((kv, va, vb),)
            for (k1, a1, b1) in us, (k2, a2, b2) in vs
                push!(stack, (k1, a1, b1, k2, a2, b2, depth + 1))
            end
            continue
        end
        pu, shu, Gu, fu = _tensor_leaf_map!(wsu, ru, tu_, p.Lu, ku, cu, ua, ub)
        pv, shv, Gv, fv = _tensor_leaf_map!(wsv, rv, tv_, p.Lv, kv, cv, va, vb)
        nu, nv = length(pu), length(pv)
        K = view(Kb, 1:nu, 1:nv); T1 = view(Tb, :, 1:nv)
        @inbounds for b in 1:nv, a in 1:nu
            K[a, b] = kern(sep(pu[a] - shu, pv[b] - shv))
        end
        mul!(T1, Gu, K)
        mul!(M, T1, Gv', fu * fv, true)
    end
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
    tensor_isfar(p, xl; quad_tol=1e-12, η_gcq=2.0, gcq_far=:max)

Whether the panel's own rule may be used for target `xl` (in p's anchor frame).
A Gauss-Legendre direction with n points has error ≈ ρ^{-2n}, with ρ the
Bernstein-ellipse parameter of the target for that direction (its distance from
the panel in the other two directions counted as the ellipse's minor axis); the
sum over the GL directions must be below `quad_tol`. A GCQ direction (span of
polynomial degree ≈ 20 times the edge powers) requires a distance of at least
`η_gcq` lengths: of the panel's larger side (`gcq_far = :max`, the default), or of
the GCQ direction itself (`gcq_far = :direction`; for a corner panel, GCQ in both
directions, the larger side). The node rule of an edge or corner panel, for a
density s^pw × smooth, is accurate to about 5e-10 at 1, and 1e-10 at 2, lengths of
its GCQ direction once the GL direction is resolved, so `:direction` with
η_gcq = 2 keeps quad_tol ≥ 1e-10 while skipping the long sides of edge panels.
"""
function tensor_isfar(p::TensorPanel, xl; quad_tol = 1e-12, η_gcq = 2.0, gcq_far::Symbol = :max)
    if is_gcq(p.ru) || is_gcq(p.rv)
        Lg = gcq_far === :max ? max(p.Lu, p.Lv) :
             gcq_far === :direction ? (is_gcq(p.ru) && is_gcq(p.rv) ? max(p.Lu, p.Lv) : is_gcq(p.ru) ? p.Lu : p.Lv) :
             throw(ArgumentError("gcq_far must be :max or :direction, got :$gcq_far"))
        _tensor_rect_dist(p, xl) <= η_gcq * Lg && return false
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

# the distinct near rows: uid[k] is the row of pair k, reps[u] the first pair of row u
function _tensor_near_unique(panels, pairs)
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
    return uid, reps
end

"""
    laplace3d_DT_tensor_corrections(interface; reuse=true, quad_tol=1e-12, η_gcq=2.0, gcq_far=:max, sources=nothing)

Sparse correction C such that `laplace3d_DT_fmm3d(interface, tol) + C` is the
near-corrected D^T operator in DielectricBIE's sign convention
(entries laplace3d_grad(y_j, x_i, n_i) w_j). Pairs that `tensor_isfar` rejects at
`quad_tol` (and `η_gcq`, `gcq_far`) get the product-integration row. With `sources` (panel indices) only the
columns of those source panels are computed, the others are zero. Returns (C, stats).
"""
function laplace3d_DT_tensor_corrections(interface::DielectricInterface{TensorPanel{T}, T};
                                         reuse::Bool = true, quad_tol = 1e-12, η_gcq = 2.0,
                                         gcq_far::Symbol = :max, sources = nothing) where T
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
                tensor_isfar(ps, xl; quad_tol, η_gcq, gcq_far) || push!(near[it], (i, js, xl, pt.normal))
            end
        end
    end
    pairs = reduce(vcat, near)
    uid, reps = reuse ? _tensor_near_unique(panels, pairs) : (collect(eachindex(pairs)), collect(eachindex(pairs)))
    rows = Vector{Vector{T}}(undef, length(reps))
    foreach(p -> (_tensor_near_tables(p.ru); _tensor_near_tables(p.rv)), panels)
    chunks = collect(Iterators.partition(eachindex(reps), 256))
    _blas_single() do
        Threads.@threads :dynamic for ch in chunks
            ws = _TensorRowWS(panels)
            for u in ch
                _, js, xl, nx = pairs[reps[u]]
                rows[u] = tensor_near_row(d -> _tensor_dnG(nx, d), panels[js], xl; ws)
            end
        end
    end
    # correction = exact (BI sign: -row) - plain, plain in absolute coordinates as FMM; assembled column by column:
    # every column of source panel js holds the panel's near targets, in increasing order
    X = reduce(vcat, [p.points for p in panels])
    # FMM3D skips pairs closer than ≈ 2^-51 × (box size); the plain term must skip the
    # same pairs (GCQ rules for p1 → -1 put a node within ~1e-20 of the edge, which
    # can meet its mirror node across the edge, or coincide with it in absolute
    # coordinates). Pairs in the ambiguous band [2^-54, 2^-48] × box are counted.
    box = maximum(maximum(x[d] for x in X) - minimum(x[d] for x in X) for d in 1:3)
    rcut = 2.0^-54 * box
    ambiguous = Threads.Atomic{Int}(0)
    bysrc = [Int[] for _ in panels]
    for k in eachindex(pairs)
        push!(bysrc[pairs[k][2]], k)
    end
    N = offs[end]
    colptr = Vector{Int}(undef, N + 1); colptr[1] = 1
    for js in eachindex(panels), c in rng[js]
        colptr[c + 1] = colptr[c] + length(bysrc[js])
    end
    rowval = Vector{Int}(undef, colptr[end] - 1); nzval = Vector{T}(undef, colptr[end] - 1)
    Threads.@threads :dynamic for js in eachindex(panels)
        ks = bysrc[js]
        isempty(ks) && continue
        sort!(ks; by = k -> pairs[k][1])
        ps = panels[js]
        for (t, k) in enumerate(ks)
            i, _, _, ni = pairs[k]
            xi = X[i]
            row = rows[uid[k]]
            for (jj, c) in enumerate(rng[js])
                idx = colptr[c] + t - 1
                dr = norm(xi .- ps.points[jj])
                dr < 2.0^-48 * box && dr >= rcut && Threads.atomic_add!(ambiguous, 1)
                plain = dr < rcut ? zero(T) : laplace3d_grad(ps.points[jj], xi, ni) * ps.weights[jj]
                rowval[idx] = i; nzval[idx] = -row[jj] - plain
            end
        end
    end
    ambiguous[] > 0 && @warn "$(ambiguous[]) near pairs closer than 2^-48 × box size: FMM3D may or may not skip them"
    return SparseMatrixCSC(N, N, colptr, rowval, nzval), (pairs = length(pairs), unique = length(reps))
end

"""
    potential_tensor(interface, σ, targets; quad_tol=1e-12, η_gcq=2.0, gcq_far=:max)

Single-layer potential u(x) = ∫ σ(y) / (4π|x - y|) dA(y) at off-surface `targets`
(absolute coordinates). Panels that `tensor_isfar` accepts use their own rule; the
others the product-integration row of `tensor_near_row` (accurate for targets close
to edges and corners, down to the floating-point resolution of the target position).
"""
function potential_tensor(interface::DielectricInterface{TensorPanel{T}, T}, σ::AbstractVector,
                          targets::AbstractVector; quad_tol = 1e-12, η_gcq = 2.0,
                          gcq_far::Symbol = :max) where T
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
            if tensor_isfar(ps, xl; quad_tol, η_gcq, gcq_far)
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
