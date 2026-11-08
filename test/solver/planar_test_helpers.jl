# Shared fixtures of the flat-panel near and self tests (planar_endpoint_gcq.jl, planar_self_endpoint_gcq.jl): GCQ rules,
# bilinear flat panels, and references for rows of a panel's parent interpolant that are independent of the near
# tables, the walk and duffy_rule. Included by each of those files (functions only, so including it twice is harmless).
using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra

isdefined(@__MODULE__, :EP_RULES) ||
    (global EP_RULES = Dict{Tuple{Float64, Float64}, Tuple{DB.EdgeGCQRule, DB.TensorRule1D}}())
"The GCQ rule of the density powers [p] at tol, and its TensorRule1D (cached)."
ep_rule(p, tol) = get!(() -> (e = DB.edge_gcq_rule([p]; tol); (e, DB.gcq_tensor_rule(e))), EP_RULES, (p, tol))

"A flat panel y = anchor + u A + v B + u v C (A, B, C in a plane), rule ru in u and rv in v."
function ep_panel(anchor, A, B, C, ru, rv)
    lmap = (u, v) -> Tuple(u .* A .+ v .* B .+ (u * v) .* C)
    return CurvedPanel(anchor, lmap, ru, rv; kind = :edge, dmap = (u, v) -> (Tuple(A .+ v .* C), Tuple(B .+ u .* C)))
end

# The reference integrals accumulate at the fine nodes of both directions (F; Fa the same of |kernel × basis|, whose
# 100 ε is the rounding floor of any quadrature of the row) and map to the basis by P at the end (pl_row).
pl_fine(P) = (zeros(size(P.ru.P, 1), size(P.rv.P, 1)), zeros(size(P.ru.P, 1), size(P.rv.P, 1)))
pl_row(P, F, Fa) = (vec(transpose(P.ru.P' * F * P.rv.P)), vec(transpose(abs.(P.ru.P)' * Fa * abs.(P.rv.P))))
pl_block(P, k, l) = ((k - 1) * length(P.ru.z0) .+ (1:length(P.ru.z0)), (l - 1) * length(P.rv.z0) .+ (1:length(P.rv.z0)))

# the Lagrange basis of fine segment k of rule r at the points x, times x^pw (nz × npoints)
pl_basis(r, k, x) = (sa = r.segs[k][1]; sb = r.segs[k][2];
                     DB._tensor_lagrange_mat(r.z0, r.bw, 2 .* (x .- sa) ./ (sb - sa) .- 1) .* (r.pw == 0 ? 1.0 : (x .^ r.pw)'))

# The boxes (k, ua, ub, l, va, vb) of fine pairs at the target x = y(ur, vr) + wt (normal nrm for Dᵀ, nothing for the
# single layer): each halved, the physically longer side, until it is c radii from the target, M-point Gauss-Legendre
# per direction (Gauss-Jacobi for the weight s^pw on a box touching s = 0). Points are offsets from the box's dyadic
# corner, so the parameter steps (ua - ur) + offset are exact relative to the box beside the target.
function pl_boxes!(F, Fa, P, g, ur, vr, wt, nrm, boxes; c = 6.0, M = 20)
    ru, rv = P.ru, P.rv
    gl = DB.gausslegendre(M)
    gj = (ru.pw == 0 ? gl : DB.gaussjacobi(M, 0.0, ru.pw), rv.pw == 0 ? gl : DB.gaussjacobi(M, 0.0, rv.pw))
    function dir(r, q, k, a, b)
        o, w = r.pw != 0 && a == 0 ? (b .* (1 .+ q[1]) ./ 2, (b / 2)^(r.pw + 1) .* q[2]) :
                                     ((b - a) .* (1 .+ gl[1]) ./ 2, (b - a) / 2 .* gl[2] .* (r.pw == 0 ? 1.0 : (a .+ (b - a) .* (1 .+ gl[1]) ./ 2) .^ r.pw))
        x = a .+ o
        sa, sb = r.segs[k]
        return x, o, w, DB._tensor_lagrange_mat(r.z0, r.bw, 2 .* (x .- sa) ./ (sb - sa) .- 1)
    end
    stack = [(b..., 0) for b in boxes]
    while !isempty(stack)
        k, ua, ub, l, va, vb, depth = pop!(stack)
        um = (ua + ub) / 2; vm = (va + vb) / 2
        rad = maximum(norm(DB._flat_step(g, um, vm, s - um, s2 - vm)) for (s, s2) in ((ua, va), (ub, va), (ub, vb), (ua, vb)))
        if norm(wt .- DB._flat_step(g, ur, vr, um - ur, vm - vr)) >= c * rad || depth >= 60
            xu, ou, wu, Lu = dir(ru, gj[1], k, ua, ub)
            xv, ov, wv, Lv = dir(rv, gj[2], l, va, vb)
            K = [begin
                     d = wt .- DB._flat_step(g, ur, vr, (ua - ur) + ou[i], (va - vr) + ov[j])
                     J = norm(cross(collect(g.a .+ xv[j] .* g.c), collect(g.b .+ xu[i] .* g.c)))
                     wu[i] * wv[j] * J * (nrm === nothing ? 1 / (4π * norm(d)) : dot(nrm, d) / (4π * norm(d)^3))
                 end for i in eachindex(xu), j in eachindex(xv)]
            bu, bv = pl_block(P, k, l)
            F[bu, bv] .+= Lu * K * Lv'
            Fa[bu, bv] .+= abs.(Lu) * abs.(K) * abs.(Lv)'
        else
            lu = norm(DB._flat_step(g, ua, vm, ub - ua, 0.0)); lv = norm(DB._flat_step(g, um, va, 0.0, vb - va))
            us = lv > 2lu ? ((ua, ub),) : ((ua, um), (um, ub))
            vs = lu > 2lv ? ((va, vb),) : ((va, vm), (vm, vb))
            for (a1, b1) in us, (a2, b2) in vs
                push!(stack, (k, a1, b1, l, a2, b2, depth + 1))
            end
        end
    end
end

# The single layer at a point (u0, v0) of the panel itself over a rectangle Q = (ua, ub, va, vb) of the fine pair (k, l)
# holding it: a fan of 4 triangles from the target to Q's sides, each by Duffy's map y = x0 + ρ (p(t) - x0) (the 1/r
# singularity cancels against ρ; on a bilinear panel y - x0 = ρ [D_u (a + (v0 + ρ D_v) c) + D_v (b + u0 c)] exactly),
# M-point Gauss-Legendre in ρ, and in t along the side halved until a piece is `grade` × its distance from the target
# (in the panel's metric at the target), M-point Gauss-Legendre per piece. Independent of duffy_rule and its region.
function pl_singular!(F, Fa, P, g, u0, v0, k, l, Q; M = 20, grade = 0.25)
    ru, rv = P.ru, P.rv
    ua, ub, va, vb = Q
    tg, wg = DB.gausslegendre(M)
    yu0 = g.a .+ v0 .* g.c; yv0 = g.b .+ u0 .* g.c
    len(du, dv) = norm(du .* yu0 .+ dv .* yv0)
    U = Float64[]; V = Float64[]; W = Float64[]
    corners = ((ua, va), (ub, va), (ub, vb), (ua, vb))
    for s in 1:4
        p1, p2 = corners[s], corners[mod1(s + 1, 4)]
        d0 = (p1[1] - u0, p1[2] - v0); e = (p2[1] - p1[1], p2[2] - p1[2])
        det = abs(d0[1] * e[2] - d0[2] * e[1])
        det == 0 && continue
        # the closest point of the side piece [α, β] to the target, in the metric at the target
        G = (dot(yu0, yu0), dot(yu0, yv0), dot(yv0, yv0))
        ip(x, y) = x[1] * y[1] * G[1] + (x[1] * y[2] + x[2] * y[1]) * G[2] + x[2] * y[2] * G[3]
        tstar = -ip(d0, e) / ip(e, e)
        pieces = [(0.0, 1.0)]; done = Tuple{Float64, Float64}[]
        while !isempty(pieces)
            α, β = pop!(pieces)
            t = clamp(tstar, α, β)
            if len(e...) * (β - α) <= grade * len((d0 .+ t .* e)...) || β - α < 1e-30
                push!(done, (α, β))
            else
                push!(pieces, (α, (α + β) / 2), ((α + β) / 2, β))
            end
        end
        for (α, β) in done, j in 1:M
            t = α + (β - α) * (1 + tg[j]) / 2; wt = (β - α) / 2 * wg[j]
            D = d0 .+ t .* e
            for i in 1:M
                ρ = (1 + tg[i]) / 2; wρ = wg[i] / 2
                u = u0 + ρ * D[1]; v = v0 + ρ * D[2]
                sep = D[1] .* (g.a .+ (v0 + ρ * D[2]) .* g.c) .+ D[2] .* (g.b .+ u0 .* g.c)      # (y - x0) / ρ
                J = norm(cross(collect(g.a .+ v .* g.c), collect(g.b .+ u .* g.c)))
                push!(U, u); push!(V, v); push!(W, wρ * wt * det * J / (4π * norm(sep)))
            end
        end
    end
    Lu = pl_basis(ru, k, U); Lv = pl_basis(rv, l, V)
    bu, bv = pl_block(P, k, l)
    F[bu, bv] .+= Lu * Diagonal(W) * Lv'
    Fa[bu, bv] .+= abs.(Lu) * Diagonal(abs.(W)) * abs.(Lv)'
end

"The rows of P at the targets T (DB._FlatTargets, off the panel) by pl_boxes! over every fine pair: (rows, |rows|)."
function ep_ref_rows(P, T; c = 6.0, M = 20)
    g = DB._flat_geom(P)
    allb = [(k, sa, sb, l, ta, tb) for (k, (sa, sb)) in enumerate(P.ru.segs) for (l, (ta, tb)) in enumerate(P.rv.segs)]
    out = zeros(length(T.ur), size(P.ru.P, 2) * size(P.rv.P, 2)); mag = zero(out)
    for t in eachindex(T.ur)
        F, Fa = pl_fine(P)
        pl_boxes!(F, Fa, P, g, T.ur[t], T.vr[t], T.w[:, t], T.nrm === nothing ? nothing : T.nrm[:, t], allb; c, M)
        out[t, :], mag[t, :] = pl_row(P, F, Fa)
    end
    return out, mag
end

"""
The single-layer self row of P at its point (u0, v0) in the fine pair (k, l): every other fine pair by pl_boxes!, the
pair (k, l) itself by pl_singular!. Returns (row, |row|) in the node basis.
"""
function sp_self_ref(P, u0, v0, k, l; c = 6.0, M = 20, grade = 0.25)
    g = DB._flat_geom(P); F, Fa = pl_fine(P)
    other = [(k2, sa, sb, l2, ta, tb) for (k2, (sa, sb)) in enumerate(P.ru.segs) for (l2, (ta, tb)) in enumerate(P.rv.segs)
             if (k2, l2) != (k, l)]
    pl_boxes!(F, Fa, P, g, u0, v0, zeros(3), nothing, other; c, M)
    pl_singular!(F, Fa, P, g, u0, v0, k, l, (P.ru.segs[k]..., P.rv.segs[l]...); M, grade)
    return pl_row(P, F, Fa)
end

"""
The panel's far-field node rule against the reference at the near/far boundary (1.5 diameters, above the centre and in
the plane across the u = 0 side): the error the GCQ discretization already makes on the rows of its span. Returns
(cardinal: max_j |Δ_j| / Σ_j |row_j|, the density σ: |Δ·σ| / |row|·|σ|), the worst of the two targets; the single layer
(nrm = nothing) or Dᵀ with the target normal nrm.
"""
function pl_far_error(P, σ; nrm = nothing)
    g = DB._flat_geom(P); n = normalize(collect(P.normals[1])); d = g.diam
    ctr = collect(g.o .+ DB._flat_step(g, 0.0, 0.0, 0.5, 0.5))                  # in the anchor frame, as P.lpoints
    xs = [ctr .+ 1.5d .* n, ctr .- (1.5d + 0.5 * norm(collect(g.a))) .* normalize(collect(g.a))]
    N = nrm === nothing ? nothing : hcat(nrm, nrm)
    rf, _ = ep_ref_rows(P, DB._FlatTargets(zeros(2), zeros(2), reduce(hcat, [x .- collect(g.o) for x in xs]), N, [(0, 0), (0, 0)]))
    K(x, y) = (r = x .- collect(y); nrm === nothing ? 1 / (4π * norm(r)) : dot(nrm, r) / (4π * norm(r)^3))
    nod = [P.weights[j] * K(xs[i], P.lpoints[j]) for i in 1:2, j in eachindex(P.weights)]
    card = maximum(maximum(abs.(nod[i, :] .- rf[i, :])) / sum(abs, rf[i, :]) for i in 1:2)
    dens = maximum(abs(dot(nod[i, :] .- rf[i, :], σ)) / dot(abs.(rf[i, :]), abs.(σ)) for i in 1:2)
    return card, dens
end

# ---- references for rows of the GLOBAL interpolant (plan 2026-11-05-global-gcq-local-duffy): the density from the
# global GCQ formula (an implementation of its own: plain powers and a dense solve), any partition of the panel.

# The cardinal values of the global interpolant cost ≈ eps × cond(Hn) in Float64: a cardinal function is a combination
# of the family's powers with coefficients up to ~1/tol (measured pointwise ≤ 0.02 eps cond, rows of random densities up
# to 0.01 eps cond). Two Float64 evaluations of the same row (different rounding, points moved by an ulp) agree on
# cardinal and random densities to this floor only, physical densities (the family's) to rounding. The piecewise
# interpolant's P holds the same rounding (it is this formula at the fine nodes), so it is no closer to the exact
# interpolant there; it is only consistent with itself.
gl_noise(P) = 0.05 * eps() * maximum((DB.is_gcq(r) ? (g = DB.gcq_global_interp(r); g.S[1] / g.S[end]) : 1.0) for r in (P.ru, P.rv))

"(npoints × n) values of the full density at x from its node values: Lagrange (GL), the global GCQ formula (GCQ)."
function gl_E(r, x)
    DB.is_gcq(r) || return DB.tensor_rule_interp(r, x)
    g = DB.gcq_global_interp(r)
    return Diagonal(x .^ g.pw) * (([t^qk for t in x, qk in g.q] * g.V) / g.Hn) * Diagonal(g.x .^ (-g.pw))
end

# pl_boxes! for the global density: boxes (ua, ub, va, vb) (any, a GCQ direction's box touching 0 by Gauss-Jacobi for
# s^pw), halved until c radii from the target, M-point rules per direction; accumulates row (nu × nv) and |row|;
# the single layer, or Dᵀ with the target normal `nrm`.
function gl_boxes!(F, Fa, P, g, ur, vr, wt, boxes; c = 6.0, M = 20, E = (r, x, d) -> gl_E(r, x), nrm = nothing)
    ru, rv = P.ru, P.rv
    gl = DB.gausslegendre(M)
    gj = (ru.pw == 0 ? gl : DB.gaussjacobi(M, 0.0, ru.pw), rv.pw == 0 ? gl : DB.gaussjacobi(M, 0.0, rv.pw))
    function dir(r, q, a, b)
        if DB.is_gcq(r) && r.pw != 0 && a == 0
            o = b .* (1 .+ q[1]) ./ 2
            return o, o, (b / 2)^(r.pw + 1) .* q[2] ./ o .^ r.pw            # weights for the full density
        end
        o = (b - a) .* (1 .+ gl[1]) ./ 2
        return a .+ o, o, (b - a) / 2 .* gl[2]
    end
    stack = [(b..., 0) for b in boxes]
    while !isempty(stack)
        ua, ub, va, vb, depth = pop!(stack)
        um = (ua + ub) / 2; vm = (va + vb) / 2
        rad = maximum(norm(DB._flat_step(g, um, vm, s - um, s2 - vm)) for (s, s2) in ((ua, va), (ub, va), (ub, vb), (ua, vb)))
        if norm(wt .- DB._flat_step(g, ur, vr, um - ur, vm - vr)) >= c * rad || depth >= 60
            xu, ou, wu = dir(ru, gj[1], ua, ub)
            xv, ov, wv = dir(rv, gj[2], va, vb)
            K = [begin
                     d = wt .- DB._flat_step(g, ur, vr, (ua - ur) + ou[i], (va - vr) + ov[j])
                     J = norm(cross(collect(g.a .+ xv[j] .* g.c), collect(g.b .+ xu[i] .* g.c)))
                     wu[i] * wv[j] * J * (nrm === nothing ? 1 / (4π * norm(d)) : dot(nrm, d) / (4π * norm(d)^3))
                 end for i in eachindex(xu), j in eachindex(xv)]
            Eu = E(ru, xu, 1); Ev = E(rv, xv, 2)
            F .+= Eu' * K * Ev; Fa .+= abs.(Eu)' * abs.(K) * abs.(Ev)
        else
            lu = norm(DB._flat_step(g, ua, vm, ub - ua, 0.0)); lv = norm(DB._flat_step(g, um, va, 0.0, vb - va))
            us = lv > 2lu ? ((ua, ub),) : ((ua, um), (um, ub))
            vs = lu > 2lv ? ((va, vb),) : ((va, vm), (vm, vb))
            for (a1, b1) in us, (a2, b2) in vs
                push!(stack, (a1, b1, a2, b2, depth + 1))
            end
        end
    end
end

# pl_singular! for the global density on a rectangle Q holding the target (any rectangle: the density is one formula)
function gl_singular!(F, Fa, P, g, u0, v0, Q; M = 20, grade = 0.25, E = (r, x, d) -> gl_E(r, x))
    ua, ub, va, vb = Q
    tg, wg = DB.gausslegendre(M)
    yu0 = g.a .+ v0 .* g.c; yv0 = g.b .+ u0 .* g.c
    len(du, dv) = norm(du .* yu0 .+ dv .* yv0)
    G = (dot(yu0, yu0), dot(yu0, yv0), dot(yv0, yv0))
    ip(x, y) = x[1] * y[1] * G[1] + (x[1] * y[2] + x[2] * y[1]) * G[2] + x[2] * y[2] * G[3]
    U = Float64[]; V = Float64[]; W = Float64[]
    corners = ((ua, va), (ub, va), (ub, vb), (ua, vb))
    for s in 1:4
        p1, p2 = corners[s], corners[mod1(s + 1, 4)]
        d0 = (p1[1] - u0, p1[2] - v0); e = (p2[1] - p1[1], p2[2] - p1[2])
        det = abs(d0[1] * e[2] - d0[2] * e[1])
        det == 0 && continue
        tstar = -ip(d0, e) / ip(e, e)
        pieces = [(0.0, 1.0)]; done = Tuple{Float64, Float64}[]
        while !isempty(pieces)
            α, β = pop!(pieces)
            t = clamp(tstar, α, β)
            (len(e...) * (β - α) <= grade * len((d0 .+ t .* e)...) || β - α < 1e-30) ? push!(done, (α, β)) :
                push!(pieces, (α, (α + β) / 2), ((α + β) / 2, β))
        end
        for (α, β) in done, j in 1:M
            t = α + (β - α) * (1 + tg[j]) / 2; wt = (β - α) / 2 * wg[j]
            D = d0 .+ t .* e
            # radially: the density varies on the scale of the distance to a GCQ edge, so ρ is graded towards it as well
            for (ρ, wρ) in gl_radial(D, u0, v0, P, tg, wg)
                u = u0 + ρ * D[1]; v = v0 + ρ * D[2]
                sep = D[1] .* (g.a .+ (v0 + ρ * D[2]) .* g.c) .+ D[2] .* (g.b .+ u0 .* g.c)
                J = norm(cross(collect(g.a .+ v .* g.c), collect(g.b .+ u .* g.c)))
                push!(U, u); push!(V, v); push!(W, wρ * wt * det * J / (4π * norm(sep)))
            end
        end
    end
    Eu = E(P.ru, U, 1); Ev = E(P.rv, V, 2)
    F .+= Eu' * Diagonal(W) * Ev; Fa .+= abs.(Eu)' * Diagonal(abs.(W)) * abs.(Ev)
end

# radial pieces of [0, 1] along the ray (u0, v0) + ρ D: halved towards ρ = 1 while a piece is longer than `grade` × the
# distance of its far end from a GCQ edge (u = 0 or v = 0) beyond the ray's end; M points per piece
function gl_radial(D, u0, v0, P, tg, wg; grade = 0.5)
    out = Tuple{Float64, Float64}[]
    pieces = [(0.0, 1.0)]
    while !isempty(pieces)
        α, β = pop!(pieces)
        dist = Inf
        DB.is_gcq(P.ru) && D[1] < 0 && (dist = min(dist, (u0 + β * D[1]) / -D[1]))
        DB.is_gcq(P.rv) && D[2] < 0 && (dist = min(dist, (v0 + β * D[2]) / -D[2]))
        if β - α > grade * dist && β - α > 1e-3
            push!(pieces, (α, (α + β) / 2), ((α + β) / 2, β))
        else
            for i in eachindex(tg)
                push!(out, (α + (β - α) * (1 + tg[i]) / 2, (β - α) / 2 * wg[i]))
            end
        end
    end
    return out
end

# a square around (u0, v0) in the panel's metric there, of side the cell's shorter physical side, clipped to the cell (k, l):
# the graded fan of gl_singular! on an elongated cell (aspect up to ~1e13 beside a GCQ edge) loses accuracy, so the
# rest of the cell is left to gl_boxes!
function gl_balanced(P, u0, v0, k, l)
    g = DB._flat_geom(P)
    (sa, sb), (ta, tb) = P.ru.segs[k], P.rv.segs[l]
    yu = g.a .+ v0 .* g.c; yv = g.b .+ u0 .* g.c
    lu, lv = norm(yu), norm(yv); h = min((sb - sa) * lu, (tb - ta) * lv) / 2
    return (max(sa, u0 - h / lu), min(sb, u0 + h / lu), max(ta, v0 - h / lv), min(tb, v0 + h / lv))
end

"""
The single-layer row of the GLOBAL density of P at its point (u0, v0): the rectangle Q (default a balanced square around
it in its fine pair (k, l), gl_balanced) by gl_singular!, the rest of the panel as the boxes of the fine grid outside Q by gl_boxes!. Returns
(row, |row|) in the node basis, ordering (i_u - 1) n_v + j_v. `E(r, x, d)` (default gl_E(r, x)) gives the basis values in
direction d (1: u, 2: v): one returning a single column (an analytic factor) integrates a tensor-product density.
"""
function gl_self_ref(P, u0, v0, k, l; Q = gl_balanced(P, u0, v0, k, l), c = 6.0, M = 20, grade = 0.25,
                     E = (r, x, d) -> gl_E(r, x))
    g = DB._flat_geom(P)
    nu, nv = size(E(P.ru, [0.5], 1), 2), size(E(P.rv, [0.5], 2), 2)
    F = zeros(nu, nv); Fa = zeros(nu, nv)
    ua, ub, va, vb = Q
    # the fine grid's boxes, cut by Q's lines, outside Q
    cuts(r, a, b) = sort(unique(vcat([s for (s, _) in r.segs], 1.0, a, b)))
    cu = cuts(P.ru, ua, ub); cv = cuts(P.rv, va, vb)
    boxes = [(cu[i], cu[i + 1], cv[j], cv[j + 1]) for i in 1:length(cu)-1 for j in 1:length(cv)-1
             if !(cu[i] >= ua && cu[i + 1] <= ub && cv[j] >= va && cv[j + 1] <= vb)]
    gl_boxes!(F, Fa, P, g, u0, v0, zeros(3), boxes; c, M, E)
    gl_singular!(F, Fa, P, g, u0, v0, Q; M, grade, E)
    return vec(transpose(F)), vec(transpose(Fa))
end

"The single layer of σ = 1 on the planar polygon with corners c (in order) at x0 on its plane (normal n): closed form."
function gl_polygon_slp(c, x0, n)
    s = 0.0
    for i in eachindex(c)
        a, b = c[i], c[mod1(i + 1, length(c))]
        e = normalize(b - a); f = a + dot(x0 - a, e) * e
        h = norm(x0 - f)
        h < 1e-300 && continue
        s += sign(dot(cross(a - x0, b - x0), n)) * h * (asinh(dot(b - f, e) / h) - asinh(dot(a - f, e) / h))
    end
    return abs(s) / (4π)
end
