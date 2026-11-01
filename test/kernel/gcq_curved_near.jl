# The curved GCQ row engine (src/kernel/gcq_curved_near.jl): tensor_near_row's quadtree on the exact map of a
# `CurvedPanel` for targets off the panel, checked against tensor_near_row on an affine panel and against brute-force
# Gauss on the lens; for targets on it (own nodes), the Duffy region plus the quadtree, checked against an exact
# identity on the sphere, tensor_near_row on an affine panel and an adaptive reference on the lens. Last, what the
# mesher's map closures, the engine's geometry and a row allocate. No SCTL.
using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using FastGaussQuadrature: gausslegendre, gaussjacobi
using HCubature
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

# Brute-force rows (:dt, :sl) of P for targets X (P's anchor frame) with normals N: nq-point Gauss-Legendre on each fine
# u-segment × nq points on each of 8 equal v-pieces, Gauss-Jacobi for the weight u^pw on the weighted edge segment (the
# cardinal functions divided by u^pw there), the cardinal functions from tensor_rule_interp and plain differences of
# positions (the targets are far enough away for both)
function gcq_brute_rows(P, X, N; nq = 48)
    ru, rv = P.ru, P.rv
    t, w = gausslegendre(nq)
    us = Float64[]; wu = Float64[]; Eus = Matrix{Float64}[]
    for (sa, sb) in ru.segs
        if sa == 0 && ru.pw != 0
            tj, wj = gaussjacobi(nq, 0.0, ru.pw)
            u = sb .* (1 .+ tj) ./ 2
            append!(us, u); append!(wu, (sb / 2)^(ru.pw + 1) .* wj)
            push!(Eus, BI.tensor_rule_interp(ru, u) ./ u .^ ru.pw)
        else
            u = sa .+ (sb - sa) .* (1 .+ t) ./ 2
            append!(us, u); append!(wu, (sb - sa) / 2 .* w)
            push!(Eus, BI.tensor_rule_interp(ru, u))
        end
    end
    Eu = reduce(vcat, Eus)
    vs = reduce(vcat, [(k - 1 .+ (1 .+ t) ./ 2) ./ 8 for k in 1:8]); wv = repeat(w ./ 16, 8)
    Ev = BI.tensor_rule_interp(rv, vs)
    Y = Matrix{NTuple{3, Float64}}(undef, length(us), length(vs)); JW = zeros(length(us), length(vs))
    Threads.@threads for a in eachindex(us)
        for b in eachindex(vs)
            Y[a, b] = BI.curved_lpoint(P, us[a], vs[b])
            JW[a, b] = BI.curved_frame(P, us[a], vs[b])[4] * wu[a] * wv[b]
        end
    end
    return map(eachindex(X)) do i
        Kd = similar(JW); Ks = similar(JW)
        for j in eachindex(JW)
            d = X[i] .- Y[j]; r = norm(d)
            Kd[j] = JW[j] * dot(N[i], d) / (4π * r^3); Ks[j] = JW[j] / (4π * r)
        end
        (vec(permutedims(Eu' * Kd * Ev)), vec(permutedims(Eu' * Ks * Ev)))
    end
end

@testset "gcq_curved_row: off-panel targets" begin
    # bounds: about 10× the largest value measured at -t 1, 2 and 8 (OMP_NUM_THREADS alike), which differ at rounding
    # (a) an affine CurvedPanel on a GCQ edge TensorPanel of a box: the same rows as tensor_near_row
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 8,
                                      rule = edge_gcq_rule(10.0, 1.0; tol = 1e-8))
    Pt = first(p for p in itf.panels if p.kind == :edge)
    @test Pt.ru.pw != 0
    o, eu, ev, Lu, Lv, nP = Pt.o, Pt.eu, Pt.ev, Pt.Lu, Pt.Lv, Pt.normal
    Pc = BI.CurvedPanel(Pt.anchor, (u, v) -> o .+ (u * Lu) .* eu .+ (v * Lv) .* ev, Pt.ru, Pt.rv; kind = :edge,
                        flip = dot(cross(collect(eu), collect(ev)), collect(nP)) < 0, dmap = (u, v) -> (Lu .* eu, Lv .* ev))
    @test all(n -> isapprox(collect(n), collect(nP); atol = 1e-15), Pc.normals)
    # Dᵀ: 20 targets on the box's face across the panel's edge (u = 0), 1e-7 … 1e-1 from it, alternately at the panel's
    # node rows and between them; the target normal is that face's (outward, -eu)
    vpos = sort([Pt.rv.x; (Pt.rv.x[1:end-1] .+ Pt.rv.x[2:end]) ./ 2])
    nt = .-eu
    kdt(d) = dot(nt, d) / (4π * norm(d)^3)
    worst_dt = 0.0
    for (i, δ) in enumerate(10.0 .^ range(-7, -1, length = 20))
        x = o .+ (vpos[mod1(i, length(vpos))] * Lv) .* ev .- δ .* nP
        rt = tensor_near_row(kdt, Pt, x)
        rc = BI.gcq_curved_row(:dt, Pc, x; n = nt)
        worst_dt = max(worst_dt, maximum(abs, rc .- rt) / maximum(abs, rt))
    end
    @test worst_dt <= 1e-13                                             # 9.6e-15 measured
    # single layer: 10 targets 1e-6 … 1e-1 above the plane, over points near the edge (one inside the weighted edge
    # segment [0, 2^-K]), in the middle and at the far end, at node rows and between them
    ksl(d) = 1 / (4π * norm(d))
    upos = [5e-10, Pt.ru.x[1], (Pt.ru.x[1] + Pt.ru.x[2]) / 2, Pt.ru.x[4], (Pt.ru.x[9] + Pt.ru.x[10]) / 2, Pt.ru.x[end]]
    @test upos[1] < Pt.ru.segs[1][2]
    worst_sl = 0.0
    for (i, h) in enumerate(10.0 .^ range(-6, -1, length = 10))
        x = o .+ (upos[mod1(i, length(upos))] * Lu) .* eu .+ (vpos[mod1(3i, length(vpos))] * Lv) .* ev .+ h .* nP
        rt = tensor_near_row(ksl, Pt, x)
        rc = BI.gcq_curved_row(:sl, Pc, x)
        worst_sl = max(worst_sl, maximum(abs, rc .- rt) / maximum(abs, rt))
    end
    @test worst_sl <= 2e-14                                             # 2.1e-15

    # (b) the lens edge panel (curved, weighted edge rule) against brute force: 4 nodes of neighbouring panels (the
    # same patch, the patch beside, the cap's top GL panel, the other cap across the rim) and 4 points off the surface,
    # all at least 0.05 × P's diameter from P
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    ps = m.interface.panels
    kP = findfirst(p -> p.kind == :edge, ps)
    P = ps[kP]
    @test P.ru.pw != 0
    diam = BI._curved_diam(P, 0.0, 1.0, 0.0, 1.0)
    offset(T, j) = (T.anchor .- P.anchor) .+ T.lpoints[j]
    X = NTuple{3, Float64}[]; N = NTuple{3, Float64}[]
    for k in (2, 8, 12, 13)
        T = ps[k]
        ds = [BI._curved_closest(P, offset(T, j))[3] for j in eachindex(T.lpoints)]
        @test minimum(ds) < 0.05 * diam                                 # a neighbour of P
        j = argmin(j -> ds[j] >= 0.05 * diam ? ds[j] : Inf, eachindex(ds))
        push!(X, offset(T, j)); push!(N, T.normals[j])
    end
    for (u, v, s) in ((0.02, 0.5, 0.05), (0.5, 0.05, -0.05), (0.3, 0.8, 0.1), (0.9, 0.3, -0.3))
        _, _, nn, _ = BI.curved_frame(P, u, v)
        push!(X, BI.curved_lpoint(P, u, v) .+ (s * diam) .* nn); push!(N, nn)
    end
    ref = gcq_brute_rows(P, X, N)
    worst = [0.0, 0.0]
    for i in eachindex(X)
        rd = BI.gcq_curved_row(:dt, P, X[i]; n = N[i]); rs = BI.gcq_curved_row(:sl, P, X[i])
        worst[1] = max(worst[1], maximum(abs, rd .- ref[i][1]) / maximum(abs, ref[i][1]))
        worst[2] = max(worst[2], maximum(abs, rs .- ref[i][2]) / maximum(abs, ref[i][2]))
    end
    @test worst[1] <= 9e-14                                             # 8.8e-15 measured (Dᵀ)
    @test worst[2] <= 5e-14                                             # 5.0e-15 (single layer)

    # (c) across the rim: the nodes of the other cap's edge panel Q nearest P (as in test/ext/sctl_curved.jl), offsets
    # from the anchors' difference; all rows finite, and the 4 targets farthest from the rim against brute force
    mid(p, u, v) = p.anchor .+ BI.curved_lpoint(p, u, v)
    kQ = argmin(k -> m.panel_patch[k] in 6:10 && ps[k].kind == :edge ?
                     norm(collect(mid(ps[k], 0.0, 0.5) .- mid(P, 0.0, 0.5))) : Inf, eachindex(ps))
    Q = ps[kQ]
    XQ = [offset(Q, j) for j in eachindex(Q.lpoints)]
    rows = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(XQ))
    Threads.@threads for j in eachindex(XQ)
        rows[j] = (BI.gcq_curved_row(:dt, P, XQ[j]; n = Q.normals[j]), BI.gcq_curved_row(:sl, P, XQ[j]))
    end
    @test all(r -> all(isfinite, r[1]) && all(isfinite, r[2]), rows)
    far = sortperm([BI._curved_closest(P, x)[3] for x in XQ])[end-3:end]
    refQ = gcq_brute_rows(P, XQ[far], Q.normals[far]; nq = 64)
    worst_rim = [0.0, 0.0]
    for (i, j) in enumerate(far), s in 1:2
        worst_rim[s] = max(worst_rim[s], abs(sum(rows[j][s]) - sum(refQ[i][s])) / abs(sum(refQ[i][s])))
    end
    @test worst_rim[1] <= 6e-15                                         # 5.8e-16 measured (Dᵀ)
    @test worst_rim[2] <= 9e-15                                         # 8.7e-16 (single layer)

    # (d) Review Focus 2: a target on P (here a node) passed without its parameters throws, never a wrong row
    @test_throws ArgumentError BI.gcq_curved_row(:sl, P, P.lpoints[5])

    # (e) targets across the rim within a few widths of P's weighted edge segment [0, 2^-K] (6e-8 here): points of Q
    # 6.9e-8, 6.3e-10 and 7.5e-13 from P, where the quadtree would take the whole edge segment as a leaf. Its own rule
    # (interpolatory product weights for u^pw, exact to degree nz - 1) was off by up to 2.3e-7 of max|row| there; the
    # segment is now split, and the rows converge in η
    cases = [(uq, vq, kind) for (uq, vq) in ((1e-7, 0.5), (1e-9, 0.3), (1e-12, 0.77)) for kind in (:dt, :sl)]
    err_edge = zeros(length(cases))
    Threads.@threads for c in eachindex(cases)                         # (an η = 6 row takes 2-4 s)
        uq, vq, kind = cases[c]
        x = (Q.anchor .- P.anchor) .+ BI.curved_lpoint(Q, uq, vq)
        nq = BI.curved_frame(Q, uq, vq)[3]
        r6 = BI.gcq_curved_row(kind, P, x; n = nq, η = 6.0)
        err_edge[c] = maximum(abs, BI.gcq_curved_row(kind, P, x; n = nq) .- r6) / maximum(abs, r6)
    end
    worst_edge = maximum(err_edge)
    @test worst_edge <= 7e-14                                           # 7.3e-15 measured (2.3e-7 before the split)
end

# An adaptive reference for the Dᵀ rows of targets on a curved GCQ panel P (its nodes), independent of the engine's
# fixed rules and of the panel's lmap and dmap: the map is the patch's, f ∘ pa (absolute), with its tangents and second
# derivatives by dual numbers; y - x is the tangents' path integral (4 … 24 Gauss points by length) below a physical
# distance of 0.2, and n·(y - x) the second-derivative integral ∫₀¹ (1 - σ) n·y''(σ) dσ below 1e-2. The target's whole
# fine segment [sa, sb] × [0, 1] is integrated by nested adaptive Gauss-Kronrod (HCubature.hquadrature) in polar
# coordinates about the target (along each side, split at the foot of the perpendicular), every other segment by
# HCubature.hcubature on v-strips graded toward the target (the weighted edge segment after u = δ w^(1/(1 + pw)), which
# takes out the weight). Ported from the spike scripts/curved3d/53_gcq_duffy_spike.jl (ref_c, polar_adaptive) of the
# study repository. Dᵀ only (the operator's self rows): the single layer, at about twice the cost, would not fit the
# testset's CI budget; its rows share every rule but the kernel with the Dᵀ rows and are checked in (a) and (c).
const GCQ_REF_GL = Dict(n => ((t, w) = gausslegendre(n); ((1 .+ t) ./ 2, w ./ 2)) for n in (4, 8, 12, 24))
# Σ_j |node-rule Dᵀ entry| of P's row for the target x (P's frame) with normal n: the row's scale, for the reference's
# absolute tolerance
function node_rule_scale(P, x, n)
    s = 0.0
    for (y, w) in zip(P.lpoints, P.weights)
        d = x .- y; r2 = dot(d, d)
        r2 == 0 && continue
        s += abs(w * dot(n, d) / (4π * r2 * sqrt(r2)))
    end
    return s
end
gcq_ref_nsel(λ) = λ < 1e-4 ? 4 : λ < 1e-2 ? 8 : λ < 0.1 ? 12 : 24
struct GCQRefTarget{F, PA, PP}
    f::F; pa::PA; P::PP
    c::NTuple{2, Float64}; k0::Int                       # the target's parameters and fine u-segment
    x::NTuple{3, Float64}; n::NTuple{3, Float64}         # its position (f ∘ pa) and normal
    su::Float64; sv::Float64; FF::NTuple{3, Float64}     # |y_u|, |y_v| and the first fundamental form there
end
function gcq_ref_geo(f, pa, u, v)
    du = BI.ForwardDiff.Dual{Nothing}(u, 1.0, 0.0); dv = BI.ForwardDiff.Dual{Nothing}(v, 0.0, 1.0)
    y = f(pa(du, dv)...)
    val(k) = BI.ForwardDiff.value(y[k]); part(k, j) = BI.ForwardDiff.partials(y[k], j)
    return (val(1), val(2), val(3)), (part(1, 1), part(2, 1), part(3, 1)), (part(1, 2), part(2, 2), part(3, 2))
end
function GCQRefTarget(f, pa, P, a, b)
    c = (P.ru.x[a], P.rv.x[b])
    k0 = clamp(searchsortedlast([sa for (sa, _) in P.ru.segs], c[1]), 1, length(P.ru.segs))
    x, yu, yv = gcq_ref_geo(f, pa, c...)
    return GCQRefTarget(f, pa, P, c, k0, x, P.normals[(a - 1) * length(P.rv.x) + b], sqrt(dot(yu, yu)),
                        sqrt(dot(yv, yv)), (dot(yu, yu), dot(yu, yv), dot(yv, yv)))
end
# y(c + Δ) - y(c), and n·(y(c + Δ) - y(c))
function gcq_ref_pdiff(T, Δ)
    σs, ws = GCQ_REF_GL[gcq_ref_nsel(hypot(Δ...))]
    s = (0.0, 0.0, 0.0)
    for (σ, w) in zip(σs, ws)
        _, yu, yv = gcq_ref_geo(T.f, T.pa, T.c[1] + σ * Δ[1], T.c[2] + σ * Δ[2])
        s = s .+ w .* (yu .* Δ[1] .+ yv .* Δ[2])
    end
    return s
end
function gcq_ref_nd(T, Δ)
    σs, ws = GCQ_REF_GL[gcq_ref_nsel(hypot(Δ...))]
    nf(t) = dot(T.n, T.f(T.pa(T.c[1] + t * Δ[1], T.c[2] + t * Δ[2])...))
    d2(σ) = BI.ForwardDiff.derivative(t -> BI.ForwardDiff.derivative(nf, t), σ)
    return sum(w * (1 - σ) * d2(σ) for (σ, w) in zip(σs, ws))
end
# K_dt J φ at c + Δ (φ flattened nu × nv), φ_u on fine segment k (the weight u^pw left out if !weight)
function gcq_ref_f(T, Δ, k; weight = true)
    P = T.P; ru, rv = P.ru, P.rv
    z = (T.c[1] + Δ[1], T.c[2] + Δ[2])
    y, yu, yv = gcq_ref_geo(T.f, T.pa, z...)
    J = norm(cross(collect(yu), collect(yv)))
    λ = hypot(Δ[1] * T.su, Δ[2] * T.sv)
    d = λ >= 0.2 ? y .- T.x : gcq_ref_pdiff(T, Δ)                          # y - x
    r2 = dot(d, d)
    nd = λ < 1e-2 ? gcq_ref_nd(T, Δ) : dot(T.n, d)
    sa, sb = ru.segs[k]; nz = length(ru.z0)
    ℓ = ru.bw ./ ((2 * (z[1] - sa) / (sb - sa) - 1) .- ru.z0)
    eu = view(ru.P, (k - 1) * nz .+ (1:nz), :)' * (ℓ ./ sum(ℓ))
    weight && ru.pw != 0 && (eu .*= z[1]^ru.pw)
    ℓv = rv.bw ./ ((2 * z[2] - 1) .- rv.z0); ev = rv.P' * (ℓv ./ sum(ℓv))
    return vec((-J * nd / (4π * r2 * sqrt(r2))) .* eu .* ev')
end
function gcq_ref_polar(T, R, tol, atol; maxevals = 10^6)
    ua, ub, va, vb = R; c = T.c
    corners = ((ua, va), (ub, va), (ub, vb), (ua, vb))
    I = zeros(length(T.P.ru.x) * length(T.P.rv.x))
    for i in 1:4
        p = corners[i]; q = corners[mod1(i + 1, 4)]
        p0 = (p[1] - c[1], p[2] - c[2]); e = (q[1] - p[1], q[2] - p[2])
        det = abs(p0[1] * e[2] - p0[2] * e[1])
        det == 0 && continue
        g11, g12, g22 = T.FF
        A = g11 * e[1]^2 + 2g12 * e[1] * e[2] + g22 * e[2]^2
        B = g11 * p0[1] * e[1] + g12 * (p0[1] * e[2] + p0[2] * e[1]) + g22 * p0[2] * e[2]
        tst = -B / A
        brk = 0 < tst < 1 ? (0.0, tst, 1.0) : (0.0, 1.0)
        inner = t -> begin
            dz = (p0[1] + t * e[1], p0[2] + t * e[2])
            hquadrature(ρ -> (ρ * det) .* gcq_ref_f(T, (ρ * dz[1], ρ * dz[2]), T.k0), 0.0, 1.0;
                        rtol = tol / 10, atol = atol / 10, maxevals)[1]
        end
        for j in 1:length(brk)-1
            I .+= hquadrature(inner, brk[j], brk[j+1]; rtol = tol, atol, maxevals)[1]
        end
    end
    return I
end
# the reference Dᵀ row of the node (a, b) of P, in P's node order
function gcq_ref_row(T::GCQRefTarget, tol, atol; maxevals = 2 * 10^6)
    P = T.P; ru = P.ru; c = T.c
    sa, sb = ru.segs[T.k0]
    I = gcq_ref_polar(T, (sa, sb, 0.0, 1.0), tol, atol)
    for k in eachindex(ru.segs)
        k == T.k0 && continue
        sa, sb = ru.segs[k]
        # v-breakpoints graded toward v0 (v0 ± ℓ 4^j, ℓ the segment's width or u-distance, whichever is larger), so that
        # hcubature starts from boxes of moderate aspect ratio along these long strips
        ℓ = max(sb - sa, max(sa - c[1], c[1] - sb, 0.0)) * T.su / T.sv
        vbr = [c[2]]
        for sgn in (-1, 1), j in 0:40
            v = c[2] + sgn * ℓ * 4.0^j
            0 < v < 1 || break
            push!(vbr, v)
        end
        vbr = sort(vcat(vbr, 0.0, 1.0))
        for (va, vb) in zip(vbr[1:end-1], vbr[2:end])
            if sa == 0 && ru.pw != 0
                δ = sb; γ = 1 / (1 + ru.pw)                                   # u = δ w^γ: u^pw du = δ^(1+pw) γ dw
                f = x -> (δ^(1 + ru.pw) * γ) .* gcq_ref_f(T, (δ * x[1]^γ - c[1], x[2] - c[2]), k; weight = false)
                I .+= hcubature(f, (0.0, va), (1.0, vb); rtol = tol, atol, maxevals)[1]
            else
                f = x -> gcq_ref_f(T, (x[1] - c[1], x[2] - c[2]), k)
                I .+= hcubature(f, (sa, va), (sb, vb); rtol = tol, atol, maxevals)[1]
            end
        end
    end
    return vec(permutedims(reshape(I, length(ru.x), length(P.rv.x))))
end

@testset "gcq_curved_row: targets on the panel" begin
    # bounds: about 10× the largest value measured at -t 1, 2 and 8 (OMP_NUM_THREADS alike), below the plan's 1e-12 (a),
    # 1e-13 (c) and 2e-12 (d)
    # (a) the split sphere's thin GCQ edge panel (as in test/ext/sctl_curved.jl): on the unit sphere n_x = ±x, so
    # n_x·(x - y) = ±|x - y|²/2 and K_dt = ±K_sl/2 pointwise. The Dᵀ row takes n·(x - y) from the second derivatives
    # (the Duffy region and leaf points near the target) or from x - y, the single layer only |x - y|: the two rows,
    # from the same points, check that n·(x - y). Own nodes nearest the edge and farthest from it, at both v-ends, and
    # points of the panel that are not nodes: in fine segment 2 (next to the weighted edge segment), on the boundary of
    # segments 1 and 2 (u0 = s1), on that of segments 5 and 6 at v0 = 0, and 1e-5 from v = 0 in the last segment
    gs = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 2.0, 8.0, 1.0))
    ms = curved_gcq_mesh(gs; nlev = 3, n_quad = 8, tol = 1e-12)
    pss = ms.interface.panels
    S = pss[findfirst(k -> ms.panel_patch[k] <= 5 && pss[k].kind == :edge, eachindex(pss))]       # upper cap
    nus, nvs, ss = length(S.ru.x), length(S.rv.x), S.ru.segs
    sph = [[(S.ru.x[a], S.rv.x[b]) for a in (1, nus) for b in (1, nvs)];
           [(1.5 * ss[1][2], 0.37), (ss[1][2], 0.37), (ss[5][2], 0.0), (0.77, 1e-5)]]
    err_sphere = zeros(length(sph)); nx = zeros(length(sph))
    Threads.@threads for q in eachindex(sph)                           # (a Dᵀ row nearest the edge takes 0.7 s)
        on = sph[q]
        x = BI.curved_lpoint(S, on...); n = BI.curved_frame(S, on...)[3]
        nx[q] = dot(n, S.anchor .+ x)
        rd = BI.gcq_curved_row(:dt, S, x; n, on)
        rs = BI.gcq_curved_row(:sl, S, x; on)
        err_sphere[q] = maximum(abs, rd .- sign(nx[q]) .* rs ./ 2) / maximum(abs, rs)     # outward (+1) or inward (-1)
    end
    @test all(t -> abs(t) ≈ 1, nx)
    @test maximum(err_sphere) <= 7e-14                                 # 6.5e-15 measured (4.6e-13: hess cutoff 1e-2)

    # (c) the affine CurvedPanel of (off-panel targets, a) on the box's GCQ edge TensorPanel: single-layer rows of own
    # nodes against tensor_near_row, whose quadtree recurses toward a target on the panel to maxdepth. Nodes: the
    # innermost (in v's middle and at its end), the u-nodes nearest their fine segment's end (0.7 % and 8.6 % of its
    # width), one in the middle, and one whose Duffy square is clipped at v = 0
    affine(T) = BI.CurvedPanel(T.anchor, (u, v) -> T.o .+ (u * T.Lu) .* T.eu .+ (v * T.Lv) .* T.ev, T.ru, T.rv;
                               kind = T.kind, flip = dot(cross(collect(T.eu), collect(T.ev)), collect(T.normal)) < 0,
                               dmap = (u, v) -> (T.Lu .* T.eu, T.Lv .* T.ev))
    boxpanels(nq) = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = nq,
                                                rule = edge_gcq_rule(10.0, 1.0; tol = 1e-8)).panels
    ksl(d) = 1 / (4π * norm(d))
    function flat_err(T, C, on; ref = nothing)                         # C's single-layer row against tensor_near_row's
        x = BI.curved_lpoint(C, on...)
        r = BI.gcq_curved_row(:sl, C, x; on)
        rt = ref === nothing ? tensor_near_row(ksl, T, x) : ref
        return maximum(abs, r .- rt) / maximum(abs, rt)
    end
    Pt = first(p for p in boxpanels(8) if p.kind == :edge)
    Pc = affine(Pt)
    worst_flat = maximum(((a, b),) -> flat_err(Pt, Pc, (Pc.ru.x[a], Pc.rv.x[b])),
                         ((1, 4), (1, 1), (20, 5), (8, 8), (13, 4), (11, 1)))
    @test worst_flat <= 4e-14                                           # 3.3e-15 measured
    # points that are not nodes: 1e-5 from v = 0 in the last segment (R's v = 0 side next to the target gets
    # 2 DUFFY_NSIDE points; 1.6e-9 with DUFFY_NSIDE), on the boundary of segments 5 and 6 at v0 = 0
    e_flat2 = maximum(on -> flat_err(Pt, Pc, on), ((0.77, 1e-5), (Pc.ru.segs[5][2], 0.0)))
    @test e_flat2 <= 6e-14                                              # 6.3e-15 measured
    # next to the weighted edge segment: in segment 2, and on its boundary with segment 1 (u0 = s1). tensor_near_row
    # takes that segment whole as a leaf (its interpolatory rule; 1.6e-12 … 1.9e-12 off here, moving with η), so the
    # reference is the curved row at η = 6
    s1 = Pc.ru.segs[1][2]
    e_flat3 = maximum(((1.5 * s1, 0.37), (s1, 0.37))) do on
        flat_err(Pt, Pc, on; ref = BI.gcq_curved_row(:sl, Pc, BI.curved_lpoint(Pc, on...); on, η = 6.0))
    end
    @test e_flat3 <= 2e-13                                              # 1.8e-14 measured
    # n_quad 16: a v-end own node of the last segment and a point near its corner (u0, v0) = (0.501, 0.001), where 16
    # Gauss points in ρ left 6.9e-14 and 1.4e-12 (now 20, from the basis' degree)
    Pt16 = first(p for p in boxpanels(16) if p.kind == :edge); Pc16 = affine(Pt16)
    e_flat4 = maximum(on -> flat_err(Pt16, Pc16, on), ((Pc16.ru.x[11], Pc16.rv.x[1]), (0.501, 0.001)))
    @test e_flat4 <= 1e-13                                              # 9.7e-15 measured
    # panels of a 1 × 2 × 0.5 box, where |y_u| ≠ |y_v|, so that R takes the cuts of the quadtree's boxes in all its
    # forms: Gauss-Legendre × Gauss-Legendre panels (one segment in u: k = 1, pw = 0) with Lu/Lv = 1/2 (R strictly
    # inside the root box in v: it keeps a piece below and one above) and 4 (Δv ≥ 1: R is the whole panel), and the
    # last segment of GCQ edge panels with Lu/Lv = 4 (R is all of segment k) and 1/4 (cut from the box peeled off
    # the cluster)
    pbox = single_dielectric_box3d_gcq(1.0, 2.0, 0.5, 10.0, 1.0; nlev = 2, n_quad = 8,
                                       rule = edge_gcq_rule(10.0, 1.0; tol = 1e-8)).panels
    e_flat5 = maximum(((:gl, 0.5), (:gl, 4.0), (:edge, 4.0), (:edge, 0.25))) do (kind, ratio)
        T = first(p for p in pbox if p.kind == kind && p.Lu / p.Lv ≈ ratio)
        C = affine(T)
        flat_err(T, C, (C.ru.x[kind === :gl ? 4 : end], C.rv.x[4]))
    end
    @test e_flat5 <= 1.2e-13                                            # 1.1e-14 measured

    # (d) the lens edge panel of (off-panel targets, b) (n_quad 8): Dᵀ rows of own nodes against the adaptive reference
    # above at rtol 1e-13. Nodes: the innermost u-node (fine segment 11, [3.1e-5, 6.1e-5]); the u-node nearest its
    # segment's end (0.7 % of its width); a node whose Duffy square is clipped at v = 0; a middle node
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    ps = m.interface.panels
    kP = findfirst(p -> p.kind == :edge, ps)
    P = ps[kP]
    f, pa = g.patches[m.panel_patch[kP]].map, m.panel_param[kP]
    nu, nv = length(P.ru.x), length(P.rv.x)
    nodes = [(1, 4), (nu, 5), (12, 1), (8, 4)]
    @test nu == 16 && P.ru.segs[end] == (0.5, 1.0) && 1 - P.ru.x[nu] < 0.01 * 0.5     # the segment-extreme node
    yu, yv = BI._tangents(P, P.ru.x[12], P.rv.x[1])
    @test P.ru.x[12] > 0.5 && P.rv.x[1] < 0.5 * norm(collect(yu)) / norm(collect(yv)) / 2  # (12, 1)'s square: clipped
    # _curved_nd_hess from lmap twice (a panel without dmap) and from dmap: 2.6e-16 apart at most (measured)
    Pn = BI.CurvedPanel(P.anchor, P.lmap, P.ru, P.rv; kind = P.kind, flip = P.flip)
    c1, n1 = (P.ru.x[1], P.rv.x[4]), P.normals[4]
    @test all(Δ -> isapprox(BI._curved_nd_hess(Pn, c1, Δ, n1), BI._curved_nd_hess(P, c1, Δ, n1); rtol = 3e-15),
              ((1e-7, 2e-7), (-3e-3, 1e-2), (0.2, -0.1)))
    err_lens = zeros(length(nodes))
    Threads.@threads for q in eachindex(nodes)                         # (the reference takes 2-13 s per row)
        a, b = nodes[q]
        i = (a - 1) * nv + b
        x = P.lpoints[i]; n = P.normals[i]
        ref = gcq_ref_row(GCQRefTarget(f, pa, P, a, b), 1e-13, 1e-13 * node_rule_scale(P, x, n))
        rd = BI.gcq_curved_row(:dt, P, x; n, on = (P.ru.x[a], P.rv.x[b]))
        err_lens[q] = maximum(abs, rd .- ref) / maximum(abs, ref)
    end
    # measured: 4.9e-14 (innermost; 8.2e-13 with the hess cutoff at 1e-2), 6.3e-15 (segment-extreme; 1.4e-13 before its
    # near side got 2 DUFFY_NSIDE points), 1.7e-14 (clipped), 7.5e-15 (middle). The single layer, against the same
    # reference with its kernel (not in the test: 6-17 s more per row): 4.9e-14, 6.2e-15, 3.9e-15, 4.1e-15 (-t 8). The
    # reference itself, at rtol 1e-13 against 1e-14 on the innermost row: 2.0e-14 (Dᵀ), 2.4e-14 (single layer)
    @test maximum(err_lens) <= 5e-13

    # (e) Review Focus 3: a target on the panel within the weighted edge segment [0, 2^-K] (and exactly on the edge,
    # u = 0) throws, naming the segment
    s1 = P.ru.segs[1][2]
    for on in ((s1 / 2, 0.5), (0.0, 0.3))
        err = try
            BI.gcq_curved_row(:dt, P, BI.curved_lpoint(P, on...); n = BI.curved_frame(P, on...)[3], on)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("edge segment", err.msg)
    end

    # (f) a box of the quadtree overlapping R splits whatever its distance (Hazard H4): at (0.375, 0.45), with η = 0.05,
    # a cluster box overlapping R is not near; against η = 1.5 the row moves by 9.5e-9 (coarse leaves), by 0.18 when
    # that box is a leaf (R counted twice; measured with the forced split taken out). x within 1e-13 × diam of
    # curved_lpoint(P, on) is replaced by it (the row is the same); x farther, or n not normal to P at `on`, throws
    xh = BI.curved_lpoint(Pc, 0.375, 0.45)
    rh = BI.gcq_curved_row(:sl, Pc, xh; on = (0.375, 0.45))
    e_eta = maximum(abs, BI.gcq_curved_row(:sl, Pc, xh; on = (0.375, 0.45), η = 0.05) .- rh) / maximum(abs, rh)
    @test e_eta <= 1e-7                                                 # 9.5e-9 measured
    on14 = (Pc.ru.x[1], Pc.rv.x[4]); x14 = BI.curved_lpoint(Pc, on14...)
    r14 = BI.gcq_curved_row(:sl, Pc, x14; on = on14)
    @test BI.gcq_curved_row(:sl, Pc, x14 .+ (1e-15, -1e-15, 1e-15); on = on14) == r14
    @test_throws ArgumentError BI.gcq_curved_row(:sl, Pc, x14 .+ (1e-9, 0.0, 0.0); on = on14)
    on84 = (P.ru.x[8], P.rv.x[4]); n84 = P.normals[(8 - 1) * nv + 4]
    @test_throws ArgumentError BI.gcq_curved_row(:dt, P, P.lpoints[(8 - 1) * nv + 4]; on = on84,
                                                 n = normalize(collect(n84) .+ 1e-6))
end

# bytes allocated by f(args...), after a warm-up call (a function barrier: the testset's locals are not typed)
_allocs(f::F, args...) where F = (f(args...); @allocated f(args...))

@testset "gcq_curved_row: allocation" begin
    # the lens edge panel of (b) and (d): the mesher's map closures and the engine's geometry allocate nothing (before
    # Task 5b they collected the patch map into arrays: lmap 3.1 kB, dmap 0.5 kB, _curved_nd_hess 2.0 … 16.1 kB per
    # call, the own row below 421 MB, the off-panel row 29 MB)
    g = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(g; nlev = 0, n_quad = 8, tol = 1e-6)
    ps = m.interface.panels
    gcq = findall(p -> p.kind == :edge, ps)
    P = ps[gcq[1]]
    for (u, v) in ((1e-9, 0.3), (0.37, 0.61), (1.0, 1.0))
        @test _allocs(P.lmap, u, v) == 0
        @test _allocs(P.dmap, u, v) == 0
        @test _allocs(BI.curved_lpoint, P, u, v) == 0
        @test _allocs(BI._tangents, P, u, v) == 0
    end
    c = (0.37, 0.61); n = BI.curved_frame(P, c...)[3]
    for Δ in ((1e-7, 2e-7), (-3e-3, 1e-2), (0.2, -0.1))                # 2, 8 and 16 σ-points
        @test _allocs(BI._curved_step, P, c, Δ) == 0
        @test _allocs(BI._curved_nd_hess, P, c, Δ, n) == 0
    end
    # rows: the buffers (16.8 kB), the result, and for an own node the Duffy rule's points and basis matrices. Measured:
    # 1.73 MB (own, innermost u-node), 20.8 kB (off the panel: a node of the next GCQ panel)
    own = (P, x, n, on) -> BI.gcq_curved_row(:dt, P, x; n, on)
    @test _allocs(own, P, P.lpoints[4], P.normals[4], (P.ru.x[1], P.rv.x[4])) <= 3.5e6
    Q = ps[gcq[2]]
    off = (P, x, n) -> BI.gcq_curved_row(:dt, P, x; n)
    @test _allocs(off, P, (Q.anchor .- P.anchor) .+ Q.lpoints[1], Q.normals[1]) <= 4.2e4
end
