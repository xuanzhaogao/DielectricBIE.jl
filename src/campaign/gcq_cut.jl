# GCQ campaign mode: pair densities cut by dielectric faces.
#
# Where a face runs through a pair density rho, the screened source rho / eps(x) jumps, and every spectral or
# point-charge rule on the screened samples converges only like O(h) to O(h^2). Instead, every spectral computation
# acts on the smooth, unscreened rho, and the faces enter through quadrature:
#
# - V_vol[a, b] = <rho_a, N[rho_b / eps]> = int phi0_a rho_b / eps dx (symmetry of N; phi0_a = N[rho_a] is smooth): the
#   samples of phi0_a rho_b with the cut-cell screening dv m(x), m = the sum over the two sides of 1/eps_side times the
#   side's weights (the trapezoid weight, corrected on the few lattice levels next to the face).
#   For box regions this is a tensor product of 1D rules along the cut axes (each needs the grid aligned with it).
# - F far from the density: point charges dv m rho (high order for a smooth kernel).
# - F near the density, either
#   - (one horizontal face, interface nodes near it horizontal) u_inc = sum_side N[chi_side rho] / eps_side with
#     N[chi_hi rho] = phi0 - N[chi_lo rho], and dz N[chi_side rho] on the face or across it by the in-plane Fourier
#     transform of the half-space Green's function, exact in z (_gcq_halfspace_dz); or
#   - (any cut) the surface corrections of gcq_surface.jl: grad N[a rho] from psi = grad N[rho] and layer potentials of
#     psi and its tangential derivatives on the interface rectangles.
# - G (the potential of the test density at the nodes) needs no change: phi0_a is smooth.

"""
    _cutcell_correction(θ; p = 7) -> (δ, mlo)

Lattice levels at depths (m - 1 + θ) h into one side of a cut (m ∈ ℤ, m ≥ 1 on that side, θ ∈ [0, 1)). The rule
integrating, on every cell and on the partial cell [0, θ h], the degree-`p` interpolant of the p + 1 levels centred on
it (from both sides: the integrand is smooth across the cut, only the indicator jumps) has weights
h ([m ≥ 1] + δ_m), with δ nonzero only for m in mlo:mlo+p. Returns δ on that range and mlo.
"""
function _cutcell_correction(θ::Real; p::Int = 7)
    0 <= θ < 1 || throw(ArgumentError("θ must be in [0, 1)"))
    m0 = (p - 1) ÷ 2; L = 3p + 10
    ix(m) = m + L + 1
    z = [m - 1 + θ for m in -L:L]
    w = zeros(2L + 1); gx, gw = gausslegendre(8)
    cells = [(0.0, z[ix(1)], 0); [(z[ix(m)], z[ix(m + 1)], m) for m in 1:L-p]]       # (lo, hi, left level)
    for (lo, hi, c) in cells
        hi > lo || continue
        nodes = (c - m0):(c - m0 + p)
        for (t, wt) in zip(gx, gw)
            zz = (lo + hi) / 2 + (hi - lo) / 2 * t
            for m in nodes
                lm = 1.0
                for q in nodes
                    q == m || (lm *= (zz - z[ix(q)]) / (z[ix(m)] - z[ix(q)]))
                end
                w[ix(m)] += wt * (hi - lo) / 2 * lm
            end
        end
    end
    mlo = -m0
    return [w[ix(m)] - (m >= 1) for m in mlo:mlo+p], mlo
end

"""
    GCQCutClass

Pairs of one shape with the same cut-cell screening `m` at the shape's support points (the weights of V_vol and of
the far field of F, per unit dv). If one horizontal face cuts them (`plane`), its height `zf` in the frame of the
shape's representative placement and 1/ε below and above it (`a_lo`, `a_hi`); otherwise these are NaN.
"""
struct GCQCutClass
    shape::Int
    plane::Bool
    zf::Float64
    a_lo::Float64
    a_hi::Float64
    m::Vector{Float64}
end

# 1D cut-cell weights (per unit h) of the interval (lo, hi] at the lattice positions t (spacing h along the line):
# the rule of int_{-inf}^{hi} minus that of int_{-inf}^{lo}
function _cutcell_interval(t, lo, hi, h; p::Int = 7)
    w = zeros(length(t))
    for (z, sg) in ((hi, 1.0), (lo, -1.0))
        dep = (z .- t) ./ h                                     # depth below z, in levels: m - 1 + θ
        θ = mod(dep[1], 1.0); (θ > 1 - 1e-9 || θ < 1e-9) && (θ = 0.0)
        δ, mlo = _cutcell_correction(θ; p)
        for i in eachindex(t)
            mi = round(Int, dep[i] - θ) + 1
            w[i] += sg * ((mi >= 1) + (mlo <= mi <= mlo + p ? δ[mi - mlo + 1] : 0.0))
        end
    end
    return w
end

# the cut-cell screening at the support points X (absolute) of a cut density: a0 + sum over boxes of (a_b - a0) times
# the product over the axes of the 1D interval weights (the indicator along an axis whose faces are out of reach).
# A face within reach must be normal to an axis along which the lattice has exactly one vector.
function _gcq_cut_m(c::CampaignInput, dg, X; p::Int = 7)
    _, ea, eb, ec = _grid_frame(dg); E = (ea, eb, ec); sc = maximum(norm, E); tl = 1e-12 * sc
    a0 = 1 / c.eps_out; n = size(X, 2); m = fill(a0, n)
    reach = (p + 2) * sc
    for (bi, b) in enumerate(c.boxes)
        ab = 1 / c.epses[bi]; ab == a0 && continue
        lo, hi = _box_lohi(b)
        W = ones(n)
        for d in 1:3
            xd = view(X, d, :)
            if !any(x -> abs(x - lo[d]) < reach || abs(x - hi[d]) < reach, xd)
                W .*= (xd .> lo[d]) .& (xd .<= hi[d])
                continue
            end
            al = [abs(e[d]) > tl for e in E]
            count(al) == 1 || error("GCQ campaign mode: a density cut by a face normal to axis $d needs a grid with " *
                                    "exactly one lattice vector not parallel to that face")
            W .*= _cutcell_interval(xd, lo[d], hi[d], abs(E[findfirst(al)][d]); p)
        end
        m .+= (ab - a0) .* W
    end
    return m
end

# one horizontal face cutting the density (support gidx + dq, screening iv): (zf in the shape's frame, a_lo, a_hi),
# or nothing
function _gcq_horizontal_cut(c::CampaignInput, dg, gidx, dq, iv)
    o, ea, eb, ec = _grid_frame(dg)
    tl = 1e-12 * maximum(norm, (ea, eb, ec))
    abs(ea[3]) <= tl && abs(eb[3]) <= tl || return nothing
    lev = Dict{Int, Float64}()
    for (g, v) in zip(gidx, iv)
        v0 = get!(lev, g[3] + dq[3], v)
        v0 == v || return nothing                                      # not constant on a level: not horizontal
    end
    zk(k) = o[3] + (k - 1) * ec[3]
    ks = sort!(collect(Base.keys(lev)); by = zk)
    sw = findall(i -> lev[ks[i]] != lev[ks[i + 1]], 1:length(ks)-1)
    length(sw) == 1 || return nothing
    za, zb = zk(ks[sw[1]]), zk(ks[sw[1] + 1])
    tz = 1e-9 * max(1.0, abs(za), abs(zb))
    faces = unique(round.([b.center[3] + s * b.Lz / 2 for b in c.boxes, s in (-1, 1)
                           if za - tz <= b.center[3] + s * b.Lz / 2 <= zb + tz]; sigdigits = 12))
    length(faces) == 1 || return nothing
    return faces[1] - _grid_shift(dg, dq)[3], lev[ks[sw[1]]], lev[ks[sw[1] + 1]]
end

# the screening of every pair: cls[q] = 0 and inveps[q] = 1/ε for a density inside one region, or cls[q] = k for a
# density cut by faces (classes[k]: pairs of one shape with the same cut-cell screening)
function _gcq_screening(c::CampaignInput, dg, shapes, shape_of, delta; p::Int = 7)
    o, ea, eb, ec = _grid_frame(dg)
    P = length(shape_of); cls = zeros(Int, P); inveps = zeros(P)
    classes = GCQCutClass[]; cid = Dict{Any, Int}()
    inbox(b, x) = all(abs(x[d] - b.center[d]) <= (b.Lx, b.Ly, b.Lz)[d] / 2 for d in 1:3)
    ones_src(X) = VolumeSource(X, ones(size(X, 2)), ones(size(X, 2)))
    screen(X) = screened_volume_source(c.boxes, c.epses, c.eps_out, ones_src(X), SharpScreening()).density
    for q in 1:P
        sh = shapes[shape_of[q]]
        isempty(sh.gidx) && continue                     # a pair whose orbitals do not overlap: zero density
        # inside one box (all corners of the support's lattice bounding box) or outside every box: one region
        lo = ntuple(d -> minimum(g[d] for g in sh.gidx) + delta[q][d], 3)
        hi = ntuple(d -> maximum(g[d] for g in sh.gidx) + delta[q][d], 3)
        corners = reduce(hcat, [o .+ (i - 1) .* ea .+ (j - 1) .* eb .+ (k - 1) .* ec
                                for i in (lo[1], hi[1]), j in (lo[2], hi[2]), k in (lo[3], hi[3])])
        if any(b -> all(x -> inbox(b, x), eachcol(corners)), c.boxes)
            inveps[q] = screen(corners[:, 1:1])[1]; continue
        end
        X = _grid_positions(dg, [g .+ delta[q] for g in sh.gidx])
        iv = screen(X)
        if all(==(iv[1]), iv)
            inveps[q] = iv[1]; continue
        end
        m = _gcq_cut_m(c, dg, X; p)
        hc = _gcq_horizontal_cut(c, dg, sh.gidx, delta[q], iv)
        key = (shape_of[q], round.(m; sigdigits = 12))
        cls[q] = get!(cid, key) do
            push!(classes, hc === nothing ? GCQCutClass(shape_of[q], false, NaN, NaN, NaN, m) :
                                            GCQCutClass(shape_of[q], true, hc..., m))
            length(classes)
        end
    end
    return cls, inveps, classes
end

# sums over the nonzero vectors R of the 2D lattice (A1, A2): S3 = sum 1/|R|^3, S5 = sum 1/|R|^5 and
# T5 = sum R R' / |R|^7, over the disc |R| < Rc exactly and beyond it as integrals
function _lattice_sums(A1, A2; nr = 300)
    Ac = abs(A1[1] * A2[2] - A1[2] * A2[1])
    Rc = nr * min(norm(A1), norm(A2))
    n1 = ceil(Int, Rc * norm(A2) / Ac) + 1; n2 = ceil(Int, Rc * norm(A1) / Ac) + 1
    S3 = 0.0; S5 = 0.0; T = zeros(2, 2)
    for a in -n1:n1, b in -n2:n2
        (a == 0 && b == 0) && continue
        x = a * A1[1] + b * A2[1]; y = a * A1[2] + b * A2[2]; r = hypot(x, y)
        r < Rc || continue
        S3 += 1 / r^3; S5 += 1 / r^5; r7 = r^7
        T[1, 1] += x * x / r7; T[1, 2] += x * y / r7; T[2, 2] += y * y / r7
    end
    t5 = 2π / (3Ac * Rc^3)
    T[2, 1] = T[1, 2]; T[1, 1] += t5 / 2; T[2, 2] += t5 / 2
    return S3 + 2π / (Ac * Rc), S5 + t5, T
end

"""
    _gcq_halfspace_dz(Fd, o, ea, eb, ec, zf, side, Y; pad = 4, p = 7, nq = 8, tol = 1e-13)

∂z N[χ ρ] at the targets `Y` (3 × n, on the plane z = `zf` or on the other side of it), with ρ the density samples
`Fd[i, j, k]` at o + (i-1) ea + (j-1) eb + (k-1) ec (ea, eb horizontal; zero beyond the array) and χ the side
`side` of the plane (-1: z < zf, +1: z > zf). By the in-plane Fourier transform of the half-space Green's function,
exact in z: for a target at height dy across the plane,

    (∂z N[χ ρ])^(k) = side / 2 ∫ exp(-|k| (d' + dy)) ρ^(k, z') dz'      (d': the depth of z' into the side),

the z'-integral by the centred degree-`p` interpolants of the levels on `nq`-point Gauss-Legendre cells (the partial
cell at the plane included), ρ^ by an FFT on the horizontal lattice padded `pad` times, the periodic images removed
to second order in (extent / period), and a type-2 NUFFT to the targets.
"""
function _gcq_halfspace_dz(Fd::AbstractArray{<:Real, 3}, o, ea, eb, ec, zf, side, Y; pad = 4, p::Int = 7, nq::Int = 8,
                           tol = 1e-13)
    side in (-1, 1) || throw(ArgumentError("side must be -1 or 1"))
    out = zeros(size(Y, 2)); isempty(out) && return out
    hz = abs(ec[3]); n1, n2, n3 = size(Fd)
    dep(k) = side * (o[3] + (k - 1) * ec[3] - zf)                     # depth of level k into the side
    ks = sort(1:n3; by = dep); dd = dep.(ks)
    i1 = findfirst(>(1e-12 * hz), dd)
    i1 === nothing && return out                                       # no level on the side: no density there
    m0 = (p - 1) ÷ 2; L = n3 - i1 + 1
    kof(m) = (i = i1 + m - 1; 1 <= i <= n3 ? ks[i] : 0)                # the level of depth index m (0: none)
    d(m) = dd[i1] + (m - 1) * hz
    # the horizontal lattice, padded to hold the density's and the targets' horizontal extent `pad` times
    B = [ea[1] eb[1]; ea[2] eb[2]]; dA = abs(det(B)); R = inv(B)'      # columns of R: the dual vectors
    ξ = R' * (Y[1:2, :] .- o[1:2])                                     # targets in lattice coordinates
    sξ = R' * ec[1:2]                                                  # horizontal drift per level
    span(r, n) = (lo = min(0.0, minimum(view(ξ, r, :)), (n3 - 1) * min(0.0, sξ[r]));
                  hi = max(n - 1.0, maximum(view(ξ, r, :)), n - 1 + (n3 - 1) * max(0.0, sξ[r])); hi - lo + 1)
    M1 = nextprod([2, 3, 5], ceil(Int, pad * span(1, n1))); M2 = nextprod([2, 3, 5], ceil(Int, pad * span(2, n2)))
    mt(m, M) = m < (M + 1) ÷ 2 ? m : m - M
    K1 = [2π * (R[1, 1] * mt(a, M1) / M1 + R[1, 2] * mt(b, M2) / M2) for a in 0:M1-1, b in 0:M2-1]
    K2 = [2π * (R[2, 1] * mt(a, M1) / M1 + R[2, 2] * mt(b, M2) / M2) for a in 0:M1-1, b in 0:M2-1]
    Kn = hypot.(K1, K2)
    c0 = zeros(ComplexF64, M1, M2); buf = zeros(ComplexF64, M1, M2)
    plan = FFTW.plan_fft!(buf)
    X1 = [o[1] + (i - 1) * ea[1] + (j - 1) * eb[1] for i in 1:n1, j in 1:n2]
    X2 = [o[2] + (i - 1) * ea[2] + (j - 1) * eb[2] for i in 1:n1, j in 1:n2]
    mom = NTuple{5, Any}[]                                             # per z' node: (depth, weight, moments 0, 1, 2)
    gx, gw = gausslegendre(nq)
    cells = [(0.0, d(1), 0); [(d(m), d(m + 1), m) for m in 1:L-1]]
    for (lo, hi, c) in cells
        hi > lo || continue
        nodes = (c - m0):(c - m0 + p)
        for (t, wt) in zip(gx, gw)
            dq = (lo + hi) / 2 + (hi - lo) / 2 * t; wq = wt * (hi - lo) / 2
            fill!(buf, 0)
            for m in nodes
                kof(m) == 0 && continue
                lm = 1.0
                for q in nodes
                    q == m || (lm *= (dq - d(q)) / (d(m) - d(q)))
                end
                @views buf[1:n1, 1:n2] .+= lm .* Fd[:, :, kof(m)]
            end
            tq = (zf + side * dq - o[3]) / ec[3]                           # level coordinate: horizontal drift tq ec
            sx, sy = tq * ec[1], tq * ec[2]
            Gr = real.(view(buf, 1:n1, 1:n2))
            s0 = sum(Gr) * dA; g1 = sum(Gr .* X1) * dA; g2 = sum(Gr .* X2) * dA
            g11 = sum(Gr .* X1 .* X1) * dA; g12 = sum(Gr .* X1 .* X2) * dA; g22 = sum(Gr .* X2 .* X2) * dA
            s1 = [g1 + sx * s0, g2 + sy * s0]                              # moments of the level, drift included
            s2 = [g11 + 2sx * g1 + sx^2 * s0  g12 + sx * g2 + sy * g1 + sx * sy * s0;
                  0.0  g22 + 2sy * g2 + sy^2 * s0]; s2[2, 1] = s2[1, 2]
            push!(mom, (dq, wq, s0, s1, s2))
            plan * buf
            @. c0 += (side * wq / 2 * dA) * exp(-Kn * dq - im * (K1 * sx + K2 * sy)) * buf
        end
    end
    # the images R != 0 of the periodic sum: the kernel side (dy + d') / (4 pi r^3), r^2 = |R + ρ|^2 + (dy + d')^2,
    # ρ = y - x horizontal, expanded in 1/|R|:  side (dy + d') / (4 pi) [S3 - 3/2 S5 ((dy + d')^2 + |ρ|^2) + 15/2 ρ' T5 ρ]
    S3, S5, T5 = _lattice_sums(M1 .* ea[1:2], M2 .* eb[1:2])
    Bm = 7.5 .* T5 .- 1.5 .* S5 .* [1.0 0.0; 0.0 1.0]
    function images(y, dyy)
        acc = 0.0
        for (dq, wq, s0, s1, s2) in mom
            Δ = dyy + dq
            ρBρ = dot(y, Bm * y) * s0 - 2 * dot(y, Bm * s1) + sum(Bm .* s2)          # int over the level of ρ' Bm ρ
            acc += wq * Δ * ((S3 - 1.5 * S5 * Δ^2) * s0 + ρBρ)
        end
        return side * acc / (4π)
    end
    dy = [-side * (Y[3, j] - zf) for j in axes(Y, 2)]
    minimum(dy) >= -1e-9 * hz || throw(ArgumentError("a target is on the source side of the plane"))
    dy .= max.(dy, 0.0)
    for dg in unique(round.(dy; sigdigits = 12))
        js = findall(j -> round(dy[j]; sigdigits = 12) == dg, axes(Y, 2))
        fk = circshift(c0 .* exp.(-Kn .* dg), (M1 ÷ 2, M2 ÷ 2))     # centred modes, FINUFFT's default order
        x1 = mod.(2π .* ξ[1, js] ./ M1 .+ π, 2π) .- π; x2 = mod.(2π .* ξ[2, js] ./ M2 .+ π, 2π) .- π
        v = FINUFFT.nufft2d2(x1, x2, 1, tol, fk; nthreads = Threads.nthreads())
        out[js] .= real.(vec(v)) ./ (M1 * M2 * dA)
        Threads.@threads for t in eachindex(js)
            out[js[t]] -= images(Y[1:2, js[t]], dg)
        end
    end
    return out
end

# the shape's samples as a dense array over its lattice bounding box: (Fd, o, ea, eb, ec), rho = tw / dv
function _gcq_shape_array(dg, sh::GCQPairShape)
    o, ea, eb, ec = _grid_frame(dg); _, dv = _grid_basis(dg)
    lo = ntuple(d -> minimum(g[d] for g in sh.gidx), 3); hi = ntuple(d -> maximum(g[d] for g in sh.gidx), 3)
    Fd = zeros(hi .- lo .+ 1)
    for (g, w) in zip(sh.gidx, sh.tw)
        Fd[(g .- lo .+ 1)...] = w / dv
    end
    return Fd, o .+ (lo[1] - 1) .* ea .+ (lo[2] - 1) .* eb .+ (lo[3] - 1) .* ec, ea, eb, ec
end

# F (n . grad u_inc) of the pairs in the blocks `blk` (columns `cols` of Floc) of the targets T (the interface nodes
# shifted into the shape's frame per pair, N per block) of the cut class cl. g0 = grad phi0 of the shape at T (TKM
# near, point charges far). `method = :surface`: g at every node by the surface corrections (`surf` with the targets
# `nodes`, `atr` the trace coefficients there, `src[k]` psi and its Hessian at surf's nodes for block k); `:halfspace`: point charges with the cut-cell screening far from the shape
# and the half-space formula near it (`near`).
function _gcq_cut_rhs!(Floc, cols, blk, cl::GCQCutClass, dg, sh::GCQPairShape, sv0, T, g0, near, nrm, N; far, tol,
                       method::Symbol = :halfspace, surf = nothing, atr = nothing, nodes = nothing, src = nothing)
    if method === :surface
        for (bk, k) in enumerate(blk)
            Ψs, Hs = src[k]
            g = atr' .* g0[:, (k - 1) * N .+ (1:N)] .+ _gcq_surface_layers(surf, nodes, Ψs, Hs; tol = max(tol, 1e-14),
                                                                         far = far(max(tol, 1e-14)))
            for j in 1:N
                Floc[j, cols[bk]] = nrm[1, j] * g[1, j] + nrm[2, j] * g[2, j] + nrm[3, j] * g[3, j]
            end
        end
        return
    end
    fi = Int[]; ni = Int[]
    for k in blk, j in 1:N
        i = (k - 1) * N + j
        push!(near[i] ? ni : fi, i)
    end
    nodej(i) = mod1(i, N); colof = Dict(zip(blk, cols)); col(i) = colof[cld(i, N)]
    if !isempty(fi)                                    # far: point charges with the cut-cell screening
        g = point_charge_field(far(tol), sv0.positions, sv0.weights .* sv0.density .* cl.m, T[:, fi]; grad = true)
        for (n, i) in enumerate(fi)
            j = nodej(i)
            Floc[j, col(i)] = nrm[1, j] * g[1, n] + nrm[2, j] * g[2, n] + nrm[3, j] * g[3, n]
        end
    end
    isempty(ni) && return
    cl.plane && all(i -> abs(abs(nrm[3, nodej(i)]) - 1) < 1e-12, ni) ||
        error("GCQ campaign mode: the half-space formula needs one horizontal face and the nodes near the density " *
              "on horizontal faces (use cut_rhs = :surface)")
    Fd, o, ea, eb, ec = _gcq_shape_array(dg, sh)
    up = [i for i in ni if T[3, i] >= cl.zf]; dn = [i for i in ni if T[3, i] < cl.zf]
    # above the face: u = a_hi phi0 + (a_lo - a_hi) N[chi_lo rho]; below: u = a_lo phi0 + (a_hi - a_lo) N[chi_hi rho]
    for (is, side, a0, a1) in ((up, -1, cl.a_hi, cl.a_lo), (dn, 1, cl.a_lo, cl.a_hi))
        isempty(is) && continue
        h = _gcq_halfspace_dz(Fd, o, ea, eb, ec, cl.zf, side, T[:, is])
        for (n, i) in enumerate(is)
            j = nodej(i)
            Floc[j, col(i)] = nrm[3, j] * (a0 * g0[3, i] + (a1 - a0) * h[n])
        end
    end
    return
end
