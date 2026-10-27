# GCQ campaign mode, near/far form of V (`vform = :nearfar`): no P × P array is formed.
#
#   V = sym(V_vol + V_layer),   sym(M) = (M + M') / 2.
#
# Tiles: the pairs are grouped by the centres of their supports into cubes of side `tile`. Every pair density lies in
# the proxy box of its tile, half-widths w about the tile centre (the same w for every tile). Tiles A and B are near
# when their proxy boxes are closer than `rnear` or touch (a function of the tile offset only).
#
# V_vol, near tile pairs: exact, by the shapes' volume solves of gcq_four_index restricted to the near keys (a region of
#   constant size about each source shape, instead of every placement of every pair).
# V_vol, far tile pairs: the moment far field of PDMK-ISDF (isdf/local_vnear.jl):
#     V_vol[a, b] ≈ M_a' C_k Mε_b,   C_k = Pm' T_k Pm,   k = t_B - t_A,
#   M_a the Cartesian moments (total degree ≤ p, in units of `hm`) of w ρ_a about its tile centre, Mε_b those of the
#   screened density (ρ_b / ε_b, or ρ_b m for a pair cut by faces), Pm = pinv of the monomials at the (p + 1)^3
#   Chebyshev proxies of the box (min-norm equivalent charges e = Pm M), T_k = 1 / (4π |y_i - y_j - k tile|) between the
#   proxies. The moments of a pair are those of its shape (computed once) shifted to the tile centre; one core per far
#   tile offset.
# V_layer, `layer = :lowrank`: the low-rank factors of gcq_four_index, V_layer = (G' Y) R, kept as Gy = G' Y (P × r) and
#   R (r × P); near entries added to the stored near part.
# V_layer, `layer = :translate`: the induced response taken invariant under in-plane translations (exact for boxes that
#   are infinite in plane; an approximation near the box edges):
#   - near entries: one solve per group of translates (shape, cut class, vertical grid step) at the group's reference
#     placement near the middle of the lattice; its induced potential S σ_g at the points of every near pair shifted
#     there: V_layer[a, b] = <ρ_a shifted by (ref_g - δ_b), S σ_g>;
#   - far entries: V_layer[a, b] ≈ M_a' L_(k, z_B) Mε_b, L = Pm' Φ Pm, Φ[i, j] the induced potential at proxy i of the
#     tile at c0 - k tile from a unit charge at proxy j of the central tile c0 of the level z_B: the response of the
#     equivalent charges Pm of the n_mom moments, of numerical rank r_p ≤ n_mom (tolerance `tau`), r_p solves.
#   The number of solves does not grow with the lattice. The error decays only like a power of the distance from the
#   box edges (relative Frobenius 1.4e-2 at 3 Å, 1.6e-4 at 24 Å on a 6 × 6 blob lattice).
#
# Storage O(P): near entries (P times the pairs in the near tiles), 2 moment vectors per pair, the cores, the factors.
# Time: the volume part is linear; the layer part is linear with `layer = :translate`. With `layer = :lowrank`, F and G
# (interface nodes × pairs), the range of W^{1/2} F (eigen of the P × P Gram matrix) and its r ∝ (orbitals) solves are not.

"""
    GCQNearFarV

The four-index matrix of [`gcq_four_index`](@ref) with `vform = :nearfar`: exact near entries, a moment far field for
the volume part, and for the layer part its low-rank factors (`layer = :lowrank`) or translation-invariant far cores
(`layer = :translate`). `V[a, b]` gives one entry, `Matrix(V)` the dense matrix (for checks on small campaigns),
`size(V)` is `(P, P)`.

Fields: `pair_ids`; the tiles (`tile` side, `origin`, `tiles` integer indices, `tile_of` per pair, `members` per tile),
the proxy half-widths `w`, `rnear`, `p`, `hm`; `near` (sparse P × P, sym(V_vol + V_layer) on the pairs of near
tiles); `mom`, `momε` (n_mom × P); `core` (far tile offset `k` => C_k, kept for one of ±k; C_{-k} = C_k'); `layer`
(`:lowrank` or `:translate`); `Gy` (P × r), `R` (r × P) of the low-rank layer part (empty otherwise); `lcore`
((k..., z_B) => L of the translated layer part, empty otherwise).
"""
struct GCQNearFarV
    pair_ids::Vector{Tuple{Int, Int}}
    tile::Float64
    origin::NTuple{3, Float64}
    tiles::Vector{NTuple{3, Int}}
    tile_of::Vector{Int}
    members::Vector{Vector{Int}}
    w::NTuple{3, Float64}
    rnear::Float64
    p::Int
    hm::Float64
    near::SparseMatrixCSC{Float64, Int}
    mom::Matrix{Float64}
    momε::Matrix{Float64}
    core::Dict{NTuple{3, Int}, Matrix{Float64}}
    layer::Symbol
    Gy::Matrix{Float64}
    R::Matrix{Float64}
    lcore::Dict{NTuple{4, Int}, Matrix{Float64}}
end

Base.size(V::GCQNearFarV) = (n = length(V.pair_ids); (n, n))
Base.size(V::GCQNearFarV, d::Integer) = size(V)[d]

# near: proxy boxes closer than rnear, or overlapping or touching (gap 0) whatever rnear
_nf_isnear(k, tile, w, rnear) = (g = norm(ntuple(d -> max(abs(k[d]) * tile - 2w[d], 0.0), 3)); g == 0 || g < rnear)
# C_k for any far offset k (stored for one of ±k)
_nf_core(V::GCQNearFarV, k) = haskey(V.core, k) ? V.core[k] : transpose(V.core[.-k])
_nf_lkey(V::GCQNearFarV, A, B) = (k = V.tiles[B] .- V.tiles[A]; (k..., V.tiles[B][3]))

function Base.getindex(V::GCQNearFarV, a::Integer, b::Integer)
    A, B = V.tile_of[a], V.tile_of[b]
    k = V.tiles[B] .- V.tiles[A]
    _nf_isnear(k, V.tile, V.w, V.rnear) && return V.near[a, b]
    C = _nf_core(V, k)
    v = (dot(view(V.mom, :, a), C, view(V.momε, :, b)) + dot(view(V.momε, :, a), C, view(V.mom, :, b))) / 2
    if V.layer === :lowrank
        v += (dot(view(V.Gy, a, :), view(V.R, :, b)) + dot(view(V.Gy, b, :), view(V.R, :, a))) / 2
    else
        v += (dot(view(V.mom, :, a), V.lcore[_nf_lkey(V, A, B)], view(V.momε, :, b)) +
              dot(view(V.mom, :, b), V.lcore[_nf_lkey(V, B, A)], view(V.momε, :, a))) / 2
    end
    return v
end

"""
    Matrix(V::GCQNearFarV)

The dense P × P matrix (far tile blocks formed from the moments): for checks on campaigns small enough to hold it.
"""
function Base.Matrix(V::GCQNearFarV)
    M = Matrix(V.near)
    L = V.layer === :lowrank ? V.Gy * V.R : nothing
    for (A, ma) in enumerate(V.members), (B, mb) in enumerate(V.members)
        k = V.tiles[B] .- V.tiles[A]
        _nf_isnear(k, V.tile, V.w, V.rnear) && continue
        C = _nf_core(V, k)
        blk = (V.mom[:, ma]' * C * V.momε[:, mb] .+ V.momε[:, ma]' * C * V.mom[:, mb]) ./ 2
        if L === nothing
            blk .+= (V.mom[:, ma]' * V.lcore[_nf_lkey(V, A, B)] * V.momε[:, mb] .+
                     (V.mom[:, mb]' * V.lcore[_nf_lkey(V, B, A)] * V.momε[:, ma])') ./ 2
        else
            blk .+= (L[ma, mb] .+ L[mb, ma]') ./ 2
        end
        M[ma, mb] .= blk
    end
    return M
end

"""
    nearfar_bytes(V::GCQNearFarV)

Storage of the parts of `V` in bytes: `(; near, moments, cores, layer)`.
"""
nearfar_bytes(V::GCQNearFarV) = (near = 16.0 * nnz(V.near), moments = 8.0 * (length(V.mom) + length(V.momε)),
                                 cores = 8.0 * sum(length, values(V.core); init = 0),
                                 layer = 8.0 * (length(V.Gy) + length(V.R) + sum(length, values(V.lcore); init = 0)))

# Cartesian exponents of total degree ≤ p, by degree
_nf_exponents(p::Int) = [(a, b, d - a - b) for d in 0:p for a in d:-1:0 for b in (d - a):-1:0]

# moments sum_n v_n ((x_n - c) / hm)^e, for the exponents `exps` (one vector)
function _nf_moments(X, v, c, hm, exps)
    p = maximum(sum, exps; init = 0)
    nt = Threads.maxthreadid()
    acc = [zeros(length(exps)) for _ in 1:nt]
    chunks = collect(Iterators.partition(axes(X, 2), max(1, cld(size(X, 2), 4nt))))
    Threads.@threads :static for ch in chunks
        m = acc[Threads.threadid()]; pw = zeros(p + 1, 3)
        @inbounds for n in ch
            for d in 1:3
                u = (X[d, n] - c[d]) / hm; pw[1, d] = 1.0
                for o in 1:p
                    pw[o + 1, d] = pw[o, d] * u
                end
            end
            for (i, e) in enumerate(exps)
                m[i] += v[n] * pw[e[1] + 1, 1] * pw[e[2] + 1, 2] * pw[e[3] + 1, 3]
            end
        end
    end
    return reduce(+, acc)
end

# moments about c - δ hm from moments about c: (u + δ)^α = sum_{β ≤ α} binom(α, β) δ^(α - β) u^β
function _nf_shift_matrix(δ, exps)
    idx = Dict(e => i for (i, e) in enumerate(exps))
    S = zeros(length(exps), length(exps))
    for (i, a) in enumerate(exps), b1 in 0:a[1], b2 in 0:a[2], b3 in 0:a[3]
        S[i, idx[(b1, b2, b3)]] = binomial(a[1], b1) * binomial(a[2], b2) * binomial(a[3], b3) *
                                  δ[1]^(a[1] - b1) * δ[2]^(a[2] - b2) * δ[3]^(a[3] - b3)
    end
    return S
end

# (p + 1)^3 Chebyshev proxies of the box with half-widths w about 0, and Pm = pinv(monomials at the proxies)
function _nf_proxies(p, w, hm, exps)
    n = p + 1
    ch(h) = [h * cos(π * (2i - 1) / (2n)) for i in 1:n]
    cx, cy, cz = ch(w[1]), ch(w[2]), ch(w[3])
    px = reduce(hcat, [[x, y, z] for x in cx for y in cy for z in cz])
    Mono = [prod((px[d, j] / hm)^e[d] for d in 1:3) for e in exps, j in axes(px, 2)]
    return px, pinv(Mono)
end

# the tile offsets t_B - t_A: every offset within the bounding box of the tiles when there are fewer of those than tile
# pairs (exact for a full lattice, O(tiles)), else the offsets of all tile pairs
function _nf_offsets(tiles)
    lo = ntuple(d -> minimum(t -> t[d], tiles), 3); hi = ntuple(d -> maximum(t -> t[d], tiles), 3)
    prod(2 .* (hi .- lo) .+ 1) <= length(tiles)^2 &&
        return vec(collect(Iterators.product((lo[d] - hi[d]:hi[d] - lo[d] for d in 1:3)...)))
    return unique(B .- A for A in tiles for B in tiles)
end

# C_k = Pm' T_k Pm, T_k[i, j] = 1 / (4π |y_i - y_j - o|)
function _nf_core_matrix(px, Pm, o)
    n = size(px, 2); T = zeros(n, n)
    Threads.@threads for j in 1:n
        @inbounds for i in 1:n
            T[i, j] = 1 / (4π * sqrt((px[1, i] - px[1, j] - o[1])^2 + (px[2, i] - px[2, j] - o[2])^2 +
                                     (px[3, i] - px[3, j] - o[3])^2))
        end
    end
    return Pm' * T * Pm
end

"""
    _gcq_nearfar_tiles(dg, shapes, shape_of, delta; tile, rnear) -> NamedTuple

The tiles of the pairs (by the centres of their supports' bounding boxes), the shape boxes, the proxy half-widths w
(the largest extent of a pair's support from its tile centre, per axis), the near pairs (a, b) of near tiles (both
orders), and the far tile offsets (one of ±k).
`tile = nothing`: the largest extent of a shape's support along x or y; `rnear = nothing`: the largest half-width.
"""
function _gcq_nearfar_tiles(dg, shapes, shape_of, delta; tile = nothing, rnear = nothing)
    P = length(shape_of)
    lo = Vector{Vector{Float64}}(undef, length(shapes)); hi = similar(lo)
    for (s, sh) in enumerate(shapes)
        X = _grid_positions(dg, sh.gidx)
        lo[s] = isempty(sh.gidx) ? zeros(3) : vec(minimum(X; dims = 2))
        hi[s] = isempty(sh.gidx) ? zeros(3) : vec(maximum(X; dims = 2))
    end
    shift = [_grid_shift(dg, delta[a]) for a in 1:P]
    ctr = [(lo[shape_of[a]] .+ hi[shape_of[a]]) ./ 2 .+ shift[a] for a in 1:P]
    h = tile === nothing ? maximum(s -> max(hi[s][1] - lo[s][1], hi[s][2] - lo[s][2]), eachindex(shapes)) : Float64(tile)
    h > 0 || throw(ArgumentError("tile must be positive"))
    org = ntuple(d -> minimum(c[d] for c in ctr), 3)
    tid = Dict{NTuple{3, Int}, Int}(); tiles = NTuple{3, Int}[]; members = Vector{Int}[]
    tile_of = map(1:P) do a
        t = ntuple(d -> floor(Int, (ctr[a][d] - org[d]) / h), 3)
        i = get!(tid, t) do
            push!(tiles, t); push!(members, Int[]); length(tiles)
        end
        push!(members[i], a); i
    end
    tc(i) = ntuple(d -> org[d] + (tiles[i][d] + 0.5) * h, 3)
    w = ntuple(3) do d
        m = h / 2
        for a in 1:P
            isempty(shapes[shape_of[a]].gidx) && continue
            t = tc(tile_of[a])[d]
            m = max(m, abs(lo[shape_of[a]][d] + shift[a][d] - t), abs(hi[shape_of[a]][d] + shift[a][d] - t))
        end
        m * (1 + 1e-12)
    end
    rn = rnear === nothing ? maximum(w) : Float64(rnear)
    # near offsets: |k_d| tile < 2 w_d + rnear in each direction is necessary
    kmax = ntuple(d -> ceil(Int, (2w[d] + rn) / h), 3)
    nearoffs = [k for k in Iterators.product((-m:m for m in kmax)...) if _nf_isnear(k, h, w, rn)]
    ab = NTuple{2, Int}[]
    for (A, t) in enumerate(tiles), k in nearoffs
        B = get(tid, t .+ k, 0)
        B == 0 && continue
        for a in members[A], b in members[B]
            push!(ab, (a, b))
        end
    end
    faroffs = Set{NTuple{3, Int}}()
    for k in _nf_offsets(tiles)
        (_nf_isnear(k, h, w, rn) || .-k in faroffs) || push!(faroffs, k)
    end
    return (; tile = h, origin = org, tiles, tile_of, members, centres = [tc(i) for i in eachindex(tiles)], w,
              rnear = rn, ab, faroffs = collect(faroffs), shift, shape_lo = lo, shape_hi = hi)
end

"""
    _gcq_nearfar_moments(dg, shapes, shape_of, tl, cls, inveps, classes; p, hm) -> (mom, momε, exps)

The moments (n_mom × P) of every pair about its tile centre, bare and screened (ρ/ε inside one region, ρ m for a cut
pair), from the moments of each shape (and cut class) about the centre of its box, shifted.
"""
function _gcq_nearfar_moments(dg, shapes, shape_of, tl, cls, inveps, classes; p, hm)
    exps = _nf_exponents(p); P = length(shape_of)
    cs = [(tl.shape_lo[s] .+ tl.shape_hi[s]) ./ 2 for s in eachindex(shapes)]
    ms = [isempty(sh.gidx) ? zeros(length(exps)) : _nf_moments(_grid_positions(dg, sh.gidx), sh.tw, cs[s], hm, exps)
          for (s, sh) in enumerate(shapes)]
    mc = [_nf_moments(_grid_positions(dg, shapes[cl.shape].gidx), shapes[cl.shape].tw .* cl.m, cs[cl.shape], hm, exps)
          for cl in classes]
    mom = zeros(length(exps), P); momε = zeros(length(exps), P)
    Threads.@threads for a in 1:P
        s = shape_of[a]
        isempty(shapes[s].gidx) && continue
        S = _nf_shift_matrix((cs[s] .+ tl.shift[a] .- tl.centres[tl.tile_of[a]]) ./ hm, exps)
        mom[:, a] .= S * ms[s]
        momε[:, a] .= cls[a] == 0 ? inveps[a] .* mom[:, a] : S * mc[cls[a]]
    end
    return mom, momε, exps
end

"""
    _gcq_translate_groups(dg, shapes, shape_of, delta, cls) -> (; group_of, members, ref)

The groups of translates of `layer = :translate`: pairs of one shape, cut class and vertical grid step (non-empty
densities only; `group_of` is 0 for the others), each with the reference pair `ref[g]` nearest the mean placement of
its group. The first two grid vectors must be horizontal (the translations are in plane).
"""
function _gcq_translate_groups(dg, shapes, shape_of, delta, cls)
    _, ea, eb, _ = _grid_frame(dg)
    abs(ea[3]) + abs(eb[3]) <= 1e-12 * (norm(ea) + norm(eb)) ||
        throw(ArgumentError("layer = :translate needs a grid whose first two vectors are horizontal"))
    gid = Dict{NTuple{3, Int}, Int}(); members = Vector{Int}[]
    group_of = zeros(Int, length(shape_of))
    for a in eachindex(shape_of)
        isempty(shapes[shape_of[a]].gidx) && continue
        group_of[a] = get!(gid, (shape_of[a], cls[a], delta[a][3])) do
            push!(members, Int[]); length(members)
        end
        push!(members[group_of[a]], a)
    end
    ref = map(members) do ms
        m = sum(q -> _grid_shift(dg, delta[q]), ms) ./ length(ms)
        ms[argmin([norm(_grid_shift(dg, delta[q]) .- m) for q in ms])]
    end
    return (; group_of, members, ref)
end

# translated layer part, near entries: for (a, b) of `ab`, <ρ_a shifted by ref - δ_b, S σ_g> (g the group of b, σ_g its
# reference solve), by key (g, shape of a, δ_a - δ_b) on the union of the shifted points
function _gcq_layer_translate_near(lay, shapes, shape_of, dg, ab)
    lt, delta = lay.lt, lay.delta
    kid = Dict{Tuple{Int, Int, NTuple{3, Int}}, Int}(); ki = zeros(Int, length(ab))
    for (i, (a, b)) in enumerate(ab)
        g = lt.group_of[b]
        (g == 0 || isempty(shapes[shape_of[a]].gidx)) && continue
        ki[i] = get!(kid, (g, shape_of[a], delta[a] .- delta[b]), length(kid) + 1)
    end
    kl = collect(kid)
    pid = Dict{NTuple{3, Int}, Int}()
    for ((g, sa, D), _) in kl, h in shapes[sa].gidx
        get!(pid, h .+ delta[lt.ref[g]] .+ D, length(pid) + 1)
    end
    pts = Vector{NTuple{3, Int}}(undef, length(pid))
    for (h, i) in pid
        pts[i] = h
    end
    Pop = potential_operator(lay.ev, lay.itf, _grid_positions(dg, pts))
    Φ = reduce(hcat, [Pop * lay.Σ[:, g] for g in axes(lay.Σ, 2)]; init = zeros(length(pts), 0))
    vals = zeros(length(kl))
    Threads.@threads for q in eachindex(kl)
        (g, sa, D), id = kl[q]
        sh = shapes[sa]; o = delta[lt.ref[g]] .+ D; acc = 0.0
        @inbounds for (h, t) in zip(sh.gidx, sh.tw)
            acc += t * Φ[pid[h .+ o], g]
        end
        vals[id] = acc
    end
    return [ki[i] == 0 ? 0.0 : vals[ki[i]] for i in eachindex(ab)], length(pts)
end

# translated layer part, far cores L_(k, z) = Pm' Φ Pm for every far tile pair (both orders), Φ[i, j] the induced
# potential at proxy i of the tile c0 - k tile from a unit charge at proxy j of the central tile c0 of level z; the
# responses by the range of W^{1/2} F Pm (the fields of the moments' equivalent charges) at tolerance tau
function _gcq_layer_translate_far(lay, tl, px, Pm)
    itf, A = lay.itf, lay.A
    nodes, nrm = _interface_targets_normals(itf); w = all_weights(itf); d = sqrt.(abs.(w)); N = size(nodes, 2)
    zs = Set(t[3] for t in tl.tiles)
    keys = [(k..., zb) for k in _nf_offsets(tl.tiles) if !_nf_isnear(k, tl.tile, tl.w, tl.rnear)
            for zb in zs if zb - k[3] in zs]
    lcore = Dict{NTuple{4, Int}, Matrix{Float64}}(); nsolve = 0; npx = size(px, 2)
    for zl in unique(t[3] for t in tl.tiles)
        kz = [key for key in keys if key[4] == zl]
        isempty(kz) && continue
        lev = [t for t in tl.tiles if t[3] == zl]
        c0 = [tl.origin[d_] + (round(Int, sum(t[d_] for t in lev) / length(lev)) + 0.5) * tl.tile for d_ in 1:2]
        Ys = px .+ [c0; tl.origin[3] + (zl + 0.5) * tl.tile]
        # F of the unit proxy charges: n . grad (1 / 4π r) at the nodes
        F = zeros(N, npx)
        Threads.@threads for j in 1:npx
            @inbounds for i in 1:N
                r1 = nodes[1, i] - Ys[1, j]; r2 = nodes[2, i] - Ys[2, j]; r3 = nodes[3, i] - Ys[3, j]
                F[i, j] = -(nrm[1, i] * r1 + nrm[2, i] * r2 + nrm[3, i] * r3) / (4π * sqrt(r1^2 + r2^2 + r3^2)^3)
            end
        end
        local Q, Rq, Y
        _gcq_dense() do
            B = (d .* F) * Pm       # only the equivalent charges of the moments are ever needed: n_mom columns
            E = eigen(Symmetric(B' * B)); ord = sortperm(E.values; rev = true)
            sv = sqrt.(max.(E.values[ord], 0.0)); r = max(count(>(lay.tau * sv[1]), sv), 1)
            Q = B * (E.vectors[:, ord[1:r]] ./ sv[1:r]'); Rq = Q' * B
        end
        Y = first(solve_dielectric_box3d_gcq_block(A, Q ./ d, itf; rtol = lay.gmres_rtol, itmax = 500,
                                                   block = lay.gmres_block))
        nsolve += size(Q, 2)
        Z = reduce(hcat, [Ys .- collect(key[1:3]) .* tl.tile for key in kz])
        Pop = potential_operator(lay.ev, itf, Z)
        Φ = reduce(hcat, [Pop * Y[:, j] for j in axes(Y, 2)]) * Rq           # (npx |kz|) × n_mom
        for (m, key) in enumerate(kz)
            lcore[key] = Pm' * Φ[(m - 1) * npx .+ (1:npx), :]
        end
    end
    return lcore, nsolve
end

# part 1 of gcq_four_index with vform = :nearfar: near entries from the volume keys of the near pairs tl.ab and the
# layer part (factors, or translated), moments, far cores
function _gcq_assemble_nearfar(pids, dg, shapes, shape_of, tl, cls, inveps, classes, vals, ksrc, kidx, kfac, G, Y, R;
                               p, stats, st, T0, r, N, npanels, ntargets, skin, nnear, lay = nothing)
    P = length(pids)
    local Gy, near, asym, mom, momε, core
    lcore = Dict{NTuple{4, Int}, Matrix{Float64}}(); npts = 0; nsolve = 0
    t = @elapsed begin
        vv = [vals[ksrc[i]][kidx[i]] * kfac[i] for i in eachindex(tl.ab)]
        if lay === nothing
            Gy = _gcq_dense(() -> G' * Y)
            ly = zeros(length(tl.ab))
            Threads.@threads for i in eachindex(tl.ab)
                a, b = tl.ab[i]
                ly[i] = dot(view(Gy, a, :), view(R, :, b))
            end
        else
            Gy = zeros(P, 0); R = zeros(0, P)
            ly, npts = _gcq_layer_translate_near(lay, shapes, shape_of, dg, tl.ab)
        end
        Vn = sparse(first.(tl.ab), last.(tl.ab), vv .+ ly, P, P)
        asym = maximum(abs, Vn - Vn'; init = 0.0) / max(maximum(abs, Vn; init = 0.0), floatmin())
        near = (Vn + Vn') ./ 2
    end
    st("near entries ($(length(tl.ab))" * (lay === nothing ? "" : ", layer part at $npts translated points") *
       "; max |V - V'| / max |V| = $(round(asym; sigdigits = 2)) on them)", t)
    local px, Pm
    t = @elapsed begin
        mom, momε, exps = _gcq_nearfar_moments(dg, shapes, shape_of, tl, cls, inveps, classes; p, hm = tl.tile)
        px, Pm = _nf_proxies(p, tl.w, tl.tile, exps)
        core = Dict(k => _nf_core_matrix(px, Pm, k .* tl.tile) for k in tl.faroffs)
    end
    st("moments (degree $p) and $(length(core)) far cores", t)
    if lay !== nothing
        t = @elapsed ((lcore, nsolve) = _gcq_layer_translate_far(lay, tl, px, Pm))
        st("layer far cores: $(length(lcore)) offsets, $nsolve proxy solves", t)
    end
    V = GCQNearFarV(pids, tl.tile, tl.origin, tl.tiles, tl.tile_of, tl.members, tl.w, tl.rnear, p, tl.tile, near,
                    mom, momε, core, lay === nothing ? :lowrank : :translate, Gy, Matrix(R), lcore)
    st("assembled near/far V", time() - T0)
    onsite_norm2 = Dict(pids[a][1] => sum(shapes[shape_of[a]].tw) for a in 1:P if pids[a][1] == pids[a][2])
    return (; pair_ids = pids, V,
              stats = (; times = stats, total = time() - T0, rank = r, N, npanels, nshapes = length(shapes), ntargets,
                       target_skin = skin, near_pairs = nnear, asym, onsite_norm2, cut_pairs = count(>(0), cls),
                       cut_classes = length(classes), ntiles = length(tl.tiles), nnear_entries = length(tl.ab),
                       nfar_offsets = length(core), proxy_solves = nsolve, bytes = nearfar_bytes(V)))
end
