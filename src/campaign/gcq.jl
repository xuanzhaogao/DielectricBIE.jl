# GCQ campaign mode: the whole four-index matrix of a campaign on ONE GCQ interface.
#
#   V[a, b] = <rho_a, Phi_b>,   Phi_b = Phi_vol_b + S sigma_b,   A sigma_b = F_b,
#   V = V_vol + V_layer,        V_layer = G' A^{-1} F,   G[j, a] = w_j phi_a(y_j)   (symmetry of the single layer)
#
# 1. Shapes: on a lattice every pair density is a grid translate of a few shapes (template_i, template_j,
#    step_j - step_i), each truncated on its own support (support_rtol). The targets are the union of all supports.
# 2. One GCQ interface, refined for the envelope of all pair densities (RHS) and for all targets (target_skin), and its
#    operator A. The interface is far from every target, so the layer part is the node rule: G' sigma.
# 3. F and G of every pair by one point-charge FMM per shape over the interface nodes shifted by each pair's offset.
# 4. Low rank: W^{1/2} F ~= Q (Q' W^{1/2} F) at tolerance tau (rank r), Y = A^{-1} W^{-1/2} Q (r GMRES solves),
#    V_layer = (G' Y)(Q' W^{1/2} F).
# 5. V_vol by shapes: one screened volume solve per source shape (TKM near the shape, FMM elsewhere) at the grid points
#    needed, and one dot product per distinct (shape, shape, shift); symmetric, so each pair of shapes is done once.
#
# The mode needs every pair density inside one dielectric box (the same screening for all, which makes the shapes
# exact translates and V_vol symmetric); it checks this. The work can be split over `nparts` cooperating processes
# (e.g. one or two per socket; OpenMP above one socket slows the FMMs), coordinated through files in `sync`.

"""
    GCQPairShape

A pair-density shape: representative pair `rep`, the grid steps `anchor` of its first orbital, its support `gidx`
(global grid indices, at the representative placement) and `tw` = w .* rho on it.
"""
struct GCQPairShape
    rep::Tuple{Int, Int}
    anchor::NTuple{3, Int}
    gidx::Vector{NTuple{3, Int}}
    tw::Vector{Float64}
end

# grid_point(dg, i, j, k) = o + (i-1) ea + (j-1) eb + (k-1) ec
function _grid_frame(dg)
    At, Bt, Ct = true_cell_vectors(dg)
    return collect(Float64, Tuple(dg.origin)), collect(At) ./ dg.nx, collect(Bt) ./ dg.ny, collect(Ct) ./ dg.nz
end
_grid_shift(dg, δ) = (f = _grid_frame(dg); δ[1] .* f[2] .+ δ[2] .* f[3] .+ δ[3] .* f[4])
function _grid_basis(dg)
    _, ea, eb, ec = _grid_frame(dg)
    lb = (Tuple(ea), Tuple(eb), Tuple(ec))
    return lb, abs(det(hcat(ea, eb, ec)))
end
_grid_positions(dg, gidx) = (X = Matrix{Float64}(undef, 3, length(gidx));
    for (k, g) in enumerate(gidx); X[:, k] .= grid_point(dg, g[1], g[2], g[3]); end; X)

"""
    gcq_pair_shapes(c, centers, pairs; support_rtol, support = :global) -> (shapes, shape_of, delta, dg)

The shapes of `pairs`: pair `p` is `shapes[shape_of[p]]` shifted by `delta[p]` grid steps. Each pair density is kept
where `|rho| >= support_rtol * m`, with `m` the largest |rho| of all pairs (`support = :global`, the default: one
absolute threshold, as the batch pipeline's envelope truncation) or of the pair itself (`support = :pair`, which keeps
large regions of negligible density for weakly overlapping pairs).
"""
function gcq_pair_shapes(c::CampaignInput, centers::Vector{CenterInfo}, pairs;
                         support_rtol = get(c.solve, "support_rtol", 1e-6), support::Symbol = :global)
    support in (:global, :pair) || throw(ArgumentError("support must be :global or :pair"))
    byid = Dict(ct.id => ct for ct in centers)
    grids = [t[2] for t in load_templates!(c)]
    sid = Dict{Any, Int}(); shapes = GCQPairShape[]
    shape_of = zeros(Int, length(pairs)); delta = Vector{NTuple{3, Int}}(undef, length(pairs))
    for (p, (i, j)) in enumerate(pairs)
        key = (byid[i].template_id, byid[j].template_id, byid[j].steps .- byid[i].steps)
        s = get!(sid, key) do
            insts = Dict(o => OrbitalInstance(o, byid[o].template_id, byid[o].steps) for o in unique([i, j]))
            # the per-pair cut keeps every point the global one keeps (a pair's maximum is at most the global one)
            b = assemble_lattice_batch(grids, insts, [(i, j)]; support_rtol)
            push!(shapes, GCQPairShape((i, j), byid[i].steps, b.gidx, b.weights .* b.densities[:, 1]))
            length(shapes)
        end
        shape_of[p] = s; delta[p] = byid[i].steps .- shapes[s].anchor
    end
    if support === :global && support_rtol > 0          # one absolute threshold (tw = w rho, w uniform)
        m = maximum(sh -> maximum(abs, sh.tw; init = 0.0), shapes)
        shapes = map(shapes) do sh
            keep = findall(t -> abs(t) >= support_rtol * m, sh.tw)
            GCQPairShape(sh.rep, sh.anchor, sh.gidx[keep], sh.tw[keep])
        end
    end
    return shapes, shape_of, delta, grids[1]
end

# the dielectric box holding every pair density (each support's bounding box inside it), or an error
function _gcq_uniform_box(c::CampaignInput, dg, shapes, shape_of, delta)
    o, ea, eb, ec = _grid_frame(dg)
    inbox(b, x) = all(abs(x[d] - b.center[d]) <= (b.Lx, b.Ly, b.Lz)[d] / 2 for d in 1:3)
    which = nothing
    for p in eachindex(shape_of)
        sh = shapes[shape_of[p]]
        isempty(sh.gidx) && continue                     # a pair whose orbitals do not overlap: zero density
        lo = ntuple(d -> minimum(g[d] for g in sh.gidx) + delta[p][d], 3)
        hi = ntuple(d -> maximum(g[d] for g in sh.gidx) + delta[p][d], 3)
        corners = [o .+ (i - 1) .* ea .+ (j - 1) .* eb .+ (k - 1) .* ec for i in (lo[1], hi[1]), j in (lo[2], hi[2]), k in (lo[3], hi[3])]
        bs = [b for b in eachindex(c.boxes) if all(x -> inbox(c.boxes[b], x), corners)]
        length(bs) == 1 && (which === nothing || which == bs[1]) ||
            error("GCQ campaign mode: pair $p is not inside one dielectric box with all the others; " *
                  "the shapes need the same screening for every pair")
        which = bs[1]
    end
    return which
end

# screened VolumeSource of a shape shifted by delta (grid steps)
function _gcq_shape_source(c::CampaignInput, dg, sh::GCQPairShape, delta = (0, 0, 0))
    lb, dv = _grid_basis(dg)
    X = _grid_positions(dg, [g .+ delta for g in sh.gidx])
    vs = VolumeSource(X, fill(dv, length(sh.tw)), sh.tw ./ dv; lattice_basis = lb)
    return screened_volume_source(c.boxes, c.epses, c.eps_out, vs, SharpScreening())
end

# all targets (the union of the shifted supports, as grid points) and the envelope max_p |tw_p| on them
function _gcq_targets(shapes, shape_of, delta, dg)
    env = Dict{NTuple{3, Int}, Float64}()
    for p in eachindex(shape_of)
        sh = shapes[shape_of[p]]
        for (g, w) in zip(sh.gidx, sh.tw)
            k = g .+ delta[p]; env[k] = max(get(env, k, 0.0), abs(w))
        end
    end
    gidx = sort!(collect(keys(env)))
    return _grid_positions(dg, gidx), [env[g] for g in gidx]
end

"""
    gcq_campaign_interface(c, X, env, dg; nlev, n_quad, rhs_tol, edge_tol, quad_tol, target_skin, far)

The GCQ interface of the campaign's boxes, RHS-refined for the envelope density `env` at the points `X` and refined
for the targets `X` (`target_skin`); `far(tol)` is the point-charge far field of the refinement.
"""
function gcq_campaign_interface(c::CampaignInput, X, env, dg; nlev, n_quad, rhs_tol, edge_tol, quad_tol, target_skin, far)
    lb, dv = _grid_basis(dg)
    envs = VolumeSource(X, fill(dv, length(env)), env ./ dv; lattice_basis = lb)
    senv = screened_volume_source(c.boxes, c.epses, c.eps_out, envs, SharpScreening())
    srcpos, srcq = _volume_source_fmm_sources(senv)
    kmax = _estimate_tkm3dc_kmax(senv)
    rhs_batch = (Pts, Nrm) -> (T = reduce(hcat, collect.(Pts)); N = reduce(hcat, collect.(Nrm));
        first(_rhs_volume_targets_hybrid(srcpos, srcq, T, N, 1.0, rhs_tol / 10, kmax,
                                         _classify_near_far_targets(T, senv); far = far(rhs_tol / 10))))
    mesh = multi_dielectric_box3d_gcq(c.boxes, c.epses, c.eps_out; nlev, n_quad, tol = edge_tol,
        rhs_batch, rhs_atol = rhs_tol, max_depth = Int(get(c.solve, "max_depth", 30)),
        targets = X, target_tol = quad_tol, target_eta = 2.0, target_far = :direction, target_skin)
    return mesh.interface
end

# the target_skin default: 3 grid steps, but at most 1/8 of the smallest distance from a target to a box face (the
# face's rectangle, not its plane: e.g. a junction plane running under a slab does not cut the slab's targets)
function _gcq_default_skin(c::CampaignInput, X, dg)
    dmin = Inf
    for b in c.boxes, a in 1:3, sg in (-1, 1)
        L = (b.Lx, b.Ly, b.Lz)
        f = b.center[a] + sg * L[a] / 2
        l1, l2 = mod1(a + 1, 3), mod1(a + 2, 3)
        d = Threads.Atomic{Float64}(Inf)
        Threads.@threads for t in axes(X, 2)
            e1 = max(0.0, abs(X[l1, t] - b.center[l1]) - L[l1] / 2)
            e2 = max(0.0, abs(X[l2, t] - b.center[l2]) - L[l2] / 2)
            Threads.atomic_min!(d, sqrt((X[a, t] - f)^2 + e1^2 + e2^2))
        end
        dmin = min(dmin, d[])
    end
    return min(3 * minimum(norm, _grid_frame(dg)[2:4]), dmin / 8)
end

# (panel, target) pairs left near after the refinement, on at most nsample targets
function _gcq_near_pairs(itf, X; quad_tol, nsample = 20_000)
    ts = size(X, 2) <= nsample ? collect(axes(X, 2)) : round.(Int, range(1, size(X, 2), length = nsample))
    cnt = Threads.Atomic{Int}(0)
    Threads.@threads for p in itf.panels
        n = count(t -> !tensor_isfar(p, Tuple(view(X, :, t)) .- p.anchor; quad_tol, gcq_far = :direction), ts)
        Threads.atomic_add!(cnt, n)
    end
    return cnt[]
end

# screened volume potential of sv at the points X: TKM field near the source, point-charge far field elsewhere
function _gcq_volume_potential(sv, X; tol, c_pad, far)
    geom = near_field_geometry(sv; c_pad)
    near = zeros(Bool, size(X, 2))           # a Vector{Bool}: a BitVector is not safe for threaded writes
    Threads.@threads for i in axes(X, 2)
        near[i] = in_near_region(geom, X, i)
    end
    ni = findall(near); fi = findall(!, near)
    Φ = zeros(size(X, 2))
    if !isempty(ni)
        fld = PrecomputedVolumeField(sv; tol, c_pad, compute_grad = false)
        Φ[ni] .= volume_field_potential(fld, X[:, ni])
    end
    isempty(fi) || (Φ[fi] .= point_charge_field(far(tol), sv.positions, sv.weights .* sv.density, X[:, fi]))
    return Φ
end

# distinct keys of V_vol: keys[s] maps (s', D) to its index among the values of source shape s;
# V_vol[a, b] = vals[ksrc[a, b]][kidx[a, b]]. Symmetric (one dielectric region): C(s_a, s_b, D) = C(s_b, s_a, -D), so a
# key is kept with the larger shape index as the source.
function _gcq_volume_keys(S, shape_of, delta)
    P = length(shape_of)
    keys = [Dict{Tuple{Int, NTuple{3, Int}}, Int}() for _ in 1:S]
    ksrc = Matrix{Int}(undef, P, P); kidx = Matrix{Int}(undef, P, P)
    for b in 1:P, a in 1:P
        sa, sb, D = shape_of[a], shape_of[b], delta[a] .- delta[b]
        src, key = sa <= sb ? (sb, (sa, D)) : (sa, (sb, .-D))
        ksrc[a, b] = src; kidx[a, b] = get!(keys[src], key, length(keys[src]) + 1)
    end
    return keys, ksrc, kidx
end

# the values of the keys of the source shapes sbs (Dict sb => values): one volume solve per source shape at the grid
# points its keys touch, then the dot products
function _gcq_volume_values(c, dg, shapes, keys, sbs; tol, c_pad, far)
    blo = [ntuple(d -> minimum((g[d] for g in sh.gidx); init = 0), 3) for sh in shapes]
    bhi = [ntuple(d -> maximum((g[d] for g in sh.gidx); init = 0), 3) for sh in shapes]
    o, ea, eb, ec = _grid_frame(dg)
    out = Dict{Int, Vector{Float64}}()
    for sb in sbs
        vals = zeros(length(keys[sb])); out[sb] = vals
        # empty shapes (non-overlapping orbitals) give zero values
        kl = [kv for kv in keys[sb] if !isempty(shapes[kv[1][1]].gidx)]
        (isempty(kl) || isempty(shapes[sb].gidx)) && continue
        lo = fill(typemax(Int), 3); hi = fill(typemin(Int), 3)
        for ((sa, D), _) in kl, d in 1:3
            lo[d] = min(lo[d], blo[sa][d] + D[d]); hi[d] = max(hi[d], bhi[sa][d] + D[d])
        end
        dims = Tuple(hi .- lo .+ 1)
        lin(g) = (g[1] - lo[1] + 1) + dims[1] * ((g[2] - lo[2]) + dims[2] * (g[3] - lo[3]))
        need = zeros(Bool, prod(dims))
        Threads.@threads for q in eachindex(kl)
            (sa, D), _ = kl[q]
            @inbounds for h in shapes[sa].gidx
                need[lin(h .+ D)] = true
            end
        end
        idx = findall(need)
        X = Matrix{Float64}(undef, 3, length(idx))
        Threads.@threads for m in eachindex(idx)
            l = idx[m] - 1
            i = l % dims[1]; j = (l ÷ dims[1]) % dims[2]; k = l ÷ (dims[1] * dims[2])
            gi, gj, gk = lo[1] + i - 1, lo[2] + j - 1, lo[3] + k - 1          # grid_point is 1-based
            @inbounds for d in 1:3
                X[d, m] = o[d] + gi * ea[d] + gj * eb[d] + gk * ec[d]
            end
        end
        Φ = zeros(prod(dims))
        Φ[idx] .= _gcq_volume_potential(_gcq_shape_source(c, dg, shapes[sb]), X; tol, c_pad, far)
        Threads.@threads for q in eachindex(kl)
            (sa, D), id = kl[q]
            sh = shapes[sa]; acc = 0.0
            @inbounds for (h, w) in zip(sh.gidx, sh.tw)
                acc += w * Φ[lin(h .+ D)]
            end
            vals[id] = acc
        end
    end
    return out
end

# greedy largest-first partition of items with costs into n parts
function _gcq_balance(costs, n)
    parts = [Int[] for _ in 1:n]; load = zeros(n)
    for i in sortperm(costs; rev = true)
        k = argmin(load); push!(parts[k], i); load[k] += costs[i]
    end
    return parts
end

_gcq_far(kind::Symbol) = kind === :sctl ? (tol -> SCTLFarField(tol = tol)) :
                         kind === :fmm3d ? (tol -> FMM3DFarField(tol)) :
                         throw(ArgumentError("far must be :sctl or :fmm3d, got :$kind"))

function _gcq_put(sync, part, name, x)
    mkpath(sync)
    tmp = joinpath(sync, "$name.tmp$part"); serialize(tmp, x); mv(tmp, joinpath(sync, name); force = true)
end
function _gcq_take(sync, name)
    f = joinpath(sync, name)
    while !isfile(f); sleep(0.1); end
    return deserialize(f)
end

"""
    gcq_four_index(c::CampaignInput; part = 1, nparts = 1, sync = nothing, far = :sctl, pairs = nothing,
                   nlev = 2, n_quad, rhs_tol, lhs_tol, gmres_rtol, volume_tol, quad_tol = 1e-8, edge_tol = 1e-8,
                   tau = 1e-3, fg_tol = 1e-4, target_skin = nothing, c_pad = c.c_pad, verbose = true)

The four-index matrix of campaign `c` on one GCQ interface (see the file header). Returns
`(; pair_ids, V, V_vol, V_layer, stats)` (raw units, `V` symmetrized) on part 1 and `nothing` on the others.

- `pairs`: the pair list (default: as `prepare`, `pair_overrides` or every pair within `neighbor_cutoff`).
- `support`: the truncation of the pair densities (`:global` or `:pair`, see [`gcq_pair_shapes`](@ref)).
- `far = :sctl` (SCTL's ParticleFMM; needs `using SCTL`) or `:fmm3d`, for the operator and for every point-charge
  field.
- Tolerances: `n_quad`, `rhs_tol`, `lhs_tol` (operator far field), `gmres_rtol` and `volume_tol` default to the
  campaign's `[solve]` values. `quad_tol` and `edge_tol` set the GCQ near quadrature and edge rules, `fg_tol` the F
  and G fields, `tau` the low-rank truncation (relative to the largest singular value of W^{1/2} F).
- `target_skin`: the cell of the target reduction of the interface refinement (default 3 grid steps, at most 1/8 of
  the smallest target-to-face distance). The layer part needs every target far from every panel; part 1 counts the
  near pairs left on a sample and warns.
- `part`, `nparts`, `sync`: run as part `part` of `nparts` processes started together, exchanging data through files
  in the directory `sync` (`/dev/shm/...` on one node, a shared directory across nodes). Shapes and solves are split
  over the parts; part 1 assembles. With `nparts = 1`, `sync` is unused.

Each part uses its own threads: on a two-socket node, run one or two parts per socket (pinned, e.g. with
`numactl --cpunodebind`), with `OMP_NUM_THREADS` at most the cores of a socket.
"""
function gcq_four_index(c::CampaignInput; part::Int = 1, nparts::Int = 1, sync = nothing, far::Symbol = :sctl,
                        pairs = nothing, nlev::Int = 2, n_quad::Int = Int(get(c.solve, "n_quad", 6)),
                        rhs_tol::Real = get(c.solve, "rhs_tol", 1e-5), lhs_tol::Real = get(c.solve, "lhs_tol", 1e-5),
                        gmres_rtol::Real = get(c.solve, "gmres_rtol", 1e-5), volume_tol::Real = get(c.solve, "volume_tol", 1e-5),
                        quad_tol::Real = 1e-8, edge_tol::Real = 1e-8, tau::Real = 1e-3, fg_tol::Real = 1e-4,
                        target_skin = nothing, c_pad::Real = c.c_pad, support::Symbol = :global, verbose::Bool = true)
    1 <= part <= nparts || throw(ArgumentError("part must be in 1:nparts"))
    nparts > 1 && sync === nothing && throw(ArgumentError("nparts > 1 needs a sync directory"))
    ff = _gcq_far(far)
    T0 = time(); stats = Pair{String, Float64}[]
    st(l, t) = (push!(stats, l => t); verbose && part == 1 && (@info "gcq_four_index: $l" t; flush(stderr)))
    # 1. pairs, shapes, targets
    local centers, pids, shapes, shape_of, delta, dg, X, env
    t = @elapsed begin
        centers = enumerate_centers(c)
        pids = pairs !== nothing ? collect(pairs) :
               c.pair_overrides !== nothing ? c.pair_overrides : enumerate_pairs(centers, c.neighbor_cutoff)
        shapes, shape_of, delta, dg = gcq_pair_shapes(c, centers, pids; support)
        _gcq_uniform_box(c, dg, shapes, shape_of, delta)
        X, env = _gcq_targets(shapes, shape_of, delta, dg)
    end
    P = length(pids); S = length(shapes)
    st("$P pairs, $S shapes, $(size(X, 2)) targets", t)
    skin = target_skin === nothing ? _gcq_default_skin(c, X, dg) : Float64(target_skin)
    # 2. interface and operator
    # one interface for every part: part 1 builds it (the refinement sums are not bit-reproducible across processes)
    local itf, A
    t = @elapsed if part == 1
        itf = gcq_campaign_interface(c, X, env, dg; nlev, n_quad, rhs_tol, edge_tol, quad_tol, target_skin = skin, far = ff)
        nparts > 1 && _gcq_put(sync, part, "interface", itf)
    else
        itf = _gcq_take(sync, "interface")
    end
    st("GCQ interface: N = $(num_points(itf)), $(length(itf.panels)) panels", t)
    t = @elapsed A = dielectric_operator(itf; far = ff(lhs_tol), near = GCQNearQuad(quad_tol = quad_tol, η_gcq = 2.0, gcq_far = :direction))
    st("operator", t)
    nodes, nrm = _interface_targets_normals(itf); w = all_weights(itf); N = size(nodes, 2); d = sqrt.(abs.(w))
    # the layer part G' sigma is the node rule: every target must be far from every panel
    nnear = part == 1 ? _gcq_near_pairs(itf, X; quad_tol) : 0
    nnear > 0 && @warn "gcq_four_index: $nnear (panel, target) pairs are near on a sample of the targets: the layer " *
                       "part is inaccurate there (lower target_skin)"
    # 3. F, G of this part's shapes
    pos = [findall(==(s), shape_of) for s in 1:S]
    myshapes = _gcq_balance(length.(pos), nparts)[part]
    mypairs = reduce(vcat, [pos[s] for s in myshapes]; init = Int[])
    Floc = zeros(N, length(mypairs)); Gloc = zeros(N, length(mypairs)); col = Dict(p => k for (k, p) in enumerate(mypairs))
    t = @elapsed for s in myshapes
        ps = pos[s]
        isempty(shapes[s].gidx) && continue             # zero density: F = G = 0
        sv = _gcq_shape_source(c, dg, shapes[s])
        T = reduce(hcat, [nodes .- _grid_shift(dg, delta[q]) for q in ps])        # the shape stays, the nodes move
        g = point_charge_field(ff(fg_tol), sv.positions, sv.weights .* sv.density, T; grad = true)
        u = point_charge_field(ff(fg_tol), sv.positions, shapes[s].tw, T)
        Threads.@threads for k in eachindex(ps)
            ck = col[ps[k]]; off = (k - 1) * N
            @inbounds for j in 1:N
                Floc[j, ck] = nrm[1, j] * g[1, off + j] + nrm[2, j] * g[2, off + j] + nrm[3, j] * g[3, off + j]
                Gloc[j, ck] = w[j] * u[off + j]
            end
        end
    end
    st("F, G of $(length(myshapes)) shapes", t)
    # 4. range of W^{1/2} F (part 1), solves of this part's columns
    local F, G, Q, R, r
    if part == 1
        F = zeros(N, P); G = zeros(N, P); F[:, mypairs] .= Floc; G[:, mypairs] .= Gloc
        for k in 2:nparts
            pk, Fk, Gk = _gcq_take(sync, "fg-$k"); F[:, pk] .= Fk; G[:, pk] .= Gk
        end
        t = @elapsed _gcq_dense() do
            B = d .* F
            E = eigen(Symmetric(B' * B)); ord = sortperm(E.values; rev = true)
            sv = sqrt.(max.(E.values[ord], 0.0)); r = max(count(>(tau * sv[1]), sv), 1)
            Q = B * (E.vectors[:, ord[1:r]] ./ sv[1:r]')
            R = Q' * B
        end
        nparts > 1 && _gcq_put(sync, part, "Q", Q)
        st("range of F: rank $r at tau $tau", t)
    else
        _gcq_put(sync, part, "fg-$part", (mypairs, Floc, Gloc))
        Q = _gcq_take(sync, "Q"); r = size(Q, 2)
    end
    Floc = Gloc = nothing
    mycols = collect(part:nparts:r)
    Yloc = zeros(N, length(mycols))
    t = @elapsed for (k, j) in enumerate(mycols)
        Yloc[:, k] .= first(solve_dielectric_box3d_gcq(A, Q[:, j] ./ d, itf; rtol = gmres_rtol, itmax = 500))
    end
    st("$(length(mycols)) GMRES solves", t)
    # 5. V_vol of this part's source shapes
    keys, ksrc, kidx = _gcq_volume_keys(S, shape_of, delta)
    vcost = [sum((length(shapes[sa].gidx) for ((sa, _), _) in keys[s]); init = 0) for s in 1:S]
    local vals
    t = @elapsed vals = _gcq_volume_values(c, dg, shapes, keys, _gcq_balance(vcost, nparts)[part]; tol = volume_tol,
                                           c_pad, far = ff)
    st("V_vol of $(length(vals)) source shapes", t)
    if part != 1
        _gcq_put(sync, part, "res-$part", (mycols, Yloc, vals))
        return nothing
    end
    # assemble
    Y = zeros(N, r); Y[:, mycols] .= Yloc
    for k in 2:nparts
        ck, Yk, vk = _gcq_take(sync, "res-$k"); Y[:, ck] .= Yk; merge!(vals, vk)
    end
    Vlay = _gcq_dense(() -> (G' * Y) * R)
    Vvol = [vals[ksrc[a, b]][kidx[a, b]] for a in 1:P, b in 1:P]
    V = Vvol .+ Vlay
    asym = maximum(abs, V - V') / maximum(abs, V)
    st("assembled V (max |V - V'| / max |V| = $(round(asym; sigdigits = 2)))", time() - T0)
    onsite_norm2 = Dict(pids[a][1] => sum(shapes[shape_of[a]].tw) for a in 1:P if pids[a][1] == pids[a][2])
    return (; pair_ids = pids, V = (V + V') / 2, V_vol = Vvol, V_layer = Vlay,
              stats = (; times = stats, total = time() - T0, rank = r, N, npanels = length(itf.panels), nshapes = S,
                       ntargets = size(X, 2), target_skin = skin, near_pairs = nnear, asym, onsite_norm2))
end

_gcq_dense(f) = (nb = BLAS.get_num_threads(); BLAS.set_num_threads(Threads.nthreads());
                 try f() finally BLAS.set_num_threads(nb) end)

"""
    run_gcq_campaign(c::CampaignInput; dir = joinpath(c.root, "gcq"), kwargs...)

[`gcq_four_index`](@ref) for campaign `c`, written (by part 1) as `assemble_v` writes the batch pipeline's result:
`V_full.tsv` (raw), `V_full_eV.jls` and `V_full_eV.tsv` (eV, orbital-normalized) and `report.txt`, under `dir`.
`kwargs` go to `gcq_four_index` (for several parts: `part`, `nparts`, `sync`). Returns its result (part 1).
"""
function run_gcq_campaign(c::CampaignInput; dir::AbstractString = joinpath(c.root, "gcq"), kwargs...)
    res = gcq_four_index(c; kwargs...)
    res === nothing && return nothing
    mkpath(dir)
    pids = res.pair_ids; n = length(pids)
    write_v_table(joinpath(dir, "V_full.tsv"), pids, res.V)
    E2 = 14.3996
    orbn2 = res.stats.onsite_norm2
    all(haskey(orbn2, p[1]) && haskey(orbn2, p[2]) for p in pids) ||
        error("run_gcq_campaign: the eV normalization needs an onsite (i,i) pair for every orbital")
    nrm = [sqrt(orbn2[p[1]] * orbn2[p[2]]) for p in pids]
    V_eV = [res.V[a, b] * 4π * E2 / (nrm[a] * nrm[b]) for a in 1:n, b in 1:n]
    _atomic_serialize(joinpath(dir, "V_full_eV.jls"),
        (; pair_ids = pids, V = V_eV, orbital_norms = orbn2, unit = "eV", E2_eV_Ang = E2, normalization = "orbital",
           method = "gcq", stats = res.stats))
    write_v_table(joinpath(dir, "V_full_eV.tsv"), pids, V_eV)
    report = sprint() do io
        println(io, "campaign: $(c.name) (GCQ campaign mode)")
        println(io, "pairs: $n   shapes: $(res.stats.nshapes)   targets: $(res.stats.ntargets)")
        println(io, "interface: N = $(res.stats.N), $(res.stats.npanels) panels; low rank $(res.stats.rank)")
        println(io, "max|V| (raw): $(maximum(abs, res.V))")
        println(io, "max rel asymmetry before symmetrizing: $(res.stats.asym)")
        println(io, "max|V| (eV):  $(maximum(abs, V_eV))")
        println(io, "onsite V[1,1] (eV): $(V_eV[1, 1])")
        println(io, "time: $(round(res.stats.total; digits = 1)) s")
        for (l, t) in res.stats.times
            println(io, "  ", rpad(l, 60), round(t; digits = 2), " s")
        end
    end
    _atomic_write_text(io -> print(io, report), joinpath(dir, "report.txt"))
    return res
end
