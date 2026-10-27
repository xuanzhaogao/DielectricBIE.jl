# GCQ campaign mode: the whole four-index matrix of a campaign on ONE GCQ interface.
#
#   V[a, b] = <rho_a, Phi_b>,   Phi_b = Phi_vol_b + S sigma_b,   A sigma_b = F_b,
#   V = V_vol + V_layer,        V_layer = G' A^{-1} F,   G[j, a] = w_j phi_a(y_j)   (symmetry of the single layer)
#
# 1. Shapes: on a lattice every pair density is a grid translate of a few shapes (template_i, template_j,
#    step_j - step_i), each truncated on its own support (support_rtol). The targets are the union of all supports.
# 2. One GCQ interface, refined for the envelope of all pair densities (RHS) and for all targets (target_skin), and its
#    operator A. The interface is far from every target, so the layer part is the node rule: G' sigma.
# 3. F and G of every pair by one volume field per shape (TKM near the density, point charges far) over the interface
#    nodes shifted by each pair's offset.
# 4. Low rank: W^{1/2} F ~= Q (Q' W^{1/2} F) at tolerance tau (rank r), Y = A^{-1} W^{-1/2} Q (r GMRES solves),
#    V_layer = (G' Y)(Q' W^{1/2} F).
# 5. V_vol by shapes: one volume solve per source shape (TKM near the shape, FMM elsewhere) at the grid points needed,
#    and one dot product per distinct (shape, shape, shift, screening). For densities inside one region V_vol is the
#    unscreened overlap over eps, symmetric, so each pair of shapes is done once.
#
# A pair density is inside one region, or cut by one horizontal face (gcq_cut.jl: cut-cell quadrature and a
# half-space formula near the face); the mode checks this. The work can be split over `nparts` cooperating processes
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

# screened VolumeSource of a shape shifted by delta (grid steps)
function _gcq_shape_source(c::CampaignInput, dg, sh::GCQPairShape, delta = (0, 0, 0))
    lb, dv = _grid_basis(dg)
    X = _grid_positions(dg, [g .+ delta for g in sh.gidx])
    vs = VolumeSource(X, fill(dv, length(sh.tw)), sh.tw ./ dv; lattice_basis = lb)
    return screened_volume_source(c.boxes, c.epses, c.eps_out, vs, SharpScreening())
end

# the bare (unscreened) VolumeSource of a shape, at its representative placement
function _gcq_shape_source0(dg, sh::GCQPairShape)
    lb, dv = _grid_basis(dg)
    return VolumeSource(_grid_positions(dg, sh.gidx), fill(dv, length(sh.tw)), sh.tw ./ dv; lattice_basis = lb)
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
function gcq_campaign_interface(c::CampaignInput, X, env, dg; nlev, n_quad, rhs_tol, edge_tol, quad_tol, target_skin, far,
                                refine_targets::Bool = true, pot_tol = rhs_tol)
    lb, dv = _grid_basis(dg)
    envs = VolumeSource(X, fill(dv, length(env)), env ./ dv; lattice_basis = lb)
    senv = screened_volume_source(c.boxes, c.epses, c.eps_out, envs, SharpScreening())
    srcpos, srcq = _volume_source_fmm_sources(senv)
    kmax = _estimate_tkm3dc_kmax(senv)
    # the panels must interpolate the RHS (n . grad phi, the solve) and the potential itself (the layer part G' sigma is
    # the node rule of sigma phi): one scalar, each part over its tolerance, refined to 1
    rhs_batch = (Pts, Nrm) -> (T = reduce(hcat, collect.(Pts)); N = reduce(hcat, collect.(Nrm));
        first(_rhs_volume_targets_hybrid(srcpos, srcq, T, N, 1.0, rhs_tol / 10, kmax,
                                         _classify_near_far_targets(T, senv); far = far(rhs_tol / 10))) ./ rhs_tol .+
        _gcq_volume_potential(senv, T; tol = pot_tol / 10, c_pad = c.c_pad, far) ./ pot_tol)
    mesh = multi_dielectric_box3d_gcq(c.boxes, c.epses, c.eps_out; nlev, n_quad, tol = edge_tol,
        rhs_batch, rhs_atol = 1.0, max_depth = Int(get(c.solve, "max_depth", 30)),
        (refine_targets ? (; targets = X, target_tol = quad_tol, target_eta = 2.0, target_far = :direction, target_skin) : (;))...)
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

# gradient of the screened volume potential of sv at the points X (3 x n): TKM field near the source, far field elsewhere
function _gcq_volume_gradient(sv, X; tol, c_pad, far)
    geom = near_field_geometry(sv; c_pad)
    near = zeros(Bool, size(X, 2))                     # a Vector{Bool}: a BitVector is not safe for threaded writes
    Threads.@threads for i in axes(X, 2)
        near[i] = in_near_region(geom, X, i)
    end
    ni = findall(near); fi = findall(!, near)
    g = zeros(3, size(X, 2))
    if !isempty(ni)
        fld = PrecomputedVolumeField(sv; tol, c_pad, compute_pot = false)
        g[:, ni] .= volume_field_gradient(fld, X[:, ni])
    end
    isempty(fi) || (g[:, fi] .= point_charge_field(far(tol), sv.positions, sv.weights .* sv.density, X[:, fi]; grad = true))
    return g
end

# Hessian (6 × n: xx, yy, zz, xy, xz, yz) of the volume potential of sv at the points X: TKM near the source, point
# charges elsewhere
function _gcq_volume_hessian(sv, X; tol, c_pad, far = FMM3DFarField)
    fld = PrecomputedVolumeField(sv; tol, c_pad, compute_grad = false)
    return volume_field_hessian(fld, X; far = far(tol))
end

# distinct keys of V_vol[a, b] = <rho_a, N[rho_b / eps]>: keys[s] maps (s', D, k) to its index among the values of
# source shape s, the bare potential of s at the points of s' shifted by D, summed with the weights tw (k = 0) or
# tw .* classes[k].m (k > 0); V_vol[a, b] = vals[ksrc[a, b]][kidx[a, b]] * kfac[a, b].
# - b inside one region: C(s_a, s_b, D) / eps_b, with C the bare overlap. C(s_a, s_b, D) = C(s_b, s_a, -D), so a key
#   is kept with the larger shape index as the source.
# - b cut: by the symmetry of N, int phi0_a rho_b / eps: the source is s_a, at b's points with b's cut-cell screening.
function _gcq_volume_keys(S, shape_of, delta, cls, inveps)
    P = length(shape_of)
    keys = [Dict{Tuple{Int, NTuple{3, Int}, Int}, Int}() for _ in 1:S]
    ksrc = Matrix{Int}(undef, P, P); kidx = Matrix{Int}(undef, P, P); kfac = ones(P, P)
    for b in 1:P, a in 1:P
        ksrc[a, b], kidx[a, b], kfac[a, b] = _gcq_volume_key!(keys, a, b, shape_of, delta, cls, inveps)
    end
    return keys, ksrc, kidx, kfac
end
# the same for the entries (a, b) of the list `ab` only (vectors over ab)
function _gcq_volume_keys(S, shape_of, delta, cls, inveps, ab::AbstractVector{NTuple{2, Int}})
    keys = [Dict{Tuple{Int, NTuple{3, Int}, Int}, Int}() for _ in 1:S]
    n = length(ab); ksrc = Vector{Int}(undef, n); kidx = Vector{Int}(undef, n); kfac = ones(n)
    for (i, (a, b)) in enumerate(ab)
        ksrc[i], kidx[i], kfac[i] = _gcq_volume_key!(keys, a, b, shape_of, delta, cls, inveps)
    end
    return keys, ksrc, kidx, kfac
end
function _gcq_volume_key!(keys, a, b, shape_of, delta, cls, inveps)
    sa, sb, D = shape_of[a], shape_of[b], delta[a] .- delta[b]
    if cls[b] == 0
        src, key = sa <= sb ? (sb, (sa, D, 0)) : (sa, (sb, .-D, 0))
        fac = inveps[b]
    else
        src, key = sa, (sb, .-D, cls[b]); fac = 1.0
    end
    return src, get!(keys[src], key, length(keys[src]) + 1), fac
end

# the values of the keys of the source shapes sbs (Dict sb => values): one volume solve per source shape at the grid
# points its keys touch, then the dot products
function _gcq_volume_values(c, dg, shapes, keys, sbs, classes; tol, c_pad, far)
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
        for ((sa, D, _), _) in kl, d in 1:3
            lo[d] = min(lo[d], blo[sa][d] + D[d]); hi[d] = max(hi[d], bhi[sa][d] + D[d])
        end
        dims = Tuple(hi .- lo .+ 1)
        lin(g) = (g[1] - lo[1] + 1) + dims[1] * ((g[2] - lo[2]) + dims[2] * (g[3] - lo[3]))
        need = zeros(Bool, prod(dims))
        Threads.@threads for q in eachindex(kl)
            (sa, D, _), _ = kl[q]
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
        Φ[idx] .= _gcq_volume_potential(_gcq_shape_source0(dg, shapes[sb]), X; tol, c_pad, far)
        Threads.@threads for q in eachindex(kl)
            (sa, D, k), id = kl[q]
            sh = shapes[sa]; acc = 0.0
            if k == 0
                @inbounds for (h, w) in zip(sh.gidx, sh.tw)
                    acc += w * Φ[lin(h .+ D)]
                end
            else
                m = classes[k].m
                @inbounds for (n, h) in enumerate(sh.gidx)
                    acc += sh.tw[n] * m[n] * Φ[lin(h .+ D)]
                end
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
- `refine_targets` (default `false`): also refine the interface for the targets (the support points), as
  `GCQPotential` needs for near-free node rules. Not needed here: the layer part is G' σ = ∫ σ φ_a dA with φ_a
  evaluated accurately at the nodes (TKM near the density), the node rule of a smooth integrand that the RHS
  refinement resolves. On Sec. 5.3 both give the same V to 2.3e-6, without on a 43% smaller interface. A density cut
  by a face needs `false`.
- Densities cut by faces: a pair density may cross any box faces, provided the grid has exactly one lattice vector
  not parallel to each face that comes within reach (an orthogonal grid for vertical faces; a grid with horizontal
  first and second vectors for horizontal faces only). Its volume part and the far field of its RHS use cut-cell
  quadrature (a tensor product of 1D rules, `_cutcell_interval`); the RHS at the nodes near it depends on `cut_rhs`:
  - `:auto` (default): the half-space formula (`_gcq_halfspace_dz`) for a class cut by one horizontal face with the
    nodes near it on horizontal faces, the surface corrections otherwise;
  - `:surface`: the surface corrections for every cut class (`gcq_surface.jl`: double and single layers of ψ and its
    Hessian on SCTL Gauss-Legendre elements of `cut_order`² nodes on the interface rectangles, at most
    `cut_κ (d + cut_L0)` in size at distance d from the cut densities (`cut_L0` = 8 grid steps by default), SCTL's near
    quadrature at `cut_tol`; needs `using SCTL`);
  - `:halfspace`: the half-space formula, an error where it does not apply.
  See `gcq_cut.jl` and `gcq_surface.jl`.
- `field_derivs`: how the TKM field of each shape gives the potential, gradient and Hessian at the nodes and the
  surface-correction nodes: `:spectral` (default; one type-2 NUFFT per component, the derivatives as ik and -k k
  multipliers) or `:window` (one scalar grid per shape, derivatives of the gathering window:
  `PrecomputedVolumeField(...; window_derivatives = true)`; one field serves every quantity and batch).
  `field_window` (a NamedTuple) passes window options to it, e.g. `(; window = :pswf)` or
  `(; window_shape = 1.1, window_ns = 16)`.
- `target_skin`: the cell of the target reduction of the interface refinement (default 3 grid steps, at most 1/8 of
  the smallest target-to-face distance). The layer part needs every target far from every panel; part 1 counts the
  near pairs left on a sample and warns.
- `gmres_block`: the solves go through block GMRES, this many right-hand sides at a time (one batched FMM per
  iteration for the block; [`solve_dielectric_box3d_gcq_block`](@ref)); 1 solves them one by one.
- `vform = :nearfar`: `V` as a [`GCQNearFarV`](@ref) (exact near entries, a moment far field) instead of a dense
  P × P matrix; `tile`, `rnear` and `p_far` set its tiles, near test and moment degree (see `gcq_nearfar.jl`).
  `layer = :translate` (with `vform = :nearfar`, `nparts = 1`) takes the induced response invariant under in-plane
  translations: a number of solves independent of the lattice size, exact for a laterally infinite box.
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
                        target_skin = nothing, c_pad::Real = c.c_pad, support::Symbol = :global,
                        refine_targets::Bool = false, cut_rhs::Symbol = :auto, cut_order::Int = 8, cut_κ::Real = 0.25,
                        cut_L0 = nothing, cut_tol::Real = 1e-10, field_derivs::Symbol = :spectral,
                        field_window::NamedTuple = (;), vform::Symbol = :dense, tile = nothing, rnear = nothing,
                        p_far::Int = 8, layer::Symbol = :lowrank, gmres_block::Int = 32, verbose::Bool = true)
    vform in (:dense, :nearfar) || throw(ArgumentError("vform must be :dense or :nearfar"))
    layer in (:lowrank, :translate) || throw(ArgumentError("layer must be :lowrank or :translate"))
    layer === :translate && vform !== :nearfar && throw(ArgumentError("layer = :translate needs vform = :nearfar"))
    layer === :translate && nparts > 1 && throw(ArgumentError("layer = :translate runs in one process (nparts = 1)"))
    cut_rhs in (:auto, :surface, :halfspace) || throw(ArgumentError("cut_rhs must be :auto, :surface or :halfspace"))
    field_derivs in (:spectral, :window) || throw(ArgumentError("field_derivs must be :spectral or :window"))
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
        cls, inveps, classes = _gcq_screening(c, dg, shapes, shape_of, delta)
        refine_targets && !isempty(classes) &&
            throw(ArgumentError("refine_targets = true needs every pair density inside one region (none cut by a face)"))
        X, env = _gcq_targets(shapes, shape_of, delta, dg)
    end
    P = length(pids); S = length(shapes)
    st("$P pairs, $S shapes, $(size(X, 2)) targets, $(count(>(0), cls)) cut pairs in $(length(classes)) classes", t)
    local tl; lt = nothing
    if vform === :nearfar
        t = @elapsed tl = _gcq_nearfar_tiles(dg, shapes, shape_of, delta; tile, rnear)
        st("$(length(tl.tiles)) tiles of side $(round(tl.tile; sigdigits = 4)), proxy half-widths " *
           "$(round.(tl.w; sigdigits = 4)), rnear $(round(tl.rnear; sigdigits = 4)): $(length(tl.ab)) near pairs, " *
           "$(length(tl.faroffs)) far offsets", t)
        if layer === :translate
            lt = _gcq_translate_groups(dg, shapes, shape_of, delta, cls)
            st("layer by translation: $(length(lt.ref)) groups of translates", 0.0)
        end
    end
    skin = target_skin === nothing ? _gcq_default_skin(c, X, dg) : Float64(target_skin)
    # 2. interface and operator
    # one interface for every part: part 1 builds it (the refinement sums are not bit-reproducible across processes)
    local itf, A
    t = @elapsed if part == 1
        itf = gcq_campaign_interface(c, X, env, dg; nlev, n_quad, rhs_tol, edge_tol, quad_tol, target_skin = skin, far = ff,
                                     refine_targets)
        nparts > 1 && _gcq_put(sync, part, "interface", itf)
    else
        itf = _gcq_take(sync, "interface")
    end
    st("GCQ interface: N = $(num_points(itf)), $(length(itf.panels)) panels", t)
    t = @elapsed A = dielectric_operator(itf; far = ff(lhs_tol), near = GCQNearQuad(quad_tol = quad_tol, η_gcq = 2.0, gcq_far = :direction))
    st("operator", t)
    nodes, nrm = _interface_targets_normals(itf); w = all_weights(itf); N = size(nodes, 2); d = sqrt.(abs.(w))
    # the layer part G' sigma is the node rule: every target must be far from every panel
    nnear = part == 1 && refine_targets ? _gcq_near_pairs(itf, X; quad_tol) : 0
    nnear > 0 && @warn "gcq_four_index: $nnear (panel, target) pairs are near on a sample of the targets: the layer " *
                       "part is inaccurate there (lower target_skin)"
    # 3. F, G of this part's shapes
    pos = [findall(==(s), shape_of) for s in 1:S]
    myshapes = _gcq_balance(length.(pos), nparts)[part]
    # the pairs whose F is needed: all (low rank), or the reference pair of each group of translates
    fpos = lt === nothing ? pos : (refs = Set(lt.ref); [filter(in(refs), pos[s]) for s in 1:S])
    mypairs = reduce(vcat, [fpos[s] for s in myshapes]; init = Int[])
    # cut pairs: the RHS method of each class near the density, and the surface corrections (their own source surface,
    # graded toward every cut density; corrections at all interface nodes)
    meth = Dict{Int, Symbol}(); surf = nothing; atr = nothing; snodes = nodes
    t = @elapsed if !isempty(classes)
        for s in myshapes
            any(q -> cls[q] > 0, pos[s]) || continue
            geom = near_field_geometry(_gcq_shape_source0(dg, shapes[s]); c_pad)
            for q in pos[s]
                cls[q] > 0 || continue
                Tq = nodes .- _grid_shift(dg, delta[q])
                ok = classes[cls[q]].plane && all(j -> !in_near_region(geom, Tq, j) || abs(abs(nrm[3, j]) - 1) < 1e-12, 1:N)
                mq = cut_rhs === :auto ? (ok ? :halfspace : :surface) : cut_rhs
                meth[cls[q]] = mq === :surface || get(meth, cls[q], :halfspace) === :surface ? :surface : :halfspace
            end
        end
        if any(==(:surface), values(meth))
            o, ea, eb, ec = _grid_frame(dg)
            env = map(findall(>(0), cls)) do q
                gi = shapes[shape_of[q]].gidx
                lo = ntuple(d -> minimum(g[d] for g in gi) + delta[q][d], 3)
                hi = ntuple(d -> maximum(g[d] for g in gi) + delta[q][d], 3)
                C = reduce(hcat, [o .+ (i - 1) .* ea .+ (j - 1) .* eb .+ (k - 1) .* ec
                                  for i in (lo[1], hi[1]), j in (lo[2], hi[2]), k in (lo[3], hi[3])])
                (Tuple(minimum(C; dims = 2)), Tuple(maximum(C; dims = 2)))
            end
            L0 = cut_L0 === nothing ? 8 * minimum(norm, (ea, eb, ec)) : Float64(cut_L0)
            Base.get_extension(@__MODULE__, :DielectricBIESCTLExt) === nothing &&
                throw(ArgumentError("the surface corrections of cut densities (cut_rhs = :surface) need `using SCTL`"))
            rects = _gcq_surface_rects(_gcq_interface_rects(c.boxes, c.epses, c.eps_out), env; κ = cut_κ, L0)
            snodes = _gcq_surface_snap(rects, nodes, nrm)
            surf = gcq_surface_corrections(rects, snodes; order = cut_order, tol = cut_tol)
            atr = _gcq_surface_trace(itf)
        end
    end
    surf === nothing ||
        st("surface corrections: $(size(surf.X, 2)) source nodes, for $(count(==(:surface), values(meth))) classes", t)
    Floc = zeros(N, length(mypairs)); Gloc = zeros(N, length(mypairs)); col = Dict(p => k for (k, p) in enumerate(mypairs))
    t = @elapsed for s in myshapes
        ps = fpos[s]
        (isempty(ps) || isempty(shapes[s].gidx)) && continue     # zero density: F = G = 0
        sv0 = _gcq_shape_source0(dg, shapes[s])                                    # the bare density
        T = reduce(hcat, [nodes .- _grid_shift(dg, delta[q]) for q in ps])        # the shape stays, the nodes move
        # g, u: gradient and potential of the bare shape, near it (nodes inside its padded box) by its TKM field,
        # elsewhere by point charges. G' sigma = int sigma phi_a dA is then the node rule of a smooth integrand, with no
        # condition on the distance from the density to the interface. F = n . g / eps for a pair inside one region
        fld = field_derivs === :window ? PrecomputedVolumeField(sv0; tol = fg_tol, c_pad, window_derivatives = true, field_window...) : nothing
        if fld === nothing
            g = _gcq_volume_gradient(sv0, T; tol = fg_tol, c_pad, far = ff)
            u = _gcq_volume_potential(sv0, T; tol = fg_tol, c_pad, far = ff)
        else
            r = volume_field_derivatives(fld, T; pot = true, grad = true, hess = false, far = ff(fg_tol))
            g, u = r.grad, r.pot
        end
        Threads.@threads for k in eachindex(ps)
            ck = col[ps[k]]; off = (k - 1) * N; a = inveps[ps[k]]
            @inbounds for j in 1:N
                Floc[j, ck] = a * (nrm[1, j] * g[1, off + j] + nrm[2, j] * g[2, off + j] + nrm[3, j] * g[3, off + j])
                Gloc[j, ck] = w[j] * u[off + j]
            end
        end
        # pairs cut by faces, by class: F by the cut-cell screening far from the shape, near it by the half-space
        # formula, or (:surface) everywhere by the surface corrections of g
        ks = findall(k -> cls[ps[k]] > 0, eachindex(ps))
        if !isempty(ks)
            geom = near_field_geometry(sv0; c_pad)
            near = zeros(Bool, size(T, 2))
            Threads.@threads for i in axes(T, 2)
                near[i] = in_near_region(geom, T, i)
            end
            # psi and its Hessian at the source nodes of the surface corrections, for the :surface pairs (one evaluation)
            sk = [k for k in ks if meth[cls[ps[k]]] === :surface]
            src = Dict{Int, NTuple{2, Matrix{Float64}}}()
            if !isempty(sk)
                ns = size(surf.X, 2)
                Ts = reduce(hcat, [surf.X .- _grid_shift(dg, delta[ps[k]]) for k in sk])
                if fld === nothing
                    Ψa = _gcq_volume_gradient(sv0, Ts; tol = fg_tol, c_pad, far = ff)
                    Ha = _gcq_volume_hessian(sv0, Ts; tol = fg_tol, c_pad, far = ff)
                else
                    r = volume_field_derivatives(fld, Ts; pot = false, grad = true, hess = true, far = ff(fg_tol))
                    Ψa, Ha = r.grad, r.hess
                end
                for (i, k) in enumerate(sk)
                    src[k] = (Ψa[:, (i - 1) * ns .+ (1:ns)], Ha[:, (i - 1) * ns .+ (1:ns)])
                end
            end
            for kc in unique(cls[ps[ks]])
                blk = [k for k in ks if cls[ps[k]] == kc]
                _gcq_cut_rhs!(Floc, [col[ps[k]] for k in blk], blk, classes[kc], dg, shapes[s], sv0, T, g, near, nrm, N;
                              far = ff, tol = fg_tol, method = meth[kc], surf, atr, nodes = snodes, src)
            end
        end
    end
    st("F, G of $(length(myshapes)) shapes", t)
    # 4. range of W^{1/2} F (part 1), solves of this part's columns; or (layer = :translate) one solve per group
    local F, G, Q, R, r; Σ = nothing
    if lt !== nothing
        r = length(lt.ref); G = nothing; R = nothing; mycols = Int[]; Yloc = zeros(N, 0)
        t = @elapsed begin
            Σ = first(solve_dielectric_box3d_gcq_block(A, Floc[:, [col[q] for q in lt.ref]], itf; rtol = gmres_rtol,
                                                       itmax = 500, block = gmres_block))
        end
        st("$r GMRES solves (one per group of translates)", t)
    elseif part == 1
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
    if lt === nothing
        mycols = collect(part:nparts:r)
        Yloc = zeros(N, length(mycols))
        t = @elapsed (Yloc .= first(solve_dielectric_box3d_gcq_block(A, Q[:, mycols] ./ d, itf; rtol = gmres_rtol,
                                                                    itmax = 500, block = gmres_block)))
        st("$(length(mycols)) GMRES solves", t)
    end
    # 5. V_vol of this part's source shapes
    keys, ksrc, kidx, kfac = vform === :nearfar ? _gcq_volume_keys(S, shape_of, delta, cls, inveps, tl.ab) :
                                                  _gcq_volume_keys(S, shape_of, delta, cls, inveps)
    vcost = [sum((length(shapes[sa].gidx) for ((sa, _, _), _) in keys[s]); init = 0) for s in 1:S]
    local vals
    t = @elapsed vals = _gcq_volume_values(c, dg, shapes, keys, _gcq_balance(vcost, nparts)[part], classes;
                                           tol = volume_tol, c_pad, far = ff)
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
    lay = lt === nothing ? nothing :
          (; lt, Σ, itf, A, delta, gmres_rtol, gmres_block, tau, far = ff(fg_tol),
             ev = GCQPotential(quad_tol = quad_tol, η_gcq = 2.0, gcq_far = :direction, far = ff(fg_tol)))
    vform === :nearfar && return _gcq_assemble_nearfar(pids, dg, shapes, shape_of, tl, cls, inveps, classes, vals,
                                                       ksrc, kidx, kfac, G, Y, R; p = p_far, stats, st, T0, r, N,
                                                       npanels = length(itf.panels), ntargets = size(X, 2), skin, nnear,
                                                       lay)
    Vlay = _gcq_dense(() -> (G' * Y) * R)
    Vvol = [vals[ksrc[a, b]][kidx[a, b]] * kfac[a, b] for a in 1:P, b in 1:P]
    V = Vvol .+ Vlay
    asym = maximum(abs, V - V') / maximum(abs, V)
    st("assembled V (max |V - V'| / max |V| = $(round(asym; sigdigits = 2)))", time() - T0)
    onsite_norm2 = Dict(pids[a][1] => sum(shapes[shape_of[a]].tw) for a in 1:P if pids[a][1] == pids[a][2])
    return (; pair_ids = pids, V = (V + V') / 2, V_vol = Vvol, V_layer = Vlay,
              stats = (; times = stats, total = time() - T0, rank = r, N, npanels = length(itf.panels), nshapes = S,
                       ntargets = size(X, 2), target_skin = skin, near_pairs = nnear, asym, onsite_norm2,
                       cut_pairs = count(>(0), cls), cut_classes = length(classes)))
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
