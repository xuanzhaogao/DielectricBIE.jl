# Several axis-aligned boxes with GCQ edge panels, touching along faces, edges or vertices.
#
# The boxes are put on the rectilinear grid of all their face coordinates. Every grid cell has one
# permittivity (a box's, or the exterior `eps_out`), and every cell face with different
# permittivities on its two sides is an interface rectangle, meshed once: a face shared by two
# boxes is one ε_A | ε_B interface, not two coincident box-to-exterior surfaces. Each rectangle is
# meshed like a face of `single_dielectric_box3d_gcq`, in four quadrants, one per corner.
#
# Each grid edge is a 2D junction of the four cells around it. Its sectors (equal neighbouring
# quadrants merged) give the density powers (`junction_powers`; a perfectly conducting sector is
# a Dirichlet condition) and a weighted GCQ rule, used across the edge by every panel touching
# it. A straight interface (two half-planes, e.g. where a grid line crosses a face) is not
# singular, and panels are Gauss-Legendre across it. A quadrant is refined toward its corner
# `nlev` times only at a true vertex (the cells around the grid vertex are not invariant along
# any axis); at a point inside an edge or a face it is a single panel.
#
# The bodies of the total-charge constraints are the clusters of boxes joined by shared faces.
# Gauss's law fixes the charge inside a closed surface lying in the exterior, and a surface around
# one box of a face-sharing pair would cut through its neighbour. Boxes that touch only along
# edges or at vertices stay separate bodies.

"""
    MultiBoxGCQ

GCQ mesh of several boxes from `multi_dielectric_box3d_gcq`:
- `interface` : the `DielectricInterface` (every interface rectangle meshed once)
- `bodies`    : node range of each cluster (boxes joined by shared faces)
- `clusters`  : box indices of each cluster
- `epses, eps_out` : box permittivities and the exterior one
- `grid, cellbox`  : grid coordinates per axis and the box index of each cell (0 = exterior)
"""
struct MultiBoxGCQ{I <: DielectricInterface}
    interface::I
    bodies::Vector{UnitRange{Int}}
    clusters::Vector{Vector{Int}}
    epses::Vector{Float64}
    eps_out::Float64
    grid::NTuple{3, Vector{Float64}}
    cellbox::Array{Int, 3}
end

Base.show(io::IO, m::MultiBoxGCQ) =
    print(io, "MultiBoxGCQ with $(length(m.epses)) boxes in $(length(m.clusters)) clusters, ",
          "$(length(m.interface.panels)) panels")

num_points(m::MultiBoxGCQ) = num_points(m.interface)

# density powers p = γ - 1 of a junction; a perfectly conducting sector (ε = Inf) is a Dirichlet
# condition, and each arc of dielectric sectors between conductors is a separate wedge
function _junction_density_powers(angs, es; gmax)
    k = findfirst(isinf, es)
    k === nothing && return junction_powers(angs, es; gmax) .- 1
    a = circshift(angs, -k); e = circshift(es, -k)       # a conductor sector last
    γ = Float64[]; arc = Int[]
    for j in eachindex(e)
        if isinf(e[j])
            isempty(arc) || append!(γ, junction_powers([a[arc]; 2pi - sum(a[arc])], [e[arc]; Inf]; gmax))
            empty!(arc)
        else
            push!(arc, j)
        end
    end
    return sort(γ) .- 1
end

"""
    multi_dielectric_box3d_gcq(boxes, epses, eps_out = 1.0; nlev = 2, n_quad = 8, tol = 1e-10,
                               qmax = nothing, gmax = 20.0, rhs = nothing, rhs_batch = nothing,
                               rhs_atol = 1e-10, max_depth = 30) -> MultiBoxGCQ

Boxes `boxes = [(center = (x, y, z), Lx, Ly, Lz), ...]` (as for `multi_dielectric_box3d`) with
permittivities `epses` (`Inf` = perfect conductor) in a medium `eps_out`, meshed with
`TensorPanel`s. Boxes may share faces (meshed once, as ε_A | ε_B interfaces), edges or vertices,
but not overlap. Every edge gets the weighted GCQ rule of its own junction
(`edge_gcq_rule(junction_powers(...) .- 1; tol, qmax)`, powers up to `gmax`); equal junctions
share one rule. `nlev`, `n_quad`, `rhs`, `rhs_atol` and `max_depth` are as in
`single_dielectric_box3d_gcq`; quadrants are refined toward true vertices only. `rhs_batch(points, normals)`
(vectors of 3-tuples, returning the values) replaces the pointwise `rhs` by one evaluation per refinement level
over all panels, e.g. an FMM for a volume source. The face
coordinates of all boxes define the grid, so boxes that are offset against each other create
smaller rectangles. Solve with `solve_dielectric_boxes_gcq(mesh, sources)`.

`targets` (3 × M, points off the interface where the potential will be evaluated) switches on a
target-driven refinement after the RHS one: every panel that a target is near (`tensor_isfar` with
`quad_tol = target_tol`, `η_gcq = target_eta`, `gcq_far = target_far`) is split in the near direction(s),
up to `target_depth` levels and down to sides of `target_min_size`. For many targets (e.g. millions of
density points), `target_skin = δ` refines for the targets closest to the interface planes only (per cell of side
δ across each plane, the nearest one on each side): a target left near is corrected exactly by `GCQPotential`. Every target is then far from every
panel, so `GCQPotential` with the same three parameters needs no near corrections for the GCQ panels (their node
rule, i.e. the FMM, is their potential); its Gauss-Legendre × Gauss-Legendre panels still get SCTL's corrections,
whose near radius is not `tensor_isfar`'s, of the size of `target_tol`.

Last, Gauss-Legendre × Gauss-Legendre panels are cut along their longer side to aspect ratio ≤ `GL_MAX_ASPECT`.
"""
function multi_dielectric_box3d_gcq(boxes::AbstractVector, epses::AbstractVector{<:Real},
                                    eps_out::Real = 1.0; nlev::Int = 2, n_quad = 8,
                                    tol::Real = 1e-10, qmax = nothing, gmax::Real = 20.0,
                                    rhs = nothing, rhs_batch = nothing, rhs_atol::Real = 1e-10, max_depth::Int = 30,
                                    targets = nothing, target_tol::Real = 1e-10, target_eta::Real = 2.0,
                                    target_far::Symbol = :direction, target_depth::Int = 20, target_min_size::Real = 0.0,
                                    target_skin::Real = 0.0)
    length(boxes) == length(epses) || throw(ArgumentError("boxes and epses must have equal length"))
    isempty(boxes) && throw(ArgumentError("no boxes"))
    isinf(eps_out) && throw(ArgumentError("eps_out must be finite"))
    T = Float64
    epses = Float64.(collect(epses)); eps_out = Float64(eps_out)
    lo = [ntuple(k -> T(b.center[k]) - T((b.Lx, b.Ly, b.Lz)[k]) / 2, 3) for b in boxes]
    hi = [ntuple(k -> T(b.center[k]) + T((b.Lx, b.Ly, b.Lz)[k]) / 2, 3) for b in boxes]
    scale = maximum(maximum(hi[i] .- lo[i]) for i in eachindex(boxes))
    htol = 1e-12 * scale
    # grid coordinates per axis (coordinates within htol merged)
    grid = ntuple(3) do k
        v = sort(vcat([l[k] for l in lo], [h[k] for h in hi]))
        g = [v[1]]
        for x in v[2:end]
            x - g[end] > htol && push!(g, x)
        end
        g
    end
    gidx(k, x) = (i = argmin(abs.(grid[k] .- x)); i)
    ncell = ntuple(k -> length(grid[k]) - 1, 3)
    cellbox = zeros(Int, ncell...)
    for b in eachindex(boxes)
        rg = ntuple(k -> gidx(k, lo[b][k]):gidx(k, hi[b][k])-1, 3)
        all(!isempty, rg) || throw(ArgumentError("box $b has zero size"))
        for I in CartesianIndices(rg)
            cellbox[I] == 0 || throw(ArgumentError("boxes $(cellbox[I]) and $b overlap"))
            cellbox[I] = b
        end
    end
    inside(I) = all(1 <= I[k] <= ncell[k] for k in 1:3)
    boxof(I) = inside(I) ? cellbox[I...] : 0
    mat(I) = (b = boxof(I); b == 0 ? eps_out : epses[b])
    unit(k) = ntuple(i -> i == k ? one(T) : zero(T), 3)
    shift(I, k, d) = ntuple(i -> i == k ? I[i] + d : I[i], 3)

    # clusters: boxes joined by shared faces (union-find; with equal permittivities the shared
    # face is not an interface, but the two boxes still form one body)
    parent = collect(eachindex(boxes))
    root(b) = (while parent[b] != b; parent[b] = parent[parent[b]]; b = parent[b]; end; b)
    for ax in 1:3, I in CartesianIndices(ncell)
        b1 = cellbox[I]; b2 = boxof(shift(Tuple(I), ax, 1))
        (b1 != 0 && b2 != 0 && b1 != b2) || continue
        parent[root(b1)] = root(b2)
    end
    roots = unique(root.(eachindex(boxes)))
    clusters = [[b for b in eachindex(boxes) if root(b) == r] for r in roots]
    clusterof = Dict(b => c for (c, cl) in enumerate(clusters) for b in cl)

    # GCQ rules: one per distinct junction (sectors up to rotation and reflection)
    glcache = Dict{Int, TensorRule1D}()
    glk(k) = (n = n_quad isa Integer ? n_quad : n_quad(k); get!(() -> gl_tensor_rule(n), glcache, n))
    rules = Dict{Any, TensorRule1D}()
    edgecache = Dict{Tuple{Int, NTuple{3, Int}}, Union{Nothing, TensorRule1D}}()
    # rule across the grid edge along axis e from grid vertex V to V + unit(e); nothing if smooth
    function edgerule(e, V)
        get!(edgecache, (e, V)) do
            a, b = mod1(e + 1, 3), mod1(e + 2, 3)
            c0 = ntuple(i -> i == e ? V[e] : V[i] - 1, 3)                 # cell at (-a, -b)
            # quadrants counter-clockwise in the (a, b) plane, from (+a, +b)
            q = [mat(shift(shift(c0, a, da), b, db)) for (da, db) in ((1, 1), (0, 1), (0, 0), (1, 0))]
            k0 = findfirst(k -> q[k] != q[mod1(k - 1, 4)], 1:4)
            k0 === nothing && return nothing
            angs = Float64[]; es = Float64[]
            for x in circshift(q, 1 - k0)
                if !isempty(es) && es[end] == x
                    angs[end] += pi / 2
                else
                    push!(angs, pi / 2); push!(es, x)
                end
            end
            length(es) == 2 && angs[1] ≈ pi && return nothing            # straight interface
            n = length(es)
            sec = [(round(angs[j], digits = 12), es[j]) for j in 1:n]
            key = minimum(vcat([circshift(sec, j) for j in 0:n-1], [circshift(reverse(sec), j) for j in 0:n-1]))
            get!(rules, key) do
                gcq_tensor_rule(edge_gcq_rule(_junction_density_powers(angs, es; gmax); tol, qmax))
            end
        end
    end
    # a grid vertex is a true vertex unless the cells around it are invariant along some axis
    function truevertex(V)
        oct(d) = mat(ntuple(i -> V[i] - 1 + d[i], 3))
        ds = [(d1, d2, d3) for d1 in 0:1, d2 in 0:1, d3 in 0:1]
        return !any(k -> all(oct(d) == oct(ntuple(i -> i == k ? 1 : d[i], 3)) for d in ds if d[k] == 0), 1:3)
    end

    planeid(ax, x) = Int(hash((ax, round(x, digits = 12))) >> 2)
    cpanels = [TensorPanel{T}[] for _ in clusters]
    ceps_in = [T[] for _ in clusters]; ceps_out = [T[] for _ in clusters]
    pending = Tuple{Int, T, T, Vector{TensorPanel{T}}, TensorRule1D}[]   # quadrants for the batched RHS refinement
    for ax in 1:3
        t1, t2 = [k for k in 1:3 if k != ax]
        for p in eachindex(grid[ax]), i in 1:ncell[t1], j in 1:ncell[t2]
            Ip = ntuple(k -> k == ax ? p : (k == t1 ? i : j), 3)            # cell on the + side
            Im = shift(Ip, ax, -1)
            mat(Im) == mat(Ip) && continue
            # normal out of the box with the larger index (the exterior has index 0): outward on
            # outer faces, as in single_dielectric_box3d_gcq, so τ has one sign per body.
            # Mixed orientations give an equivalent system whose rows of flipped sign spread the
            # spectrum to both sides of 0 and slow GMRES (and defeat near-row reuse).
            bm, bp = boxof(Im), boxof(Ip)
            sg = bm > bp ? 1 : -1
            nrm = sg .* unit(ax)
            em, ep = sg > 0 ? (mat(Im), mat(Ip)) : (mat(Ip), mat(Im))    # eps_in, eps_out
            c = clusterof[max(bm, bp)]
            fid = planeid(ax, grid[ax][p])
            a1 = (grid[t1][i+1] - grid[t1][i]) / 2
            a2 = (grid[t2][j+1] - grid[t2][j]) / 2
            for s1 in 0:1, s2 in 0:1
                V = ntuple(k -> k == ax ? p : (k == t1 ? i + s1 : j + s2), 3)
                cpt = ntuple(k -> grid[k][V[k]], 3)
                d1 = (1 - 2s1) .* unit(t1)                                 # into the face
                d2 = (1 - 2s2) .* unit(t2)
                g1 = edgerule(t1, ntuple(k -> k == t1 ? i : V[k], 3))      # edge along d1
                g2 = edgerule(t2, ntuple(k -> k == t2 ? j : V[k], 3))      # edge along d2
                nl = truevertex(V) ? nlev : 0
                quad = TensorPanel{T}[]
                b1, b2 = a1, a2
                for k in 1:nl
                    b1, b2 = b1 / 2, b2 / 2
                    gl = glk(k)
                    push!(quad, TensorPanel(nrm, cpt, b1 .* d1, d2, d1, b2, b1, something(g1, gl), gl,
                                            g1 === nothing ? :gl : :edge, fid))
                    push!(quad, TensorPanel(nrm, cpt, b2 .* d2, d1, d2, b1, b2, something(g2, gl), gl,
                                            g2 === nothing ? :gl : :edge, fid))
                    push!(quad, TensorPanel(nrm, cpt, b1 .* d1 .+ b2 .* d2, d1, d2, b1, b2, gl, gl, :gl, fid))
                end
                # corner: u along d1 is measured from the edge along d2, v from the edge along d1
                glc = glk(max(nl, 1))
                push!(quad, TensorPanel(nrm, cpt, (zero(T), zero(T), zero(T)), d1, d2, b1, b2,
                                        something(g2, glc), something(g1, glc),
                                        g1 === nothing && g2 === nothing ? :gl : :corner, fid))
                # a pointwise rhs refines here; rhs_batch and targets refine below, all quadrants together
                if rhs !== nothing && rhs_batch === nothing
                    quad = _gcq_rhs_refine(quad, rhs, T(rhs_atol), max_depth, glc, glc, glc)
                end
                push!(pending, (c, em, ep, quad, glc))
            end
        end
    end
    quads = [q for (_, _, _, q, _) in pending]
    if rhs_batch !== nothing
        quads = _gcq_rhs_refine_batched([(q, glc, glc) for (q, (_, _, _, _, glc)) in zip(quads, pending)],
                                        rhs_batch, T(rhs_atol), max_depth)
    end
    if targets !== nothing
        quads = _gcq_target_refine([(q, glc, glc) for (q, (_, _, _, _, glc)) in zip(quads, pending)], targets;
                                   quad_tol = target_tol, η_gcq = target_eta, gcq_far = target_far,
                                   max_depth = target_depth, min_size = target_min_size, skin = target_skin)
    end
    # GL×GL panels cut to the aspect limit after all refinement, before the per-panel permittivities and the bodies
    quads = map(_split_gl_aspect, quads)
    for ((c, em, ep, _, _), quad) in zip(pending, quads)
        append!(cpanels[c], quad)
        append!(ceps_in[c], fill(em, length(quad)))
        append!(ceps_out[c], fill(ep, length(quad)))
    end
    interface = DielectricInterface(reduce(vcat, cpanels), reduce(vcat, ceps_in), reduce(vcat, ceps_out))
    offs = cumsum([0; [sum(num_points, ps; init = 0) for ps in cpanels]])
    bodies = [offs[c]+1:offs[c+1] for c in eachindex(clusters)]
    return MultiBoxGCQ(interface, bodies, clusters, epses, eps_out, grid, cellbox)
end

# box index of the cell containing x (0 = exterior); error if x lies on an interface
function _multibox_locate(m::MultiBoxGCQ, x; rtol = 1e-10)
    h = rtol * maximum(g[end] - g[1] for g in m.grid)
    ncell = size(m.cellbox)
    boxof(I) = all(1 <= I[k] <= ncell[k] for k in 1:3) ? m.cellbox[I...] : 0
    matof(I) = (b = boxof(I); b == 0 ? m.eps_out : m.epses[b])
    # cells whose closure contains x: on a grid plane, the cells on both sides of it
    cand = ntuple(3) do k
        g = argmin(abs.(m.grid[k] .- x[k]))
        abs(m.grid[k][g] - x[k]) <= h ? (g - 1:g) : (searchsortedlast(m.grid[k], x[k]):searchsortedlast(m.grid[k], x[k]))
    end
    cells = [Tuple(I) for I in CartesianIndices(cand)]
    all(matof(I) == matof(cells[1]) for I in cells) ||
        throw(ArgumentError("source point $x lies on an interface"))
    return maximum(boxof, cells)
end

"""
    source_permittivity(mesh::MultiBoxGCQ, x)

Permittivity at the point `x` (a box's, or `eps_out`); throws if `x` lies on an interface.
"""
function source_permittivity(m::MultiBoxGCQ, x)
    b = _multibox_locate(m, x)
    return b == 0 ? m.eps_out : m.epses[b]
end

"""
    box_charge_constraints(mesh::MultiBoxGCQ, sources)

Total-charge constraints `[(range_c, Q_c), ...]`, one per cluster of face-sharing boxes:
Q_c = Σ_{sources q inside the cluster} q (1/ε_out − 1/ε(q)), the charge inside a closed surface
around the cluster. A source inside a perfect conductor is not allowed.
"""
function box_charge_constraints(m::MultiBoxGCQ, sources)
    srcs = sources isa PointSource ? [sources] : collect(sources)
    Q = zeros(length(m.clusters))
    for s in srcs
        b = _multibox_locate(m, s.point)
        b == 0 && continue
        isinf(m.epses[b]) && throw(ArgumentError("a source inside a perfect conductor is not allowed"))
        c = findfirst(cl -> b in cl, m.clusters)
        Q[c] += s.charge * (1 / m.eps_out - 1 / m.epses[b])
    end
    return [(m.bodies[c], Q[c]) for c in eachindex(m.clusters)]
end
