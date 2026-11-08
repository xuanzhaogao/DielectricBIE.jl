# Polyhedra from OBJ triangle meshes, meshed with CurvedPanels: GCQ edge panels, dyadic grading toward vertices.
#
# Each triangle (i, j, k) is split into three quadrilaterals, one per vertex: vertex, midpoint of one edge, centroid,
# midpoint of the other edge, as the bilinear patch Q(U, V) = c + U A + V B + U V C with the edges at V = 0 (side 3)
# and U = 0 (side 1) and seams to the centroid at U = 1 and V = 1. `CurvedGeometry` of these patches gives every
# half-edge its junction (sector angles, permittivities) and hence its density powers. Each quadrilateral is then
# split like a face quadrant of `single_dielectric_box3d_gcq`: 2×2 toward the vertex `nlev` times, the two children
# touching an edge are edge panels (weighted GCQ across, Gauss-Legendre along), the third is GL×GL, and the last
# vertex child is the corner panel: GCQ × GCQ (`corner = :gcq`, each direction the rule of the edge it is measured
# from, as in the cube) or GL × GL (`corner = :gl`). Offsets from a panel's anchor are formed from parameter differences of the
# bilinear map, so nodes near an edge keep their offsets to full relative precision.

"""
    read_obj(f; snap = Float64[]) -> (V, F)

Vertices (3 × nv) and triangles (3 × nf, 1-based) of an OBJ file. Coordinates within 1e-5 of ±x for x in `snap`
are replaced by ±x (OBJ files often carry irrational coordinates to six digits).
"""
function read_obj(f; snap = Float64[])
    V = Vector{Float64}[]; F = Vector{Int}[]
    for l in eachline(f)
        c = split(l)
        isempty(c) && continue
        if c[1] == "v"
            x = parse.(Float64, c[2:4])
            for s in snap, i in 1:3
                abs(abs(x[i]) - s) < 1e-5 && (x[i] = sign(x[i]) * s)
            end
            push!(V, x)
        elseif c[1] == "f"
            push!(F, [parse(Int, split(t, '/')[1]) for t in c[2:end]])
        end
    end
    all(length.(F) .== 3) || throw(ArgumentError("only triangles are supported"))
    return reduce(hcat, V), reduce(hcat, F)
end

"""
    orient_outward(V, F) -> (F, volume)

Check that the triangles `F` (3 × nf) form a closed, consistently oriented surface (every edge used once in each
direction) and orient them outward; returns the triangles and the enclosed volume.
"""
function orient_outward(V, F)
    E = Dict{Tuple{Int, Int}, Int}()
    for t in axes(F, 2), a in 1:3
        e = (F[a, t], F[mod1(a + 1, 3), t])
        E[e] = get(E, e, 0) + 1
    end
    for (e, n) in E
        n == 1 && get(E, reverse(e), 0) == 1 || throw(ArgumentError("edge $e: the surface is not closed and consistently oriented"))
    end
    vol = sum(dot(V[:, F[1, t]], cross(V[:, F[2, t]], V[:, F[3, t]])) for t in axes(F, 2)) / 6
    return vol > 0 ? (F, vol) : (F[[1, 3, 2], :], -vol)
end

# bilinear quadrilateral c, c + A, c + A + B + C, c + B (A, B, C as tuples), in plain coordinates
struct BilinearQuad
    c::NTuple{3, Float64}
    A::NTuple{3, Float64}
    B::NTuple{3, Float64}
    C::NTuple{3, Float64}
end
(q::BilinearQuad)(U, V) = q.c .+ U .* q.A .+ V .* q.B .+ (U * V) .* q.C
# Q(U, V) - Q(Ua, Va) from the parameter differences
_bilinear_offset(q::BilinearQuad, Ua, Va, dU, dV) = dU .* (q.A .+ (Va + dV) .* q.C) .+ dV .* (q.B .+ Ua .* q.C)

"""
    polyhedron_patches(V, F; eps_in, eps_out = 1.0) -> (patches, quads)

Three `SurfacePatch`es per outward-oriented triangle (see the file header) and their `BilinearQuad`s.
"""
function polyhedron_patches(V, F; eps_in::Real, eps_out::Real = 1.0)
    quads = BilinearQuad[]
    for t in axes(F, 2), a in 1:3
        i, j, k = F[a, t], F[mod1(a + 1, 3), t], F[mod1(a + 2, 3), t]
        c = V[:, i]; m1 = (V[:, i] .+ V[:, j]) ./ 2; m2 = (V[:, i] .+ V[:, k]) ./ 2
        g = (V[:, i] .+ V[:, j] .+ V[:, k]) ./ 3
        A = m1 .- c; B = m2 .- c
        push!(quads, BilinearQuad(Tuple(c), Tuple(A), Tuple(B), Tuple(g .- c .- A .- B)))
    end
    # x_U × x_V ∝ (V_j - V_i) × (V_k - V_i): outward for an outward triangle
    patches = [SurfacePatch(q; eps_in, eps_out) for q in quads]
    return patches, quads
end

"""
    polyhedron_quad_rects(nlev) -> [(U0, U1, V0, V1, s)]

The panels of one quadrilateral of `polyhedron_patches` in its parameters, graded toward the vertex (0, 0): `s` = 3 an edge panel on V = 0,
1 on U = 0, 0 GL × GL, -1 the corner panel.
"""
function polyhedron_quad_rects(nlev::Int)
    rects = Tuple{Float64, Float64, Float64, Float64, Int}[]
    a = 1.0
    for _ in 1:nlev
        b = a / 2
        push!(rects, (b, a, 0.0, b, 3), (0.0, b, b, a, 1), (b, a, b, a, 0))
        a = b
    end
    push!(rects, (0.0, a, 0.0, a, -1))
    return rects
end

"""
    polyhedron_mesh(g, quads; nlev = 4, n_quad = 10, n_corner = n_quad, tol = 1e-10, gmax = 20.0, edge = :gcq,
                    corner = :gl, ninterp = 14, grid_tol = :auto, density_tol = nothing,
                    floor = 0.0)

Mesh the patches of `polyhedron_patches` (the file header). `edge = :gl` puts Gauss-Legendre on the edge panels
too (an h-graded reference, with `nlev` levels toward the vertices only: use it with care). `corner = :gcq` gives
the corner panels the GCQ rule in both directions (`n_corner` is then unused; needs `PlanarProductNearQuad`).
`ninterp` is the GCQ rules' interpolation points per fine segment (`edge_gcq_rule`; the nodes are unchanged);
`grid_tol` (default `:auto`: `tol`, the fine grid resolves the family to `tol`, study 73; `density_tol / 10` with
`density_tol`; `nothing` the bare dyadic grid, `edge_gcq_rule`) and `density_tol` set the component-resolved training
and the certified rank, and `floor` keeps the GCQ nodes
at s ≥ floor (`edge_gcq_rule`; these change the rules' nodes, hence the unknowns). Returns
(interface, rules, powers): the `DielectricInterface` of `CurvedPanel`s, the GCQ rule of every edge curve and
the density powers of every curve (rules are shared between curves of equal powers).
"""
function polyhedron_mesh(g::CurvedGeometry, quads; nlev::Int = 4, n_quad::Int = 10, n_corner::Int = n_quad,
                         tol::Real = 1e-10, gmax::Real = 20.0, edge::Symbol = :gcq, corner::Symbol = :gl,
                         ninterp::Int = 14, grid_tol::Union{Nothing, Real, Symbol} = :auto,
                         density_tol::Union{Nothing, Real} = nothing, floor::Real = 0.0)
    corner in (:gcq, :gl) || throw(ArgumentError("corner must be :gcq or :gl"))
    powers = [c.smooth ? Float64[] : _junction_density_powers(c.angles, c.eps; gmax) for c in g.curves]
    rgl = gl_tensor_rule(n_quad); rcorner = gl_tensor_rule(n_corner)
    cache = Dict{Vector{Float64}, TensorRule1D}()
    grid_tol === :auto && (grid_tol = density_tol === nothing ? Float64(tol) : nothing)
    grid_tol isa Symbol && throw(ArgumentError("grid_tol must be a positive number, nothing or :auto, got :$grid_tol"))
    rule(i) = edge == :gl ? rgl : get!(() -> gcq_tensor_rule(edge_gcq_rule(powers[i]; tol, ninterp, grid_tol, density_tol, floor)),
                                       cache, round.(powers[i], digits = 9))
    panels = CurvedPanel[]; ein = Float64[]; eout = Float64[]
    for (k, P) in enumerate(g.patches)
        q = quads[k]
        sc = g.side_curve[:, k]
        g.curves[sc[1]].smooth || g.curves[sc[3]].smooth ||
            (g.curves[sc[2]].smooth && g.curves[sc[4]].smooth) ||
            throw(ArgumentError("patch $k: expected edges at U = 0, V = 0 and seams at U = 1, V = 1"))
        for (U0, U1, V0, V1, s) in polyhedron_quad_rects(nlev)
            dU = U1 - U0; dV = V1 - V0
            xU(V) = q.A .+ V .* q.C; xV(U) = q.B .+ U .* q.C
            if s == 3            # edge V = 0 at the panel's u = 0: U = U0 + v dU, V = u dV
                anchor = q(U0, 0.0)
                lmap = (u, v) -> _bilinear_offset(q, U0, 0.0, v * dU, u * dV)
                dmap = (u, v) -> (dV .* xV(U0 + v * dU), dU .* xU(u * dV))
                ru, rv, kind, flip = rule(sc[3]), rgl, :edge, true
            elseif s == 1        # edge U = 0 at the panel's u = 0: U = u dU, V = V0 + v dV
                anchor = q(0.0, V0)
                lmap = (u, v) -> _bilinear_offset(q, 0.0, V0, u * dU, v * dV)
                dmap = (u, v) -> (dU .* xU(V0 + v * dV), dV .* xV(u * dU))
                ru, rv, kind, flip = rule(sc[1]), rgl, :edge, false
            else                 # GL × GL (s = -1: the corner panel at the vertex)
                anchor = q(U0, V0)
                lmap = (u, v) -> _bilinear_offset(q, U0, V0, u * dU, v * dV)
                dmap = (u, v) -> (dU .* xU(V0 + v * dV), dV .* xV(U0 + u * dU))
                if s == -1 && corner == :gcq && edge == :gcq
                    # u is measured from the edge U = 0, v from the edge V = 0
                    ru, rv, kind, flip = rule(sc[1]), rule(sc[3]), :corner, false
                else
                    r = s == -1 ? rcorner : rgl
                    ru, rv, kind, flip = r, r, :gl, false
                end
            end
            edge == :gl && (kind = :gl)
            push!(panels, CurvedPanel(anchor, lmap, ru, rv; kind, flip, dmap))
            push!(ein, P.eps_in); push!(eout, P.eps_out)
        end
    end
    itf = DielectricInterface{CurvedPanel, Float64}(panels, ein, eout)
    return itf, cache, powers
end
