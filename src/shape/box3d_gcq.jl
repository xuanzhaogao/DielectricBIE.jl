# Single dielectric box with GCQ edge panels.
#
# Each face is split into four quadrants, one per box vertex. Each quadrant
# [0,a1]×[0,a2] (local axes along the two edges leaving the vertex) is split
# 2×2 toward the vertex `nlev` times; at each level the two children touching
# one edge are edge panels (weighted GCQ across the edge × Gauss-Legendre along
# it), the fourth child is Gauss-Legendre × Gauss-Legendre, and the final
# vertex child is the corner panel: GCQ × GCQ (`corner = :gcq`, the edge rule in
# both directions) or GL × GL (`corner = :gl`). There is no refinement along
# the edges: the GCQ rule resolves the edge singularity on panels of any length.
#
# With `rhs` given, panels are then split until their tensor rule interpolates
# rhs(point, normal) to `rhs_atol` (absolute, on a 10 × 10 grid of interior test
# points, as in `single_dielectric_box3d_rhs_adaptive`): GL panels split 2×2; an
# edge panel splits along the edge (two edge panels) and/or across it (an edge
# panel next to the edge and a GL panel beyond it), in the direction that is
# unresolved; a corner panel splits like one more vertex level.
#
# Each edge may have its own rule (`edge_rule`), e.g. when several boxes meet along
# an edge; a GCQ × GCQ corner panel then takes, in each direction, the rule of the
# edge it is measured from. Plane ids are geometric (axis, coordinate), so faces of
# different boxes lying in one plane are recognised as coplanar (D^T vanishes).

"""
    single_dielectric_box3d_gcq(Lx, Ly, Lz, eps_in, eps_out; nlev=2, n_quad=8,
                                corner=:gcq, tol=1e-10, rule=nothing, corner_rule=nothing,
                                edge_rule=nothing, center=(0,0,0),
                                rhs=nothing, rhs_atol=1e-10, max_depth=30)

Box [-Lx/2, Lx/2] × [-Ly/2, Ly/2] × [-Lz/2, Lz/2] with permittivity `eps_in`
in a medium `eps_out`, discretized with `TensorPanel`s (see the file header).
`n_quad` is the Gauss-Legendre order: an Int, or a function `level -> order`
for the panels created at refinement level 1..nlev (the corner panel has level
nlev). `rule` may pass a prebuilt `EdgeGCQRule`; by default one is built for
the right-angle edge with `edge_gcq_rule(eps_in, eps_out; tol)`. A GCQ × GCQ
corner panel uses `corner_rule` if given, else `rule`. `edge_rule(m, axis)`, if
given, returns the `EdgeGCQRule` for the edge with midpoint `m` along coordinate
`axis` (or `nothing` for the default); the box is centred at `center`. `rhs(p, n)`
switches on the RHS-adaptive splitting to `rhs_atol`.
"""
function single_dielectric_box3d_gcq(Lx::T, Ly::T, Lz::T, eps_in::T, eps_out::T;
                                     nlev::Int = 2, n_quad = 8, corner::Symbol = :gcq,
                                     tol::Real = 1e-10, rule = nothing, corner_rule = nothing,
                                     edge_rule = nothing, center = (zero(T), zero(T), zero(T)),
                                     rhs = nothing, rhs_atol::Real = 1e-10,
                                     max_depth::Int = 30) where T
    corner in (:gcq, :gl) || throw(ArgumentError("corner must be :gcq or :gl"))
    er = rule === nothing ? edge_gcq_rule(eps_in, eps_out; tol) : rule
    gq = gcq_tensor_rule(er)
    glcache = Dict{Int, TensorRule1D}()
    glk(k) = (n = n_quad isa Integer ? n_quad : n_quad(k); get!(() -> gl_tensor_rule(n), glcache, n))
    rc = corner == :gl ? glk(max(nlev, 1)) : corner_rule === nothing ? gq : gcq_tensor_rule(corner_rule)
    h = (Lx / 2, Ly / 2, Lz / 2)
    unit(k) = ntuple(i -> i == k ? one(T) : zero(T), 3)
    # tensor rule of the edge leaving vertex c along d (one TensorRule1D per EdgeGCQRule,
    # so near rows of equal rules are shared)
    trules = IdDict{Any, TensorRule1D}()
    erule(c, d) = begin
        edge_rule === nothing && return gq
        ax = findfirst(!=(0), d)
        r = edge_rule(c .+ h[ax] .* d, ax)
        r === nothing ? gq : get!(() -> gcq_tensor_rule(r), trules, r)
    end
    faces = [(3, +1), (3, -1), (1, +1), (1, -1), (2, +1), (2, -1)]    # (axis, sign)
    planeid(ax, x) = Int(hash((ax, round(x, digits = 12))) >> 2)
    panels = TensorPanel{T}[]
    for (ax, sg) in faces
        nrm = sg .* unit(ax)
        fid = planeid(ax, center[ax] + sg * h[ax])
        t1, t2 = [k for k in 1:3 if k != ax]
        for s1 in (-1, 1), s2 in (-1, 1)
            c = ntuple(i -> center[i] + (i == ax ? sg * h[ax] : (i == t1 ? s1 * h[t1] : s2 * h[t2])), 3)
            d1 = (-s1) .* unit(t1)            # along the edge leaving c, into the face
            d2 = (-s2) .* unit(t2)
            g1, g2 = erule(c, d1), erule(c, d2)   # rules across the edges along d1, d2
            quad = TensorPanel{T}[]
            a1, a2 = h[t1], h[t2]
            for k in 1:nlev
                b1, b2 = a1 / 2, a2 / 2
                gl = glk(k)
                # child touching the edge along d1 (β = 0): GCQ across = d2
                push!(quad, TensorPanel(nrm, c, b1 .* d1, d2, d1, b2, b1, g1, gl, :edge, fid))
                # child touching the edge along d2 (α = 0): GCQ across = d1
                push!(quad, TensorPanel(nrm, c, b2 .* d2, d1, d2, b1, b2, g2, gl, :edge, fid))
                # interior child
                push!(quad, TensorPanel(nrm, c, b1 .* d1 .+ b2 .* d2, d1, d2, b1, b2, gl, gl, :gl, fid))
                a1, a2 = b1, b2
            end
            # corner: u along d1 is measured from the edge along d2, v from the edge along d1
            cu, cv = corner == :gl ? (rc, rc) : (corner_rule === nothing ? (g2, g1) : (rc, rc))
            push!(quad, TensorPanel(nrm, c, (zero(T), zero(T), zero(T)), d1, d2, a1, a2, cu, cv, :corner, fid))
            append!(panels, rhs === nothing ? quad :
                    _gcq_rhs_refine(quad, rhs, T(rhs_atol), max_depth, gq, rc, glk(max(nlev, 1))))
        end
    end
    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end

const _GCQ_TEST_PTS = collect(((1:10) .- 0.5) ./ 10)

# max interpolation error of rhs on panel p: total, in u only, in v only
function _gcq_rhs_errors(p::TensorPanel, rhs)
    nu, nv = length(p.ru.x), length(p.rv.x)
    lp(u, v) = p.anchor .+ tensor_lpoint(p, u, v)
    F = [rhs(lp(u, v), p.normal) for u in p.ru.x, v in p.rv.x]
    Eu = tensor_rule_interp(p.ru, _GCQ_TEST_PTS); Ev = tensor_rule_interp(p.rv, _GCQ_TEST_PTS)
    Ftu = [rhs(lp(u, v), p.normal) for u in _GCQ_TEST_PTS, v in p.rv.x]      # exact in v (nodes)
    Ftv = [rhs(lp(u, v), p.normal) for u in p.ru.x, v in _GCQ_TEST_PTS]      # exact in u (nodes)
    Ft = [rhs(lp(u, v), p.normal) for u in _GCQ_TEST_PTS, v in _GCQ_TEST_PTS]
    eu = maximum(abs, Eu * F .- Ftu)
    ev = maximum(abs, F * Ev' .- Ftv)
    e = maximum(abs, Eu * F * Ev' .- Ft)
    return e, eu, ev
end

function _gcq_rhs_refine(panels::Vector{TensorPanel{T}}, rhs, atol::T, max_depth::Int,
                         gq::TensorRule1D, rc::TensorRule1D, glc::TensorRule1D) where T
    out = TensorPanel{T}[]
    stack = [(p, 0) for p in panels]
    while !isempty(stack)
        p, depth = pop!(stack)
        e, eu, ev = _gcq_rhs_errors(p, rhs)
        if e <= atol || depth >= max_depth
            push!(out, p); continue
        end
        hu, hv = p.Lu / 2, p.Lv / 2
        mk(o, eu, ev, Lu, Lv, ru, rv, kind) = TensorPanel(p.normal, p.anchor, o, eu, ev, Lu, Lv, ru, rv, kind, p.plane)
        kids = TensorPanel{T}[]
        if p.kind == :gl
            for (du, dv) in ((0, 0), (1, 0), (0, 1), (1, 1))
                push!(kids, mk(p.o .+ (du * hu) .* p.eu .+ (dv * hv) .* p.ev, p.eu, p.ev, hu, hv, p.ru, p.rv, :gl))
            end
        elseif p.kind == :edge
            # u = across the edge (GCQ, edge at u = 0), v = along it (GL)
            split_u = eu > atol / 2; split_v = ev > atol / 2 || !split_u
            vs = split_v ? ((0, hv), (1, hv)) : ((0, p.Lv),)
            for (dv, Lv) in vs
                ov = p.o .+ (dv * hv) .* p.ev
                if split_u
                    push!(kids, mk(ov, p.eu, p.ev, hu, Lv, p.ru, p.rv, :edge))
                    push!(kids, mk(ov .+ hu .* p.eu, p.eu, p.ev, hu, Lv, p.rv, p.rv, :gl))
                else
                    push!(kids, mk(ov, p.eu, p.ev, p.Lu, Lv, p.ru, p.rv, :edge))
                end
            end
        else
            # corner: one more vertex level (axes d1 = eu, d2 = ev from the vertex)
            # the child along eu touches the edge along eu (rule p.rv), the child along
            # ev the edge along ev (rule p.ru); a GL × GL corner falls back to gq
            push!(kids, mk(p.o .+ hu .* p.eu, p.ev, p.eu, hv, hu, is_gcq(p.rv) ? p.rv : gq, glc, :edge))
            push!(kids, mk(p.o .+ hv .* p.ev, p.eu, p.ev, hu, hv, is_gcq(p.ru) ? p.ru : gq, glc, :edge))
            push!(kids, mk(p.o .+ hu .* p.eu .+ hv .* p.ev, p.eu, p.ev, hu, hv, glc, glc, :gl))
            push!(kids, mk(p.o, p.eu, p.ev, hu, hv, p.ru, p.rv, :corner))
        end
        append!(stack, [(k, depth + 1) for k in kids])
    end
    return out
end
