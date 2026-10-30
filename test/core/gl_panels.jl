using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "shape", "lens_patches.jl"))

# every GL×GL panel within the aspect limit (with the relative slack of check_gl_aspect)
gl_aspect_ok(panels) = all(panel_aspect(p) <= 4 * (1 + 1e-8) for p in panels if is_gl_panel(p))
box_area(L) = 2 * (L[1] * L[2] + L[2] * L[3] + L[3] * L[1])

@testset "is_gl_panel" begin
    itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 4.0, 1.0; nlev = 2)
    @test all(is_gl_panel(p) == (p.kind == :gl) for p in itf.panels)
    @test count(is_gl_panel, itf.panels) == 6 * 4 * 2                     # one per face, quadrant and level
    itc = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 4.0, 1.0; nlev = 2, corner = :gl)
    @test all(is_gl_panel, [p for p in itc.panels if p.kind == :corner])
    @test all(is_gl_panel(p) == (p.kind != :edge) for p in itc.panels)
    flat = single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.4, 4.0, 1.0, Float64)
    @test all(is_gl_panel, flat.panels)
    # curved: the GCQ edge panels are not GL, the others are; with edge = :gl every panel is
    l = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(l; nlev = 2, n_quad = 8)
    @test all(is_gl_panel(p) == (p.kind == :gl) for p in m.interface.panels)
    @test all(is_gl_panel, curved_gcq_mesh(l; nlev = 2, n_quad = 8, edge = :gl).interface.panels)
end

@testset "aspect guard" begin
    @test GL_MAX_ASPECT === 4.0
    r = BI.gl_tensor_rule(8)
    rg = BI.gcq_tensor_rule(edge_gcq_rule(4.0, 1.0; tol = 1e-8))
    tp(Lu, Lv; ru = r) = TensorPanel((0.0, 0.0, 1.0), (1.0, 2.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                     Lu, Lv, ru, r, :gl, 7)
    @test panel_aspect(tp(4.0, 1.0)) == 4.0 && panel_aspect(tp(1.0, 4.0)) == 4.0
    @test check_gl_aspect(tp(4.0, 1.0), 1) === nothing
    @test check_gl_aspect(tp(4.0 * (1 + 1e-9), 1.0), 1) === nothing
    @test_throws ArgumentError check_gl_aspect(tp(4.1, 1.0), 1)
    @test_throws ArgumentError check_gl_aspect(tp(1.0, 4.1), 1)
    err = try check_gl_aspect(tp(4.1, 1.0), 12); nothing; catch e; e; end
    @test err isa ArgumentError &&
          err.msg == "panel 12: GL×GL aspect ratio $(4.1 / 1.0) exceeds GL_MAX_ASPECT = 4 (large aspect ratios are not supported yet)"
    @test check_gl_aspect(tp(10.0, 1.0; ru = rg), 1) === nothing             # GCQ panels have no limit
    # flat panels: the corners' sides; curved panels: mean chords (not arc lengths)
    ns, ws = BI.gausslegendre(4)
    fp = BI.rect_panel3d_discretize((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 0.5, 0.0), (0.0, 0.5, 0.0), ns, ws, (0.0, 0.0, 1.0))
    @test panel_aspect(fp) ≈ 6
    @test_throws ArgumentError check_gl_aspect(fp, 1)
    @test panel_aspect(CurvedPanel((0.0, 0.0, 0.0), (u, v) -> promote(2u, v / 2, 0.0), r, r)) ≈ 4
    arc = CurvedPanel((1.0, 0.0, 0.0), (u, v) -> promote(cos(π / 2 * u) - 1, sin(π / 2 * u), v / 4), r, r)
    @test panel_aspect(arc) ≈ sqrt(2) / 0.25
    @test_throws ArgumentError check_gl_aspect(arc, 1)

    # _split_gl_aspect: a GL panel above the limit in ceil(aspect / 4) equal GL pieces along its longer side
    p10 = tp(10.0, 1.0); p9 = tp(1.0, 9.0); p4 = tp(4.0, 1.0); pg = tp(10.0, 1.0; ru = rg)
    pc = TensorPanel(p10.normal, p10.anchor, p10.o, p10.eu, p10.ev, 10.0, 1.0, r, r, :corner, 7)   # GL corner
    q = BI._split_gl_aspect([p10, p9, p4, pg, pc])
    @test q isa Vector{TensorPanel{Float64}} && length(q) == 3 + 3 + 1 + 1 + 3
    @test q[7] === p4 && q[8] === pg                                       # within the limit, or GCQ: unchanged
    for (k, p) in ((1, p10), (4, p9), (9, pc))
        pieces = q[k:k+2]
        @test all(x -> x.kind == :gl && x.normal == p.normal && x.anchor == p.anchor && x.plane == p.plane &&
                       x.ru === p.ru && x.rv === p.rv && x.eu == p.eu && x.ev == p.ev, pieces)
        @test all(x -> panel_aspect(x) ≈ 10 / 3 || panel_aspect(x) ≈ 3, pieces)
        # the pieces tile the panel
        @test sum(sum(x.weights) for x in pieces) ≈ sum(p.weights) rtol = 1e-14
        @test pieces[1].corners[1] == p.corners[1] && collect(pieces[3].corners[3]) ≈ collect(p.corners[3])
        @test all(collect(pieces[i].corners[p.Lu > p.Lv ? 2 : 4]) ≈ collect(pieces[i+1].corners[1]) for i in 1:2)
    end
end

@testset "meshers keep GL aspect ≤ 4" begin
    # lens: without the split its thin GL panels next to the edge panels reach aspect 17, 71 (nlev = 4, 6) and,
    # at the rim itself with edge = :gl, 286, in 88, 120 and 152 panels
    l = CurvedGeometry(lens_patches(1.0, π / 3, π / 3, 4.0, 4.0, 1.0))
    for (kw, n0) in (((nlev = 4,), 88), ((nlev = 6,), 120), ((nlev = 8, edge = :gl), 152))
        m = curved_gcq_mesh(l; kw...)
        ps = m.interface.panels
        @test gl_aspect_ok(ps)
        @test length(ps) > n0
        @test count(p -> !is_gl_panel(p), ps) == (haskey(kw, :edge) ? 0 : 8 * 2)   # GCQ panels are never split
        # the per-panel data follow the split panels: patch, permittivities, nodes = the patch map at panel_param
        @test length(m.panel_patch) == length(m.panel_param) == length(m.interface.eps_in) == length(ps)
        @test all(m.interface.eps_in[k] == l.patches[m.panel_patch[k]].eps_in &&
                  m.interface.eps_out[k] == l.patches[m.panel_patch[k]].eps_out for k in eachindex(ps))
        @test all(all(isapprox.(p.points[i], l.patches[m.panel_patch[k]].map(m.panel_param[k](u, v)...); atol = 1e-13))
                  for (k, p) in enumerate(ps) for (i, (u, v)) in enumerate((u, v) for u in p.ru.x for v in p.rv.x))
        offs = cumsum([0; [BI.num_points(p) for p in ps]])
        @test all(offs[k]+1:offs[k+1] ⊆ m.ranges[m.panel_patch[k]] for k in eachindex(ps))
        @test first(m.ranges[1]) == 1 && last(m.ranges[end]) == offs[end] &&
              all(last(m.ranges[i]) + 1 == first(m.ranges[i+1]) for i in 1:length(m.ranges)-1)
    end

    # edge = :gl: the GL panels at the rim (U = 0 of the side charts) are split along it; every piece keeps the rim
    # at its u = 0, and its offsets from an anchor on the rim stay cancellation-free (u = 1e-12 vs BigFloat)
    m = curved_gcq_mesh(l; nlev = 8, edge = :gl)
    for k in (1, 2, 3, 4, 6, 7, 8, 9)
        rim = [j for j in findall(==(k), m.panel_patch) if m.panel_param[j](0.0, 0.5)[1] == 0]
        @test length(rim) > 2                                                # m = 2 panels along before the split
        @test all(j -> m.panel_param[j](0.0, 0.3)[1] == 0 && m.panel_param[j](1.0, 0.3)[1] ≈ 2.0^-8, rim)
        sort!(rim; by = j -> m.panel_param[j](0.0, 0.0)[2])
        @test m.panel_param[rim[1]](0.0, 0.0)[2] == 0 && m.panel_param[rim[end]](0.0, 1.0)[2] ≈ 1
        @test all(m.panel_param[rim[i]](0.0, 1.0)[2] ≈ m.panel_param[rim[i+1]](0.0, 0.0)[2] for i in 1:length(rim)-1)
        j = rim[2]; p = m.interface.panels[j]; pa = m.panel_param[j]
        U, V = pa(1e-12, 0.3); Ua, Va = pa(0.0, 0.0)
        o = BI.curved_lpoint(p, 1e-12, 0.3)
        ob = setprecision(BigFloat, 256) do
            f = l.patches[k].map
            Float64.(f(big(U), big(V)) .- f(big(Ua), big(Va)))
        end
        @test all(isapprox.(o, ob; rtol = 1e-13, atol = 1e-30))
    end

    # curved default order: 16-point Gauss-Legendre in every GL direction
    md = curved_gcq_mesh(l)
    @test all(p -> length(p.rv.x) == 16 && (BI.is_gcq(p.ru) || length(p.ru.x) == 16), md.interface.panels)

    # boxes: the 4 × 1 × 1 box is at the limit (aspect 4); the 5 × 1 × 1 box has 32 GL panels of aspect 5 on its
    # long faces (and 16 GL corner panels with corner = :gl), each split in two
    rhs = (p, n) -> exp(-20 * sum(abs2, p .- (1.9, 0.0, 0.0)))
    for (L, kw, np) in (((4.0, 1.0, 1.0), (nlev = 2,), 168), ((4.0, 1.0, 1.0), (nlev = 2, rhs, rhs_atol = 1e-8), 296),
                        ((5.0, 1.0, 1.0), (nlev = 2,), 168 + 32), ((5.0, 1.0, 1.0), (nlev = 2, corner = :gl), 168 + 32 + 16))
        itf = single_dielectric_box3d_gcq(L..., 4.0, 1.0; kw...)
        @test gl_aspect_ok(itf.panels)
        @test length(itf.panels) == np == length(itf.eps_in) == length(itf.eps_out)
        @test sum(BI.all_weights(itf)) ≈ box_area(L) rtol = 1e-12
        haskey(kw, :rhs) || @test box_node_orbits(itf, L...) isa Tuple       # still symmetric to the last bit
    end

    # two 5 × 1 × 1 boxes sharing the face y = 1/2 (308 panels without the split): permittivities and bodies follow
    # the split panels
    bx = [(center = (0.0, 0.0, 0.0), Lx = 5.0, Ly = 1.0, Lz = 1.0), (center = (0.0, 1.0, 0.0), Lx = 5.0, Ly = 1.0, Lz = 1.0)]
    mb = multi_dielectric_box3d_gcq(bx, [2.0, 10.0]; nlev = 2, n_quad = 6)
    ps = mb.interface.panels
    @test gl_aspect_ok(ps) && length(ps) > 308
    @test length(mb.interface.eps_in) == length(mb.interface.eps_out) == length(ps)
    @test mb.bodies == [1:BI.num_points(mb)]
    @test sum(BI.all_weights(mb.interface)) ≈ 2 * box_area((5.0, 1.0, 1.0)) - 5 rtol = 1e-12   # the shared face once
    for (k, p) in enumerate(ps)
        c = p.anchor .+ BI.tensor_lpoint(p, 0.5, 0.5)
        @test mb.interface.eps_in[k] == source_permittivity(mb, c .- 1e-3 .* p.normal)
        @test mb.interface.eps_out[k] == source_permittivity(mb, c .+ 1e-3 .* p.normal)
    end
end

@testset "split meshes keep their area" begin
    # the unit sphere graded toward its equator with Gauss-Legendre there: thin GL panels at the equator, split
    s = CurvedGeometry(lens_patches(1.0, π / 2, π / 2, 4.0, 4.0, 1.0))
    m = curved_gcq_mesh(s; grade = c -> abs(c.midpoint[3]) < 1e-12, edge = :gl, nlev = 8)
    @test length(m.interface.panels) > 152 && gl_aspect_ok(m.interface.panels)
    @test sum(BI.all_weights(m.interface)) ≈ 4π rtol = 1e-10
    # GCQ boxes: the exact surface area
    for L in ((4.0, 1.0, 1.0), (5.0, 1.0, 1.0), (1.0, 7.0, 0.5))
        itf = single_dielectric_box3d_gcq(L..., 4.0, 1.0; nlev = 2)
        @test gl_aspect_ok(itf.panels)
        @test sum(BI.all_weights(itf)) ≈ box_area(L) rtol = 1e-12
    end
end
