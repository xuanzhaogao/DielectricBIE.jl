using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra, SparseArrays
using Test

# the regular octahedron conv(±e_i): 8 triangles, 12 edges with dihedral angle arccos(-1/3)
const OCT_V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
# outward: (i, j, k) is positively oriented when the signs of ±x, ±y, ±z multiply to +1
const OCT_F = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                           for i in (1, 2), j in (3, 4), k in (5, 6)])

@testset "OBJ input and orientation" begin
    f = tempname() * ".obj"
    open(f, "w") do io
        for x in eachcol(OCT_V)
            println(io, "v ", join(x .* 0.618034, " "))         # six-digit golden ratio, snapped below
        end
        for t in eachcol(OCT_F)
            println(io, "f ", join(["$i/$i" for i in t], " "))
        end
    end
    V, F = read_obj(f; snap = [(sqrt(5) - 1) / 2])
    @test V == OCT_V .* ((sqrt(5) - 1) / 2)
    F2, vol = orient_outward(V, F)
    @test vol ≈ 4 / 3 * ((sqrt(5) - 1) / 2)^3
    @test orient_outward(V, F2[[1, 3, 2], :])[1] == F2
    @test_throws ArgumentError orient_outward(V, F[:, 1:7])            # not closed
end

@testset "polyhedron mesh and planar near quadrature" begin
    F, _ = orient_outward(OCT_V, OCT_F)
    patches, quads = polyhedron_patches(OCT_V, F; eps_in = 2.0)
    g = CurvedGeometry(patches)
    @test length(polyhedron_quad_rects(2)) == 7
    itf, rules, powers = polyhedron_mesh(g, quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    @test length(itf.panels) == 8 * 3 * 4
    @test sum(DB.all_weights(itf)) ≈ 8 * sqrt(3) / 2 rtol = 1e-7         # area (8 triangles of side √2), to the rule's tol
    @test length(rules) == 1                                              # one junction for all 12 edges
    # rows reused between congruent panels agree with rows computed per panel: on the 6 corner panels of two
    # triangles sharing an edge (the three quadrilaterals of an equilateral triangle are congruent)
    m0, _, _ = polyhedron_mesh(g, quads; nlev = 0, tol = 1e-4, corner = :gcq)
    itf0 = DB.DielectricInterface{CurvedPanel, Float64}(m0.panels[1:6], m0.eps_in[1:6], m0.eps_out[1:6])
    C1 = DB.near_corrections(PlanarProductNearQuad(), itf0)
    C0 = DB.near_corrections(PlanarProductNearQuad(symmetry = false), itf0)
    @test norm(C1 - C0, Inf) < 1e-12 * norm(C0, Inf)
    # a charge inside: the total induced charge is 1 - 1/ε
    src = PointSource((0.1, 0.05, 0.02), 1.0)
    A = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = PlanarProductNearQuad())
    σ, st = solve_dielectric_box3d_gcq(A, rhs_dielectric_box3d(itf, src, 2.0), itf; rtol = 1e-12)
    @test st.solved
    @test abs(dot(DB.all_weights(itf), σ) / 0.5 - 1) < 1e-4
    # single-layer potential: continuous across a face, and the node rule far away
    c = sum(OCT_V[:, F[:, 1]]; dims = 2)[:] ./ 3; n = normalize(c)
    u = planar_slp_potential(itf, σ, hcat(c .+ 1e-8 .* n, c .- 1e-8 .* n, [0.0, 0.0, 20.0]))
    @test abs(u[1] - u[2]) < 1e-7 * abs(u[1])
    X = reduce(vcat, [p.points for p in itf.panels]); w = DB.all_weights(itf)
    @test u[3] ≈ sum(w[j] * σ[j] / (4π * norm(collect(X[j]) .- [0.0, 0.0, 20.0])) for j in eachindex(w)) rtol = 1e-13
end

# ∫ dA / (4π |x0 - y|) over the convex polygon c (vertices in order) for x0 in its plane (n its normal): signed triangles
# (x0, a, b), each h (asinh(t_b / h) - asinh(t_a / h)), h the distance of x0 to the line ab, t along it from the foot
function polygon_slp(c, x0, n)
    s = 0.0
    for k in eachindex(c)
        a = c[k]; b = c[mod1(k + 1, length(c))]
        e = (b - a) / norm(b - a)
        f = a + dot(x0 - a, e) * e
        h = norm(x0 - f)
        h == 0 && continue
        s += sign(dot(cross(a - x0, b - x0), n)) * h * (asinh(dot(b - f, e) / h) - asinh(dot(a - f, e) / h))
    end
    return abs(s) / (4π)
end

@testset "single layer on the surface" begin
    # a GL × GL panel, σ = 1: against the closed form at its own nodes
    F, _ = orient_outward(OCT_V, OCT_F)
    patches, quads = polyhedron_patches(OCT_V, F; eps_in = 2.0)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    P = first(P for P in itf.panels if P.kind == :gl)
    one = DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0])
    u = planar_slp_nodes(one, ones(DB.num_points(P)))
    c = [collect(DB.curved_lpoint(P, s, t)) for (s, t) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    ue = [polygon_slp(c, collect(y .- P.anchor), collect(P.normals[1])) for y in P.points]
    @test maximum(abs.(u .- ue) ./ ue) < 1e-12
    # GCQ edge and corner panels (PEC rules of a cube, GCQ in one or both directions) as CurvedPanels of the same map
    # and rules, a random density: against potential_tensor's product integration on the TensorPanels
    rule = DB.edge_gcq_rule(DB.edge_density_powers(Inf, 1.0; gmax = 2.0); tol = 1e-6, qmax = 12.0)
    cube = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, Inf, 1.0; nlev = 1, n_quad = 6, rule)
    for kind in (:edge, :corner)
        T = first(P for P in cube.panels if P.kind == kind)
        C = CurvedPanel(T.anchor, (u, v) -> T.o .+ (u * T.Lu) .* T.eu .+ (v * T.Lv) .* T.ev, T.ru, T.rv; kind,
                        dmap = (u, v) -> (T.Lu .* T.eu, T.Lv .* T.ev))
        @test maximum(norm(collect(x .- y)) for (x, y) in zip(C.points, T.points)) < 1e-14
        σ = randn(DB.num_points(T))
        ut = DB.potential_tensor(DB.DielectricInterface{typeof(T), Float64}([T], [Inf], [1.0]), σ, T.points; quad_tol = 1e-16)
        # (the moment endpoint rule and own-cell self rows: exact for every density of the span; the defaults, scaled
        # GCQ endpoints and 3 × 3 self rows, to the rule's tolerance for resolved densities: planar_block_self.jl)
        uc = planar_slp_nodes(DB.DielectricInterface{CurvedPanel, Float64}([C], [Inf], [1.0]), σ;
                              q = PlanarProductNearQuad(endpoint_rule = :moment, self = :cell))
        @test maximum(abs.(uc .- ut)) < 1e-12 * maximum(abs.(ut))
    end
    # σ = 1 (in the GCQ span to the rule's tol) on GCQ × GL and GCQ × GCQ panels: the closed form at their own nodes
    itf12, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 6, tol = 1e-12, corner = :gcq)
    for kind in (:edge, :corner)
        P = first(P for P in itf12.panels if P.kind == kind)
        u = planar_slp_nodes(DB.DielectricInterface{CurvedPanel, Float64}([P], [2.0], [1.0]), ones(DB.num_points(P)))
        c = [collect(DB.curved_lpoint(P, s, t)) for (s, t) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
        ue = [polygon_slp(c, collect(y), collect(P.normals[1])) for y in P.lpoints]
        @test maximum(abs.(u .- ue) ./ ue) < 1e-11
    end
end

# The exact panel geometry of polyhedron_mesh in BigFloat: the point (u, v) of panel K (the bilinear quadrilaterals'
# Float64 data taken as exact, the panels in polyhedron_mesh's order)
function exact_point(quads, nlev, K, u, v)
    rects = polyhedron_quad_rects(nlev)
    k, r = divrem(K - 1, length(rects)) .+ 1
    q = quads[k]; U0, U1, V0, V1, s = big.(rects[r])
    U, V = s == 3 ? (U0 + big(v) * (U1 - U0), big(u) * V1) : s == 1 ? (big(u) * U1, V0 + big(v) * (V1 - V0)) :
                    (U0 + big(u) * (U1 - U0), V0 + big(v) * (V1 - V0))
    return big.(collect(q.c)) .+ U .* big.(collect(q.A)) .+ V .* big.(collect(q.B)) .+ (U * V) .* big.(collect(q.C))
end

# ∫_P (x - y) / |x - y|³ dA(y) over the planar polygon c (vertices in order, BigFloat): n Ω + Σ_e ν_e ln((R_a + R_b + L) /
# (R_a + R_b - L)), Ω = ∫ n·(x - y) / |x - y|³ dA the signed solid angle (fan of triangles, Van Oosterom-Strackee),
# ν_e the in-plane outward normal of side e (the in-plane part by the divergence theorem in the plane)
function polygon_field(c, x)
    n = cross(c[2] - c[1], c[3] - c[1]); n = n / norm(n)
    Ω = sum(2 * atan(dot(a, cross(b, d)), norm(a) * norm(b) * norm(d) + dot(a, b) * norm(d) + dot(a, d) * norm(b) +
                                          dot(b, d) * norm(a))
            for (a, b, d) in ((c[1] - x, c[k] - x, c[k + 1] - x) for k in 2:length(c) - 1))
    F = -Ω .* n
    for k in eachindex(c)
        a = c[k]; b = c[mod1(k + 1, length(c))]
        L = norm(b - a); Ra = norm(a - x); Rb = norm(b - x)
        F = F .+ (cross(b - a, n) / L) .* log((Ra + Rb + L) / (Ra + Rb - L))
    end
    return F
end

@testset "Dᵀ rows at nodes beside an edge" begin
    setprecision(BigFloat, 256) do
        # the field of a polygon against its far-field limit (the closed form's signs)
        c = [big.([0.0, 0.0, 0.0]), big.([1.0, 0.0, 0.0]), big.([1.0, 1.0, 0.0]), big.([0.0, 1.0, 0.0])]
        x = big.([0.5, 0.5, 1000.0]); F = polygon_field(c, x)
        @test Float64(norm(F .- (x .- big.([0.5, 0.5, 0.0])) ./ norm(x .- big.([0.5, 0.5, 0.0]))^3)) < 1e-5 * Float64(norm(F))
        # two faces of a perfect-conductor octahedron sharing an edge, as given and turned and moved away from the
        # origin (generic coordinates): σ = 1 (in the GCQ span, p1 < 0), the Dᵀ row sums at the nodes beside the edge on
        # the other face (down to 2.6e-7 of the panel from it) against the closed form. Differences of Float64 positions
        # left 1.2e-11 and 2.7e-11; the check itself is good to ~2e-12 (the node rule, up to ~2e4, added back to C). Both
        # endpoint rules: σ = 1 is in the GCQ family, which the scaled GCQ rule of endpoint_rule = :gcq integrates to tol
        F0, _ = orient_outward(OCT_V, OCT_F)
        ta = 1; tb = findfirst(t -> t != ta && length(intersect(F0[:, t], F0[:, ta])) == 2, axes(F0, 2))
        ax = normalize([0.3, -0.81, 0.47]); K = [0 -ax[3] ax[2]; ax[3] 0 -ax[1]; -ax[2] ax[1] 0]
        rot = I + sin(0.7310281) * K + (1 - cos(0.7310281)) * K^2
        for V in (OCT_V, rot * OCT_V .+ [37.3, -51.9, 93.1]), rule in (:moment, :gcq)
            patches, quads = polyhedron_patches(V, F0; eps_in = Inf)
            nlev = 1; npq = length(polyhedron_quad_rects(nlev))
            itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev, n_quad = 6, tol = 1e-14, corner = :gcq)
            sel = [K for t in (ta, tb) for k in 3(t - 1) + 1:3t for K in (k - 1) * npq + 1:k * npq]
            sub = DB.DielectricInterface{CurvedPanel, Float64}(itf.panels[sel], itf.eps_in[sel], itf.eps_out[sel])
            C = DB.near_corrections(PlanarProductNearQuad(endpoint_rule = rule), sub)
            X, N = DB._interface_targets_normals(sub); X = Matrix{Float64}(X); N = Matrix{Float64}(N)
            offs = cumsum([0; DB.num_points.(sub.panels)])
            owner(t) = searchsortedlast(offs, t - 1)
            worst = 0.0; count = 0
            for (k, P) in enumerate(sub.panels)
                P.kind == :gl && continue
                cols = offs[k] + 1:offs[k + 1]
                corners = [exact_point(quads, nlev, sel[k], s, t) for (s, t) in ((0, 0), (1, 0), (1, 1), (0, 1))]
                ts = unique(C[:, cols].rowval)
                # targets on the other triangle within 1e-4 of P's diameter of P
                diam = maximum(norm(collect(a .- b)) for a in P.points for b in P.points)
                ts = filter(ts) do t
                    (sel[owner(t)] - 1) ÷ (3npq) != (sel[k] - 1) ÷ (3npq) &&
                        minimum(norm(X[:, t] .- collect(y)) for y in P.points) < 1e-4 * diam
                end
                isempty(ts) && continue
                rows = Matrix(C[ts, cols]) .+ DB.dt_node_rule_block(sub, k, X[:, ts], N[:, ts])
                for (r, t) in enumerate(ts)
                    Pt = sub.panels[owner(t)]; j = t - offs[owner(t)]; nv = length(Pt.rv.x)
                    xe = exact_point(quads, nlev, sel[owner(t)], Pt.ru.x[(j - 1) ÷ nv + 1], Pt.rv.x[(j - 1) % nv + 1])
                    ref = Float64(dot(big.(N[:, t]), polygon_field(corners, xe)) / (4 * big(π)))
                    worst = max(worst, abs(sum(rows[r, :]) - ref) / max(1.0, abs(ref)))
                    count += 1
                end
            end
            @test count > 50
            @test worst < 5e-12
        end
    end
end

@testset "Dᵀ near corrections: the in-plane nodes of small panels get the exact row 0" begin
    # Panels 3e-4 across (nlev = 12) at O(1) positions with non-dyadic coordinates: their nodes lie off the panel's
    # plane by the rounding of absolute positions (~3e-17, 1.1e-13 of the diameter). Dᵀ vanishes in the plane, so a
    # panel's own nodes and the other nodes in its plane get the exact row 0, not a product integral (an own node's
    # would hit r = 0: NaN); their correction is minus the node rule, which cancels the far field's term for the pair
    # (not 0, as the nodes lie off the plane).
    patches, quads = polyhedron_patches(OCT_V, OCT_F; eps_in = Inf)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 12, n_quad = 4, tol = 1e-4, corner = :gcq)
    k = findlast(P -> P.kind == :gl, itf.panels)               # a GL panel of the last level, at a vertex
    P = itf.panels[k]
    @test DB._flat_geom(P).diam < 1e-3
    X = reduce(hcat, [collect(y) for y in P.points])
    @test maximum(abs.(X' * collect(P.normals[1]) .- dot(collect(P.anchor), collect(P.normals[1])))) > 1e-13 * DB._flat_geom(P).diam
    one = DB.DielectricInterface{CurvedPanel, Float64}([P], [Inf], [1.0])
    C = near_corrections(PlanarProductNearQuad(), one)
    n = DB.num_points(P)
    @test nnz(C) == n^2                                        # every node is a target, all in the panel's plane
    @test all(isfinite, nonzeros(C))
    @test Matrix(C) == -DB.dt_node_rule_block(one, 1, collect(1:n))   # minus the node rule, bit for bit
end
