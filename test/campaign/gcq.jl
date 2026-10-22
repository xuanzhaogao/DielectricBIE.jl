using DielectricBIE, Test, LinearAlgebra, Serialization
import DielectricBIE as BI
include(joinpath(@__DIR__, "fixture_campaign.jl"))

# the fixture's Gaussian-blob template on an n^3 grid spanning the same 3.0 cell (n = 6 in fixture_campaign.jl)
function write_fine_xsf(path, n)
    open(path, "w") do io
        println(io, "CRYSTAL\nPRIMVEC\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nPRIMCOORD\n1 1\nX 1.5 1.5 1.5")
        println(io, "BEGIN_BLOCK_DATAGRID_3D\n g\nBEGIN_DATAGRID_3D_g\n$n $n $n\n0.0 0.0 0.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0")
        vals = [exp(-(((i-1)/n*3.0 - 1.5)^2 + ((j-1)/n*3.0 - 1.5)^2 + ((k-1)/n*3.0 - 1.5)^2) / (2 * 0.4^2))
                for i in 1:n, j in 1:n, k in 1:n]
        for chunk in Iterators.partition(vec(vals), n)
            println(io, join(string.(chunk), " "))
        end
        println(io, "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D")
    end
end

@testset "GCQ campaign mode" begin
    # the fixture with a third orbital half a cell from each: overlapping frames, so the neighbour pairs (1,2) and (2,3)
    # are one non-empty shape, translated; (1,3) does not overlap (an empty pair)
    dir = mktempdir()
    toml = write_fixture_campaign(dir)
    write_fine_xsf(joinpath(dir, "orb.xsf"), 15)          # resolve the blob (the 6-point fixture grid does not)
    # three orbitals 8 grid steps apart along x (positions on the grid: centroid + whole steps)
    dgf = BI.read_xsf(joinpath(dir, "orb.xsf"))[2]
    c0 = BI.density_centroid(dgf); ex = BI.true_cell_vectors(dgf)[1][1] / dgf.nx
    orbs = join(["[[orbital]]\ntype = 1\nx = $(c0[1] + 8k * ex)\ny = $(c0[2])\nz = $(c0[3])\n" for k in 0:2], "\n")
    txt = read(toml, String)
    i1 = findfirst("[[orbital]]", txt).start; i2 = findfirst("[pairing]", txt).start
    write(toml, txt[1:i1-1] * orbs * "\n" * txt[i2:end])
    c = load_campaign(toml)
    kw = (; far = :fmm3d, nlev = 1, n_quad = 4, rhs_tol = 1e-4, lhs_tol = 1e-9, gmres_rtol = 1e-10, volume_tol = 1e-8,
            quad_tol = 1e-6, edge_tol = 1e-6, tau = 1e-10, fg_tol = 1e-10, verbose = false)
    out = joinpath(dir, "gcq_out")
    res = run_gcq_campaign(c; kw..., dir = out)
    # the batch pipeline's output files
    @test all(isfile(joinpath(out, f)) for f in ("V_full.tsv", "V_full_eV.jls", "V_full_eV.tsv", "report.txt"))
    r = deserialize(joinpath(out, "V_full_eV.jls"))
    @test r.unit == "eV" && r.method == "gcq" && r.V ≈ r.V'
    pids = res.pair_ids; P = length(pids)
    @test pids == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    @test res.stats.nshapes == 3                         # onsite, (i, i+1) and the empty (1,3)
    @test res.stats.asym < 1e-6
    # with the target refinement: no target is near a panel, and the same V (the layer part G' sigma is exact either way)
    rt = gcq_four_index(c; kw..., refine_targets = true)
    @test rt.stats.near_pairs == 0 && rt.stats.N > res.stats.N
    @test maximum(abs, rt.V - res.V) < 1e-6 * maximum(abs, res.V)
    @test res.V ≈ res.V'
    @test all(res.V[a, a] > 0 for a in 1:P if pids[a][1] == pids[a][2])
    @test all(iszero, res.V[3, :]) && maximum(abs, res.V[2, :]) > 0        # the empty pair (1,3), a non-empty one

    # reference: every pair on its own, the same interface, the layer part by GCQPotential at the pair's points
    centers = BI.enumerate_centers(c)
    shapes, shape_of, delta, dg = gcq_pair_shapes(c, centers, pids)
    X, env = BI._gcq_targets(shapes, shape_of, delta, dg)
    ff = BI._gcq_far(:fmm3d)
    skin = res.stats.target_skin
    itf = BI.gcq_campaign_interface(c, X, env, dg; nlev = 1, n_quad = 4, rhs_tol = 1e-4, edge_tol = 1e-6, quad_tol = 1e-6,
                                    target_skin = skin, far = ff, refine_targets = false)
    @test BI.num_points(itf) == res.stats.N
    A = dielectric_operator(itf; far = ff(1e-9), near = GCQNearQuad(quad_tol = 1e-6, η_gcq = 2.0, gcq_far = :direction))
    ev = GCQPotential(quad_tol = 1e-6, η_gcq = 2.0, gcq_far = :direction, far = FMM3DFarField(1e-12))
    pts = [BI._grid_positions(dg, [g .+ delta[p] for g in shapes[shape_of[p]].gidx]) for p in 1:P]
    tws = [shapes[shape_of[p]].tw for p in 1:P]
    Vvol = zeros(P, P); Vlay = zeros(P, P)
    for b in 1:P
        isempty(tws[b]) && continue                      # the empty pair: zero row and column
        svb = BI._gcq_shape_source(c, dg, shapes[shape_of[b]], delta[b])
        F = rhs_dielectric_box3d(itf, svb, 1.0, FMM3DFarField(1e-12))
        σ, _ = solve_dielectric_box3d_gcq(A, F, itf; rtol = 1e-10)
        for a in 1:P
            isempty(tws[a]) && continue
            Vvol[a, b] = dot(tws[a], BI._gcq_volume_potential(svb, pts[a]; tol = 1e-8, c_pad = c.c_pad, far = ff))
            Vlay[a, b] = dot(tws[a], potential_operator(ev, itf, pts[a]) * σ)
        end
    end
    sc = maximum(abs, Vvol)
    # (the reference's RHS and test sums are point charges; the mode uses the TKM field near the density)
    @test maximum(abs, res.V_layer - Vlay) < 1e-6 * sc
    # the volume part takes each entry with the larger shape index as the source (V_vol is symmetric up to the TKM error)
    @test maximum(abs, res.V_vol - [shape_of[a] <= shape_of[b] ? Vvol[a, b] : Vvol[b, a] for a in 1:P, b in 1:P]) < 1e-8 * sc
    @test maximum(abs, res.V_vol - Vvol) < 1e-3 * sc
    @test maximum(abs, res.V_layer) > 1e-3 * sc                        # the interface matters in this check

    # two cooperating parts, part 2 in a separate process, give the same V
    sync = mktempdir()
    code = "using DielectricBIE; gcq_four_index(load_campaign(\"$toml\"); far = :fmm3d, nlev = 1, n_quad = 4, " *
           "rhs_tol = 1e-4, lhs_tol = 1e-9, gmres_rtol = 1e-10, volume_tol = 1e-8, quad_tol = 1e-6, edge_tol = 1e-6, " *
           "tau = 1e-10, fg_tol = 1e-10, verbose = false, part = 2, nparts = 2, sync = \"$sync\")"
    cmd = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
                 "JULIA_NUM_THREADS" => string(Threads.nthreads()))
    p2 = run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = false)
    r1 = gcq_four_index(c; kw..., part = 1, nparts = 2, sync)
    wait(p2)
    @test success(p2)
    @test maximum(abs, r1.V - res.V) / maximum(abs, res.V) < 1e-12
    @test_throws ArgumentError gcq_four_index(c; kw..., part = 1, nparts = 2)          # no sync directory

end
