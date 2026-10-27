using DielectricBIE, Test, LinearAlgebra
import DielectricBIE as BI
include(joinpath(@__DIR__, "fixture_campaign.jl"))

# the fixture's Gaussian blob on an n^3 grid of the same 3.0 cell
function write_blob_xsf(path, n)
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

@testset "GCQ campaign mode, near/far V" begin
    # moment shift: moments about c' from those about c equal the direct moments about c'
    exps = BI._nf_exponents(4)
    X = randn(3, 50); v = randn(50); c = [0.1, -0.2, 0.3]; c2 = [0.5, 0.4, -0.6]; hm = 0.7
    @test BI._nf_shift_matrix((c .- c2) ./ hm, exps) * BI._nf_moments(X, v, c, hm, exps) ≈ BI._nf_moments(X, v, c2, hm, exps)
    # far core: e' T e' of the equivalent charges reproduces sum v_i v'_j / (4π|x_i - y_j|) for well-separated clusters
    exps = BI._nf_exponents(8); w = (1.0, 1.0, 1.0)
    px, Pm = BI._nf_proxies(8, w, 1.0, exps)
    Y = 0.9 .* (2 .* rand(3, 40) .- 1); u = rand(40); o = [4.0, 1.0, 0.0]
    C = BI._nf_core_matrix(px, Pm, o)
    ex = sum(v[i] * u[j] / (4π * norm(0.9 .* X[:, i] ./ maximum(abs, X) .- Y[:, j] .- o)) for i in 1:50, j in 1:40)
    m1 = BI._nf_moments(0.9 .* X ./ maximum(abs, X), v, zeros(3), 1.0, exps); m2 = BI._nf_moments(Y, u, zeros(3), 1.0, exps)
    @test abs(dot(m1, C, m2) - ex) < 1e-5 * sum(abs, v) * sum(abs, u) / (4π * 2)

    # four orbitals along x, 12 grid steps apart, in one dielectric box: one orbital per tile, rnear = 0, so the
    # pairs of tiles two or more apart are far
    dir = mktempdir()
    toml = write_fixture_campaign(dir)
    write_blob_xsf(joinpath(dir, "orb.xsf"), 15)
    dg = BI.read_xsf(joinpath(dir, "orb.xsf"))[2]
    c0 = BI.density_centroid(dg); ex1 = BI.true_cell_vectors(dg)[1][1] / dg.nx
    orbs = join(["[[orbital]]\ntype = 1\nx = $(c0[1] + 12k * ex1)\ny = $(c0[2])\nz = $(c0[3])\n" for k in 0:3], "\n")
    txt = read(toml, String)
    i1 = findfirst("[[orbital]]", txt).start; i2 = findfirst("[pairing]", txt).start
    txt = txt[1:i1-1] * orbs * "\n" * txt[i2:end]
    xm = c0[1] + 18ex1
    txt = replace(txt, r"boxes = .*" => "boxes = [[$xm, $(c0[2]), $(c0[3]), $(36ex1 + 8), 8.0, 6.0, 2.0]]")
    write(toml, txt)
    c = load_campaign(toml)
    kw = (; far = :fmm3d, nlev = 1, n_quad = 4, rhs_tol = 1e-4, lhs_tol = 1e-9, gmres_rtol = 1e-10, volume_tol = 1e-8,
            quad_tol = 1e-6, edge_tol = 1e-6, tau = 1e-10, fg_tol = 1e-10, verbose = false)
    rd = gcq_four_index(c; kw...)
    rn = gcq_four_index(c; kw..., vform = :nearfar, tile = 12ex1, rnear = 0.0)
    V = rn.V; P = size(V, 1)
    @test rn.pair_ids == rd.pair_ids && size(V) == size(rd.V)
    @test length(V.tiles) == 4
    D = Matrix(V)
    @test all(D[a, b] ≈ V[a, b] for a in 1:P, b in 1:P)
    @test D ≈ D'
    far = [!BI._nf_isnear(V.tiles[V.tile_of[b]] .- V.tiles[V.tile_of[a]], V.tile, V.w, V.rnear) for a in 1:P, b in 1:P]
    @test count(far) > 0 && count(.!far) > 0
    sc = maximum(abs, rd.V)
    @test maximum(abs, D[.!far] .- rd.V[.!far]) < 1e-7 * sc       # near: the same solves on fewer points
    @test maximum(abs, D[far] .- rd.V[far]) < 1e-4 * sc            # far: the moment field (p = 8)
    @test rn.stats.nnear_entries == count(.!far)
    @test_throws ArgumentError gcq_four_index(c; kw..., vform = :sparse)
    @test_throws ArgumentError gcq_four_index(c; kw..., layer = :translate)       # needs vform = :nearfar

    # layer = :translate where no translation happens: two orbitals stacked vertically (one pair per group of
    # translates, three tile levels), so the translated layer part equals the exact one, near entries to the solver
    # tolerance and far entries to the moment accuracy
    dir = mktempdir()
    toml = write_fixture_campaign(dir)
    write_blob_xsf(joinpath(dir, "orb.xsf"), 15)
    dz = 24ex1
    orbs = join(["[[orbital]]\ntype = 1\nx = $(c0[1])\ny = $(c0[2])\nz = $(c0[3] + k * dz)\n" for k in 0:1], "\n")
    txt = read(toml, String)
    i1 = findfirst("[[orbital]]", txt).start; i2 = findfirst("[pairing]", txt).start
    txt = txt[1:i1-1] * orbs * "\n" * txt[i2:end]
    txt = replace(txt, r"neighbor_cutoff = .*" => "neighbor_cutoff = 100.0")
    txt = replace(txt, r"boxes = .*" => "boxes = [[$(c0[1]), $(c0[2]), $(c0[3] + dz / 2), 6.0, 6.0, $(dz + 6), 2.0]]")
    write(toml, txt)
    c = load_campaign(toml)
    kw = (; kw..., gmres_rtol = 1e-8, tau = 1e-8, vform = :nearfar, tile = 6ex1, rnear = 0.0, p_far = 2)
    rl = gcq_four_index(c; kw...)
    rt = gcq_four_index(c; kw..., layer = :translate)
    V = rt.V; P = size(V, 1)
    @test V.layer === :translate && length(V.tiles) == 3
    @test rt.stats.proxy_solves <= 3 * length(BI._nf_exponents(2))       # n_mom per level, not (p + 1)^3
    Dl = Matrix(rl.V); Dt = Matrix(V)
    @test all(Dt[a, b] ≈ V[a, b] for a in 1:P, b in 1:P)
    @test Dt ≈ Dt'
    far = [!BI._nf_isnear(V.tiles[V.tile_of[b]] .- V.tiles[V.tile_of[a]], V.tile, V.w, V.rnear) for a in 1:P, b in 1:P]
    @test count(far) > 0
    sc = maximum(abs, Dl)
    @test maximum(abs, Dt[.!far] .- Dl[.!far]) < 1e-5 * sc
    @test maximum(abs, Dt[far] .- Dl[far]) < 1e-3 * sc
end
