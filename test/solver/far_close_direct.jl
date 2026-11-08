using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra
using Random
using Test

@testset "_close_direct!: the node-rule Dᵀ of the close nodes, added at every node" begin
    rng = MersenneTwister(3)
    n = 300
    S = randn(rng, 3, n); S[:, 7] = S[:, 2]                    # a coincident pair: its term is skipped
    Nm = reduce(hcat, [normalize(randn(rng, 3)) for _ in 1:n])
    w = rand(rng, n)
    close = sort(union(randperm(rng, n)[1:40], [2, 7]))
    for K in (1, 3)
        X = randn(rng, n, K); Y0 = randn(rng, n, K)
        ref = copy(Y0)
        for i in 1:n, j in close
            d = S[:, i] - S[:, j]; r = norm(d)
            r == 0 && continue
            for k in 1:K
                ref[i, k] += w[j] * dot(Nm[:, i], d) / (4π * r^3) * X[j, k]
            end
        end
        Y = DB._close_direct!(copy(Y0), X, S, Nm, w, close)
        @test all(isfinite, Y)
        @test maximum(abs.(Y - ref)) <= 1e-13 * maximum(abs.(ref))
        @test DB._close_direct!(copy(Y0), X, S, Nm, w, Int[]) == Y0
    end
    # every node close, on a GCQ mesh (nodes 1e-6 and less from the edges): the whole node-rule Dᵀ, the sum of
    # dt_node_rule_block over the panels (what the near corrections subtract)
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F0 = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                       for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, orient_outward(V, F0)[1]; eps_in = 2.0)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 4, tol = 1e-6, corner = :gcq)
    Xn, Nn = DB._interface_targets_normals(itf); Xn = Matrix{Float64}(Xn); Nn = Matrix{Float64}(Nn)
    N = size(Xn, 2); wn = DB.all_weights(itf)
    offs = cumsum([0; DB.num_points.(itf.panels)])
    x = randn(rng, N)
    ref = zeros(N)
    for k in eachindex(itf.panels)
        ref .+= DB.dt_node_rule_block(itf, k, Xn, Nn) * x[offs[k]+1:offs[k+1]]
    end
    y = DB._close_direct!(zeros(N, 1), reshape(x, N, 1), Xn, Nn, wn, collect(1:N))
    @test maximum(abs.(y[:, 1] - ref)) <= 1e-13 * maximum(abs.(ref))
end
