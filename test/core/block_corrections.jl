using DielectricBIE
import DielectricBIE as BI
using LinearAlgebra, SparseArrays, Random
using SCTL
using Test

@testset "BlockCorrection" begin
    rng = MersenneTwister(2)
    cols = [1:4, 5:9, 10:10, 11:16]                  # panels of different sizes
    n = 16; m = 23
    # a correction with rows near some panels only, one panel with no rows
    C = spzeros(m, n)
    for (k, cs) in enumerate(cols)
        k == 3 && continue
        rs = randperm(rng, m)[1:rand(rng, 3:12)]
        C[rs, cs] = randn(rng, length(rs), length(cs))
    end
    for chunk in (2, 256)                            # chunked work units and whole blocks
        B = BI.BlockCorrection(C, cols; chunk)
        @test size(B) == (m, n)
        @test sparse(B) == C
        @test BI.nstored(B) == nnz(C)
        x = randn(rng, n); X = randn(rng, n, 5)
        @test B * x ≈ C * x
        @test B * X ≈ C * X
        Y = randn(rng, m, 5); Y0 = copy(Y)
        mul!(Y, B, X, 2.0, 0.5)
        @test Y ≈ 0.5 * Y0 + 2.0 * (C * X)
        @test B * X[:, 1:2] ≈ C * X[:, 1:2]          # buffer resized for another number of columns
    end
    @test_throws DimensionMismatch BI.BlockCorrection(C, [1:4, 5:9])
    # from an interface: one block per panel
    itf = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 4.0, 1.0, Float64)
    Cd = BI.near_corrections(SCTLNearCorrection(tol = 1e-8), itf)
    Bd = BlockCorrection(Cd, itf)
    xd = randn(rng, BI.num_points(itf))
    @test Bd * xd ≈ Cd * xd
    @test all(cs -> cs in panel_column_ranges(itf), Bd.cols)
end
