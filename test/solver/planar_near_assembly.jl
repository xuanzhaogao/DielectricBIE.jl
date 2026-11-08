using DielectricBIE
import DielectricBIE as DB
using LinearAlgebra, SparseArrays
using Test

@testset "PlanarProductNearQuad assembly: deterministic, rows shared between congruent panels" begin
    # the octahedron at nlev 1: GL × GL, GCQ × GL and GCQ × GCQ panels, three congruent quadrilaterals per triangle
    V = [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]
    F0 = reduce(hcat, [(i == 1 ? 1 : -1) * (j == 3 ? 1 : -1) * (k == 5 ? 1 : -1) > 0 ? [i, j, k] : [i, k, j]
                       for i in (1, 2), j in (3, 4), k in (5, 6)])
    patches, quads = polyhedron_patches(V, orient_outward(V, F0)[1]; eps_in = 2.0)
    itf, _, _ = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 1, n_quad = 5, tol = 1e-6, corner = :gcq)
    for kernel in (:dt, :slp)
        C1 = DB._pn_corrections(PlanarProductNearQuad(), itf, kernel)
        # the threads take the work in any order: the same corrections, bit for bit
        @test DB._pn_corrections(PlanarProductNearQuad(), itf, kernel) == C1
        # each distinct row computed once and shared: as every row computed for its own panel
        C0 = DB._pn_corrections(PlanarProductNearQuad(symmetry = false), itf, kernel)
        @test norm(C1 - C0, Inf) < 1e-12 * norm(C0, Inf)
        # the operator's per-panel blocks hold the same correction
        @test sparse(BlockCorrection(C1, itf)) == dropzeros(C1)
        # assembled as blocks directly (dielectric_operator's and planar_slp_nodes' path): the same correction, kept
        # as it is by BlockCorrection
        Bk = DB._pn_corrections(PlanarProductNearQuad(), itf, kernel; blocks = true)
        @test Bk isa BlockCorrection
        @test sparse(Bk) == dropzeros(C1)
        @test BlockCorrection(Bk, itf) === Bk
        x = randn(size(C1, 2))
        @test norm(Bk * x - C1 * x, Inf) <= 1e-14 * norm(C1 * x, Inf)
    end
    # dielectric_operator with PlanarProductNearQuad: its correction is the blocks of near_corrections
    A = dielectric_operator(itf; far = FMM3DFarField(1e-12; close = 1e-4), near = PlanarProductNearQuad())
    @test A.corrections isa BlockCorrection
    @test sparse(A.corrections) == dropzeros(DB.near_corrections(PlanarProductNearQuad(), itf))
end
