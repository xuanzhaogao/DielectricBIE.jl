# FFTW's threads callback (src/utils/fftw_threads.jl): threaded FFTW plans run concurrently from Julia tasks, and
# FINUFFT's threaded FFTs (nested FFTW parallel loops) through it, give the serial results
using DielectricBIE, Test
import DielectricBIE as BI

@testset "FFTW threads callback" begin
    FFTW, FINUFFT = BI.FFTW, BI.FINUFFT
    A = randn(ComplexF64, 96, 80, 4)
    ref = FFTW.fft(A, (1, 2))
    p = FFTW.set_num_threads(() -> FFTW.plan_fft(A, (1, 2)), 4)
    ok = fill(false, 4 * Threads.nthreads())
    Threads.@threads for k in eachindex(ok)
        ok[k] = all(_ -> p * A ≈ ref, 1:20)
    end
    @test all(ok)
    x1 = 2π .* rand(3000) .- π; x2 = 2π .* rand(3000) .- π; fk = randn(ComplexF64, 200, 180)
    v1 = FINUFFT.nufft2d2(x1, x2, 1, 1e-12, fk; nthreads = 1)
    @test all(_ -> FINUFFT.nufft2d2(x1, x2, 1, 1e-12, fk; nthreads = Threads.nthreads()) ≈ v1, 1:20)
end
