using DielectricBIE
import DielectricBIE as DB
using Test

@testset "_tforeach: each index once, on the threads" begin
    for n in (0, 1, 5, 1000)
        hits = zeros(Int, n)
        DB._tforeach(n) do i
            hits[i] += 1                       # one call per index: no two tasks write one entry
        end
        @test all(==(1), hits)
    end
    # a given order: each of its entries once, none else
    ord = [5, 3, 9, 1]; seen = zeros(Int, 10)
    DB._tforeach(i -> (seen[i] += 1), ord)
    @test seen[ord] == ones(Int, 4) && sum(seen) == 4
    # the threads share the work: with two or more, the indices are not all done by one thread
    if Threads.nthreads() > 1
        who = zeros(Int, 2000)
        DB._tforeach(2000) do i
            who[i] = Threads.threadid()
            s = 0.0; for j in 1:20_000; s += sqrt(j + i); end; s < 0 && error()
        end
        @test length(unique(who)) > 1
    end
    # a failure in f is rethrown
    @test_throws Exception DB._tforeach(i -> i == 3 && error("boom"), 5)
end
