# Dynamic self-scheduling on Julia's threads. `Threads.@threads` (:static or :dynamic) hands each thread one
# contiguous block of the iterations, so iterations of uneven cost (panels with many near targets, rows of edge
# panels) leave most threads idle while a few finish their blocks; `:greedy` needs Julia 1.11.

"""
    _tforeach(f, order)
    _tforeach(f, n)

`f(i)` for every `i` in `order` (or `1:n`) on Julia's threads: one task per thread, each taking the next index of
`order` as it finishes one, so put the costly iterations first. A failure in `f` is rethrown.
"""
function _tforeach(f, order::AbstractVector{<:Integer})
    n = length(order)
    nt = min(Threads.nthreads(), n)
    if nt <= 1
        foreach(f, order)
        return nothing
    end
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:nt
        Threads.@spawn while true
            j = Threads.atomic_add!(next, 1)
            j > n && break
            f(order[j])
        end
    end
    return nothing
end
_tforeach(f, n::Integer) = _tforeach(f, 1:n)
