# Near corrections as dense blocks, one per source panel, stored and applied as SCTL's BoundaryIntegralOp does its
# near interactions (K_near, ComputeNearInterac): the correction builders return sparse matrices; the operators convert
# them with BlockCorrection(C, interface). An application computes every block's product (rows chunked for load
# balance, in parallel) into one contiguous buffer, then each target row sums its contributions from the buffer (in
# parallel, no two threads write one row) -- SCTL's GEMM per near block, then sort-scatter and accumulate by target.

"""
    BlockCorrection(C, interface)

A near correction `C` (sparse, `Nt × N`: targets × interface nodes) stored as dense blocks, one per source panel:
block `k` maps the nodal values of panel `k` (columns `cols[k]`) to the targets near it (`rows[k]`, the rows with a
nonzero in those columns). `mul!(Y, B, X, α, β)`, `B * x` and `B * X` apply it to a vector or an `N × K` block with
Julia threads; `sparse(B)` converts back. Dense blocks hold each (panel, near target) row in full, like the builders
produce them, and need no index per entry.
"""
mutable struct BlockCorrection{T}
    m::Int
    n::Int
    rows::Vector{Vector{Int}}
    cols::Vector{UnitRange{Int}}
    blocks::Vector{Matrix{T}}
    # work units, as SCTL's near_blk: rows chk_lo[u]:chk_hi[u] of block chk_blk[u], so that panels with many near
    # targets are split for load balance; their products go to rows off[b] + (chk_lo:chk_hi) of one contiguous buffer
    chk_blk::Vector{Int32}
    chk_lo::Vector{Int32}
    chk_hi::Vector{Int32}
    off::Vector{Int}
    # contributions to target row r: buffer rows src[j] for j in ptr[r]:ptr[r+1]-1 (SCTL's sort-scatter by target)
    ptr::Vector{Int}
    src::Vector{Int}
    buf::Matrix{T}              # all block products, reused while the number of columns is unchanged
end

"""
    panel_column_ranges(interface) -> Vector{UnitRange{Int}}

The node indices of each panel of `interface` (columns of a correction matrix).
"""
function panel_column_ranges(interface::DielectricInterface)
    offs = cumsum([0; [num_points(p) for p in interface.panels]])
    return [offs[k]+1:offs[k+1] for k in eachindex(interface.panels)]
end

BlockCorrection(C::AbstractSparseMatrix, interface::DielectricInterface; chunk::Integer = 256) =
    BlockCorrection(SparseMatrixCSC(C), panel_column_ranges(interface); chunk)

function BlockCorrection(C::SparseMatrixCSC{T}, cols::Vector{UnitRange{Int}}; chunk::Integer = 256) where T
    m, n = size(C)
    (isempty(cols) ? n == 0 : (first(cols[1]) == 1 && last(cols[end]) == n)) ||
        throw(DimensionMismatch("the panels' columns 1:$(isempty(cols) ? 0 : last(cols[end])) do not match the $n columns"))
    np = length(cols)
    rows = Vector{Vector{Int}}(undef, np)
    blocks = Vector{Matrix{T}}(undef, np)
    rv = rowvals(C); nzv = nonzeros(C)
    Threads.@threads :dynamic for k in 1:np
        # fast path: every column of the panel has the same rows (as the builders produce), copied column by column
        c1 = first(cols[k], 1)
        r1 = isempty(c1) ? Int[] : rv[nzrange(C, c1[1])]
        if !isempty(c1) && issorted(r1; lt = <=) &&
           all(c -> length(nzrange(C, c)) == length(r1) && view(rv, nzrange(C, c)) == r1, cols[k])
            B = Matrix{T}(undef, length(r1), length(cols[k]))
            for (jc, c) in enumerate(cols[k])
                copyto!(view(B, :, jc), view(nzv, nzrange(C, c)))
            end
            rows[k] = r1; blocks[k] = B
            continue
        end
        rs = Int[]
        for c in cols[k], idx in nzrange(C, c)
            push!(rs, rv[idx])
        end
        sort!(unique!(rs))
        B = zeros(T, length(rs), length(cols[k]))
        for (jc, c) in enumerate(cols[k]), idx in nzrange(C, c)
            B[searchsortedfirst(rs, rv[idx]), jc] += nzv[idx]
        end
        rows[k] = rs; blocks[k] = B
    end
    keep = findall(k -> !isempty(rows[k]), 1:np)
    rows = rows[keep]; blocks = blocks[keep]; bcols = cols[keep]
    off = cumsum([0; length.(rows)])
    chk_blk = Int32[]; chk_lo = Int32[]; chk_hi = Int32[]
    for (b, rs) in enumerate(rows), lo in 1:chunk:length(rs)
        push!(chk_blk, b); push!(chk_lo, lo); push!(chk_hi, min(lo + chunk - 1, length(rs)))
    end
    # the buffer rows of each target row, in CSR order
    cnt = zeros(Int, m + 1)
    for rs in rows, r in rs
        cnt[r + 1] += 1
    end
    ptr = cumsum(cnt) .+ 1
    src = Vector{Int}(undef, ptr[end] - 1)
    fill_at = ptr[1:m]
    for (b, rs) in enumerate(rows), (l, r) in enumerate(rs)
        src[fill_at[r]] = off[b] + l; fill_at[r] += 1
    end
    return BlockCorrection{T}(m, n, rows, bcols, blocks, chk_blk, chk_lo, chk_hi, off, ptr, src, Matrix{T}(undef, 0, 0))
end

Base.size(B::BlockCorrection) = (B.m, B.n)
Base.size(B::BlockCorrection, d::Integer) = d == 1 ? B.m : d == 2 ? B.n : 1
Base.eltype(::BlockCorrection{T}) where T = T

"Number of stored entries (the sum of the block sizes)."
nstored(B::BlockCorrection) = sum(length, B.blocks; init = 0)

function SparseArrays.sparse(B::BlockCorrection{T}) where T
    I = Int[]; J = Int[]; V = T[]
    for (rs, cs, M) in zip(B.rows, B.cols, B.blocks), (jc, c) in enumerate(cs), (l, r) in enumerate(rs)
        iszero(M[l, jc]) && continue
        push!(I, r); push!(J, c); push!(V, M[l, jc])
    end
    return sparse(I, J, V, B.m, B.n)
end

# Julia's BLAS at one thread while the per-panel products run on Julia's threads (several threads calling a
# multi-threaded BLAS at once oversubscribe the cores)
function _blas_single(f)
    nt = BLAS.get_num_threads()
    nt == 1 && return f()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(nt)
    end
end

function LinearAlgebra.mul!(Y::AbstractMatrix, B::BlockCorrection{T}, X::AbstractMatrix, α::Number, β::Number) where T
    size(X, 1) == B.n && size(Y, 1) == B.m && size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    K = size(X, 2)
    size(B.buf) == (B.off[end], K) || (B.buf = Matrix{T}(undef, B.off[end], K))
    buf = B.buf
    # 1. each work unit: (its rows of a panel's block) * (the panel's nodal values), into the contiguous buffer
    _blas_single() do
        Threads.@threads :dynamic for u in eachindex(B.chk_blk)
            b = B.chk_blk[u]; lo = B.chk_lo[u]; hi = B.chk_hi[u]
            mul!(view(buf, B.off[b]+lo:B.off[b]+hi, :), view(B.blocks[b], lo:hi, :), view(X, B.cols[b], :))
        end
    end
    # 2. each target row sums its contributions
    Threads.@threads for r in 1:B.m
        @inbounds for k in 1:K
            s = zero(T)
            for j in B.ptr[r]:B.ptr[r+1]-1
                s += buf[B.src[j], k]
            end
            Y[r, k] = iszero(β) ? α * s : β * Y[r, k] + α * s
        end
    end
    return Y
end

LinearAlgebra.mul!(Y::AbstractMatrix, B::BlockCorrection, X::AbstractMatrix) = mul!(Y, B, X, true, false)
LinearAlgebra.mul!(y::AbstractVector, B::BlockCorrection, x::AbstractVector, α::Number, β::Number) =
    (mul!(reshape(y, :, 1), B, reshape(x, :, 1), α, β); y)
LinearAlgebra.mul!(y::AbstractVector, B::BlockCorrection, x::AbstractVector) = mul!(y, B, x, true, false)
Base.:*(B::BlockCorrection{T}, x::AbstractVector) where T = mul!(Vector{T}(undef, B.m), B, x)
Base.:*(B::BlockCorrection{T}, X::AbstractMatrix) where T = mul!(Matrix{T}(undef, B.m, size(X, 2)), B, X)
