# Operators and solve for `TensorPanel` interfaces (GCQ edge panels):
#     (τ I + D^T_FMM + C) σ = rhs,   τ = (ε_out + ε_in) / (2 (ε_out - ε_in)),
# in DielectricBIE's sign convention, with C the sparse near correction of
# `laplace3d_DT_tensor_corrections`.

function _tensor_tau(interface::DielectricInterface)
    return reduce(vcat, [fill(dielectric_tau(interface.eps_in[i], interface.eps_out[i]), num_points(p))
                         for (i, p) in enumerate(interface.panels)])
end

"""
    lhs_dielectric_box3d_gcq_fmm3d(interface, fmm_tol; reuse=true, quad_tol=1e-12, η_gcq=2.0, gcq_far=:max)

FMM-accelerated operator for a `TensorPanel` interface: FMM D^T plus the sparse
near correction plus the diagonal τ. Returns (A::LinearMap, stats), where
`stats` reports the number of near pairs and of distinct near rows computed. `η_gcq` and
`gcq_far` set the near test of GCQ panels (see `tensor_isfar`).
"""
function lhs_dielectric_box3d_gcq_fmm3d(interface::DielectricInterface{TensorPanel{Float64}, Float64},
                                        fmm_tol::Float64; reuse::Bool = true, quad_tol = 1e-12, η_gcq = 2.0,
                                        gcq_far::Symbol = :max)
    D = laplace3d_DT_fmm3d(interface, fmm_tol)
    C0, stats = laplace3d_DT_tensor_corrections(interface; reuse, quad_tol, η_gcq, gcq_far)
    C = BlockCorrection(C0, interface)
    τ = _tensor_tau(interface)
    n = length(τ)
    return LinearMap{Float64}(x -> D * x .+ C * x .+ τ .* x, n, n), stats
end

"""
    lhs_dielectric_box3d_gcq_dense(interface; kw...)

Dense version of `lhs_dielectric_box3d_gcq_fmm3d` (for tests and small problems).
"""
function lhs_dielectric_box3d_gcq_dense(interface::DielectricInterface{TensorPanel{Float64}, Float64}; kw...)
    C, _ = laplace3d_DT_tensor_corrections(interface; kw...)
    A = laplace3d_DT(interface)
    A .+= C
    A[diagind(A)] .+= _tensor_tau(interface)
    return A
end

# shift of the rank-one charge constraint: moves the equilibrium eigenvalue τ + ½ to τ
const _CHARGE_SHIFT = -0.5

"""
    solve_dielectric_box3d_gcq(A, b, interface; rtol=1e-12, itmax=500, constraints=nothing, precondition=true)

GMRES on the W^{1/2}-scaled system (unknowns W^{1/2} σ, W the quadrature
weights), which keeps the unknowns O(1)-scaled when σ is singular. Returns
(σ, stats) with the Krylov stats.

`precondition` (default `true`) applies the contrast scaling of [`dielectric_diagonal_scaling`](@ref),
`1/τ` per panel, as a right preconditioner, as the flat-panel solvers do: `A P = I + (Dᵀ + C) P` then has
one eigenvalue cluster at 1 instead of one per distinct contrast (a no-op with a single contrast).
`memory` is the Krylov basis size (GMRES restarts after that many iterations); its workspace is `N × memory`, so the
default caps it at 100 instead of preallocating `itmax` vectors per solve.

`constraints` imposes each body's total charge, which high-contrast problems and
perfect conductors (`eps_in = Inf`) otherwise leave undetermined. Given the total charges
(0 for a body without free charge inside, q(1 - 1/ε) in DielectricBIE's convention for a
body of permittivity ε containing the charge q), pass `[(range_c, Q_c), ...]` with
`range_c` the indices of body c's nodes; then
    (A - ½ Σ_c u_c w_cᵀ) σ = b - ½ Σ_c u_c Q_c,   u_c = 1_c / |∂c|,  w_cᵀσ = ∫_∂c σ,
is solved instead. The rank-one terms move each body's near-zero eigenvalue τ + ½ to τ
(|τ| ≥ ½) and leave the rest of the spectrum unchanged.
`box_charge_constraints` builds the constraints for box bodies, and
`solve_dielectric_boxes_gcq` applies them by default.
"""
function solve_dielectric_box3d_gcq(A, b::AbstractVector, interface::DielectricInterface;
                                    rtol::Real = 1e-12, itmax::Int = 500, constraints = nothing,
                                    precondition::Bool = true, memory::Int = min(itmax, 100))
    w = all_weights(interface)
    d = sqrt.(abs.(w))
    n = length(b)
    Ad, bd = A, b
    if constraints !== nothing
        cs = [(collect(rg), Float64(Q), sum(w[rg])) for (rg, Q) in constraints]
        bd = copy(b)
        for (rg, Q, area) in cs
            bd[rg] .+= _CHARGE_SHIFT * Q / area
        end
        Ad = LinearMap{Float64}(n, n) do x
            y = A * x
            for (rg, _, area) in cs
                y[rg] .+= _CHARGE_SHIFT * dot(view(w, rg), view(x, rg)) / area
            end
            y
        end
    end
    # right preconditioner P: σ = P W^{-1/2} z, with z the GMRES unknowns
    p = precondition ? 1 ./ _tensor_tau(interface) : ones(n)
    As = LinearMap{Float64}(z -> d .* (Ad * (p .* (z ./ d))), n, n)
    z, stats = Krylov.gmres(As, d .* bd; rtol, atol = 0.0, itmax, memory)
    return p .* (z ./ d), stats
end

# the W^{1/2}-scaled, right-preconditioned operator of solve_dielectric_box3d_gcq on N × K blocks (one batched
# matvec of A per block)
struct _GCQScaledOp{TA}
    A::TA
    d::Vector{Float64}
    s::Vector{Float64}      # p ./ d
    n::Int
end
Base.size(op::_GCQScaledOp) = (op.n, op.n)
Base.size(op::_GCQScaledOp, k::Integer) = k <= 2 ? op.n : 1
Base.eltype(::_GCQScaledOp) = Float64
function LinearAlgebra.mul!(Y::AbstractMatrix, op::_GCQScaledOp, X::AbstractMatrix)
    Y .= op.A * (op.s .* X)
    Y .*= op.d
    return Y
end
LinearAlgebra.mul!(y::AbstractVector, op::_GCQScaledOp, x::AbstractVector) =
    (mul!(reshape(y, op.n, 1), op, reshape(x, op.n, 1)); y)

"""
    solve_dielectric_box3d_gcq_block(A, B, interface; rtol=1e-12, itmax=500, precondition=true, block=32)

[`solve_dielectric_box3d_gcq`](@ref) for the columns of `B` at once: block GMRES on the same W^{1/2}-scaled,
right-preconditioned system, `block` columns at a time, so that every iteration applies `A` to an N × `block` block
(one batched FMM) instead of one vector. `A` must accept `A * X` for a matrix `X` (as
[`dielectric_operator`](@ref) does). The tolerance is set so that every column's residual is at most `rtol` relative to
that column. Returns (Σ, iterations), the block iterations summed over the blocks.
"""
function solve_dielectric_box3d_gcq_block(A, B::AbstractMatrix, interface::DielectricInterface;
                                          rtol::Real = 1e-12, itmax::Int = 500, precondition::Bool = true,
                                          block::Int = 32)
    w = all_weights(interface)
    d = sqrt.(abs.(w))
    n = size(B, 1)
    p = precondition ? 1 ./ _tensor_tau(interface) : ones(n)
    op = _GCQScaledOp(A, d, p ./ d, n)
    Σ = zeros(n, size(B, 2)); iters = 0
    for cols in Iterators.partition(axes(B, 2), max(block, 1))
        Bd = d .* B[:, cols]
        cn = [norm(c) for c in eachcol(Bd)]
        keep = findall(>(0), cn)
        isempty(keep) && continue
        Bk = Bd[:, keep]
        # block GMRES stops on ‖R‖_F ≤ rtol_b ‖Bd‖_F: rtol_b = rtol min_j ‖b_j‖ / ‖Bd‖_F bounds every column
        Z, stats = Krylov.block_gmres(op, Bk; rtol = rtol * minimum(cn[keep]) / norm(Bk), atol = 0.0, itmax,
                                      memory = 20)
        Σ[:, collect(cols)[keep]] .= p .* (Z ./ d)
        iters += stats.niter
    end
    return Σ, iters
end
