# Operators and solve for `TensorPanel` interfaces (GCQ edge panels):
#     (τ I + D^T_FMM + C) σ = rhs,   τ = (ε_out + ε_in) / (2 (ε_out - ε_in)),
# in DielectricBIE's sign convention, with C the sparse near correction of
# `laplace3d_DT_tensor_corrections`.

function _tensor_tau(interface::DielectricInterface)
    return reduce(vcat, [fill(0.5 * (interface.eps_out[i] + interface.eps_in[i]) /
                                   (interface.eps_out[i] - interface.eps_in[i]), num_points(p))
                         for (i, p) in enumerate(interface.panels)])
end

"""
    lhs_dielectric_box3d_gcq_fmm3d(interface, fmm_tol; reuse=true, quad_tol=1e-12, η_gcq=2.0)

FMM-accelerated operator for a `TensorPanel` interface: FMM D^T plus the sparse
near correction plus the diagonal τ. Returns (A::LinearMap, stats), where
`stats` reports the number of near pairs and of distinct near rows computed.
"""
function lhs_dielectric_box3d_gcq_fmm3d(interface::DielectricInterface{TensorPanel{Float64}, Float64},
                                        fmm_tol::Float64; reuse::Bool = true, quad_tol = 1e-12, η_gcq = 2.0)
    D = laplace3d_DT_fmm3d(interface, fmm_tol)
    C, stats = laplace3d_DT_tensor_corrections(interface; reuse, quad_tol, η_gcq)
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

"""
    solve_dielectric_box3d_gcq(A, b, interface; rtol=1e-12, itmax=500)

GMRES on the W^{1/2}-scaled system (unknowns W^{1/2} σ, W the quadrature
weights), which keeps the unknowns O(1)-scaled when σ is singular. Returns
(σ, stats) with the Krylov stats.
"""
function solve_dielectric_box3d_gcq(A, b::AbstractVector, interface::DielectricInterface;
                                    rtol::Real = 1e-12, itmax::Int = 500)
    d = sqrt.(abs.(all_weights(interface)))
    n = length(b)
    As = LinearMap{Float64}(x -> d .* (A * (x ./ d)), n, n)
    ρ, stats = Krylov.gmres(As, d .* b; rtol, atol = 0.0, itmax, memory = itmax)
    return ρ ./ d, stats
end
