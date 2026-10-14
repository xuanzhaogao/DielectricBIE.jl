# Operators and solve for `TensorPanel` interfaces (GCQ edge panels):
#     (τ I + D^T_FMM + C) σ = rhs,   τ = (ε_out + ε_in) / (2 (ε_out - ε_in)),
# in DielectricBIE's sign convention, with C the sparse near correction of
# `laplace3d_DT_tensor_corrections`.

function _tensor_tau(interface::DielectricInterface)
    return reduce(vcat, [fill(dielectric_tau(interface.eps_in[i], interface.eps_out[i]), num_points(p))
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
    solve_dielectric_box3d_gcq(A, b, interface; rtol=1e-12, itmax=500, constraints=nothing)

GMRES on the W^{1/2}-scaled system (unknowns W^{1/2} σ, W the quadrature
weights), which keeps the unknowns O(1)-scaled when σ is singular. Returns
(σ, stats) with the Krylov stats.

`constraints` deflates the near-null modes of high-contrast problems: as
ε_in/ε_out → ∞ (or 0), τ → ∓1/2 and τ I + D^T has one near-zero eigenvalue per
closed body (its equilibrium charge), which stalls GMRES. Given the bodies' total
charges, which Gauss's law fixes (0 for a body without free charge inside, q(1 - 1/ε)
in DielectricBIE's convention for a body of permittivity ε containing the charge
q), pass `[(range_c, Q_c), ...]` with `range_c` the indices of body c's nodes; then
    (A + Σ_c u_c w_cᵀ) σ = b + Σ_c u_c Q_c,   u_c = 1_c / |∂c|,  w_cᵀσ = ∫_∂c σ,
is solved instead. The true solution satisfies both systems; the rank-one terms move
the near-zero eigenvalues to O(1). Without them a discretization error δ enters each body's
total charge amplified by ~ε_in/ε_out, and a perfect conductor (`eps_in = Inf`) is singular.
`box_charge_constraints` builds the constraints for box bodies, and
`solve_dielectric_boxes_gcq` applies them by default.
"""
function solve_dielectric_box3d_gcq(A, b::AbstractVector, interface::DielectricInterface;
                                    rtol::Real = 1e-12, itmax::Int = 500, constraints = nothing)
    w = all_weights(interface)
    d = sqrt.(abs.(w))
    n = length(b)
    Ad, bd = A, b
    if constraints !== nothing
        cs = [(collect(rg), Float64(Q), sum(w[rg])) for (rg, Q) in constraints]
        bd = copy(b)
        for (rg, Q, area) in cs
            bd[rg] .+= Q / area
        end
        Ad = LinearMap{Float64}(n, n) do x
            y = A * x
            for (rg, _, area) in cs
                y[rg] .+= dot(view(w, rg), view(x, rg)) / area
            end
            y
        end
    end
    As = LinearMap{Float64}(x -> d .* (Ad * (x ./ d)), n, n)
    ρ, stats = Krylov.gmres(As, d .* bd; rtol, atol = 0.0, itmax, memory = itmax)
    return ρ ./ d, stats
end
