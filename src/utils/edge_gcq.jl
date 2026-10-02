# Edge singularity powers and the weighted generalized Chebyshev quadrature
# (GCQ) rule for panels touching a dielectric edge.
#
# Near an edge the single-layer density behaves like σ ~ Σ_k c_k s^{p_k}, where
# s is the distance to the edge and p_k = γ_k - 1 with γ_k the roots of the
# wedge transmission determinant `theta_ODE_det` (corner_singularity.jl).
# `edge_gcq_rule` builds nodes and weights on [0,1] that integrate the family
#     { s^{p_k + n} } ∪ { s^n },   n = 0..nsmooth
# (edge terms times smooth factors) to a tolerance. The rule is weighted:
# σ = s^{p_1} v(s) with p_1 = min p_k, and GCQ runs for the v-family in the
# inner product with weight s^{p_1}, which only needs p_1 > -1 (plain L² GCQ
# needs every p > -1/2). The rule is invariant under s -> L s, so one rule
# serves every panel [0, L] touching an edge.

"""
    junction_powers(angles, eps; gmax=20.0, ngrid=200_000)

Potential powers γ ∈ (0, gmax] at a 2D junction of sectors with openings
`angles` (summing to 2π, in angular order) and permittivities `eps`. Simple
roots of `theta_ODE_det` are bracketed by sign changes; double roots (the
determinant touching zero, e.g. γ = 2m at a right-angle corner) are found from
local minima of |det|. The density powers are `γ .- 1`.
"""
function junction_powers(angles::AbstractVector, eps::AbstractVector;
                         gmax::Real = 20.0, ngrid::Int = 200_000)
    length(angles) == length(eps) || throw(ArgumentError("angles and eps must have equal length"))
    abs(sum(angles) - 2pi) < 1e-12 || throw(ArgumentError("angles must sum to 2π"))
    f = g -> theta_ODE_det(angles[1:end-1], eps, g)
    gs = collect(range(1e-3, gmax, length = ngrid))
    fs = f.(gs)
    fscale = maximum(abs, fs)
    roots = Float64[]
    bisect(F, a, b) = begin
        fa = F(a)
        for _ in 1:200
            m = (a + b) / 2; fm = F(m)
            (fm == 0 || b - a < 4 * Base.eps(m)) && return m
            fa * fm < 0 ? (b = m) : (a = m; fa = fm)
        end
        return (a + b) / 2
    end
    for i in 1:length(gs)-1
        fs[i] * fs[i+1] < 0 && push!(roots, bisect(f, gs[i], gs[i+1]))
    end
    h = 1e-6
    df = g -> (f(g + h) - f(g - h)) / 2h
    for i in 2:length(gs)-1
        if abs(fs[i]) < abs(fs[i-1]) && abs(fs[i]) < abs(fs[i+1]) &&
           fs[i-1] * fs[i+1] > 0 && df(gs[i-1]) * df(gs[i+1]) < 0
            g0 = bisect(df, gs[i-1], gs[i+1])
            if abs(f(g0)) < 1e-10 * fscale && all(abs.(roots .- g0) .> 1e-8)
                push!(roots, g0)
            end
        end
    end
    return sort(roots)
end

"""
    edge_density_powers(eps_in, eps_out; angle=π/2, gmax=20.0)

Density powers p_k = γ_k - 1 at the edge of a dielectric wedge of opening
`angle` (permittivity `eps_in`) in a medium `eps_out`.
"""
edge_density_powers(eps_in::Real, eps_out::Real; angle::Real = pi / 2, gmax::Real = 20.0) =
    junction_powers([2pi - angle, angle], [Float64(eps_out), Float64(eps_in)]; gmax) .- 1

"""
    EdgeGCQRule

Weighted GCQ rule on [0,1] and the data to interpolate in its span:
- `x, w`    : nodes and weights acting on density values (∫₀¹ σ ≈ Σ w_j σ(x_j))
- `pw`      : weight power p_1 (σ = x^pw v)
- `segs, z0, wz` : fine grid (segments with 14-point Gauss-Legendre on [-1,1])
- `P`       : (n_fine × n) matrix, σ node values -> fine-node values of v
- `pows`    : exponents of the integrated family
"""
struct EdgeGCQRule
    x::Vector{Float64}
    w::Vector{Float64}
    pw::Float64
    segs::Vector{Tuple{Float64, Float64}}
    z0::Vector{Float64}
    wz::Vector{Float64}
    P::Matrix{Float64}
    pows::Vector{Float64}
    tol::Float64
end

Base.show(io::IO, r::EdgeGCQRule) =
    print(io, "EdgeGCQRule with $(length(r.x)) nodes (tol $(r.tol), weight x^$(round(r.pw, digits = 4)))")

"""
    edge_gcq_rule(dpows; tol=1e-10, nsmooth=20)
    edge_gcq_rule(eps_in, eps_out; angle=π/2, tol=1e-10, nsmooth=20)

Weighted GCQ rule for the density powers `dpows` (several contrasts may be
combined with `vcat`). Steps: adaptive Gauss-Kronrod fine grid for the
weighted family, SVD of the weighted sample matrix (rank at `tol`),
column-pivoted QR for the nodes, transposed Vandermonde solve for the weights.
"""
function edge_gcq_rule(dpows::AbstractVector; tol::Real = 1e-10, nsmooth::Int = 20)
    p1 = minimum(dpows)
    p1 > -1 || throw(ArgumentError("density power $p1 is not integrable"))
    q = sort(unique(round.(vcat([(p - p1) .+ (0:nsmooth) for p in dpows]...,
                                collect((0:nsmooth) .- p1)); digits = 12)))
    q = q[q .>= -1e-12]
    fs = s -> [s^qi for qi in q]
    _, _, segbuf, _ = quadgk_segbuf_count(s -> s^p1 .* fs(s), 0.0, 1.0;
                                          atol = tol / 10, rtol = tol / 10)
    segs = sort([(sg.a, sg.b) for sg in segbuf])
    z0, w0 = gausslegendre(14)
    xg = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs])
    wg = reduce(vcat, [(b - a) / 2 .* w0 for (a, b) in segs])
    ωg = wg .* xg .^ p1
    A = reduce(vcat, [sqrt(ωj) .* fs(xj)' for (xj, ωj) in zip(xg, ωg)])
    S = svd(A)
    r = sum(S.S .> tol * S.S[1])
    U = S.U[:, 1:r]
    nodeinds = sort(qr(U', ColumnNorm()).p[1:r])
    x = xg[nodeinds]
    Is = vec(sum(sqrt.(ωg) .* U, dims = 1))
    W = sqrt.(ωg[nodeinds]) .* (U[nodeinds, :]' \ Is)
    Uf = U ./ sqrt.(ωg)
    P = (Uf / Uf[nodeinds, :]) * Diagonal(x .^ (-p1))
    return EdgeGCQRule(x, W ./ x .^ p1, p1, segs, z0, w0, P, q .+ p1, Float64(tol))
end

edge_gcq_rule(eps_in::Real, eps_out::Real; angle::Real = pi / 2, kw...) =
    edge_gcq_rule(edge_density_powers(eps_in, eps_out; angle); kw...)

"Largest relative error of the rule over its own family (∫₀¹ x^p = 1/(p+1))."
edge_gcq_family_error(r::EdgeGCQRule) =
    maximum(abs(sum(r.w .* r.x .^ p) * (p + 1) - 1) for p in r.pows)
