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
`angle` (permittivity `eps_in`) in a medium `eps_out`. A perfect conductor is
`eps_in = Inf` (or `eps_out = Inf` for a conducting exterior): the potential then
satisfies a Dirichlet condition on the other sector, of opening β, and
γ_k = kπ/β, which is also the limit of the dielectric roots as the contrast → ∞
(for a right angle: p = -1/3, 1/3, 1, 5/3, ...).
"""
function edge_density_powers(eps_in::Real, eps_out::Real; angle::Real = pi / 2, gmax::Real = 20.0)
    isinf(eps_in) && isinf(eps_out) && throw(ArgumentError("eps_in and eps_out cannot both be infinite"))
    if isinf(eps_in) || isinf(eps_out)
        β = isinf(eps_in) ? 2pi - angle : angle                  # opening of the dielectric sector
        return [k * pi / β - 1 for k in 1:floor(Int, gmax * β / pi)]
    end
    return junction_powers([2pi - angle, angle], [Float64(eps_out), Float64(eps_in)]; gmax) .- 1
end

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
    edge_gcq_rule(dpows; tol=1e-10, nsmooth=20, qmax=nothing)
    edge_gcq_rule(eps_in, eps_out; angle=π/2, tol=1e-10, nsmooth=20, qmax=nothing)

Weighted GCQ rule for the density powers `dpows` (several contrasts may be
combined with `vcat`). The family is {s^{p_k + n}} ∪ {s^n}, n = 0..nsmooth; with
`qmax` given it is instead every such exponent up to `qmax` (n as large as
needed), so `dpows` (the powers up to some gmax) choose the classes and `qmax`
alone sets the size. Steps: fine grid for the weighted family, SVD of the
weighted sample matrix (rank at `tol`), column-pivoted QR for the nodes,
transposed Vandermonde solve for the weights.

Fine grid (`grid`):
- `:graded` (default): dyadic segments [2^-k, 2^-k+1], k = 1..K, with 14-point
  Gauss-Legendre, and [0, 2^-K] with Gauss-Jacobi for the weight s^p1, so the
  weight is integrated exactly and p1 may approach -1 (checkerboard edges at high
  contrast). K is set so that the non-polynomial part of the family carries less
  than tol/10 on [0, 2^-K]: K ≈ log2(10/tol) / (1 + p1 + c), c the smallest
  non-integer exponent of v = σ/s^p1; a family needing K > 1000 (2^-K below the
  floating-point range) is refused.
- `:adaptive`: adaptive Gauss-Kronrod on s^p1 · family (fails for p1 ≲ -0.95).
In both, the interpolation matrix `P` gives v at the 14-point Gauss-Legendre
nodes of every fine segment (`segs`, `z0`).
"""
function edge_gcq_rule(dpows::AbstractVector; tol::Real = 1e-10, nsmooth::Int = 20,
                       qmax::Union{Nothing, Real} = nothing, grid::Symbol = :graded)
    p1 = minimum(dpows)
    p1 > -1 || throw(ArgumentError("density power $p1 is not integrable"))
    qmax === nothing || (nsmooth = ceil(Int, qmax - p1))
    q = sort(unique(round.(vcat([(p - p1) .+ (0:nsmooth) for p in dpows]...,
                                collect((0:nsmooth) .- p1)); digits = 12)))
    q = q[q .>= -1e-12]
    qmax === nothing || (q = q[q .+ p1 .<= qmax + 1e-12])
    fs = s -> [s^qi for qi in q]
    z0, w0 = gausslegendre(14)
    if grid == :adaptive
        _, _, segbuf, _ = quadgk_segbuf_count(s -> s^p1 .* fs(s), 0.0, 1.0;
                                              atol = tol / 10, rtol = tol / 10)
        segs = sort([(sg.a, sg.b) for sg in segbuf])
        xq = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs])
        ωq = reduce(vcat, [(b - a) / 2 .* w0 for (a, b) in segs]) .* xq .^ p1
        xi = xq
    elseif grid == :graded
        cs = [c for c in q if abs(c - round(c)) > 1e-9]
        cmin = isempty(cs) ? 1.0 : minimum(cs)
        K = max(20, ceil(Int, log2(10 / tol) / (1 + p1 + cmin)))
        K <= 1000 || throw(ArgumentError("family too singular near s = 0 for Float64 " *
                                         "(needs segments down to 2^-$K)"))
        δ = 2.0^(-K)
        segs = vcat([(0.0, δ)], [(2.0^(-k), 2.0^(-k + 1)) for k in K:-1:1])
        ngj = max(24, ceil(Int, maximum(q) / 2) + 4)
        tj, wj = gaussjacobi(ngj, 0.0, p1)               # weight (1 + t)^p1 on [-1, 1]
        xq = vcat(δ .* (1 .+ tj) ./ 2,
                  reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs[2:end]]))
        ωq = vcat((δ / 2)^(1 + p1) .* wj,
                  reduce(vcat, [(b - a) / 2 .* w0 for (a, b) in segs[2:end]]) .* xq[ngj+1:end] .^ p1)
        xi = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* z0 for (a, b) in segs])
    else
        throw(ArgumentError("grid must be :graded or :adaptive"))
    end
    A = reduce(vcat, [sqrt(ωj) .* fs(xj)' for (xj, ωj) in zip(xq, ωq)])
    S = svd(A)
    r = sum(S.S .> tol * S.S[1])
    U = S.U[:, 1:r]
    # node candidates: the graded grid's Gauss-Jacobi points integrate the weight on
    # [0, 2^-K] but are not offered as nodes (they would put a node ~1e-17 from the
    # edge, which the near-field and FMM machinery resolves poorly); the smooth factor
    # varies negligibly on [0, 2^-K], so nodes above it carry that mass as well
    cand = grid == :graded ? collect(ngj+1:length(xq)) : collect(eachindex(xq))
    nodeinds = sort(cand[qr(U[cand, :]', ColumnNorm()).p[1:r]])
    x = xq[nodeinds]
    Is = vec(sum(sqrt.(ωq) .* U, dims = 1))
    W = sqrt.(ωq[nodeinds]) .* (U[nodeinds, :]' \ Is)
    # unweighted basis F V Σ^-1 at the interpolation nodes and at the rule nodes
    B = S.V[:, 1:r] ./ S.S[1:r]'
    Ui = reduce(vcat, [fs(xj)' for xj in xi]) * B
    Un = (U ./ sqrt.(ωq))[nodeinds, :]
    P = (Ui / Un) * Diagonal(x .^ (-p1))
    return EdgeGCQRule(x, W ./ x .^ p1, p1, segs, z0, w0, P, q .+ p1, Float64(tol))
end

edge_gcq_rule(eps_in::Real, eps_out::Real; angle::Real = pi / 2, kw...) =
    edge_gcq_rule(edge_density_powers(eps_in, eps_out; angle); kw...)

"Largest relative error of the rule over its own family (∫₀¹ x^p = 1/(p+1))."
edge_gcq_family_error(r::EdgeGCQRule) =
    maximum(abs(sum(r.w .* r.x .^ p) * (p + 1) - 1) for p in r.pows)
