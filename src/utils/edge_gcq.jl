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
    # cached (one junction is ~0.5 s of determinant evaluations, and every mesh call asks for its junctions again)
    key = (Float64.(angles), Float64.(eps), Float64(gmax), ngrid)
    return copy(lock(() -> get!(() -> _junction_powers(angles, eps, gmax, ngrid), JUNCTION_POWERS_CACHE, key),
                     JUNCTION_POWERS_LOCK))
end

const JUNCTION_POWERS_CACHE = Dict{Tuple{Vector{Float64}, Vector{Float64}, Float64, Int}, Vector{Float64}}()
const JUNCTION_POWERS_LOCK = ReentrantLock()

function _junction_powers(angles, eps, gmax, ngrid)
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
    # A local minimum of |det| without a sign change on the grid is either a double
    # root or two simple roots inside one grid cell (e.g. γ1 ≈ 1 at low contrast,
    # where γ1 and 2 - γ1 nearly coincide). The extremum g0 tells them apart: if det
    # changes sign there, bracket one root on each side of it. Rounding at true
    # double roots is ~1e-30 fscale, far below the 1e-20 threshold.
    h = 1e-6
    df = g -> (f(g + h) - f(g - h)) / 2h
    for i in 2:length(gs)-1
        if abs(fs[i]) < abs(fs[i-1]) && abs(fs[i]) < abs(fs[i+1]) &&
           fs[i-1] * fs[i] > 0 && fs[i] * fs[i+1] > 0 && df(gs[i-1]) * df(gs[i+1]) < 0
            g0 = bisect(df, gs[i-1], gs[i+1])
            f0 = f(g0)
            if f0 * fs[i] < 0 && abs(f0) > 1e-20 * fscale
                append!(roots, (bisect(f, gs[i-1], g0), bisect(f, g0, gs[i+1])))
            elseif abs(f0) < 1e-10 * fscale && all(abs.(roots .- g0) .> 1e-8)
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
    edge_gcq_rule(dpows; tol=1e-10, nsmooth=20, qmax=nothing, ninterp=14)
    edge_gcq_rule(eps_in, eps_out; angle=π/2, tol=1e-10, nsmooth=20, qmax=nothing, ninterp=14)

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
nodes of every fine segment (`segs`, `z0`). With `floor` > 0 the nodes are chosen
among the fine points s ≥ floor only (the fine grid still resolves the family below).

`ninterp` sets the points per fine segment of the interpolant (the near-field density representation, degree
`ninterp - 1` per segment); the nodes, weights and retained subspace do not depend on it. The default 14 is the
construction above. Otherwise P samples the retained-space global interpolant (`GCQGlobalInterp`) at the
`ninterp`-point Gauss-Legendre nodes of every segment, in 128-bit arithmetic (~9 s for 840 exponents at 28 points).
Measured against the continuous interpolant (study 62, regularized density relative to its maximum, tol 1e-12):
- members of the trained family: 14 points leave ~1e-12 (one power) to ~2e-11 (a 40-power family), and more at a lower
  tol (~5e-11 at 1e-8, ~6e-10 at 1e-4); 20 points ~1e-15 to ~2e-13; 28 points ~1e-14;
- data outside the family (solved corner densities) and the cardinal functions of large families need more points.
  A 40-power family's cardinal functions are resolved to ~8e-3 at 20 points, ~2e-9 at 28 and ~1e-14 at 36.

Family accuracy (studies 67, 73):
- `grid_tol` (default `nothing`: the bare dyadic grid above; `density_tol / 10` with `density_tol`): before the SVD,
  halve the positive graded cells until every original regularized component f_j = s^q_j (maximum 1 on [0, 1]) is
  interpolated by the cell's 14-point polynomial to `grid_tol` at 33 Chebyshev points (an ArgumentError when 16
  halvings do not suffice). The bare dyadic grid leaves up to 7.5e-5 on an octahedron edge's 840-component family, and
  its weighted inner product then misjudges weak singular directions. `grid_tol = tol` is `polyhedron_mesh`'s default
  (the whole construction in Float64; study 73: the PEC capacitance of the cube to ~3e-15 at tol 1e-12, with
  `PlanarProductNearQuad`); the other meshes keep the bare grid (their tests were tuned on it).
- `density_tol` (`grid = :graded`; sets `grid_tol = density_tol / 10` unless given): instead of the singular-value
  rank at `tol`, the smallest rank r (from it up to `rank_max`, default 16 more, and at most the numerical rank
  σ_r > 10 eps σ_1) such that, within the selected-node range, the rule's own evaluator (`ninterp`-point Lagrange per
  fine cell through its P) and the retained global model both reconstruct every original component from its exact
  nodal values to `density_tol / 4` at 43 points per cell, the model's Lebesgue constant there is at most
  `lebesgue_max` (100) and the negative share of the regularized weights, Σ|W_i < 0| / Σ|W_i|, is at most
  `negative_weight_max` (1e-4: the stability factor Σ|W| / |ΣW| stays below 1.0002). Ranks between the
  singular-value rank and the certified one often have sign-alternating weights (shares 1e-3 to 3e-2) or Lebesgue
  constants near 200. The check is sampled in Float64 (agrees with 128-bit evaluation to 2-3 digits down to 1e-13; a
  gate below 1e-14 is not certified), not a proven bound, on the node range only. Errors on all positive cells and the
  pointwise error on the terminal element (s < 2^-K, where the evaluator's polynomial misses s^q by 1e-5 to 1e-2 of
  the component's maximum; its weighted contribution is small) are measured and reported, not certified. Without a
  passing rank the rank of smallest error (within the limits if any rank is) is kept, marked uncertified.
  `gcq_certificate(rule)` returns the record. With `grid_tol` the singular-value rank leaves ~1e-10 on the
  840-component family; `density_tol = 1e-12` takes rank 34 (the numerical rank) for ~1e-13, at a Lebesgue constant
  52 (14 at the singular-value rank) and a negative weight share 2.4e-6.
"""
function edge_gcq_rule(dpows::AbstractVector; tol::Real = 1e-10, nsmooth::Int = 20,
                       qmax::Union{Nothing, Real} = nothing, grid::Symbol = :graded, floor::Real = 0.0,
                       ninterp::Int = 14, grid_tol::Union{Nothing, Real} = nothing,
                       density_tol::Union{Nothing, Real} = nothing, rank_max::Union{Nothing, Int} = nothing,
                       lebesgue_max::Real = 100.0, negative_weight_max::Real = 1e-4)
    ninterp >= 2 || throw(ArgumentError("ninterp must be at least 2, got $ninterp"))
    grid_tol === nothing || grid_tol > 0 || throw(ArgumentError("grid_tol must be positive, got $grid_tol"))
    density_tol === nothing || density_tol > 0 || throw(ArgumentError("density_tol must be positive, got $density_tol"))
    rank_max === nothing || rank_max >= 1 || throw(ArgumentError("rank_max must be at least 1, got $rank_max"))
    density_tol !== nothing && grid != :graded && throw(ArgumentError("density_tol needs grid = :graded"))
    # a certified rank needs a component-resolved training grid (study 67: the default grid leaves 7.5e-5)
    density_tol !== nothing && grid_tol === nothing && (grid_tol = density_tol / 10)
    grid_tol !== nothing && grid != :graded && throw(ArgumentError("grid_tol needs grid = :graded"))
    p1 = minimum(dpows)
    p1 > -1 || throw(ArgumentError("density power $p1 is not integrable"))
    qmax === nothing || (nsmooth = ceil(Int, qmax - p1))
    q = sort(unique(round.(vcat([(p - p1) .+ (0:nsmooth) for p in dpows]...,
                                collect((0:nsmooth) .- p1)); digits = 12)))
    q = q[q .>= -1e-12]
    qmax === nothing || (q = q[q .+ p1 .<= qmax + 1e-12])
    fs = s -> [s^qi for qi in q]
    z0, w0 = gausslegendre(14)
    gstats = nothing
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
        gstats = grid_tol === nothing ? nothing : _gcq_resolve_segments!(segs, q, z0, Float64(grid_tol))
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
    # node candidates: the graded grid's Gauss-Jacobi points integrate the weight on
    # [0, 2^-K] but are not offered as nodes (they would put a node ~1e-17 from the
    # edge, which the near-field and FMM machinery resolves poorly); the smooth factor
    # varies negligibly on [0, 2^-K], so nodes above it carry that mass as well
    cand = grid == :graded ? collect(ngj+1:length(xq)) : collect(eachindex(xq))
    cand = filter(i -> xq[i] >= floor, cand)
    cert = Dict{String, Any}()
    zf, wf = ninterp == 14 ? (z0, w0) : gausslegendre(ninterp)
    if density_tol !== nothing
        r, cert = _gcq_density_rank(S, q, xq, ωq, cand, segs, zf, wf, r, Float64(density_tol), rank_max,
                                    Float64(lebesgue_max), Float64(negative_weight_max))
    end
    gstats === nothing || merge!(cert, gstats)
    U = S.U[:, 1:r]
    nodeinds = sort(cand[qr(U[cand, :]', ColumnNorm()).p[1:r]])
    x = xq[nodeinds]
    Is = vec(sum(sqrt.(ωq) .* U, dims = 1))
    W = sqrt.(ωq[nodeinds]) .* (U[nodeinds, :]' \ Is)
    # unweighted basis F V Σ^-1 at the interpolation nodes and at the rule nodes
    B = S.V[:, 1:r] ./ S.S[1:r]'
    Ui = reduce(vcat, [fs(xj)' for xj in xi]) * B
    Un = (U ./ sqrt.(ωq))[nodeinds, :]
    P = (Ui / Un) * Diagonal(x .^ (-p1))
    if density_tol !== nothing
        wts = W ./ x .^ p1
        cert["min_weight"] = minimum(wts); cert["cond_U_selected"] = cond(U[nodeinds, :])
        cert["moment_error"] = maximum(abs(sum(wts .* x .^ (p1 + u)) * (p1 + u + 1) - 1) for u in q)
        cert["fvs_nodal_identity"] = _gcq_nodal_identity(P, segs, z0, w0, x, p1)
    end
    gdata = GCQGlobalInterp(q, p1, x, S.V[:, 1:r], S.S[1:r]; cert)
    # the interpolation grid (zf, wf): the training's 14 points unless ninterp; P sampled from the retained model in
    # 128-bit arithmetic for ninterp != 14 and for density_tol (at its rank, the Float64 FV/S map misses cardinal
    # directions: nodal identity error up to 4e-2 on an 840-component family, study 67)
    if ninterp != 14 || density_tol !== nothing
        P = _gcq_sample_global(gdata, reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* zf for (a, b) in segs]))
    end
    if density_tol !== nothing
        # the certificate is of the rule's own evaluator (Lagrange per fine cell through this P), measured after the
        # build at the certificate's holdouts; the search above used the Float64 model at the fine nodes
        cert["map"] = "128-bit sampling of the retained model"
        en, ep, et = _gcq_evaluator_errors(P, segs, zf, wf, q, x, p1)
        cert["evaluator_error_node_range"] = en; cert["evaluator_error_positive"] = ep; cert["evaluator_error_terminal"] = et
        cert["evaluator"] = "$(length(zf))-point Lagrange per fine cell through P"
        cert["certified"] = cert["certified"] && en <= cert["gate_1d"]
    end
    rule = EdgeGCQRule(x, W ./ x .^ p1, p1, segs, zf, wf, P, q .+ p1, Float64(tol))
    _gcq_global_register!(P, gdata)
    return rule
end

# max |R(x_i) - e_i| of the regularized local interpolant (cells of the GL points z, weights w) through P at the nodes x
function _gcq_nodal_identity(P, segs, z, w, x, pw)
    bw = gl_barycentric_weights(z, w); nz = length(z)
    e = 0.0
    for (i, xi) in enumerate(x)
        k = clamp(searchsortedlast(segs, (xi, Inf)), 1, length(segs)); sa, sb = segs[k]
        row = vec(_tensor_lagrange_mat(z, bw, [2 * (xi - sa) / (sb - sa) - 1])' * P[(k-1)*nz .+ (1:nz), :]) .* x .^ pw
        row[i] -= 1
        e = max(e, maximum(abs, row))
    end
    return e
end

# the regularized global interpolant at the points ts (full nodal values -> v = σ / s^pw), f(t) V Hn⁻¹ diag(x^-pw), in
# `prec`-bit arithmetic from the stored Float64 exponents, right singular vectors and nodes (exact inputs)
function _gcq_sample_global(g, ts; prec::Int = 128)        # g: a GCQGlobalInterp
    setprecision(BigFloat, prec) do
        q = big.(g.q); V = big.(g.V); x = big.(g.x)
        M = ([xi^qi for xi in x, qi in q] * V) \ Diagonal(x .^ (-big(g.pw)))
        out = zeros(length(ts), length(x))
        for c0 in 1:256:length(ts)                              # in chunks: m BigFloat powers per point
            rows = c0:min(c0 + 255, length(ts))
            out[rows, :] .= Float64.(([max(big(t), floatmin(BigFloat))^qi for t in ts[rows], qi in q] * V) * M)
        end
        return out
    end
end

# grid_tol: halve the positive cells of `segs` (in place, segment 1 the weighted terminal element) until every regularized
# component s^q (maximum 1 on [0, 1]) is interpolated at the cell's GL points `z` to grid_tol at 33 second-kind Chebyshev
# points; on a cell [a, b] the error of s^q is b^q times its error on [a/b, 1] (cached by the ratio)
function _gcq_resolve_segments!(segs, q, z, grid_tol; N = 33, maxdepth = 16)
    bw = gl_barycentric_weights(z, gausslegendre(length(z))[2])
    τ = [-cos(π * j / (N - 1)) for j in 0:N-1]
    L = transpose(_tensor_lagrange_mat(z, bw, τ))
    cache = Dict{Float64, Vector{Float64}}()
    function err(a, b)
        ρ = a / b
        e = get!(cache, ρ) do
            sn = ρ .+ (1 - ρ) .* (1 .+ z) ./ 2; st = ρ .+ (1 - ρ) .* (1 .+ τ) ./ 2
            vec(maximum(abs.(L * [x^u for x in sn, u in q] - [x^u for x in st, u in q]); dims = 1))
        end
        return maximum(e .* b .^ q)
    end
    out = [segs[1]]; maxerr = 0.0; splits = 0
    stack = [(a, b, 0) for (a, b) in reverse(segs[2:end])]
    while !isempty(stack)
        a, b, depth = pop!(stack); e = err(a, b)
        if e <= grid_tol
            push!(out, (a, b)); maxerr = max(maxerr, e)
        else
            depth < maxdepth || throw(ArgumentError("edge_gcq_rule: the family's components are not resolved to " *
                                                    "grid_tol = $grid_tol on [$a, $b] after $maxdepth halvings (error $e)"))
            m = (a + b) / 2
            push!(stack, (m, b, depth + 1), (a, m, depth + 1)); splits += 1
        end
    end
    base = length(segs)
    resize!(segs, length(out)); segs .= out
    return Dict{String, Any}("grid_tol" => grid_tol, "cells_base" => base, "cells" => length(out), "splits" => splits,
                             "component_resolution" => maxerr)
end

# the certificate's holdouts: 43 first-kind Chebyshev points (with the ends) per positive cell, 33 Gauss-Legendre points
# on the terminal element
_gcq_cert_tau() = vcat(-1.0, [-cos((2j - 1) * π / 82) for j in 1:41], 1.0)

# max over all original components of |evaluator - s^q| for the rule's evaluator (Lagrange at the GL points zf of each
# fine cell through P, from the exact full nodal values x^(pw + q)): (selected-node range, positive cells, terminal)
function _gcq_evaluator_errors(P, segs, zf, wf, q, x, pw)
    bw = gl_barycentric_weights(zf, wf); nz = length(zf)
    τ = _gcq_cert_tau(); zt, _ = gausslegendre(33)
    Lh = _tensor_lagrange_mat(zf, bw, τ)'; Lt = _tensor_lagrange_mat(zf, bw, zt)'
    Fx = [xi^(pw + u) for xi in x, u in q]; xmin, xmax = extrema(x)
    en = ep = 0.0
    for k in 2:length(segs)
        a, b = segs[k]; t = a .+ (b - a) .* (1 .+ τ) ./ 2
        E = abs.(Lh * (P[(k-1)*nz .+ (1:nz), :] * Fx) - [ti^u for ti in t, u in q])
        ep = max(ep, maximum(E))
        inr = xmin .<= t .<= xmax
        any(inr) && (en = max(en, maximum(E[inr, :])))
    end
    t = segs[1][2] .* (1 .+ zt) ./ 2
    et = maximum(abs.(Lt * (P[1:nz, :] * Fx) - [ti^u for ti in t, u in q]))
    return en, ep, et
end

# density_tol: the smallest rank r in r0 .. rmax whose retained model (QR-selected nodes from the training points
# `cand`) and the rule's evaluator (the model at the fine GL points zf, Lagrange per cell; Float64 here, measured again
# on the built 128-bit P) reconstruct every component s^q from its exact nodal values to density_tol / 4 at the
# certificate's holdouts within the selected-node range, with the model's Lebesgue constant there at most lebesgue_max
# and the negative share of the regularized weights (Σ|W_i<0| / Σ|W_i|) at most negative_weight_max. rmax is capped at
# the numerical rank (σ_r > 10 eps σ_1) and at rank_max (default r0 + 16). Else the rank of smallest error (within the
# limits when any rank is), uncertified. A gate below the Float64 check's floor (1e-14) is never certified.
function _gcq_density_rank(S, q, xq, ωq, cand, segs, zf, wf, r0, density_tol, rank_max, lebesgue_max, negative_weight_max)
    rnum = count(>(10 * eps() * S.S[1]), S.S)
    rsv = r0
    rmax = min(length(q), length(cand), rnum, rank_max === nothing ? r0 + 16 : rank_max)
    r0 = min(r0, rmax)
    τ = _gcq_cert_tau(); zt, _ = gausslegendre(33)
    bw = gl_barycentric_weights(zf, wf); nz = length(zf)
    Lh = _tensor_lagrange_mat(zf, bw, τ)'; Lt = _tensor_lagrange_mat(zf, bw, zt)'
    th = reduce(vcat, [a .+ (b - a) .* (1 .+ τ) ./ 2 for (a, b) in segs[2:end]])
    tt = segs[1][2] .* (1 .+ zt) ./ 2
    xf = reduce(vcat, [(a + b) / 2 .+ (b - a) / 2 .* zf for (a, b) in segs])
    Fh = [t^u for t in th, u in q]; Ft = [t^u for t in tt, u in q]
    Vr = S.V[:, 1:rmax]
    Hh = Fh * Vr; Ht = Ft * Vr; Hz = [t^u for t in xf, u in q] * Vr
    sw = sqrt.(ωq)
    gate = density_tol / 4; floor64 = 1e-14
    nh = length(τ); ncell = length(segs) - 1
    best = nothing; bestkey = (2, Inf); trace = Dict{String, Any}[]
    for r in r0:rmax
        U = S.U[:, 1:r]
        ii = sort(cand[qr(U[cand, :]', ColumnNorm()).p[1:r]])
        x = xq[ii]; xmin, xmax = extrema(x)
        Fx = [xi^u for xi in x, u in q]; Hn = Fx * S.V[:, 1:r]
        C = Hn \ Fx
        Eh = abs.(Hh[:, 1:r] * C - Fh); Et = abs.(Ht[:, 1:r] * C - Ft)
        inr = xmin .<= th .<= xmax
        en = maximum(Eh[inr, :])
        Mz = Hz[:, 1:r] * C                                       # the model at the fine GL points
        Ev = abs.(reduce(vcat, [Lh * Mz[k*nz .+ (1:nz), :] for k in 1:ncell]) - Fh)
        ev = maximum(Ev[inr, :])
        Is = vec(sum(sw .* U, dims = 1))
        W = sw[ii] .* (U[ii, :]' \ Is)                            # weights of the regularized density
        negf = sum(abs, W[W .< 0]; init = 0.0) / sum(abs, W)
        Λ = maximum(sum(abs.(Hh[inr, 1:r] / Hn), dims = 2))
        within = Λ <= lebesgue_max && negf <= negative_weight_max
        pass = within && en <= gate && ev <= gate && gate >= floor64
        rec = Dict{String, Any}("rank" => r, "model_error_node_range" => en, "model_error_positive" => maximum(Eh),
                                "model_error_terminal" => maximum(Et), "evaluator_error_node_range_float64" => ev,
                                "x_min" => xmin, "x_max" => xmax, "sigma_ratio" => S.S[r] / S.S[1],
                                "lebesgue_node_range" => Λ, "negative_weight_fraction" => negf, "within_limits" => within)
        push!(trace, Dict{String, Any}(k => rec[k] for k in ("rank", "model_error_node_range", "evaluator_error_node_range_float64",
                                                              "lebesgue_node_range", "negative_weight_fraction", "sigma_ratio")))
        key = (within ? 0 : 1, max(en, ev))
        key < bestkey && (best = rec; bestkey = key)
        pass && (best = rec; break)
    end
    certified = best["within_limits"] && max(best["model_error_node_range"], best["evaluator_error_node_range_float64"]) <= gate &&
                gate >= floor64
    merge!(best, Dict{String, Any}("certified" => certified, "domain" => "selected-node range",
                                   "density_tol" => density_tol, "gate_1d" => gate, "rank_singular_value" => rsv,
                                   "rank_numerical" => rnum, "rank_max" => rmax, "lebesgue_max" => lebesgue_max,
                                   "negative_weight_max" => negative_weight_max, "float64_floor" => floor64,
                                   "holdouts_per_cell" => nh, "sampled" => true, "trace" => trace))
    return best["rank"], best
end

edge_gcq_rule(eps_in::Real, eps_out::Real; angle::Real = pi / 2, kw...) =
    edge_gcq_rule(edge_density_powers(eps_in, eps_out; angle); kw...)

"""
    GCQGlobalInterp

The global interpolant of a weighted GCQ rule in its retained subspace, one formula on the whole of (0, 1]:
    σ(s) = s^pw h(s) c,   h(s) = f(s) V,   Hn c = σ(x) ./ x.^pw,   Hn = h(x)  (the nodes x),
with f(s) = [s^q_1, …, s^q_m] the regularized family and V (m × r) its retained right singular vectors (the weighted
training SVD of `edge_gcq_rule`, rank r = number of nodes), S the retained singular values. The rule's interpolation
matrix P is this interpolant at the fine nodes (through the SVD's own nodal factor), and its piecewise evaluator
(`tensor_rule_interp`) interpolates P in each fine segment; this object evaluates the formula itself at any point
(`global_interp!`). Hn is formed directly from f and V (not through 1/S) and LU-factored once. Powers run along the
integer chains of the exponents: `pred[k]` is the exponent q[pred[k]] = q[k] - 1 - δ[k] (|δ| ≤ 1e-10: the exponents
are rounded to 12 digits, so q[k] - 1 need not be one of them exactly), s^q[k] = s^q[pred[k]] s s^δ[k] with
s^δ = 1 + δ L + (δ L)²/2 (L = log s; exact to rounding for |δ L| ≤ 1e-8); `pred[k] = 0` for a power by `^`, in the order
`order` (predecessors first). Read-only after construction (safe to share between threads).
"""
struct GCQGlobalInterp
    q::Vector{Float64}
    pw::Float64
    x::Vector{Float64}
    V::Matrix{Float64}
    S::Vector{Float64}
    Hn::Matrix{Float64}
    lu::LU{Float64, Matrix{Float64}, Vector{Int}}
    xpw::Vector{Float64}             # x .^ (-pw)
    pred::Vector{Int}
    δ::Vector{Float64}
    order::Vector{Int}
    cert::Dict{String, Any}          # edge_gcq_rule's family-accuracy record (density_tol / grid_tol), else empty
end

function GCQGlobalInterp(q::Vector{Float64}, pw::Float64, x::Vector{Float64}, V::Matrix{Float64}, S::Vector{Float64};
                         cert::Dict{String, Any} = Dict{String, Any}())
    size(V) == (length(q), length(x)) && length(S) == length(x) ||
        throw(DimensionMismatch("GCQGlobalInterp: V must be (exponents × nodes) and S one value per node"))
    pred = zeros(Int, length(q)); δ = zeros(length(q))
    for k in eachindex(q)
        j = argmin(abs.(q .- (q[k] - 1.0)))
        d = q[k] - q[j] - 1.0
        abs(d) <= 1e-10 && (pred[k] = j; δ[k] = d)
    end
    order = sortperm(q)
    Hn = [t^qk for t in x, qk in q] * V
    return GCQGlobalInterp(q, pw, x, V, S, Hn, lu(Hn), x .^ (-pw), pred, δ, order, cert)
end

Base.show(io::IO, g::GCQGlobalInterp) =
    print(io, "GCQGlobalInterp ($(length(g.q)) exponents, rank $(length(g.x)), weight x^$(round(g.pw, digits = 4)))")

# The global data of the rules built by edge_gcq_rule, by objectid of their P (shared by the EdgeGCQRule and the
# TensorRule1D made from it): read without a lock from an immutable snapshot, replaced (copy-on-write) under the lock.
# P is held weakly: a rule's data live as long as its P (meshes build their rules per call, most never use the global
# backend), and a registration drops the entries of collected rules; an entry whose P is gone does not match a new array
# that reuses its objectid. A rule whose P is not registered (deserialized, deep-copied, or assembled by hand) has no
# data, and asking for it is an error; the struct layouts of the rules are unchanged. Rules built while precompiling are
# not registered (their arrays do not survive into the session).
const GCQ_GLOBAL = Ref(Dict{UInt, Tuple{WeakRef, GCQGlobalInterp}}())
const GCQ_GLOBAL_LOCK = ReentrantLock()

function _gcq_global_register!(P::Matrix{Float64}, g::GCQGlobalInterp)
    ccall(:jl_generating_output, Cint, ()) == 1 && return g
    lock(GCQ_GLOBAL_LOCK) do
        d = filter(e -> e.second[1].value !== nothing, GCQ_GLOBAL[])
        d[objectid(P)] = (WeakRef(P), g)
        GCQ_GLOBAL[] = d
    end
    return g
end

"""
    gcq_global_interp(r) -> GCQGlobalInterp

The retained global interpolation data of a GCQ rule `r` (an `EdgeGCQRule`, or a `TensorRule1D` made from one by
`gcq_tensor_rule`). An `ArgumentError` for a rule without them: a Gauss-Legendre rule, or a GCQ rule not built by
`edge_gcq_rule` in this session (deserialized or copied, its P a different array) or whose nodes or weight power differ
from those the data were built for.
"""
function gcq_global_interp(P::Matrix{Float64}, x::AbstractVector, pw::Real)
    e = get(GCQ_GLOBAL[], objectid(P), nothing)
    (e === nothing || e[1].value !== P) && throw(ArgumentError("gcq_global_interp: no global interpolation data for this " *
        "rule (not built by edge_gcq_rule in this session: rebuild it, e.g. after deserialization)"))
    g = e[2]
    g.x == x && g.pw == pw || throw(ArgumentError("gcq_global_interp: the rule's nodes or weight power differ from " *
                                                  "those of its global interpolation data"))
    return g
end
gcq_global_interp(r::EdgeGCQRule) = gcq_global_interp(r.P, r.x, r.pw)

"""
    gcq_certificate(rule) -> Dict

The family-accuracy record of a GCQ rule built with `edge_gcq_rule(...; grid_tol, density_tol)` (empty otherwise): for
`density_tol`, "certified" (the sampled gate density_tol / 4 on the domain "selected-node range", met by the rule's
evaluator and the retained model, within the limits), "rank", "rank_singular_value", "rank_numerical",
"evaluator_error_node_range" / "_positive" / "_terminal" (the rule's evaluator through its P, measured after the
build), "model_error_node_range" / "_positive" / "_terminal" (the retained global model), "lebesgue_node_range",
"negative_weight_fraction" with their limits "lebesgue_max" / "negative_weight_max", "cond_U_selected",
"min_weight", "moment_error", "fvs_nodal_identity", "map", and "trace" (per searched rank); for `grid_tol`, "cells",
"component_resolution". Sampled values, not proven bounds. The Dict is the rule's own (shared, not a copy).
"""
gcq_certificate(r) = gcq_global_interp(r).cert

"""
    universal_gcq_rule(pmin; qmax = 20.0, h = 0.01, tol = 1e-10, floor = 1e-10, ninterp = 14)

Universal weighted GCQ rule: one rule for every density power in [pmin, qmax], for edges whose powers vary
along the curve (sector angles changing) or are not known. The family is {s^q : q ∈ pmin:h:qmax} ∪ {s^n},
refined to the step h/10 on [pmin, pmin + 10h] (the step h alone leaves ~1e-11 at exponents just above pmin),
built by `edge_gcq_rule` with the weight s^pmin. A continuous band of exponents also spans s^q log^k s, the
terms a varying power generates. Left free, the nodes reach much closer to s = 0 than a dedicated rule's (~1e-17 of
the panel at tol 1e-12, pmin = -0.3), below the rounding of absolute positions; `floor` keeps them at s ≥ floor. For
pmin ≳ -0.5 the default 1e-10 costs nothing (tol 1e-12: exponents in [pmin, 10] to ~1e-13, s^q log s to ~2e-12);
as pmin → -1 a floor costs accuracy.
"""
function universal_gcq_rule(pmin::Real; qmax::Real = 20.0, h::Real = 0.01, tol::Real = 1e-10, floor::Real = 1e-10,
                            ninterp::Int = 14)
    pmin > -1 || throw(ArgumentError("density power $pmin is not integrable"))
    0 < h < 1 || throw(ArgumentError("the step h must be in (0, 1)"))
    dpows = pmin .+ unique(vcat(collect(0:h:1-h/2), collect(0:h/10:10h)))
    return edge_gcq_rule(dpows; tol, qmax, floor, ninterp)
end

"Largest relative error of the rule over its own family (∫₀¹ x^p = 1/(p+1))."
edge_gcq_family_error(r::EdgeGCQRule) =
    maximum(abs(sum(r.w .* r.x .^ p) * (p + 1) - 1) for p in r.pows)
