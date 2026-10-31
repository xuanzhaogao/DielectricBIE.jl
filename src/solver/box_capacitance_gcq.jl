# Capacitance of a conducting box on a GCQ mesh, with the mesh's symmetry: the equilibrium density
# is invariant under the box's symmetry group, so the system is reduced to one unknown per node orbit,
# assembled exactly (node rule plus the sparse near corrections, no FMM) and solved densely.

"""
    box_node_orbits(interface, Lx, Ly, Lz; center = (0, 0, 0)) -> (orbit, reps)

Node orbits of a box mesh (`single_dielectric_box3d_gcq`) under the symmetry group of the box: the
reflections in its three mid-planes, and the permutations of the axes of equal side length (48 elements
for a cube, 16 for a square prism, 8 otherwise). `orbit[j]` is the orbit of node `j`, `reps[o]` the
first node of orbit `o`. Throws `ArgumentError` unless the mesh is symmetric to the last bit (the mirror
images of every node are nodes with the same weight), which `single_dielectric_box3d_gcq` meshes are.
"""
function box_node_orbits(interface::DielectricInterface, Lx::Real, Ly::Real, Lz::Real;
                         center = (0.0, 0.0, 0.0))
    L = (Lx, Ly, Lz)
    groups = [[k for k in 1:3 if L[k] == l] for l in unique(L)]          # axes of equal length
    perms = [p for p in _permutations3() if all(p[k] in g for g in groups for k in g)]
    # the distinct images of x under the group (the reflections and the allowed axis permutations)
    nimages(x) = length(Set(Tuple(s[k] * (x[p[k]] - center[p[k]]) for k in 1:3)
                            for p in perms, s in Iterators.product((-1, 1), (-1, 1), (-1, 1))))
    X = reduce(vcat, [p.points for p in interface.panels])
    w = all_weights(interface)
    key(x) = Tuple(reduce(vcat, [sort([abs(x[k] - center[k]) for k in g]) for g in groups]))
    ids = Dict{NTuple{3, Float64}, Int}()
    orbit = zeros(Int, length(X)); reps = Int[]
    for (j, x) in enumerate(X)
        orbit[j] = get!(ids, key(x)) do
            push!(reps, j); length(reps)
        end
    end
    # the mesh is invariant iff every orbit holds all the images of its representative, with one weight
    sz = zeros(Int, length(reps)); foreach(o -> sz[o] += 1, orbit)
    all(o -> sz[o] == nimages(X[reps[o]]), eachindex(reps)) && all(j -> w[j] == w[reps[orbit[j]]], eachindex(w)) ||
        throw(ArgumentError("the mesh is not symmetric under the box's symmetry group"))
    return orbit, reps
end

_permutations3() = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

"""
    symmetric_dense_operator(interface, orbit, reps; quad_tol = 1e-12, η_gcq = 2.0) -> M

The operator A = τ I + Dᵀ + C of a `TensorPanel` interface (`lhs_dielectric_box3d_gcq_dense`) restricted
to densities constant on the node orbits: `M[o, o′] = Σ_{j ∈ o′} A[reps[o], j]`, one row per orbit
(`box_node_orbits`). Exact for symmetric densities when the operator commutes with the symmetry. The rows
are direct sums (the node rule, no FMM) plus the near corrections of `laplace3d_DT_tensor_corrections`
(`quad_tol`, `η_gcq`); the corrections are computed for all targets and the dense matrix is only
`length(reps)²`.
"""
function symmetric_dense_operator(interface::DielectricInterface{TensorPanel{Float64}, Float64},
                                  orbit::AbstractVector{<:Integer}, reps::AbstractVector{<:Integer};
                                  quad_tol = 1e-12, η_gcq = 2.0)
    X = reduce(vcat, [p.points for p in interface.panels])
    Nx = reduce(vcat, [fill(p.normal, num_points(p)) for p in interface.panels])
    w = all_weights(interface); τ = _tensor_tau(interface)
    N = length(X); no = length(reps)
    # pairs FMM3D would skip are skipped by the near corrections' plain term too (see
    # laplace3d_DT_tensor_corrections), so the node rule skips them here
    rcut = 2.0^-54 * maximum(maximum(x[d] for x in X) - minimum(x[d] for x in X) for d in 1:3)
    M = zeros(no, no)
    Threads.@threads :dynamic for o in 1:no
        i = reps[o]; xi = X[i]; ni = Nx[i]
        @inbounds for j in 1:N
            (j == i || norm(xi .- X[j]) < rcut) && continue
            M[o, orbit[j]] += laplace3d_grad(X[j], xi, ni) * w[j]
        end
        M[o, o] += τ[i]
    end
    # near corrections of the representative rows, column by column (CSC, no transposed copy); each
    # thread owns the columns of M of its source orbits
    C, _ = laplace3d_DT_tensor_corrections(interface; quad_tol, η_gcq)
    repo = zeros(Int, N); foreach(o -> repo[reps[o]] = o, 1:no)
    cols = [Int[] for _ in 1:no]; foreach(j -> push!(cols[orbit[j]], j), 1:N)
    rv, nz = rowvals(C), nonzeros(C)
    Threads.@threads :dynamic for o′ in 1:no
        for j in cols[o′], k in nzrange(C, j)
            o = repo[rv[k]]
            o > 0 && (M[o, o′] += nz[k])
        end
    end
    return M
end

"""
    box_capacitance_gcq(Lx, Ly, Lz; nlev = 8, n_quad = 12, tol = 1e-14, qmax = 12.0, quad_tol = 1e-14,
                        η_gcq = 2.0, targets = [(0, 0, 0)]) -> NamedTuple

Capacitance of a conducting box `[-Lx/2, Lx/2] × [-Ly/2, Ly/2] × [-Lz/2, Lz/2]` in vacuum, normalized as
C / (4πε₀) (lengths in the units of `Lx`; the unit cube gives 0.66067815409957 [Helsing & Perfekt 2013]).

The equilibrium density σ of total charge Q = 1 solves the conductor equation (−½ I + Dᵀ) σ = 0 with the
total-charge constraint of `solve_dielectric_box3d_gcq`,

    (−½ I + Dᵀ − ½ u wᵀ) σ = −½ u Q,    u = 1 / |∂Ω|,

on a `single_dielectric_box3d_gcq` mesh (`nlev`, `n_quad`, `eps_in = Inf`) with the exact conductor edge
powers (`edge_gcq_rule` at `tol`, `qmax`). σ is constant on the node orbits of the box's symmetry group,
so the system is `symmetric_dense_operator` (one row per orbit; `quad_tol`, `η_gcq`) solved by LU.

Two estimates of C are returned:
- `C`: the Thomson energy Q² / (4π ∫σ Sσ), Sσ at one node per orbit (`potential_tensor` on the
  surface). The energy is stationary at the equilibrium density, so its error is second order in the
  error of σ: the default mesh (N ≈ 150k for a cube, ~40 s on 32 threads) gives the unit cube's C to
  ~5e-15, and every run with `n_quad ≥ 12`, `nlev ≥ 8` agrees to ±1.5e-15.
- `C_point`: Q / (4π U(x)) at the interior `targets` (first order in the error of σ; ~1e-13 here).

Sσ at the nodes of the Gauss-Legendre panels is SCTL's Duffy self row ([`potential_tensor`](@ref)): `n_quad` must be
one of SCTL's orders (4, 8, 12, 16, 20; otherwise an `ArgumentError`), and `using SCTL` is needed.

Returns `(; C, C_point, Q, σ, interface, orbit, reps)`, with `Q = wᵀσ` the discrete total charge.
"""
function box_capacitance_gcq(Lx::Real, Ly::Real, Lz::Real; nlev::Int = 8, n_quad::Int = 12, tol::Real = 1e-14,
                             qmax::Real = 12.0, quad_tol::Real = 1e-14, η_gcq::Real = 2.0,
                             targets = [(0.0, 0.0, 0.0)])
    rule = edge_gcq_rule(edge_density_powers(Inf, 1.0; gmax = 2.0); tol, qmax)
    itf = single_dielectric_box3d_gcq(Float64(Lx), Float64(Ly), Float64(Lz), Inf, 1.0; nlev, n_quad, rule)
    orbit, reps = box_node_orbits(itf, Lx, Ly, Lz)
    w = all_weights(itf)
    area = sum(w)
    W = zeros(length(reps)); foreach(j -> W[orbit[j]] += w[j], eachindex(w))
    M = symmetric_dense_operator(itf, orbit, reps; quad_tol, η_gcq)
    M .-= (0.5 / area) .* W'                                  # the total-charge constraint, Q = 1
    s = M \ fill(-0.5 / area, length(reps))
    σ = s[orbit]
    Q = dot(w, σ)
    X = reduce(vcat, [p.points for p in itf.panels])
    # Sσ at one node per orbit and U at the interior targets, in one call (one set of SCTL elements and near lists)
    no = length(reps)
    Uall = potential_tensor(itf, σ, vcat(X[reps], collect(targets)); quad_tol = 1e-16)
    E = sum(W[o] * s[o] * Uall[o] for o in eachindex(reps))    # ∫σ Sσ
    U = Uall[no+1:end]
    return (; C = Q^2 / (4π * E), C_point = Q ./ (4π .* U), Q, σ, interface = itf, orbit, reps)
end
