```@meta
CurrentModule = DielectricBIE
```

# User guide

This page covers the solver features beyond the [quickstart](index.md): FMM acceleration,
volume sources, many-right-hand-side solves, and visualization. For large distributed
four-index runs, see [Lattice campaigns](campaign.md).

## FMM-accelerated solves

For larger problems, replace the dense operator with an FMM-accelerated `LinearMap` and
solve with GMRES. In 3D, [`dielectric_operator`](@ref) combines an FMM far field with SCTL.jl's
near-field correction, which is needed for accuracy near the surface (see the operator backends
below).

```julia
using DielectricBIE

interface = single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs_fmm = lhs_dielectric_box2d_fmm2d(interface, 1e-12)
rhs = rhs_dielectric_box2d(interface, PointSource((0.1, 0.1), 1.0), 5.0)

# solve_gmres(A, b, atol, rtol)
sigma = solve_gmres(lhs_fmm, rhs, 1e-12, 1e-12)
```

## Boxes with GCQ edge panels

At the edges of a dielectric box the surface density is singular,
σ ~ Σₖ cₖ s^{pₖ} in the distance s to the edge. The powers pₖ come from the 2D wedge
determinant (`edge_density_powers`, `junction_powers`), and `edge_gcq_rule` builds a
weighted generalized Chebyshev quadrature (GCQ) on [0,1] that integrates the expansion
times smooth functions. `single_dielectric_box3d_gcq` uses it on `TensorPanel`s: GCQ
across each edge × Gauss–Legendre along it, GCQ × GCQ at the vertices, and dyadic
refinement toward the vertices only. Near-singular interactions are a sparse correction
added to the FMM operator.

```julia
using DielectricBIE, SCTL, LinearAlgebra   # SCTL: the near quadrature of the Gauss–Legendre panels
import DielectricBIE as BI

eps_in = 10.0
rule = edge_gcq_rule(eps_in, 1.0; tol = 1e-12)          # 28 nodes
interface = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, eps_in, 1.0;
                                        nlev = 3, n_quad = 8, rule)
A, stats = lhs_dielectric_box3d_gcq_fmm3d(interface, 1e-12)
rhs = rhs_dielectric_box3d(interface, PointSource((0.1, 0.15, 0.2), 1.0), eps_in)
sigma, _ = solve_dielectric_box3d_gcq(A, rhs, interface; rtol = 1e-12)
dot(BI.all_weights(interface), sigma) - (1 - 1 / eps_in)   # ≈ 5e-9
```

The error then falls by about 3× per vertex level, the rate of the vertex
singularity (σ ~ ρ^{−0.397} for ε = 10). Use a Gauss–Legendre order of at least 6:
lower orders cannot resolve the part of σ driven by the source.

The rule integrates the edge expansion up to a largest exponent `qmax`. The density alone is
resolved with `qmax ≈ 10`, but the far-field rows integrate the density *times* the kernel,
which adds smooth content: for targets one to two panel widths away the error is about 1e-9
at `qmax = 10` and reaches 1e-12 at `qmax ≈ 20–24`, for 4–5 more nodes per edge panel.

## Several bodies, high contrast and perfect conductors

Several boxes are meshed separately (one `single_dielectric_box3d_gcq` per body, placed with
`center`) and solved together. `solve_dielectric_boxes_gcq` joins the bodies, assembles the
FMM operator and the right-hand side (each point charge scaled by the permittivity of the
region containing it), and solves with one total-charge constraint per body:

```julia
using DielectricBIE, SCTL, LinearAlgebra

R = 1e4                                                   # contrast; Inf = perfect conductor
rule = edge_gcq_rule(R, 1.0; tol = 1e-12, qmax = 24.0)
parts = [single_dielectric_box3d_gcq(1.0, 1.0, 1.0, R, 1.0; nlev = 4, n_quad = 8, rule, center = c)
         for c in ((-0.75, 0.0, 0.0), (0.75, 0.0, 0.0))]
sol = solve_dielectric_boxes_gcq(parts, PointSource((0.0, 0.9, 0.1), 1.0))
sol.σ, sol.interface, sol.bodies                          # density, joined interface, node ranges
```

**The total-charge constraint.** At high contrast (and for `eps_in = Inf`) the plain system
does not pin down each body's total charge: its error grows like ε_in/ε_out. Gauss's law fixes
the total charge of every body,

    Q_c = Σ_{charges q inside c} q (1/ε_out − 1/ε_in(c))      (0 for a body without charge),

and the solver imposes it through a rank-one term per body, keeping the system square:

    (A − ½ Σ_c u_c w_cᵀ) σ = b − ½ Σ_c u_c Q_c,    u_c = 1_c / |∂c|,

with w_c the quadrature weights of body c. `deflate = false` gives the plain system;
`box_charge_constraints` builds the constraints for `solve_dielectric_box3d_gcq` when the
operator is assembled by hand.

**Perfect conductors.** Pass `eps_in = Inf`: `dielectric_tau` gives τ = −1/2, and
`edge_density_powers(Inf, 1.0)` gives the conductor edge powers γ_k = kπ/(2π − α) (for a right
angle p = −1/3, 1/3, 1, 5/3, …, the limit of the dielectric roots). A source inside a conductor
is rejected.

**Touching boxes.** Where two boxes share an edge, that edge is a junction of four sectors and
needs its own rule, e.g. for two boxes of permittivity R meeting along the z-axis edge with
vacuum in the other two quadrants (p₁ = (4/π) atan(1/√R) − 1 → −1 as R grows):

```julia
r_shared = edge_gcq_rule(junction_powers(fill(pi / 2, 4), [R, 1.0, R, 1.0]; gmax = 2.5) .- 1;
                         tol = 1e-12, qmax = 24.0)
shared(m, axis) = (axis == 3 && abs(m[1]) < 1e-12 && abs(m[2]) < 1e-12) ? r_shared : nothing
parts = [single_dielectric_box3d_gcq(1.0, 1.0, 1.0, R, 1.0; nlev = 4, n_quad = 8, rule,
                                     edge_rule = shared, center = c)
         for c in ((0.5, 0.5, 0.5), (-0.5, -0.5, 0.5))]
```

The shared edge itself adds no local error at any contrast; the remaining limit is the vertex
at its end, whose exponent also tends to 0 as R grows. Touching perfect conductors (p₁ = −1)
are not supported: the density is no longer integrable.

**Boxes sharing faces.** Meshing boxes separately is exact only while they touch along edges
or at vertices. A face shared by two boxes must be meshed once, as an ε_A | ε_B interface,
not as two coincident box-to-exterior surfaces. `multi_dielectric_box3d_gcq` takes the same box
list as `multi_dielectric_box3d`:
- It puts the boxes on the grid of all their face coordinates and meshes every interface
  rectangle once.
- It gives every edge the GCQ rule of its own junction, from `junction_powers`. These are the
  cube edges, the triple junctions (vacuum π, ε_A π/2, ε_B π/2) where a shared face meets the
  outside, and four-sector edges; equal junctions share one rule.
- It refines toward true vertices only.

```julia
boxes = [(center = (-0.5, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
         (center = (0.5, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
mesh = multi_dielectric_box3d_gcq(boxes, [2.0, 10.0], 1.0; nlev = 4, n_quad = 8,
                                  tol = 1e-12, qmax = 24.0)
sol = solve_dielectric_boxes_gcq(mesh, PointSource((-0.4, 0.1, 0.15), 1.0))
```

The total-charge constraints are imposed per cluster of boxes joined by shared faces
(`mesh.clusters`, node ranges `mesh.bodies`), the charge inside a closed surface around the
cluster. A surface around one box of a face-sharing pair would cut through its neighbour.
At high contrast the constraint is less effective for such a cluster than for a single box.
The near-null mode's left vector is weighted by the permittivity jumps across the shared
face, not constant over the cluster. A cluster can also have further near-null modes: two
high-contrast boxes joined through a low-contrast one carry a +q/−q mode, which the cluster's
total charge does not fix. Treat high-contrast face-sharing clusters with care.

**Capacitance of a box.** `box_capacitance_gcq` computes C / (4πε₀) of a conducting box. It
solves for the equilibrium density with total charge 1, using the conductor equation with the
total-charge constraint and the exact conductor edge powers. The density and the mesh share the
box's symmetry group: 48 elements for a cube, 16 for a square prism, 8 otherwise. So the system
is reduced to one unknown per node orbit, using `box_node_orbits` and
`symmetric_dense_operator`. It is assembled without FMM (direct node sums plus the near
corrections) and solved by LU. For the unit cube with N = 150k nodes that is 3.2k unknowns.

```julia
using DielectricBIE, SCTL

c = box_capacitance_gcq(1.0, 1.0, 1.0)       # nlev = 8, n_quad = 12, tol = 1e-14: ~25 s on 8 threads
c.C          # 0.660678154099568: the Thomson energy Q² / (4π ∫σ Sσ)
c.C_point    # Q / (4π U) at interior points (`targets`), less accurate
```

Two estimates of C are returned:
- `C` comes from the Thomson energy. The energy is stationary at the equilibrium density, so its
  error is second order in the error of σ. It converges by `nlev = 8` for `n_quad ≥ 12`, and
  agrees to about 2e-15 with Helsing & Perfekt's 0.66067815409957 (2013, stated error 1e-13).
- `C_point` divides by the potential at interior points. Its error is first order and levels off
  between 3e-15 and 3e-14.

With `n_quad = 10` the energy estimate stalls about 3.5e-14 too high: the limit there is the GL
order along the edges.

## Volume sources and precomputed fields

A `VolumeSource` is a volume charge density sampled on a quadrature grid: positions
(`3 × N`), quadrature weights, and density values. Its free-space Laplace potential can
drive the right-hand side of a dielectric problem on an interface that is adaptively
refined to resolve that source:

```julia
using DielectricBIE

# A Gaussian charge sampled on a uniform midpoint grid: positions (3×N), weights, density.
n, h = 24, 1.2 / 24
xs = range(-0.6 + h / 2, 0.6 - h / 2; length = n)
positions = reduce(hcat, [[x, y, z] for x in xs for y in xs for z in xs])
density = [exp(-sum(abs2, p) / (2 * 0.08^2)) for p in eachcol(positions)]
vs = VolumeSource(positions, fill(h^3, n^3), density)

# single_dielectric_box3d_rhs_adaptive(Lx, Ly, Lz, n_quad, source,
#                                       eps_src, l_ec, rhs_atol, eps_in, eps_out[, T])
interface = single_dielectric_box3d_rhs_adaptive(
    1.0, 1.0, 1.0, 4, vs, 1.0, 0.25, 1e-3, 4.0, 1.0, Float64)

rhs = rhs_dielectric_box3d_hybrid(interface, vs, 1.0, 1e-6)
```

When the same source is reused for many target batches or assemblies, build the field
once and reuse it. `PrecomputedVolumeField` stores a truncated spectral representation;
in-box targets are evaluated with a type-2 NUFFT and out-of-box targets with the FMM:

```julia
field = PrecomputedVolumeField(vs; tol = 1e-6)

phi  = volume_field_potential(field, targets)   # targets: 3×n  ->  length-n potential
grad = volume_field_gradient(field, targets)    #              ->  3×n gradient
rhs  = rhs_dielectric_box3d_field(interface, field, 1.0)
```

`volume_field_hessian(field, targets)` returns the Hessian (6×n: ∂xx, ∂yy, ∂zz, ∂xy, ∂xz, ∂yz). By default every
derivative is evaluated spectrally: each component is its own type-2 NUFFT of `ik`- or `-kk`-multiplied
coefficients, so every batch pays one full FFT per component.

`window_derivatives = true` (opt-in) builds one real scalar fine grid instead. Every derivative then comes from
differentiating the gathering window, and repeated batches only gather:

```julia
field = PrecomputedVolumeField(vs; tol = 1e-10, window_derivatives = true)   # window_sigma = 1.5, window_tol = tol * 1e-4
r = volume_field_derivatives(field, targets)                    # (; pot, grad, hess) in one pass
r = volume_field_derivatives(field, targets; pot = false)       # any subset
```

- **The window.** The ES window tapered as `exp(β(√(1−z²) − 1))·(1 − z²)³`, evaluated analytically together with its
  first two derivatives.
- **Why analytic.** A fitted, piecewise-polynomial window, as in FINUFFT, loses about N² of its accuracy per
  derivative. The untruncated ES has singular derivatives at the support ends, where stencils change.
- **Accuracy.** At equal `tol` it matches the spectral route on smooth fields, because both are then limited by the
  TKM field itself.
- **Limitation.** Modes at the edge of the mode box are about 100× less accurate (the window's Fourier correction is
  largest there); the TKM `kmax` keeps the spectrum decayed there.
- **Far field.** Out-of-box targets take the same point-charge far field (`far`) as the other evaluators.

In the GCQ campaign mode, `gcq_four_index(c; field_derivs = :window)` uses one such field per shape for the bare
potential and gradient at the nodes, and for ψ and H of the surface corrections.

## Many right-hand sides and four-index integrals

For several sources sharing one interface, assemble and block-solve all right-hand sides
at once, then contract into the four-index Coulomb matrix `V[a, b]`:

```julia
using DielectricBIE, SCTL

# vss::Vector{VolumeSource} on a shared interface that resolves all of them
# (e.g. built with multi_dielectric_box3d_rhs_adaptive)
sigma, stats = solve_dielectric_box3d_block(interface, vss; rtol = 1e-6)
V = four_index_matrix(interface, vss, sigma; lhs_tol = 1e-6, volume_tol = 1e-6)
```

## Operator backends: SCTL.jl and self-defined quadratures

The 3D operator `τ I + Dᵀ` is assembled from two interchangeable parts:
[`dielectric_operator`](@ref)`(interface; far, near)`.

- **`far`, a [`FarField`](@ref)**, evaluates the adjoint double layer with the smooth node rule (the
  interface's quadrature points and weights):
  - [`SCTLFarField`](@ref)`(tol = …)` is SCTL.jl's FMM, the default once SCTL is loaded. It uses PVFMM with
    OpenMP and no MPI, and keeps its octree across GMRES iterations;
  - [`FMM3DFarField`](@ref)`(tol)` is FMM3D.
- **`near`, a [`NearQuadrature`](@ref)**, returns the sparse correction `C` (accurate minus node rule) for
  nearby panel pairs:
  - [`SCTLNearCorrection`](@ref) is the default for `FlatPanel` interfaces: SCTL's near and singular quadrature
    of every panel ([`sctl_gl_corrections`](@ref)), with either far field;
  - [`GCQNearQuad`](@ref) is the default for GCQ `TensorPanel` and `CurvedPanel` interfaces: the GCQ rules for the
    edge and corner panels (product integration in Julia, [`DielectricBIE.gcq_curved_row`](@ref) on a curved panel's
    exact map), SCTL's quadrature for the Gauss–Legendre × Gauss–Legendre ones at its `quad_tol` (1e-12 by default;
    the deprecated [`SCTLCurvedNearQuad`](@ref)`(tol = t)`, the former curved default at 1e-10, is
    `GCQNearQuad(quad_tol = t)`);
  - [`SCTLNearQuad`](@ref) is SCTL's whole operator (its own FMM and singular quadrature), for `FlatPanel`
    interfaces with `n_quad` in 4, 8, 12, 16 or 20, combined with `SCTLFarField`;
  - your own quadrature works too, as described below.

Every Gauss–Legendre × Gauss–Legendre panel gets its near and self corrections from SCTL (see "SCTL's quadrature
for Gauss–Legendre panels" below), so any interface with such panels, a `FlatPanel` interface in particular, needs
`using SCTL`.

The result, a [`DielectricOperator`](@ref), supports `A * x` and `A * X` for an `N × K` block, so it serves
both `Krylov.gmres` and `Krylov.block_gmres`.

SCTL.jl is a separate, unregistered package with a C++ part. Install it once, then load it next to
DielectricBIE:

```julia
using Pkg
Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl")   # Pkg.add(url) fails on its git submodules
Pkg.build("SCTL")                                             # compiles the C++ shim (a few minutes)
```

```julia
using DielectricBIE, SCTL, Krylov

interface = single_dielectric_box3d(1.2, 0.8, 0.6, 8, 0.05, 4.0, 1.0, Float64)
rhs = rhs_dielectric_box3d(interface, PointSource((0.1, 0.1, 0.1), 1.0), 4.0)

A = dielectric_operator(interface; far = SCTLFarField(tol = 1e-10), near = SCTLNearQuad())
sigma, stats = Krylov.gmres(A, rhs; rtol = 1e-8)
```

Here are timings of `SCTLNearQuad` for this box on 32 cores of an AMD EPYC 9474F, at tolerance 1e-8,
with Julia's BLAS at one thread and the right-hand side `rhs` from the example above. The error is the
maximum relative error of `Dᵀσ` for a smooth density. The GMRES runs took 14–15 iterations.

| nodes (`n_quad`, `l_ec`) | backend (`far` + `near`) | setup | one apply | GMRES solve | error |
|:--|:--|:--|:--|:--|:--|
| 80,896 (8, 0.05) | SCTL + `SCTLNearQuad` | 0.34 s | 0.050 s | 0.73 s | 1.2e-10 |
| 288,256 (8, 0.02) | SCTL + `SCTLNearQuad` | 1.3 s | 0.128 s | 1.97 s | 1.1e-10 |

In the same runs an application with the FMM3D far field (and DielectricBIE's former near correction) took 0.49 s
and 2.0 s. `SCTLNearCorrection` uses SCTL's near and self quadrature too, stored as a sparse correction, and
combines with either far field.

Things to know about the SCTL backends:

- **Accuracy.** With 40,000 or more nodes SCTL's far field is PVFMM, whose gradients lose about 1.5 digits.
  So `SCTLFarField(tol = t)` runs PVFMM with two extra digits, and `SCTLNearQuad` builds SCTL's operator at
  accuracy `t / 100`. PVFMM reaches about 1e-11 at best. Below 40,000 nodes the far field is an exact direct
  sum.
- **Precomputation.** The first use for a given number of digits precomputes PVFMM's translation operators
  and caches them (see `SCTL.fmm_cache_dir()`). This takes seconds for these Laplace kernels. Before starting
  many parallel jobs, run one small solve first.
- **Threads.** SCTL uses OpenMP (`OMP_NUM_THREADS`). While SCTL runs, Julia's BLAS is limited to one thread.
  Krylov's vector operations are also faster with `BLAS.set_num_threads(1)` on many cores. On 96 cores the
  SCTL solve of the 288,256-node box took 0.6 s that way, against 7.3 s with 96 BLAS threads.

### Batch mode

[`solve_dielectric_box3d_block`](@ref), `solve_dielectric_lattice_batch` and the campaign solve phase take
`far` / `near` too. Block GMRES then applies the chosen operator to all `K` right-hand sides of an iteration:

```julia
sigma, stats = solve_dielectric_box3d_block(interface, vss; rtol = 1e-8,
                                            far = SCTLFarField(tol = 1e-10), near = SCTLNearQuad())
```

Without `far` and `near` the operator is FMM3D's far field at `fmm_tol` plus `SCTLNearCorrection(tol = up_tol)`.
FMM3D evaluates the `K` densities in one vectorized FMM. SCTL's FMM evaluates them one after another on the
same octree. For a [lattice campaign](campaign.md), add a `[backend]` table to the `.toml`
(`far = "sctl"`, optionally `near = "sctl"`; see [`campaign_backend`](@ref)). `run_phase` loads SCTL.jl on
every worker.

### Self-defined near quadratures (e.g. for GCQ panels)

A near quadrature only has to produce the sparse correction. Any of these work as `near`:

- a sparse matrix `C`;
- a function `interface -> C`;
- a subtype of [`NearQuadrature`](@ref) with a [`near_corrections`](@ref) method.

The rows of `C` are targets and the columns are sources, both in the node order of the interface.
`C[i, j]` is the accurate value of `Dᵀ[i, j]` minus the node-rule value
`w_j n_i·(x_i − y_j) / (4π|x_i − y_j|³)`. Pairs that are not near get no entry.

```julia
struct MyGCQ <: DielectricBIE.NearQuadrature
    quad_tol::Float64
end
DielectricBIE.near_corrections(q::MyGCQ, interface) =
    first(laplace3d_DT_tensor_corrections(interface; quad_tol = q.quad_tol))   # or your own rule

interface = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 6)
A = dielectric_operator(interface; far = SCTLFarField(tol = 1e-12), near = MyGCQ(1e-12))
```

### Building a near quadrature on SCTL's pieces

SCTL.jl exposes the steps of its own near correction as Julia data (see its manual page "Near-field pieces"). In
DielectricBIE they are available as follows:

- [`SCTLNearCorrection`](@ref)`(tol = …)` is SCTL's complete correction as a sparse `C`, usable with either far field.
- [`sctl_near_list`](@ref)`(interface; tol, radius)` is SCTL's classification alone: which interface nodes are
  near which panel. It works for any panel type when you pass the radii.
- [`dt_node_rule_block`](@ref)`(interface, k, targets)` is the node-rule block that a correction subtracts.

So a quadrature written in Julia only has to produce the accurate rows. This example uses SCTL's classification
with a dense composite Gauss rule for the rows: 4 × 4 squares of 24 × 24 points on each panel, times the panel's
Lagrange basis, so that the rows act on nodal values. That is accurate for targets at least one node offset from
the panel, as are the targets of a box off the panel's plane. Targets in the panel's plane keep the node rule,
since `Dᵀ` vanishes there:

```julia
using DielectricBIE, SCTL
import DielectricBIE as BI
using LinearAlgebra

interface = single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 4.0, 1.0, Float64)
nl = sctl_near_list(interface; tol = 1e-10)              # SCTL's near test
X, Nrm = BI._interface_targets_normals(interface)
owner = reduce(vcat, [fill(k, BI.num_points(p)) for (k, p) in enumerate(interface.panels)])

coplanar(p, q) = abs(dot(p.normal, q.normal)) > 1 - 1e-12 && abs(dot(p.normal, q.points[1] .- p.points[1])) < 1e-12

# Dᵀ rows of panel p at targets Xt with normals Nt (3 × m): your quadrature goes here
function dense_rows(p, Xt, Nt; m = 4, q = 24)
    a, b, c, d = p.corners
    cc = (a .+ b .+ c .+ d) ./ 4; bma = b .- a; dma = d .- a
    g, w = BI.gausslegendre(q)
    us = [(-1 + (2s - 1 + g[i]) / m, w[i] / m) for s in 1:m for i in 1:q]   # composite rule on [-1, 1]
    B = zeros(size(Xt, 2), p.n_quad^2)
    for (u, wu) in us, (v, wv) in us
        ℓ = kron(BI.barycentric_row(p.gl_xs, p.bary_weights, u), BI.barycentric_row(p.gl_xs, p.bary_weights, v))
        y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
        for t in axes(Xt, 2)
            r = Xt[:, t] .- y
            B[t, :] .+= wu * wv * norm(bma) * norm(dma) / 4 * dot(Nt[:, t], r) / (4π * norm(r)^3) .* ℓ
        end
    end
    return B
end

accurate = map(eachindex(interface.panels)) do k
    p = interface.panels[k]
    ts = SCTL.near_targets(nl, k)
    B = dt_node_rule_block(interface, k, ts)               # start from the node rule
    for k2 in unique(owner[ts])
        coplanar(p, interface.panels[k2]) && continue      # Dᵀ = 0 in the panel's plane
        rows = findall(t -> owner[t] == k2, ts)
        B[rows, :] = dense_rows(p, X[:, ts[rows]], Nrm[:, ts[rows]])
    end
    B
end
plain = [dt_node_rule_block(interface, k, SCTL.near_targets(nl, k)) for k in eachindex(interface.panels)]
C = SCTL.assemble_near_correction(nl, accurate, [BI.num_points(p) for p in interface.panels]; subtract = plain)

A = dielectric_operator(interface; far = SCTLFarField(tol = 1e-12), near = C)
```

On this box, `A` agrees with `near = SCTLNearCorrection(tol = 1e-12)` to 1e-13. That comparison is part of the
tests and is a quick check of a new rule. For panels that are not `FlatPanel`s, give the radii:
`sctl_near_list(interface; radius = r)`, with one radius per node or one for all.

### SCTL's quadrature for Gauss–Legendre panels

Every Gauss–Legendre × Gauss–Legendre panel ([`is_gl_panel`](@ref): a `FlatPanel`, or a `TensorPanel` or
`CurvedPanel` without a GCQ rule) gets its near and self corrections from SCTL, in every near quadrature:
[`SCTLNearCorrection`](@ref) and [`GCQNearQuad`](@ref) both call
[`sctl_gl_corrections`](@ref)`(interface, panels; tol)` for those panels, and `GCQNearQuad` keeps its own rules,
in Julia, for the GCQ edge and corner panels, whose densities are singular (on `TensorPanel` and `CurvedPanel`
interfaces alike: one path per panel family). The same holds for the single-layer potential
([`sctl_gl_potential_corrections`](@ref), used by [`SCTLPotential`](@ref), [`GCQPotential`](@ref) and
[`potential_tensor`](@ref)). So SCTL is required for any interface with such panels: without `using SCTL` these
calls throw an error naming the package.

Each panel is one SCTL element over its parameter square:
- **Elements.** When the panel's node count per direction is one of SCTL's orders 4, 8, 12, 16 or 20, the element
  is the panel itself. Otherwise, for a `FlatPanel` or a GL `TensorPanel` only, it is a proxy element of the next
  order, with the density interpolated from the panel's nodes (exact for its polynomial); a target is near the
  panel within the near radius of the panel's own node rule. A `CurvedPanel` needs the same SCTL order in u and v
  (4, 8, 12, 16 or 20 nodes per direction): any other count is an `ArgumentError`.
- **Self rows.** When the element is the panel, the panel's own nodes are passed to SCTL with the element nodes'
  exact coordinates, so they get SCTL's Duffy rule (the singular self rows), not a one-sided near-singular limit. A
  proxy element has no self rows at the panel's nodes (`Dᵀ` vanishes there on a flat panel): the single-layer
  potential at such a node is an `ArgumentError`.
- **Curved panels.** A curved panel's element is SCTL's interpolant of its map at the element's order, so the
  accuracy also depends on the panel's size and order: on the unit sphere with 40 panels `Dᵀ1` is off by 2e-4 at
  8 nodes per direction, 5e-9 at 16 and 9e-12 at 20.
- **Aspect ratio.** SCTL's rule on one element is accurate for moderate aspect ratios only. The meshers split GL
  panels to aspect ratio at most [`GL_MAX_ASPECT`](@ref) = 4 ([`panel_aspect`](@ref)), and the near quadrature
  refuses a larger one with an `ArgumentError` naming the panel ([`check_gl_aspect`](@ref)). GCQ panels have no
  such limit.

```julia
interface = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 2, n_quad = 8)
gl = findall(is_gl_panel, interface.panels)
C_gl = sctl_gl_corrections(interface, gl; tol = 1e-12)        # the GL panels' columns of GCQNearQuad(quad_tol = 1e-12)
```

A quadrature that can compute only some panels' columns can implement
`near_corrections(q, interface, sources)`, as `GCQNearQuad` does. Otherwise the default computes all columns and
drops the others.

The GCQ solvers take the same keywords:
`solve_dielectric_boxes_gcq(parts, sources; far = SCTLFarField(tol = 1e-12), near = MyGCQ(1e-12))`.

## Flat-faced polyhedra

A closed triangle mesh, for example from an OBJ file, is meshed with `CurvedPanel`s. Each
triangle is split into three quadrilaterals, one per vertex. `CurvedGeometry` finds the junction
of every edge (its sector angles and permittivities) and so the edge's density powers. Each
quadrilateral is then graded `nlev` times toward its vertex, like a face quadrant of
`single_dielectric_box3d_gcq`:
- the panels along an edge get the edge's GCQ rule across it and Gauss–Legendre along it;
- with `corner = :gcq`, the panel at the vertex gets the GCQ rule in both directions.

```julia
V, F = read_obj("model.obj"; snap = [(1 + sqrt(5)) / 2])    # snap six-digit coordinates to exact values
F, vol = orient_outward(V, F)                                 # closed and consistently oriented, outward
patches, quads = polyhedron_patches(V, F; eps_in = 2.0)
itf, rules, powers = polyhedron_mesh(CurvedGeometry(patches), quads; nlev = 2, n_quad = 8, tol = 1e-8,
                                     corner = :gcq)
A = dielectric_operator(itf; far = FMM3DFarField(1e-12), near = PlanarProductNearQuad())
σ, st = solve_dielectric_box3d_gcq(A, rhs_dielectric_box3d(itf, PointSource((0.1, 0.2, 0.15), 1.0), 2.0), itf)
u = planar_slp_potential(itf, σ, T)                           # Sσ at the targets T (3 × m), near the surface too
```

**The default pipeline (study 73).** All in Float64:
1. `polyhedron_mesh` builds its GCQ rules with `grid_tol = tol` (its default): `edge_gcq_rule` refines the graded fine
   grid until every component of the family is interpolated to `tol`, then takes the singular-value rank, the
   pivoted-QR nodes, the weights and the Float64 map P from the node values to the fine grid. (`edge_gcq_rule` alone,
   and the box and curved meshes, keep the bare dyadic grid.)
2. Near rows (`PlanarProductNearQuad()`): P maps the node values to the fine grid, every fine cell carries its local
   polynomial, and the cells at an edge are clustered into a smaller GCQ panel, the rule scaled to [0, s_k]
   (`endpoint_rule = :gcq`).
3. Single-layer self rows: one Duffy rule over the 3 × 3 fine cells around the node (`self = :block`); the rest of
   the panel by the walk from the whole panel with the Duffy region left out, so the cells at an edge stay clustered
   beyond it (study 75: 2.4× fewer leaves at a corner node beside the vertex, the same energy).

As a perfect conductor (Dᵀ solve with Q = 1, direct far field, Thomson energy Q²/(4πE)), the unit cube's capacitance
comes to −2.7e-15 of the reference at n_quad = 12, nlev = 12, and the octahedron conv(±e_i) to C = 0.72048161272881. The
floor is set by n_quad (2.9e-12, 3e-14, ≤ 3e-15 at 8, 10, 12), not by the edge rules or the self rows
(`dielectric_edge_singularity/results/polyhedra3d/73_pec_capacitance_simple`).

**Near quadrature.** `PlanarProductNearQuad` computes the near corrections of flat panels by
product integration, so it also handles GCQ in both directions.
- Dᵀ vanishes for targets in a panel's plane, so on-surface rows need no singular quadrature.
- A target off the plane gets the exact row of the panel's interpolation space, by adaptive
  refinement over clusters of the GCQ segments and parts of segments.
- Separations come from parameter differences of the panel's bilinear map, and a node beside an
  edge is placed by its own panel's parameters, so its distance from the edge is exact however
  small (nodes come within ~1e-7 of the panel size of a GCQ edge).
- Rows are computed once per congruent panel class and relative target position, so a symmetric
  polyhedron costs about one symmetry orbit.

**Endpoint rule.** A leaf on an endpoint cluster [0, s_k] of a GCQ direction uses, by default
(`endpoint_rule = :gcq`), the panel's own GCQ rule scaled to [0, s_k]: nodes s_k x, weights s_k w
(density weights, so the factor s^p is not applied again), and the parent basis interpolated at the
scaled nodes. `endpoint_rule = :moment` instead uses an 18-point rule with the exact moments of the
panel's basis, exact for every density of the span.
- Each GCQ direction takes its own rule, so a leaf is GL × GL, scaled GCQ × GL, GL × scaled GCQ or
  scaled GCQ × scaled GCQ, H_u (K J) H_vᵀ. This holds on GCQ × GL and GCQ × GCQ panels, for
  off-surface rows and for the walk of a self row. GL × GL panels have no clusters.
- The cutoff of each target is the first admissible cluster along the walk, the first one far
  enough for the adaptive refinement. It depends on the target and the transverse box. The segments
  beyond the cutoff and the target's other boxes are integrated as before.
- A self row at an owned node is the disjoint sum ∫_{Ω∖C} + ∫_{C∖R} + ∫_R, where C is the node's
  fine cell and R is its Duffy region. The first term comes from the walk with the selected
  endpoint rule. C∖R is at most two strips; they lie inside C, so they have no clusters and are
  the same under both rules. ∫_R uses the Duffy rule, unchanged. A cluster holding
  C is peeled for that node only, so each target's leaves are its own whatever targets share the
  walk. (The illustrations of this decomposition in the project notes are schematic.)
- Self rows exist only at a panel's own nodes. No node of a GCQ rule lies in the weighted segment
  at an edge, and arbitrary on-surface targets have no owned cell, so they are not supported.
- The scaled rule integrates the GCQ family (s^p times smooth factors) to the rule's tolerance. It
  does not integrate every density of the panel's span: a cardinal basis function times the kernel
  is not in the family.

Measured on flat GCQ × GL panels (study 56 in the companion repository, η_leaf = 1). A cardinal row
is measured as max_j |Δ_j| / Σ_j |row_j|; a density σ as |Δ·σ| / (|row|·|σ|).

| | `:moment` | `:gcq`, tol 1e-12 | tol 1e-8 | tol 1e-4 |
|---|---|---|---|---|
| s^p and smooth mixtures | ≤ 4e-14 | 9e-13 | 2e-9 | 1e-5 |
| cardinal rows | ≤ 8e-15 | 1.3e-7 | 4.2e-6 | 1.2e-4 |
| random densities of the span | ≤ 4e-14 | 4.6e-7 | 1.8e-5 | 9.8e-4 |
| kernel evaluations | 1 | 1.04 | 0.98 | 0.93 |

The scaled rules have 22, 16–17 and 9–10 GCQ nodes against the moment rule's 18 points.

Warm time and the octahedron's total charge are the same within noise. At equal cost (η_leaf =
0.5) the moment rule is more accurate on cardinal rows and random densities of the span. On resolved
densities the two agree to the GCQ tolerance, and the conductor capacitance of study 73 does not
depend on the choice; `:gcq` is the default as the simpler rule (the cells at an edge as one smaller
GCQ panel).

The same holds for self rows and GCQ × GCQ corners (study 57: 2064 owned nodes of GCQ × GL and
unequal GCQ × GCQ panels, η_leaf = 1, against an independent graded-Duffy reference):

| | `:moment` | `:gcq` |
|---|---|---|
| s^p, mixed-sign densities | ≤ 1.6e-15 | ≤ 1.7e-7 (tol 1e-4), ~1e-15 (tol 1e-12) |
| cardinal rows | ≤ 6.6e-16 | ≤ 1.4e-5 |
| random densities of the span | ≤ 1.6e-15 | ≤ 1e-4 |
| kernel evaluations, time | 1 | 1.01, 0.96–1.05 |

- The Duffy region takes 57–76 % of a self row's time under either rule.
- On cardinal rows and random densities `:gcq` is no less accurate than the GCQ node rule itself is
  in the far field, on the same panel at 1.5 diameters. The tests assert this: the worst measured
  errors are 0.08 (off-surface) and 0.12 (self rows) of that error.
- It does not reach the GCQ tolerance on general densities of the span: it misses by 10³–10⁷.
  Restricting it to smaller endpoint prefixes would not fix this short of the weighted segment,
  that is, short of using the moment rule.
- On the octahedron, the corrections have the same build time, nnz and matvec cost under both
  rules.

`planar_slp_potential` evaluates the single-layer potential with the same product rows near the
panels. Its node sum is exact (direct summation by default): FMM3D's ~1e-8 relative error on
terms ~wσ/δ would spoil targets at distance δ ≪ the node spacing.

**On the surface.** `planar_slp_nodes(itf, σ)` is `u = Sσ` at the interface's own nodes. It is
the node sum without the coincident node plus the near correction `planar_slp_corrections(itf) * σ`:
```julia
u = planar_slp_nodes(itf, σ)
E = dot(BI.all_weights(itf) .* σ, u)                            # ∫σ Sσ, e.g. C = Q² / (4π E) (Thomson)
```
A node's own panel is integrated (`duffy = true`, the default) by one sinh-mapped Duffy rule over
the 3 × 3 fine cells around the node (`self = :block`, the default; a GL direction is one cell, a
GCQ direction's weighted cell at the edge is left out, and the longer side is cut to twice the
shorter around the node), with the density at each point from its own cell's polynomial, and by
the adaptive refinement elsewhere: the walk starts from the whole panel, a box disjoint from the
Duffy region is a root, a box inside it is dropped, and an overlapping box is cut at the region's
sides or has its endpoint cluster peeled, so the scaled-GCQ clusters at an edge reach past the
region (study 75). The Duffy rays cross cell boundaries, where the cells'
polynomials jump by their interpolation error: for densities the fine grid resolves, σ = 1 is
within ~1.5e-14 of the closed form at tol 1e-12 (the own-cell rows: ~1e-14) and smooth densities
agree with the own-cell rows to ~3e-15, but the rows of single cardinal functions err by up to
~1e-4. `self = :cell` keeps the Duffy region inside the node's own cell (the rest of the cell and
of the panel by the walk), exact for every density of the span.

On the small stellated dodecahedron (ε = 2, all 90 edges at 116.57°/243.43°), GCQ × GCQ corners
take ∫σ to a relative error of 2e-11 at N = 417k. GL × GL corners stall near 1e-5. As a perfect
conductor, its capacitance from the energy Q²/(4πE) self-converges to about 5e-13. The potential
at its centre gives about 5e-11.

**Three tolerances (and `ninterp`).** A GCQ rule has three separate accuracies.
- **Training tolerance (`tol`):** the rank of the weighted SVD and the depth of the dyadic grid. It sets the far-field
  node rule's accuracy on the trained family in a weighted norm. It is not a pointwise bound: the octahedron's
  40-power family (tol 1e-12) represents its own members to only ~1e-11 pointwise.
- **Interpolation accuracy:** how well the near-field density (a polynomial on each fine cell through `P`) follows the
  rule's continuous interpolant. With the default 14 points per dyadic cell, measured against a 128-bit evaluation
  (regularized density, relative to its maximum):
  - ~1e-12 for a one-power family at tol 1e-12;
  - ~2e-11 for the octahedron's family;
  - up to 1e-4 in the last cell [0.5, 1] for a solved corner density, with jumps of 2e-7 at u = 0.5.

  `polyhedron_mesh(...; ninterp = 28)` (or `edge_gcq_rule(...; ninterp)`) samples `P` from the continuous
  interpolant at 28 points per cell. That brings these to ≤ 5e-13 without changing the nodes or weights.
  - Twenty points suffice only for densities in the trained family.
  - The cardinal functions of a 40-power family need ~36 points.
  - Cost: ~9 s per 840-exponent rule to build, and 30–70% more per self row.
  - On the octahedron the solved potentials moved by 5e-11, with no gain in the total charge. The default is unchanged.
- **Integration accuracy:** the quadrature of a row for the chosen representation. It stays at rounding
  (≤ 1e-15 on physical densities) for either interpolation order.

Study 62 has the audit (`dielectric_edge_singularity/results/polyhedra3d/62_gcq_interpolation_audit`).

**Family accuracy (`grid_tol`, `density_tol`).** These options control accuracy on a rule's own
training family f_j(s) = s^q_j (the regularized original components, maximum 1 on [0, 1]). Five separate layers set it:
- **Training grid.** The bare 14-point dyadic cells (`edge_gcq_rule`'s default, `grid_tol = nothing`) resolve the
  octahedron family's components only to 7.5e-5, and even a single-power family's high powers only to ~1e-8.
  `grid_tol = tol` (`polyhedron_mesh`'s default) halves cells until every component resolves to tol (49 cells for the
  conductor cube and octahedron at tol 1e-12). A grid that is not resolved also misjudges the norms of weak singular
  directions (by 800–3400×).
- **Compression rank.** The weighted SVD tolerance is not a pointwise bound. At the singular-value rank, a resolved
  840-component rule reconstructs its family to only ~1e-10.
  - `density_tol` (which sets `grid_tol = density_tol / 10` unless given) takes the smallest rank meeting
    density_tol / 4 in the selected-node range. The gate is met by the rule's own evaluator (Lagrange per fine cell
    through its P, measured on the built rule) and by the retained global model. That rank must also keep a Lebesgue
    constant ≤ `lebesgue_max` (100), a negative share of the regularized weights ≤ `negative_weight_max` (1e-4), and
    stay within the numerical rank.
  - The 840-component family takes rank 34 instead of 28 (1.2e-13). The ranks in between often carry
    sign-alternating weights (shares of 1e-3 to 3e-2) or Lebesgue constants near 200.
  - `gcq_certificate(rule)` holds the record: certified or not, the evaluator and model errors, the limits and a
    per-rank trace.
  - Uncertified domains are reported, not certified: all positive cells, including below the first node (7e-12);
    and the terminal element, where the evaluator's polynomial misses s^q pointwise by 1e-5 to 1e-2, as for the
    default rules.
- **Map precision.** At these ranks the Float64 FV/S map misses cardinal directions (nodal identity error 4e-2), so
  `density_tol` rules sample P in 128-bit arithmetic: ~9 s for an 840-exponent rule. A stable nodal solve, cond(U_selected)
  ~10, does not make the Float64 basis construction accurate.
- **Local order:** `ninterp`, as above.
- **Endpoint integrals:** checked by integrals, not pointwise.
  - Near and self single-layer integrals of exact family products improve from ~1e-11 (self rows beside a fine
    boundary: 6e-7) to ≤ 4e-14.
  - Dᵀ just above an edge (d = 1e-6), relative to max|σ| ∫|K|, improves from 9e-11 to ≤ 1.3e-13 on family powers.
  - σ = 1 (regularized s^-p) is the exception: the terminal element leaves 3.9e-12 with the `:moment` endpoint in
    both variants, and 8.7e-13 with `:gcq` on the improved rules (5e-14 on the defaults).

A solved corner density is a useful stability test, but not an exact-family test. Its model error (vertex behaviour
is not a tensor of edge families) can exceed every layer above.

End to end, the options did not pay on the tested meshes (study 67: the octahedron at ε = 2 and 10, a cube at ε = 10):
- 33–38% more unknowns;
- +1 to +3 GMRES iterations;
- no consistent change in total charge (from 22% better to 2.5× worse);
- a worse achievable residual of the assembled operator, by 10× to 320×.

The residual floor does not depend on the FMM tolerance (1e-13 and 1e-15 give the same result). Within each
configuration it is proportional to the near-correction norm ‖C‖_F, which grows as the higher-rank nodes approach the
edge. With `floor = 2.5e-7`, ‖C‖_F and the residual fall, but the octahedron rules are then not certified and the
charge does not improve. The defaults are unchanged.
(`dielectric_edge_singularity/results/polyhedra3d/67_gcq_family_accuracy`.)

## Potentials off the surface

Potentials at points off the interface are needed for `evaluate_batch_potential`, `four_index_matrix` and a
campaign's eval phase. They are computed by a [`PotentialEvaluator`](@ref) through
[`potential_operator`](@ref)`(ev, interface, targets)`:

- **[`SCTLPotential`](@ref)`(tol = …)`**, the default (with `tol = lhs_tol`), is SCTL's `BoundaryIntegralOp` for the
  single layer with the targets set:
  1. a near test per node (the Bernstein radius at `tol`);
  2. for each near target, the panel split at the target's closest point and refined dyadically toward it;
  3. PVFMM for the far field, with the octree built once for all densities.

  It needs Gauss–Legendre × Gauss–Legendre panels ([`is_gl_panel`](@ref)) and `using SCTL`. A panel whose node
  count is not one of SCTL's orders is a proxy element of the next order, as above. A target on a node of a panel
  that is its own element (4, 8, 12, 16 or 20 nodes per direction) gets SCTL's Duffy self row.
- **[`GCQPotential`](@ref)`(quad_tol = …)`** is the evaluator for GCQ (`TensorPanel` and `CurvedPanel`) interfaces,
  off and on the surface: the node rule by a far field, SCTL's corrections ([`sctl_gl_potential_corrections`](@ref))
  for the GL×GL panels and the product integration of the edge and corner panels (`tensor_near_row` on
  `TensorPanel`s, [`DielectricBIE.gcq_curved_row`](@ref) on `CurvedPanel`s, whose targets on a panel get its
  Duffy-region row at their parameters). [`potential_tensor`](@ref) is the `TensorPanel` quadrature summed directly,
  and also works on the surface (SCTL's Duffy self rows at the GL panels' nodes).

```julia
using DielectricBIE, SCTL

Φ = evaluate_batch_potential(interface, Σ, sources, targets; lhs_tol = 1e-8, volume_tol = 1e-8)   # SCTLPotential(tol = 1e-8)
V = four_index_matrix(interface, sources, Σ; lhs_tol = 1e-8, volume_tol = 1e-8, potential = SCTLPotential(tol = 1e-10))
```

A campaign's eval phase uses `SCTLPotential(tol = lhs_tol)` (see [`campaign_potential`](@ref)).

`SCTLPotential` replaced an evaluator that corrected the FMM by adaptive cubature for targets within
`5 × (panel size) / n_quad` of a panel's center. Here is a comparison on a box of 80,896 nodes (`n_quad = 8`,
`l_ec = 0.05`), with 1.1 million targets: a cloud around the box, plus 100,000 points between 1e-4 and 1e-1 from the
surface. The run used 32 threads at tolerance 1e-8. Errors are against a tight adaptive cubature over every panel.

| evaluator | setup | per density | max error (cloud) | max error (closest 20 targets) |
|:--|:--|:--|:--|:--|
| HCubature-corrected FMM (removed) | 10.1 s | 6.5 s | 7e-10 | 8e-4 |
| `SCTLPotential` | 1.0 s | 0.14 s | 8e-12 | 2e-12 |

The removed evaluator's correction radius was measured from panel centers and didn't depend on the tolerance:
targets close to the surface near panel edges or corners got no correction (the 8e-4 above), and interfaces with
large panels relative to `n_quad` could be wrong even far from the surface (5e-4 at the grid points of the coarse
24-panel box in the tests).

## Visualization

Plotting helpers (`viz_2d`, `viz_3d`, `viz_3d_surface`, `viz_3d_interface_solution`,
`viz_3d_zslice`, `plot_campaign_geometry`) live in a Makie package extension. Install Makie
and a backend, then load the backend alongside `DielectricBIE`:

```julia
using Pkg
Pkg.add(["Makie", "CairoMakie"])
```

```julia
using CairoMakie
using DielectricBIE

interface = single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 4.0, 1.0, Float64)
fig = viz_3d(interface)
```
