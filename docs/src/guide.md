```@meta
CurrentModule = DielectricBIE
```

# User guide

This page covers the solver features beyond the [quickstart](index.md): FMM acceleration,
volume sources, many-right-hand-side solves, and visualization. For large distributed
four-index runs, see [Lattice campaigns](campaign.md).

## FMM-accelerated solves

For larger problems, replace the dense operator with an FMM-accelerated `LinearMap` and
solve with GMRES. In 3D, `lhs_dielectric_box3d_fmm3d_corrected` adds the near-field
correction needed for accuracy near the surface.

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
using DielectricBIE, LinearAlgebra
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
using DielectricBIE, LinearAlgebra

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

## Many right-hand sides and four-index integrals

For several sources sharing one interface, assemble and block-solve all right-hand sides
at once, then contract into the four-index Coulomb matrix `V[a, b]`:

```julia
using DielectricBIE

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
  - [`FMM3DFarField`](@ref)`(tol)` is FMM3D, the default;
  - [`SCTLFarField`](@ref)`(tol = …)` is SCTL.jl's FMM. It uses PVFMM with OpenMP and no MPI, and keeps its
    octree across GMRES iterations.
- **`near`, a [`NearQuadrature`](@ref)**, returns the sparse correction `C` (accurate minus node rule) for
  nearby panel pairs:
  - [`UpsampledNearQuad`](@ref) is the default for `FlatPanel` interfaces;
  - [`GCQNearQuad`](@ref) is the default for GCQ `TensorPanel` interfaces;
  - [`SCTLNearQuad`](@ref) is SCTL's singular quadrature, for `FlatPanel` interfaces with `n_quad` in 4, 8, 12,
    16 or 20, combined with `SCTLFarField`;
  - your own quadrature works too, as described below.

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

Here are timings for this box on 32 cores of an AMD EPYC 9474F, at tolerance 1e-8, with
Julia's BLAS at one thread and the right-hand side `rhs` from the example above. The error is the
maximum relative error of `Dᵀσ` for a smooth density. The GMRES runs took 14–15 iterations with
every backend.

| nodes (`n_quad`, `l_ec`) | backend (`far` + `near`) | setup | one apply | GMRES solve | error |
|:--|:--|:--|:--|:--|:--|
| 80,896 (8, 0.05) | FMM3D + `UpsampledNearQuad` | 2.8 s | 0.49 s | 7.0 s | 5e-10 |
| | SCTL + `UpsampledNearQuad` | 2.3 s | 0.039 s | 0.56 s | 1.4e-8 |
| | SCTL + `SCTLNearQuad` | 0.34 s | 0.050 s | 0.73 s | 1.2e-10 |
| 288,256 (8, 0.02) | FMM3D + `UpsampledNearQuad` | 7.0 s | 2.0 s | 30.2 s | 5e-10 |
| | SCTL + `UpsampledNearQuad` | 6.7 s | 0.086 s | 1.35 s | 1.1e-8 |
| | SCTL + `SCTLNearQuad` | 1.3 s | 0.128 s | 1.97 s | 1.1e-10 |

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

FMM3D evaluates the `K` densities in one vectorized FMM. SCTL's FMM evaluates them one after another on the
same octree. For a [lattice campaign](campaign.md), add a `[backend]` table to the `.toml`
(`far = "sctl"`, optionally `near = "sctl"`; see [`campaign_backend`](@ref)). `run_phase` then loads SCTL.jl on
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
with DielectricBIE's adaptive quadtree for the rows. The tree's moments use the panel's Lagrange basis, so they act
on nodal values. Targets in the panel's plane keep the node rule, since `Dᵀ` vanishes there:

```julia
using DielectricBIE, SCTL
import DielectricBIE as BI
using LinearAlgebra

interface = single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.5, 4.0, 1.0, Float64)
nl = sctl_near_list(interface; tol = 1e-10)              # SCTL's near test
X, Nrm = BI._interface_targets_normals(interface)
owner = reduce(vcat, [fill(k, BI.num_points(p)) for (k, p) in enumerate(interface.panels)])
cfg = BI.AdaptiveConfig(atol = 1e-12)

coplanar(p, q) = abs(dot(p.normal, q.normal)) > 1 - 1e-12 && abs(dot(p.normal, q.points[1] .- p.points[1])) < 1e-12

accurate = map(eachindex(interface.panels)) do k                   # your quadrature goes here
    p = interface.panels[k]
    ts = SCTL.near_targets(nl, k)
    B = dt_node_rule_block(interface, k, ts)               # start from the node rule
    for k2 in unique(owner[ts])
        coplanar(p, interface.panels[k2]) && continue      # Dᵀ = 0 in the panel's plane
        rows = findall(t -> owner[t] == k2, ts)
        K = zeros(length(rows), BI.num_points(p))
        BI.adaptive_block_moments_tree!(K, p, [Tuple(X[:, t]) for t in ts[rows]], interface.panels[k2].normal, :DT, cfg)
        B[rows, :] = K
    end
    B
end
plain = [dt_node_rule_block(interface, k, SCTL.near_targets(nl, k)) for k in eachindex(interface.panels)]
C = SCTL.assemble_near_correction(nl, accurate, [BI.num_points(p) for p in interface.panels]; subtract = plain)

A = dielectric_operator(interface; far = SCTLFarField(tol = 1e-12), near = C)
```

On this box, `A` agrees with `near = SCTLNearCorrection(tol = 1e-12)` to 1e-9. That comparison is part of the
tests and is a quick check of a new rule. For panels that are not `FlatPanel`s, give the radii:
`sctl_near_list(interface; radius = r)`, with one radius per node or one for all.

The GCQ solvers take the same keywords:
`solve_dielectric_boxes_gcq(parts, sources; far = SCTLFarField(tol = 1e-12), near = MyGCQ(1e-12))`.

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
