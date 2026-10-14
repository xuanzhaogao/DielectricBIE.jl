```@meta
CurrentModule = DielectricBIE
```

# DielectricBIE.jl

DielectricBIE.jl solves Laplace problems with dielectric interfaces in 2D and 3D using
boundary integral equations. It provides single- and double-layer operators, panel-based
quadrature with adaptive refinement, FMM acceleration with near-field correction, volume
charge sources, many-right-hand-side (block) solves, and a batched four-index integral
pipeline that can run distributed across a cluster.

- [User guide](guide.md) — FMM-accelerated solves, volume sources, many right-hand sides,
  four-index integrals, and visualization.
- [Lattice campaigns](campaign.md) — batched, restartable four-index runs across a cluster.
- [API reference](api.md) — every exported function and type.
- [Development](development.md) — running tests and building these docs.

## Versions

The `main` branch holds the released code, which produced the results of *Adaptive Boundary-Integral Computation of Screened Coulomb Integrals in Dielectric Environments*.
New features are developed on the [`dev`](https://github.com/xuanzhaogao/DielectricBIE.jl/tree/dev)
branch and documented in the [development docs](https://xuanzhaogao.github.io/DielectricBIE.jl/dev/):
generalized Chebyshev quadrature (GCQ) edge panels built from the exact edge-singularity powers,
several touching or separated bodies solved with a total-charge constraint per body, and perfect
conductors. They will be released from `main` once complete.

## Features

- Laplace single- and double-layer operators in 2D and 3D.
- Dielectric interface builders for box geometries, with adaptive panel refinement.
- Box meshes with generalized Chebyshev quadrature (GCQ) edge panels built from the exact
  edge-singularity powers (`edge_gcq_rule`), for one or several dielectric or perfectly
  conducting boxes, including high-contrast junctions where the edge power approaches −1.
- Total-charge constraints per body (Gauss's law) that keep high-contrast and conductor
  solves accurate (`solve_dielectric_boxes_gcq`).
- Direct and FMM-accelerated linear operators (with near-field correction) for iterative solves.
- Volume charge sources, plus a reusable precomputed spectral field (`PrecomputedVolumeField`)
  for fast repeated potential/gradient evaluation and right-hand-side assembly.
- Many-right-hand-side (block) solves and four-index Coulomb integrals.
- A batched, restartable lattice campaign pipeline for large four-index runs, with
  optional distributed/Slurm execution.
- Linear algebra helpers (`solve_lu`, `solve_gmres`).
- Optional Makie visualization extension.

The public API is exported, so `using DielectricBIE` brings it into scope directly.

## Installation

DielectricBIE.jl requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("DielectricBIE")
```

For local development, clone the repository and run:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Quickstart (2D)

```julia
using DielectricBIE

# single_dielectric_box2d(Lx, Ly, n_quad, l_panel, l_corner, eps_in, eps_out[, T])
interface = single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs = lhs_dielectric_box2d(interface)
rhs = rhs_dielectric_box2d(interface, PointSource((0.1, 0.1), 1.0), 5.0)

sigma = solve_lu(lhs, rhs)
```

## Quickstart (3D)

```julia
using DielectricBIE

# single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out[, T])
interface = single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 4.0, 1.0, Float64)
lhs = lhs_dielectric_box3d(interface)
rhs = rhs_dielectric_box3d(interface, PointSource((0.1, 0.1, 0.1), 1.0), 4.0)

sigma = solve_lu(lhs, rhs)
```

Next, see the [User guide](guide.md) for FMM-accelerated solves and volume sources.
