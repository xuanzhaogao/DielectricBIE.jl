```@meta
CurrentModule = DielectricBIE
```

# Lattice campaigns

For large four-index runs over many orbital centers on a lattice, the package provides a
file-backed pipeline driven from a TOML campaign description. It groups pair densities
into batches that share one boundary operator, solves each batch, then evaluates and
contracts every pair against a shared target set.

## Pipeline

```
prepare  →  solve_batch  →  consolidate  →  eval_batch  →  assemble_v
```

- **prepare** — enumerate centers, neighbor pairs, and batches; write the manifest.
- **solve_batch** — per batch: assemble the shared interface, block-GMRES, store σ + screened ρ.
- **consolidate** — build the shared evaluation target set and the contraction store.
- **eval_batch** — per batch: evaluate Φ at the shared targets and contract all pairs into V columns.
- **assemble_v** — gather all V columns into the dense matrix; write `V_full.tsv` + `report.txt`.

Every phase writes its outputs atomically (temp file + rename), so a run is crash-safe and
restartable: re-running a phase skips already-completed batches, and a killed job is
recovered simply by rerunning the phase. `pending_batches(c, :solve | :eval)` reports
what is left to do.

## Campaign TOML

```toml
name = "demo"                          # campaign name
root = "/path/to/output"               # output dir: manifest.tsv, batches/, V/, logs/
templates = ["orb1.xsf", "orb2.xsf"]   # type index -> .xsf path (relative to this file)

[[orbital]]                            # one entry per orbital; id = 1-based order
type = 1                               # index into `templates`
x = 0.0                                # Cartesian center in the templates' frame
y = 0.0
z = 7.5

[pairing]
neighbor_cutoff = 2.6                  # pair orbitals within this distance (default: Inf)
# pairs = [[1, 2], [1, 3]]             # OR give explicit pair overrides instead

[dielectrics]
eps_out = 1.0
boxes = [[0.0, 0.0, 7.5, 90.0, 90.0, 3.35, 3.5]]   # rows of [cx cy cz Lx Ly Lz eps]

[solve]
n_quad = 6
edge_refine_level = 2                  # or set `l_ec` directly
rhs_tol = 1e-3
lhs_tol = 1e-5
gmres_rtol = 1e-5
support_rtol = 1e-4
volume_tol = 1e-5
max_order = 8
max_depth = 128

[batching]
n_centers_per_batch = 1

[eval]
far_pad_steps = 2.0
```

## Running serially

```julia
using DielectricBIE, SCTL    # the eval phase's potential evaluator, SCTLPotential, needs SCTL

c = load_campaign("campaign.toml")
prepare(c)
for id in pending_batches(c, :solve); solve_batch(c, id); end
consolidate(c)
for id in pending_batches(c, :eval); eval_batch(c, id); end
assemble_v(c)
```

The whole pipeline can also be run in memory, without writing files, via
`four_index_integrals("campaign.toml")` (returns `(; pair_ids, V)`).

## Running distributed

The `:solve` and `:eval` phases parallelize over batches. `run_phase` is provided by a
package extension; load its weak dependencies (`Distributed` and `SlurmClusterManager`) to
enable it. Spawning policy: inside a Slurm allocation with more than one task it uses
`SlurmManager()` (one worker per task); otherwise it spawns `workers` local processes; with
`workers = 0` and no allocation it runs inline.

```julia
using DielectricBIE, Distributed, SlurmClusterManager

c = load_campaign("campaign.toml")
prepare(c)
run_phase(c, :solve; workers = 4)   # local workers (or SlurmManager inside an allocation)
consolidate(c)
run_phase(c, :eval; workers = 4)
assemble_v(c)
```

A campaign is typically driven by a small script over this API (the package itself ships
no CLI). On Slurm, run one task per node and give each task the whole node via threads:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=logs/solve_%j.out
set -euo pipefail

# Pin both thread pools — unpinned threads silently corrupt timings.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_GLUE_THREADS=8

julia --project -t "$JULIA_GLUE_THREADS" driver.jl campaign.toml solve
```

where `driver.jl` parses `<campaign.toml> <phase>` and calls the phase functions above.
Notes for cluster runs:

- **Precompile on the head process first** (`using Pkg; Pkg.precompile()` before loading
  `Distributed`/`SlurmClusterManager`) so workers don't race to precompile over a shared
  filesystem.
- **One worker per node** (`--ntasks-per-node=1`, `--cpus-per-task=<cores>`); the FMM
  saturates many cores, so node-sized tasks are the right grain.
- **Crash/walltime recovery:** just resubmit the same phase — completed batches are skipped.

## GCQ campaign mode

[`run_gcq_campaign`](@ref) (and [`gcq_four_index`](@ref) underneath) computes the whole four-index matrix of a campaign
on **one** GCQ interface (`TensorPanel`s with GCQ edge rules), instead of one interface per batch:

```julia
using DielectricBIE, SCTL

c = load_campaign("campaign.toml")
run_gcq_campaign(c)                    # -> <root>/gcq/V_full_eV.jls, V_full.tsv, V_full_eV.tsv, report.txt
```

How it works:

1. **Shapes.** On a lattice every pair density is a grid translate of a few *shapes* (template_i, template_j, relative
   step), each truncated on its own support (`support_rtol`). The targets are the union of all supports.
2. **One interface and operator.** The interface is refined for the envelope of all pair densities (the field and the
   potential of each). The induced part of V is then the node rule: by the symmetry of the single layer,
   V_layer = G' A^{-1} F, with G[j, a] = w_j φ_a(y_j). This rule needs no refinement for the targets, only accurate
   φ_a at the nodes; `refine_targets = true` refines for them too (`target_skin`), a larger mesh for the same V.
3. **F and G** of every pair by one volume field per shape (TKM near the density, point charges far from it), over the
   interface nodes shifted by each pair's offset.
4. **Low rank.** The right-hand sides are numerically low-rank: Q, the range of W^{1/2} F at tolerance `tau`, needs
   only r solves (r ≪ number of pairs).
5. **The volume part** by shapes: one screened volume solve per source shape, then one dot product per distinct
   (shape, shape, shift).

Requirements and tolerances:

- Each pair density lies inside one region, or is cut by box faces, including edges, corners and junctions of
  several boxes. A cut needs a grid with exactly one lattice vector not parallel to each face that comes within reach:
  an orthogonal grid for vertical faces, or horizontal first and second vectors for horizontal faces only. The mode
  checks this. A cut density's screened source ρ/ε jumps at the faces, so spectral and point-charge rules on its
  samples converge only like O(h). The mode keeps every spectral computation on the smooth ρ instead:
  - V_vol[a, b] = ∫ φ0_a ρ_b / ε dx (symmetry of N, φ0_a = N[ρ_a]) by cut-cell quadrature: the trapezoid weights,
    corrected on the few lattice levels next to each face by degree-7 interpolants centred on each cell (a tensor
    product of 1D rules for box regions).
  - The RHS, chosen by `cut_rhs`:
    - `:surface`: g = ∇N[aρ] (a = 1/ε) at every node, from ψ = ∇N[ρ] and its Hessian H, plus double- and single-layer
      corrections on the interface rectangles:
      g = (a_A + a_B)/2 ψ + Σ_m (a_A − a_B)_m (D_m[ψ] − S_m[Hn − tr(H) n]).
      The layers live on their own SCTL Gauss-Legendre elements (`cut_order`² nodes), graded toward the cut densities
      (size at most `cut_κ (d + cut_L0)` at distance d). SCTL's singular and near-singular quadrature (`cut_tol`)
      and one FMM evaluate them. Needs `using SCTL`.
    - `:halfspace`: for one horizontal face with the nodes near the density on horizontal faces, ∂z N[χρ] by the
      in-plane Fourier transform of the half-space Green's function, exact in z, and point charges with the cut-cell
      weights elsewhere.
    - `:auto` (the default): `:halfspace` where it applies, `:surface` elsewhere.

  Measured accuracy:
  - **Horizontal cut, Gaussian off-centre** (σ = 0.5, h = 0.1): the volume term is accurate to 7e-8 and the face
    field to 1e-8. TKM on the screened samples gives 8e-2, point charges 9e-2.
  - **Surface corrections, at points on the faces of one box and of four boxes meeting along an edge** (down to
    1e-6 from it): accurate to 2e-12 (test/ext/sctl_surface.jl), where TKM of ρ/ε is off by 1e-2 to 1e-1.
  - **Campaign whose substrate face cuts the orbitals** (test/campaign/fixture_cut.jl, σ = 0.5): V differs from its
    h = 0.067 value by 1.3e-4, 7.5e-6 and 1.1e-6 at h = 0.2, 0.133 and 0.1, with max |V − V'| / max |V| down to
    1.2e-7. `:surface` and `:halfspace` agree to 1.9e-6 at h = 0.133. TKM on the screened samples is off by 2e-2 to
    6e-2 at every h.
- The tolerances default to the campaign's `[solve]` values. Further keywords: `nlev`, `quad_tol`, `edge_tol`, `tau`,
  `fg_tol`, `refine_targets`, `target_skin`, `cut_rhs`, `cut_order`, `cut_κ`, `cut_L0`, `cut_tol`, and `far = :sctl`
  (`using SCTL`) or `:fmm3d`.

For the Sec. 5.3 campaign (198 graphene orbitals, 2623 pairs, Si|SiO₂ substrate), the comparison is:

| run | resources | wall time | cost |
|---|---|---|---|
| batch pipeline | 10 nodes | 2 h | about 2000 core-hours |
| this mode, 4 processes | one node | 160 s | about 4 core-hours |

The results agree with the batch pipeline to 1e-6 for the same pair densities.

The solves go through block GMRES, `gmres_block` right-hand sides at a time (default 32; one batched FMM per
iteration, [`solve_dielectric_box3d_gcq_block`](@ref)). On a 3 × 3 blob lattice this takes 21 solves from 42.6 s to
14.6 s with V unchanged to 2e-11; `gmres_block = 1` solves one at a time.

`tau` sets the number of solves. Every pair density with a significant charge needs about one: on n × n lattices of
asymmetric blobs the exact layer part has numerical rank n² (the number of orbitals) at 1e-6 of ‖V‖, and the
low-rank mode reaches V to 1.7e-5 at `tau = 1e-3` (100 solves, 10 × 10) and to 1.2e-6 at `tau = 1e-4` (194 solves).

### Near/far form of V

`vform = :nearfar` returns `V` as a [`GCQNearFarV`](@ref) instead of a dense P × P matrix, with the linear-cost
compression of PDMK-ISDF (see `src/campaign/gcq_nearfar.jl`). The pairs are grouped into cubic tiles (side `tile`).
The entries between near tiles (proxy boxes closer than `rnear`) are exact and stored sparse. The volume part between
far tiles is `M_a' C_k Mε_b`: Cartesian moments of degree ≤ `p_far` of each pair about its tile centre (computed once
per shape and shifted), and one core per tile offset k from equivalent charges on Chebyshev proxies. Storage and
assembly are O(P). `V[a, b]` gives one entry, `Matrix(V)` the dense matrix for checks.

On a 10 × 10 lattice (890 pairs) the near entries agree with the dense V to 5e-13 and the far entries to 2e-6, 4e-5
and 5e-4 of max |V| at `p_far` = 8, 6 and 4.

The layer part of the far entries comes from the low-rank factors (`layer = :lowrank`, the default), or
from translation (`layer = :translate`, one process). That mode treats the induced response as invariant under
in-plane translations: one solve per group of translated pairs (shape, cut class, vertical step) at a reference
placement near the middle, and n_mom solves per tile level for the far cores. The number of solves does not grow
with the lattice. This is exact only for a laterally infinite box: the side faces make the error decay like a
power of the distance to the box edge. Against the exact layer part, on a 6 × 6 lattice, the error is 1.4e-2, 5.8e-3,
9.5e-4 and 1.6e-4 (relative Frobenius) for margins of 3, 6, 12 and 24 Å. Where no translation happens (one pair per
group) it agrees to 1.3e-7 on near entries and 1.6e-5 to 4.9e-6 on far entries at `p_far` = 4 to 8.

### Several processes

The work splits over `nparts` processes started together, which exchange data through files in `sync`. Shapes,
solves and volume shapes are divided over the parts; part 1 builds the interface and assembles V. For example, four
processes on a two-socket node:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
SYNC=/dev/shm/gcq_$SLURM_JOB_ID; mkdir -p $SYNC
for k in 1 2 3 4; do
  JULIA_NUM_THREADS=24 OMP_NUM_THREADS=24 numactl --cpunodebind=$(( (k - 1) / 2 )) --membind=$(( (k - 1) / 2 )) \
    julia --project -e "using DielectricBIE, SCTL;
                        run_gcq_campaign(load_campaign(\"campaign.toml\"); part = $k, nparts = 4, sync = \"$SYNC\")" &
done
wait
```

Keep `OMP_NUM_THREADS` at or below the cores of one socket: with SCTL's FMM, more OpenMP threads than that made the
solves about 3× slower on two-socket genoa nodes. Across nodes, use a shared directory for `sync`.
