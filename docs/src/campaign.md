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
using DielectricBIE

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

- Each pair density lies inside one region, or is cut by one horizontal face (z = const; the grid's first two lattice
  vectors horizontal) whose rectangle covers it, with the interface near it only on horizontal faces; the mode checks
  this. A cut density's screened source ρ/ε jumps at the face, so spectral and point-charge rules on its samples
  converge only like O(h). The mode keeps every spectral computation on the smooth ρ instead:
  - V_vol[a, b] = ∫ φ0_a ρ_b / ε dx (symmetry of N, φ0_a = N[ρ_a]) by cut-cell quadrature: the trapezoid weights,
    corrected on the few lattice levels next to the face by degree-7 interpolants centred on each cell;
  - the RHS far from the density by point charges with the same weights;
  - the RHS at the nodes near it from ∂z N[χ ρ] on the face, by the in-plane Fourier transform of the half-space
    Green's function (exact in z), with the periodic images removed to second order.

  On a Gaussian cut off-centre (σ = 0.5, h = 0.1) the volume term is accurate to 7e-8 and the face field to 1e-8,
  where TKM on the screened samples gives 8e-2 and point charges 9e-2. On a campaign whose substrate face cuts the orbitals (test/campaign/fixture_cut.jl,
  σ = 0.5), V differs from its h = 0.067 value by 1.3e-4, 7.5e-6 and 1.1e-6 at h = 0.2, 0.133 and 0.1, with
  max |V - V'| / max |V| down to 1.2e-7; TKM on the screened samples is off by 2e-2 to 6e-2 at every h.
- The tolerances default to the campaign's `[solve]` values. Further keywords: `nlev`, `quad_tol`, `edge_tol`, `tau`,
  `fg_tol`, `refine_targets`, `target_skin`, and `far = :sctl` (`using SCTL`) or `:fmm3d`.

For the Sec. 5.3 campaign (198 graphene orbitals, 2623 pairs, Si|SiO₂ substrate), the comparison is:

| run | resources | wall time | cost |
|---|---|---|---|
| batch pipeline | 10 nodes | 2 h | about 2000 core-hours |
| this mode, 4 processes | one node | 160 s | about 4 core-hours |

The results agree with the batch pipeline to 1e-6 for the same pair densities.

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
