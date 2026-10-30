module DielectricBIE

using LinearAlgebra
using SparseArrays, StaticArrays
using QuadGK: quadgk_segbuf_count
using Serialization
using FastGaussQuadrature, LegendrePolynomials, SpecialFunctions
using Krylov, LinearMaps, HCubature
using NearestNeighbors
using ForwardDiff
using TOML

using FMM2D, FMM3D
import FFTW, FINUFFT

# truncated-kernel method for volume-source potentials (internal submodule)
include("tkm3d/TKM3D.jl")
using .TKM3D

#core types
export FlatPanel
export DielectricInterface, dielectric_tau
export rhs_approx
export interface_approx
export interface_uniform_samples
export TensorPanel, TensorRule1D, gl_tensor_rule, gcq_tensor_rule
export CurvedPanel, curved_lpoint
export EdgeGCQRule, edge_gcq_rule, universal_gcq_rule, edge_density_powers, junction_powers, edge_gcq_family_error

export PointSource, VolumeSource
export PrecomputedVolumeField, volume_field_potential, volume_field_gradient, volume_field_hessian, volume_field_derivatives,
       rhs_dielectric_box3d_field
export NearFieldGeometry, near_field_geometry, in_near_region, lattice_spacing, source_box
export screened_volume_source
export SharpScreening, SoftMixPermittivity, SoftMixInversePermittivity

# kernel functions
export laplace2d_pot, laplace2d_grad
export laplace2d_S, laplace2d_D, laplace2d_DT, laplace2d_pottrg
export laplace2d_S_fmm2d, laplace2d_DT_fmm2d, laplace2d_D_fmm2d, laplace2d_pottrg_fmm2d

export laplace3d_pot, laplace3d_grad
export laplace3d_S, laplace3d_D, laplace3d_DT, laplace3d_pottrg
export laplace3d_S_fmm3d, laplace3d_DT_fmm3d, laplace3d_D_fmm3d, laplace3d_pottrg_fmm3d
export laplace3d_DT_fmm3d_corrected
export laplace3d_DT_fmm3d_corrected_hcubature
export laplace3d_D_fmm3d_corrected
export laplace3d_pottrg_fmm3d_corrected_hcubature
export laplace3d_pottrg_near
export laplace3d_DT_tensor_corrections, tensor_near_row, potential_tensor

# shapes
export single_dielectric_box2d, multi_dielectric_box2d
export single_dielectric_box3d, multi_dielectric_box3d
export single_dielectric_box3d_rhs_adaptive, multi_dielectric_box3d_rhs_adaptive
export single_dielectric_box3d_rhs_adaptive_varquad
export single_dielectric_box3d_gcq, multi_dielectric_box3d_gcq, MultiBoxGCQ
export SurfacePatch, CurvedGeometry, CurveJunction, junction_angles, patch_frame, curved_gcq_mesh, CurvedMesh

# solvers
export lhs_dielectric_box2d, lhs_dielectric_box2d_fmm2d, rhs_dielectric_box2d
export Lhs_dielectric_box2d, Lhs_dielectric_box2d_fmm2d, Rhs_dielectric_box2d
export lhs_dielectric_box3d, lhs_dielectric_box3d_fmm3d, lhs_dielectric_box3d_fmm3d_corrected
export rhs_dielectric_box3d, rhs_dielectric_box3d_fmm3d, rhs_dielectric_box3d_hybrid
export Lhs_dielectric_box3d, Lhs_dielectric_box3d_fmm3d, Lhs_dielectric_box3d_fmm3d_corrected
export Rhs_dielectric_box3d, Rhs_dielectric_box3d_fmm3d, Rhs_dielectric_box3d_hybrid

# linear algebra
export solve_lu, solve_gmres
export lhs_dielectric_box3d_gcq_fmm3d, lhs_dielectric_box3d_gcq_dense, solve_dielectric_box3d_gcq,
       solve_dielectric_box3d_gcq_block
export join_bodies, source_permittivity, box_charge_constraints, solve_dielectric_boxes_gcq
export box_node_orbits, symmetric_dense_operator, box_capacitance_gcq

export density_centroid

# multi-rhs solver
export pair_density_source
export batched_lhs_dielectric_box3d_fmm3d_corrected, BatchedDielectricOperator
export solve_dielectric_box3d_block, dielectric_diagonal_scaling

# operator backends (far field x near quadrature)
export FarField, FMM3DFarField, SCTLFarField, far_field_operator, apply_far!, point_charge_field, layer_field
export NearQuadrature, near_corrections, UpsampledNearQuad, GCQNearQuad, SCTLNearQuad, CorrectionMatrix, FunctionNearQuad
export default_near_quadrature, default_far_field, DielectricOperator, dielectric_operator
export BlockCorrection, panel_column_ranges
export SCTLNearCorrection, sctl_near_list, dt_node_rule_block
export SCTLProxyNearQuad, default_polynomial_panel, SCTLCurvedNearQuad
export PotentialEvaluator, HCubaturePotential, SCTLPotential, GCQPotential, potential_operator
export four_index_matrix

# lattice batch
export lattice_grid_steps, OrbitalInstance
export LatticeBatch, assemble_lattice_batch
export batch_volume_sources, solve_dielectric_lattice_batch
export evaluate_batch_potential

# batch IO
export BatchResult, save_batch_result, load_batch_result, is_complete_batch

# campaign
export CampaignInput, OrbitalSpec, load_campaign
export manifest_path, centers_path, targets_path, rho_store_path, logs_dir, batch_path, v_path
export snap_orbital
export CenterInfo, BatchSpec, enumerate_centers, enumerate_pairs, build_batches
export write_centers, read_centers, write_manifest, read_manifest
export prepare, solve_batch, consolidate, eval_batch, assemble_v, pending_batches
export gcq_four_index, run_gcq_campaign, gcq_pair_shapes
export GCQNearFarV, nearfar_bytes
export four_index_integrals
export run_phase
export write_v_table

# visualization
export viz_2d, viz_3d
export viz_3d_surface
export viz_3d_interface_solution
export viz_3d_zslice
export plot_campaign_geometry

# core types
include("core/panels.jl")
include("utils/edge_gcq.jl")       # needed by the tensor panels (EdgeGCQRule)
include("core/tensor_panels.jl")
include("core/curved_panels.jl")
include("core/block_corrections.jl") # near corrections as per-panel dense blocks
include("core/sources.jl")
include("core/source_geometry.jl")

# kernel functions
include("kernel/laplace2d.jl")
include("kernel/laplace3d.jl")
include("kernel/laplace3d_near_adaptive.jl")
include("kernel/laplace3d_near.jl")
include("kernel/laplace3d_near_upsampling.jl")
include("kernel/laplace3d_near_hcubature.jl")
include("kernel/laplace3d_tensor_near.jl")

# # geometries
include("shape/box2d.jl")
include("shape/box3d.jl") # core types and geometry
include("shape/box3d_single.jl") # single-box builder
include("shape/box3d_fmm_helpers.jl") # FMM/TKM helpers (needed by rhs_adaptive)
include("shape/volume_field.jl") # precomputed spectral volume field
include("shape/box3d_rhs_adaptive.jl") # RHS-adaptive builders
include("shape/box3d_multi.jl") # multi-box builders
include("shape/box3d_gcq.jl") # single box with GCQ edge panels
include("shape/box3d_multi_gcq.jl") # several boxes with GCQ edge panels, shared faces meshed once
include("shape/curved_geometry.jl")  # parametric patches, shared curves, junctions
include("shape/curved_gcq_mesh.jl")  # CurvedPanel meshes graded toward edges

# # utilities
include("utils/linear_algebra.jl")
include("utils/corner_singularity.jl")
include("utils/fftw_threads.jl")    # a non-yielding FFTW threads callback (FFTW.jl's can crash)
include("utils/barycentric.jl")
include("utils/bernstein.jl")
include("utils/quad_order.jl")
include("utils/best_grid.jl")
include("utils/gaussians.jl")
include("utils/xsf_reader.jl")

# # solvers
include("solver/backends.jl") # operator backends: far field x near quadrature
include("solver/dielectric_box2d.jl")
include("solver/dielectric_box3d.jl")
include("solver/dielectric_box3d_gcq.jl")
include("solver/dielectric_boxes_gcq.jl")
include("solver/box_capacitance_gcq.jl")
include("solver/multi_rhs.jl")
include("solver/lattice_batch.jl")
include("solver/batch_io.jl")

# # campaign
include("campaign/toml_input.jl")
include("campaign/geometry.jl")
include("campaign/manifest.jl")
include("campaign/tasks.jl")
include("campaign/v_output.jl")
include("campaign/gcq.jl")       # GCQ campaign mode: the whole V on one GCQ interface
include("campaign/gcq_cut.jl")   # GCQ campaign mode: pair densities cut by faces
include("campaign/gcq_surface.jl")  # GCQ campaign mode: the surface corrections of the RHS of cut densities
include("campaign/gcq_nearfar.jl")  # GCQ campaign mode: V as near entries, moment far field and layer factors

# # visualization
include("visualization/viz_2d.jl")
include("visualization/viz_3d.jl")

include("precompile.jl")

__init__() = _install_fftw_spawnloop()

end
