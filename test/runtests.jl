# FMM3D, FMM2D and FINUFFT are OpenMP codes: unset, every call starts one OpenMP thread per core on top of Julia's
# threads, and an FMM of a few hundred points takes a second instead of milliseconds. Set before the libraries load.
haskey(ENV, "OMP_NUM_THREADS") || (ENV["OMP_NUM_THREADS"] = string(Threads.nthreads()))

using DielectricBIE
using SCTL       # the near and self corrections of Gauss-Legendre panels, SCTLPotential: SCTL's (DielectricBIESCTLExt)
using LinearAlgebra
using Test

@testset "DielectricBIE.jl" begin
    run_full = get(ENV, "BI_RUN_FULL_TESTS", "0") == "1"


    # utilities
    include("utils/linear_algebra.jl")
    include("utils/barycentric.jl")
    include("utils/quad_order.jl")
    include("utils/best_grid.jl")
    include("utils/edge_gcq.jl")
    include("utils/fftw_threads.jl")
    # visualization helpers
    include("visualization/extensions.jl")

    # core
    include("core/panels.jl")
    include("core/block_corrections.jl")
    include("core/sources.jl")
    include("core/source_geometry.jl")
    include("core/tensor_panels.jl")
    include("core/curved_panels.jl")
    include("core/gl_panels.jl")
    include("shape/curved_geometry.jl")
    include("shape/volume_field.jl")
    include("shape/volume_field_window.jl")

    # kernel functions
    include("kernel/laplace2d.jl")
    include("kernel/laplace3d.jl")

    # shape functions
    if run_full
        include("shape/box3d.jl")
    end

    # multi-box 3d
    include("shape/multi_box3d.jl")

    # single box with GCQ edge panels (tensor panels, FMM + near corrections)
    include("solver/dielectric_box3d_gcq.jl")
    include("solver/dielectric_boxes_gcq.jl")
    include("solver/box_capacitance_gcq.jl")
    include("solver/gcq_batched_potential.jl")
    include("solver/gcq_target_refine.jl")
    include("kernel/laplace3d_tensor_near_row.jl")
    include("solver/multi_box3d_gcq.jl")

    # multi-rhs solver
    include("solver/multi_rhs.jl")
    include("solver/multi_rhs_vector.jl")
    include("solver/lattice_batch.jl")
    include("solver/batch_io.jl")
    include("solver/deprecated_wrappers.jl")

    # operator backends (far field x near quadrature)
    include("ext/sctl.jl")
    include("ext/sctl_curved.jl")
    include("ext/sctl_far_close.jl")
    include("ext/sctl_gl.jl")
    include("ext/sctl_rounding.jl")

    # campaign
    include("campaign/toml_input.jl")
    include("campaign/geometry.jl")
    include("campaign/manifest.jl")
    include("campaign/v_output.jl")
    include("campaign/tasks.jl")
    include("campaign/gcq.jl")
    include("campaign/gcq_cut.jl")
    include("campaign/gcq_nearfar.jl")
    if run_full
        include("campaign/pipeline.jl")
        include("campaign/distributed_ext.jl")
    end

    # solver functions
    if run_full
        include("solver/dielectric_box2d.jl")
        include("solver/dielectric_box3d.jl")
    end
end
