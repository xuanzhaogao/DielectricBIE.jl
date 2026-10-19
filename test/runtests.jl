using DielectricBIE
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
    # visualization helpers
    include("visualization/extensions.jl")

    # core
    include("core/panels.jl")
    include("core/sources.jl")
    include("core/source_geometry.jl")
    include("core/tensor_panels.jl")
    include("shape/volume_field.jl")

    # kernel functions
    include("kernel/laplace2d.jl")
    include("kernel/laplace3d.jl")
    include("kernel/laplace3d_near_smoke.jl")

    # shape functions
    if run_full
        include("kernel/laplace3d_near.jl")
        include("kernel/laplace3d_near_adaptive.jl")
        include("kernel/laplace3d_near_source_cache.jl")
        include("kernel/laplace3d_near_corner_pairs.jl")
        include("kernel/laplace3d_near_touching.jl")
        include("shape/box3d.jl")
    end

    # multi-box 3d
    include("shape/multi_box3d.jl")

    # single box with GCQ edge panels (tensor panels, FMM + near corrections)
    include("solver/dielectric_box3d_gcq.jl")
    include("solver/dielectric_boxes_gcq.jl")
    include("solver/gcq_batched_potential.jl")
    include("solver/multi_box3d_gcq.jl")

    # multi-rhs solver
    include("solver/multi_rhs.jl")
    include("solver/multi_rhs_vector.jl")
    include("solver/lattice_batch.jl")
    include("solver/batch_io.jl")

    # operator backends (far field x near quadrature); the SCTL ones when SCTL.jl is installed
    include("ext/sctl.jl")

    # campaign
    include("campaign/toml_input.jl")
    include("campaign/geometry.jl")
    include("campaign/manifest.jl")
    include("campaign/v_output.jl")
    include("campaign/tasks.jl")
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
