"""
    viz_3d(interface; highlight_panel = nothing, neighbor_list = nothing, highlight_edges = false, kw...) -> Figure

3D view of the panels of an interface (or a vector of interfaces, or `viz_3d(; interfaces, sources)` with volume
sources). `highlight_panel = k` highlights panel `k`, and with `neighbor_list`, a `Dict{Tuple{Int, Int}, Int}` of
`(i, j) => value`, the panels `j` paired with `k`, colored by value (e.g. the panels holding SCTL's near targets of
panel `k`, from [`sctl_near_list`](@ref): see `examples/highlight_neighbors_demo.jl`). Requires a Makie backend
(`using CairoMakie`/`GLMakie`). Implemented in the Makie extension.
"""
function viz_3d(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_surface(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_interface_solution(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_zslice(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

"""
    plot_campaign_geometry(c::CampaignInput; kwargs...) -> Figure
    plot_campaign_geometry(toml_path::AbstractString; kwargs...) -> Figure

3D view of a campaign's geometry: each dielectric box as a wireframe cuboid colored by its
permittivity ε, plus the orbital positions as a scatter colored by sublattice `type`. Requires
a Makie backend (`using CairoMakie`/`GLMakie`). Implemented in the Makie extension.
"""
function plot_campaign_geometry(args...; kwargs...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end
