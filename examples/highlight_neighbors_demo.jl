# The near panels of one panel, by SCTL's near test (sctl_near_list): the panels holding nodes within SCTL's near
# radius of the highlighted panel, colored by how many of their nodes are near it. Needs SCTL.jl and CairoMakie in the
# environment (SCTL.jl installed as in the user guide, "Operator backends").
using DielectricBIE, SCTL
using LinearAlgebra
using CairoMakie

Lx, Ly, Lz = 10.0, 10.0, 1.0
n_quad = 8
l_ec = 0.5
eps_in, eps_out = 2.0, 1.0

interface = single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out)
n_panels = length(interface.panels)
println("panels: ", n_panels)

tol = 1e-6
nl = sctl_near_list(interface; tol)
owner = reduce(vcat, [fill(k, DielectricBIE.num_points(p)) for (k, p) in enumerate(interface.panels)])
neighbor_list = Dict{Tuple{Int, Int}, Int}()
for k in 1:n_panels, t in SCTL.near_targets(nl, k)
    j = owner[t]
    j == k && continue
    neighbor_list[(k, j)] = get(neighbor_list, (k, j), 0) + 1
end
println("near panel pairs total: ", length(neighbor_list))

panel_size(p) = norm(p.corners[2] .- p.corners[1]) * norm(p.corners[3] .- p.corners[2])

# Pick a "large" interior panel (max area, not on edge).
non_edge = [i for (i, p) in enumerate(interface.panels) if !p.is_edge]
sizes = [panel_size(interface.panels[i]) for i in non_edge]
panel_large = non_edge[argmax(sizes)]
near_of_large = [n for ((i, _), n) in neighbor_list if i == panel_large]

println("large interior -> panel #", panel_large,
        "  size=", round(panel_size(interface.panels[panel_large]); digits=3),
        "  #near panels=", length(near_of_large),
        "  near nodes per panel=", sort!(unique(near_of_large)))

all_counts = collect(values(neighbor_list))
shared_cr = (Float32(minimum(all_counts)), Float32(maximum(all_counts)))

fig = viz_3d(
    interface;
    highlight_panel = panel_large,
    neighbor_list = neighbor_list,
    show_points = false,
    fill_alpha = 0.6,
    neighbor_colorrange = shared_cr,
    add_neighbor_colorbar = false,
    size = (800, 800),
)
ax = fig.content[1]
ax.azimuth = -π/2 + deg2rad(15)
ax.elevation = deg2rad(60)

out = joinpath(@__DIR__, "highlight_neighbors_demo.png")
save(out, fig)
println("saved: ", out)
