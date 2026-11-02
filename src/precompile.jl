# Precompile workload: small instances of the expensive first calls that need no SCTL (GCQ rules, box, multi-box and
# curved meshes, a right-hand side, a curved GCQ panel's rows), so that scripts and the tests do not compile them again
# in every process. The operators, solves, potentials and the campaign need SCTL's quadrature for Gauss-Legendre
# panels: they are in the SCTL extension's workload (ext/DielectricBIESCTLExt.jl), which uses the helpers below.
using PrecompileTools: @setup_workload, @compile_workload

# the unit sphere as six cube-sphere patches, each oriented outward (also used by the SCTL extension's workload)
function _precompile_sphere()
    ps = SurfacePatch[]
    for a in 1:3, s in (-1.0, 1.0)
        # the face point p = (s, x, y), (x, s, y) or (x, y, s), projected; each branch of one type (no allocation at
        # dual numbers), and r summed in p's order
        f = (U, V) -> begin
            x = 2U - 1; y = 2V - 1
            r = sqrt(a == 1 ? s^2 + x^2 + y^2 : a == 2 ? x^2 + s^2 + y^2 : x^2 + y^2 + s^2)
            a == 1 ? (s / r, x / r, y / r) : a == 2 ? (x / r, s / r, y / r) : (x / r, y / r, s / r)
        end
        P = SurfacePatch(f; eps_in = 4.0, eps_out = 1.0)
        n = patch_frame(P, 0.5, 0.5)[3]
        push!(ps, SurfacePatch(f; eps_in = 4.0, eps_out = 1.0, flip = n[a] * s < 0))
    end
    return ps
end

# its mesh with one seam graded (GCQ edge panels, the rule of its smooth junction)
_precompile_sphere_mesh() = curved_gcq_mesh(CurvedGeometry(_precompile_sphere()); nlev = 0, n_quad = 4, tol = 1e-4,
                                            grade = c -> c.midpoint[1] > 0.5 && c.midpoint[3] > 0.5)

# a tiny campaign in `dir`: one Gaussian-blob template, two orbitals, one box; returns the TOML path
function _precompile_campaign(dir)
    xsf = joinpath(dir, "orb.xsf")
    open(xsf, "w") do io
        println(io, "CRYSTAL\nPRIMVEC\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nPRIMCOORD\n1 1\nX 1.5 1.5 1.5")
        println(io, "BEGIN_BLOCK_DATAGRID_3D\n g\nBEGIN_DATAGRID_3D_g\n6 6 6\n0.0 0.0 0.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0")
        x(i) = (i - 1) / 6 * 3.0
        vals = [exp(-((x(i) - 1.5)^2 + (x(j) - 1.5)^2 + (x(k) - 1.5)^2) / (2 * 0.4^2)) for i in 1:6, j in 1:6, k in 1:6]
        for chunk in Iterators.partition(vec(vals), 6)
            println(io, join(string.(chunk), " "))
        end
        println(io, "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D")
    end
    toml = joinpath(dir, "campaign.toml")
    open(toml, "w") do io
        println(io, "name = \"precompile\"\nroot = \"$(joinpath(dir, "out"))\"\ntemplates = [\"$xsf\"]\n")
        for xo in (1.8, 5.4)
            println(io, "[[orbital]]\ntype = 1\nx = $xo\ny = 1.8\nz = 1.8\n")
        end
        println(io, "[pairing]\nneighbor_cutoff = 4.0\n")
        println(io, "[dielectrics]\neps_out = 1.0\nboxes = [[3.6, 1.8, 1.8, 8.0, 8.0, 8.0, 2.0]]\n")
        println(io, "[solve]\nn_quad = 3\nrhs_tol = 1e-2\nlhs_tol = 1e-4\ngmres_rtol = 1e-6\nsupport_rtol = 1e-6\n" *
                    "volume_tol = 1e-6\nmax_depth = 16\n")
        println(io, "[batching]\nn_centers_per_batch = 1\n\n[eval]\nc_pad = 5.0")
    end
    return toml
end

# the campaign with loose tolerances (far = :fmm3d or :sctl)
_precompile_four_index(toml, far) =
    gcq_four_index(load_campaign(toml); far, nlev = 0, n_quad = 3, quad_tol = 1e-4, edge_tol = 1e-3, rhs_tol = 1e-2,
                   lhs_tol = 1e-4, gmres_rtol = 1e-6, tau = 1e-6, fg_tol = 1e-6, verbose = false)

@setup_workload begin
    @compile_workload begin
        edge_density_powers(10.0, 1.0)
        universal_gcq_rule(-0.3; qmax = 6.0, tol = 1e-6)
        r = edge_gcq_rule(10.0, 1.0; tol = 1e-3)
        itf = single_dielectric_box3d_gcq(1.0, 1.0, 1.0, 10.0, 1.0; nlev = 0, n_quad = 3, rule = r)
        rhs_dielectric_box3d(itf, PointSource((0.1, 0.15, 0.2), 1.0), 10.0)
        multi_dielectric_box3d_gcq([(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
                                    (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)], [4.0, 2.0], 1.0;
                                   nlev = 0, n_quad = 3, tol = 1e-3)
        # and a GCQ edge panel's rows (the pure-Julia engine of GCQNearQuad and GCQPotential on curved interfaces,
        # which the SCTL extension's workload covers only on AVX-512 CPUs): an own-node Dᵀ row and an off-panel single
        # layer row
        ps = _precompile_sphere_mesh().interface.panels
        P = ps[findfirst(!is_gl_panel, ps)]
        gcq_curved_row(:dt, P, P.lpoints[1]; n = P.normals[1], on = (P.ru.x[1], P.rv.x[1]))
        gcq_curved_row(:sl, P, P.lpoints[1] .+ 0.05 .* P.normals[1])
    end
end
