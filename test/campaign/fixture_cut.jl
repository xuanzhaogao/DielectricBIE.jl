# a campaign whose substrate face (z = zf) cuts the orbitals: a Gaussian blob (sigma) template on an n^3 grid of
# spacing L/(n-1); orbitals: three along x at the face's height, one above it
function write_blob_xsf(path, n; L = 4.0, σ = 0.5, shear = 0.0)
    open(path, "w") do io
        println(io, "CRYSTAL\nPRIMVEC\n$L 0.0 0.0\n0.0 $L 0.0\n0.0 0.0 $L\nPRIMCOORD\n1 1\nX $(L/2) $(L/2) $(L/2)")
        println(io, "BEGIN_BLOCK_DATAGRID_3D\n g\nBEGIN_DATAGRID_3D_g\n$n $n $n\n0.0 0.0 0.0\n$L 0.0 0.0\n0.0 $L 0.0\n$shear 0.0 $L")
        x(i) = (i - 1) * L / (n - 1) - L / 2
        vals = [exp(-(x(i)^2 + x(j)^2 + x(k)^2) / (2σ^2)) for i in 1:n, j in 1:n, k in 1:n]
        for chunk in Iterators.partition(vec(vals), n)
            println(io, join(string.(chunk), " "))
        end
        println(io, "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D")
    end
end
function write_cut_campaign(dir, n; L = 4.0, σ = 0.5, dzf = 0.13, eps_sub = 4.0, rhs_tol = 1e-4, box = nothing,
                            support_rtol = 1e-6, shear = 0.0)
    xsf = joinpath(dir, "orb.xsf"); write_blob_xsf(xsf, n; L, σ, shear)
    h = L / (n - 1); c0 = L / 2              # the template's centroid (symmetric blob)
    orbs = [(c0 + 1.6k, c0, c0) for k in 0:2]; push!(orbs, (c0, c0, c0 + 2.0))
    zf = c0 + dzf
    toml = joinpath(dir, "campaign.toml")
    open(toml, "w") do io
        println(io, "name = \"cut\"\nroot = \"$(joinpath(dir, "out"))\"\ntemplates = [\"$xsf\"]\n")
        for o in orbs
            println(io, "[[orbital]]\ntype = 1\nx = $(o[1])\ny = $(o[2])\nz = $(o[3])\n")
        end
        println(io, "[pairing]\nneighbor_cutoff = 4.0\n")
        b = box === nothing ? [c0 + 1.6, c0, zf - 3.0, 12.0, 10.0, 6.0, eps_sub] : box
        println(io, "[dielectrics]\neps_out = 1.0\nboxes = [$(collect(b))]\n")
        println(io, "[solve]\nn_quad = 4\nrhs_tol = $rhs_tol\nlhs_tol = 1e-9\ngmres_rtol = 1e-10\nsupport_rtol = $support_rtol\nvolume_tol = 1e-8\nmax_depth = 16\n")
        println(io, "[batching]\nn_centers_per_batch = 1\n\n[eval]\nc_pad = 5.0")
    end
    return toml
end
