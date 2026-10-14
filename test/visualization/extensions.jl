pkg_root = pkgdir(DielectricBIE)

@testset "Visualization extension wiring" begin
    project_toml = read(joinpath(pkg_root, "Project.toml"), String)
    @test occursin("\nMakie = ", project_toml)
    @test occursin("DielectricBIEMakieExt = [\"Makie\"]", project_toml)

    extension_path = joinpath(pkg_root, "ext", "DielectricBIEMakieExt.jl")

    @test isfile(extension_path)
    @test !isfile(joinpath(pkg_root, "ext", "GLMakieExt.jl"))

    if isfile(extension_path)
        extension_source = read(extension_path, String)
        @test occursin("module DielectricBIEMakieExt", extension_source)
        @test occursin("using Makie", extension_source)
        @test !occursin("using GLMakie", extension_source)
    end
end

@testset "Visualization fallback warnings" begin
    @test_logs (:warn, r"Makie.*backend") DielectricBIE.viz_2d(nothing)
    @test_logs (:warn, r"Makie.*backend") DielectricBIE.viz_3d(nothing)
    @test_logs (:warn, r"Makie.*backend") DielectricBIE.viz_3d_surface(nothing)
    @test_logs (:warn, r"Makie.*backend") DielectricBIE.viz_3d_interface_solution(nothing)
    @test_logs (:warn, r"Makie.*backend") DielectricBIE.viz_3d_zslice(nothing)
end
