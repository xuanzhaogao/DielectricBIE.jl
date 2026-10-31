# Development

Clone the repository and instantiate its dependencies:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Tests

The tests need SCTL.jl, which is not registered: it cannot be a test dependency, and `Pkg.test()` does not
work. Run `test/runtests.jl` from a shared environment holding the package, SCTL.jl and the test dependencies, as
CI does (`.github/workflows/CI.yml` lists them):

```sh
julia --project=@sctl -e 'using Pkg; Pkg.develop(path = ".");
    Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl");
    Pkg.add(["Test", "Random", "LinearAlgebra", "SparseArrays", "Serialization", "Distributed", "SlurmClusterManager",
             "FastGaussQuadrature", "HCubature", "Krylov", "LegendrePolynomials", "SpecialFunctions", "StaticArrays"]);
    Pkg.build("SCTL")'
OMP_NUM_THREADS=8 julia --project=@sctl -t 8 test/runtests.jl
```

The default run covers the fast tests. Set `BI_RUN_FULL_TESTS=1` for the full suite, including the box geometry,
2D and 3D dense solver, and campaign pipeline tests:

```sh
BI_RUN_FULL_TESTS=1 OMP_NUM_THREADS=8 julia --project=@sctl -t 8 test/runtests.jl
```

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The built site is written to `docs/build/`.

## Branches

Development happens on `dev`: open pull requests against `dev`, not `main`. `main` only
receives `dev` when a release is cut, and registered versions are tagged from it. CI runs on
pull requests and on pushes to both branches; the `dev` documentation is built from `dev`, the
`stable` documentation from release tags.
