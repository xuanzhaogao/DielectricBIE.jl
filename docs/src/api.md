```@meta
CurrentModule = DielectricBIE
```

# API reference

```@index
```

```@autodocs
Modules = [DielectricBIE]
```

## Internals: truncated-kernel method

`DielectricBIE.TKM3D` evaluates the free-space Laplace potential of volume sources
with a truncated-kernel spectral method. It is used internally by the volume-source
right-hand sides and by [`PrecomputedVolumeField`](@ref).

```@autodocs
Modules = [DielectricBIE.TKM3D]
```
