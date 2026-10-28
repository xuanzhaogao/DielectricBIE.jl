using DielectricBIE
using Documenter

DocMeta.setdocmeta!(DielectricBIE, :DocTestSetup, :(using DielectricBIE); recursive=true)

makedocs(;
    modules=[DielectricBIE],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="DielectricBIE.jl",
    format=Documenter.HTML(;
        canonical="https://xuanzhaogao.github.io/DielectricBIE.jl",
        edit_link="main",
        assets=String[],
        size_threshold=400 * 2^10,        # the single API page (every docstring) is past the 200 KiB default
        size_threshold_warn=250 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        "User guide" => "guide.md",
        "Lattice campaigns" => "campaign.md",
        "API reference" => "api.md",
        "Development" => "development.md",
    ],
)

deploydocs(;
    repo="github.com/xuanzhaogao/DielectricBIE.jl",
    devbranch="dev",
)
