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
