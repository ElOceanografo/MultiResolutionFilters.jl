using MultiResolutionFilters
using Documenter

DocMeta.setdocmeta!(MultiResolutionFilters, :DocTestSetup, :(using MultiResolutionFilters); recursive=true)

makedocs(;
    modules=[MultiResolutionFilters],
    authors="Sam src/",
    repo="https://github.com/ElOceanografo/MultiResolutionFilters.jl/blob/{commit}{path}#{line}",
    sitename="MultiResolutionFilters.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ElOceanografo.github.io/MultiResolutionFilters.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ElOceanografo/MultiResolutionFilters.jl",
)
