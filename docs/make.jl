using CMBForegrounds
using Documenter

DocMeta.setdocmeta!(CMBForegrounds, :DocTestSetup, :(using CMBForegrounds); recursive=true)

makedocs(;
    modules=[CMBForegrounds],
    authors="Marco Bonici <bonici.marco@gmail.com> and contributors",
    sitename="CMBForegrounds.jl",
    format=Documenter.HTML(;
        canonical="https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    warnonly=[:missing_docs],  # Don't error on undocumented functions, just warn
)

deploydocs(;
    repo="github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl",
    devbranch="develop",
)
