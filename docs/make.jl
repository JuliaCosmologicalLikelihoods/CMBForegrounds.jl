using CMBForegrounds
using Documenter

DocMeta.setdocmeta!(CMBForegrounds, :DocTestSetup, :(using CMBForegrounds); recursive=true)

makedocs(;
    modules=[CMBForegrounds],
    authors="Marco Bonici <bonici.marco@gmail.com> and contributors",
    sitename="CMBForegrounds.jl",
    format=Documenter.HTML(;
        canonical="https://mbonici.github.io/CMBForegrounds.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/mbonici/CMBForegrounds.jl",
    devbranch="main",
)