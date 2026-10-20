using Documenter
using mGV

makedocs(;
    sitename = "mGV",
    modules = [mGV],
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://SWIFT-WUR.github.io/mGV",
        edit_link = "main",
    ),
    pages = [
        "Introduction" => "index.md",
        "Getting started" => "getting_started.md",
        "Model input data" => "input_data.md",
        "Running on GPU" => "gpu.md",
        "Model coupling" => "coupling.md",
        "Developer documentation" => [
            "developer/index.md",
            "Optimizing GPU performance" => "developer/gpu_performance.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/SWIFT-WUR/mGV.git",
    devbranch = "main",
    push_preview = true,
)
