### Runic formatter
#### Installation

In Julia v1.12 and later, Runic can be installed as a
[Pkg app](https://pkgdocs.julialang.org/dev/apps/):

```sh
julia -e 'using Pkg; Pkg.Apps.add("Runic")'
```

Assuming `~/.julia/bin` is in your `PATH` you can now invoke `runic` from your shell, e.g.:

```sh
runic --version # Show version info
runic --help    # Show documentation
```

```sh
# Format all files in-place in the current directory (recursively)
runic --inplace .
```

For more instruction see https://github.com/fredrikekre/Runic.jl and https://github.com/fredrikekre/runic-action.
