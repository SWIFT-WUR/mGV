# Runic formatter
The mGV repo uses the Runic formatter to standardize the code, and CI checks the correct code formatting.
Runic has no configuration and the formatting rule are set in stone. 
For more instruction see https://github.com/fredrikekre/Runic.jl and https://github.com/fredrikekre/runic-action.
It is recommended to apply all the formatting within one or more commits containing only those changes. 
Adding the full formatting commit hashes into the `.git-blme-ignore-revs` file, `git blame` is instructed to ignore them.

## Installation

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
# Format a single file in-place
runic --inplace my_script.jl
```
