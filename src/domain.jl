"""
The active (land) cells of the input grid. The model only stores and computes these
cells: grid fields become vectors over the active cells, and every other dimension (snow
band, vegetation type, soil layer, ...) follows the cell dimension. Cells keep the order
of the grid, with longitude varying fastest.

Input and output files are gridded; [`to_cells`](@ref) gathers the active cells from a
gridded array, and [`to_grid!`](@ref) scatters them back.
"""
struct Domain
    grid_size::Tuple{Int, Int}  # (nx, ny) of the input grid
    cells::Vector{Int}          # linear index into the (nx, ny) grid of each active cell
end

ncells(domain::Domain) = length(domain.cells)

"""
Read the active cells from the parameter file: the cells with vegetation, where the sum of
the vegetation fractions (`Cv`) over the vegetation types is larger than zero. Missing or
NaN fractions count as zero.
"""
function Domain(config::Cfg)
    name = config.input.names.vegetation_fraction
    ds = open_param_source(config.input.paths.input_param_file)
    cv = replace(readfull(ds, name, 0.0), NaN => 0)
    close_param_source(ds)
    cv_sum = dropdims(sum(cv; dims = 3:ndims(cv)); dims = Tuple(3:ndims(cv)))
    active = cv_sum .> 1.0e-6

    domain = Domain(size(active), findall(vec(active)))
    ncells(domain) > 0 || error("The parameter file has no active cells ($name is zero everywhere)")
    nx, ny = domain.grid_size
    println("Domain: $(ncells(domain)) active cells of $nx x $ny grid cells")
    return domain
end

"""
Gather the active cells of a gridded array, of size (nx, ny, dims...), into an array of
size (ncells, dims...).
"""
function to_cells(domain::Domain, grid::AbstractArray)
    size(grid)[1:2] == domain.grid_size ||
        error("Grid of size $(size(grid)[1:2]) does not match the domain grid $(domain.grid_size)")
    rest = size(grid)[3:end]
    flat = reshape(grid, prod(domain.grid_size), prod(rest))
    return reshape(flat[domain.cells, :], ncells(domain), rest...)
end

"""
Scatter an array over the active cells, of size (ncells, dims...), into the active
cells of the gridded array `grid`, of size (nx, ny, dims...). Inactive cells are left
unchanged.
"""
function to_grid!(grid::AbstractArray, domain::Domain, cells::AbstractArray)
    flat = reshape(grid, prod(domain.grid_size), :)
    view(flat, domain.cells, :) .= reshape(cells, ncells(domain), :)
    return grid
end
