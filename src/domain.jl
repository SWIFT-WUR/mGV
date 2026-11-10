"""
The active (land) cells of the input grid. The model only stores and computes these
cells: grid fields become vectors over the active cells, and every other dimension (soil
layer, ...) follows the cell dimension. Snow bands and vegetation types are stored as
[`Tiles`](@ref) instead. Cells keep the order of the grid, with longitude varying fastest.

Input and output files are gridded; [`to_cells`](@ref) gathers the active cells from a
gridded array, and [`to_grid!`](@ref) scatters them back.
"""
struct Domain
    grid_size::Tuple{Int, Int}  # (nx, ny) of the input grid
    cells::Vector{Int}          # linear index into the (nx, ny) grid of each active cell
end

ncells(domain::Domain) = length(domain.cells)

"""
Read the active cells from the parameter file: the cells where the run cell mask
(`run_cell`) is larger than zero. Missing values count as zero.
"""
function Domain(config::Cfg)
    name = config.input.names.run_cell
    ds = open_param_source(config.input.paths.input_param_file)
    run_cell = readfull(ds, name, 0)
    close_param_source(ds)
    active = run_cell .> 0

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

"""
The snow bands and vegetation types that cover part of each cell. Most of the (band,
vegetation type) combinations of a cell have no area, so only the ones that do are stored:

- the active bands: the bands of each cell with a snow band area fraction above zero.
  Per-band fields are vectors over the active bands.
- the tiles: the vegetation types of each active band with a vegetation fraction above
  zero. Per-tile fields (and the vegetation parameters) are vectors over the tiles.

Bands are ordered by cell, and tiles by cell, then band, then vegetation type. The tiles of
a cell (or band) are therefore contiguous: `cell_tiles` and `band_tiles` hold the first tile
of each cell (band), followed by `ntiles + 1`.
"""
struct Tiles{V <: AbstractVector{Int32}}
    cell::V         # cell of each tile
    band::V         # band of each tile, an index into the active bands
    veg::V          # vegetation type of each tile, 1 to nveg
    band_cell::V    # cell of each active band
    band_number::V  # band of each active band in the input data, 1 to the number of bands
    cell_tiles::V   # the tiles of cell i are cell_tiles[i]:(cell_tiles[i + 1] - 1)
    band_tiles::V   # the tiles of band j are band_tiles[j]:(band_tiles[j + 1] - 1)
    nveg::Int       # number of vegetation types; the last one is bare soil
end

@adapt_structure Tiles

ntiles(tiles::Tiles) = length(tiles.cell)
nbands(tiles::Tiles) = length(tiles.band_cell)

"""
Find the tiles from the snow band area fraction, of size (ncells, nbands), and the
vegetation fraction, of size (ncells, nveg). Missing values should be replaced by zero.
"""
function Tiles(area_fraction::AbstractMatrix, vegetation_fraction::AbstractMatrix)
    n, nb = size(area_fraction)
    nveg = size(vegetation_fraction, 2)
    size(vegetation_fraction, 1) == n ||
        error("Vegetation fraction has $(size(vegetation_fraction, 1)) cells, area fraction $n")

    cell, band, veg = Int32[], Int32[], Int32[]
    band_cell, band_number = Int32[], Int32[]
    cell_tiles, band_tiles = Int32[1], Int32[1]
    for i in 1:n
        for b in 1:nb
            area_fraction[i, b] > 0 || continue
            push!(band_cell, i)
            push!(band_number, b)
            for v in 1:nveg
                vegetation_fraction[i, v] > 0 || continue
                push!(cell, i)
                push!(band, length(band_cell))
                push!(veg, v)
            end
            push!(band_tiles, length(cell) + 1)
        end
        push!(cell_tiles, length(cell) + 1)
    end

    tiles = Tiles(cell, band, veg, band_cell, band_number, cell_tiles, band_tiles, nveg)
    println("Tiles: $(ntiles(tiles)) tiles in $(nbands(tiles)) snow bands, of $(n * nb * nveg) possible")
    return tiles
end

"""
Gather the active bands of an array over the cells, of size (ncells, nbands), into a vector
over the active bands.
"""
function to_bands(tiles::Tiles, x::AbstractMatrix)
    return x[CartesianIndex.(tiles.band_cell, tiles.band_number)]
end

"""
Gather the tiles of an array over the cells, of size (ncells, dims..., nveg), into an array
of size (ntiles, dims...). Every band of a cell gets the same values.
"""
function to_tiles(tiles::Tiles, x::AbstractArray)
    rest = size(x)[2:(end - 1)]
    size(x, ndims(x)) == tiles.nveg ||
        error("Array of size $(size(x)) does not have $(tiles.nveg) vegetation types")
    flat = reshape(x, size(x, 1), prod(rest), tiles.nveg)
    out = similar(x, ntiles(tiles), prod(rest))
    for (t, (i, v)) in enumerate(zip(tiles.cell, tiles.veg))
        out[t, :] .= @view flat[i, :, v]
    end
    return reshape(out, ntiles(tiles), rest...)
end
