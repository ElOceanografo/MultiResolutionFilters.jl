# using MultiResolutionFilters
# using Test
using Distributions
using RegionTrees
using Morton
using StaticArrays
using RxInfer
import RegionTrees: AbstractRefinery, needs_refinement, refine_data

struct RWRefinery <: AbstractRefinery
    tolerance::Float64
    σ::Float64
    d::Int
end

# These two methods are all we need to implement
function needs_refinement(r::RWRefinery, cell)
    maximum(cell.boundary.widths) > r.tolerance
end

function refine_data(r::RWRefinery, cell::Cell, indices)
    boundary = child_boundary(cell, indices)
    i = 2^r.d *(index(cell)-1) + cartesian2morton([indices...]) + 1
    # println(index(cell), " ", indices, " ", cartesian2morton([indices...]), " ", i)
    area = prod(boundary.widths)
    return (i=i, x=cell.data.x + r.σ * randn())
end

nleaves(tree) = length(collect(allleaves(tree)))
ncells(tree) = length(collect(allcells(tree)))
index(cell) = cell.data.i
data(cell) = cell.data.x
coordinates(cell) = cell.boundary.origin
# Now we can use our refinery to create the entire tree, with
# all cells split automatically:
r = RWRefinery(0.05, 0.1, 2)
root = Cell(SVector(0., 0), SVector(1., 1), (i=1, x=0.0))
adaptivesampling!(root, r)
length(collect(allcells(root)))
length(unique(index.(allcells(root))))


z = data.(allleaves(root))
xy = coordinates.(allleaves(root))
x = first.(xy)
y = last.(xy)
using Plots
scatter(x, y, zcolor=z, markerstrokewidth=0, shape=:square, aspect_ratio=:equal,
    markersize=5)

length(collect(allcells(root)))
length(unique(index.(allcells(root))))

@model function treemodel(tree, nc, nl, σ)
    y = datavar(nl)
    x = randomvar(nc)
    
    x[1] ~ NormalMeanVariance(0.0, 1.0)
    for cell in collect(allcells(tree))[2:end]
        i = index(cell)
        i_parent = index(parent(cell))
        x[i] ~ NormalMeanVariance(x[i_parent], σ^2)
    end

    for (j, leaf) in enumerate(alleaves(tree))
        y[i] ~ NormalMeanVariance(x[index(leaf)], 0.1)
    end
    
end