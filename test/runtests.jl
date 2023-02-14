# using MultiResolutionFilters
# using Test
using Distributions
using RegionTrees
using Morton
using StaticArrays
using Missings
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
    return (i=i, x=cell.data.x + r.σ * area^0.1 * randn())
end

nleaves(tree) = length(collect(allleaves(tree)))
ncells(tree) = length(collect(allcells(tree)))
index(cell) = cell.data.i
data(cell) = cell.data.x
coordinates(cell) = cell.boundary.origin
# Now we can use our refinery to create the entire tree, with
# all cells split automatically:
r = RWRefinery(0.01, 0.1, 2)
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
    markersize=3)

length(collect(allcells(root)))
length(unique(index.(allcells(root))))

# from https://biaslab.github.io/RxInfer.jl/stable/examples/Handling%20Missing%20Data/
@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_τ::Missing) = missing
@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::Missing, q_τ::Any) = missing
@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::Missing, q_τ::PointMass) = missing
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::Missing) = missing
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Missing, q_μ::Any) = missing
@rule typeof(+)(:in1, Marginalisation) (m_out::Missing, m_in2::Any) = missing
@rule typeof(+)(:in1, Marginalisation) (m_out::Any, m_in2::Missing) = missing

@model function treemodel(tree, σ)
    y = datavar(Float64, nleaves(tree)) where {allow_missing = true}
    x = randomvar(ncells(tree)) 
    
    x[1] ~ Normal(mean = 0.0, precision = 1/5)
    for cell in collect(allcells(tree))[2:end]
        i = index(cell)
        i_parent = index(parent(cell))
        v = 
        x[i] ~ Normal(mean = x[i_parent], precision = σ^-2)
    end

    leaves = collect(allleaves(tree))
    for i in 1:length(leaves)
        leaf = leaves[i]
        y[i] ~ Normal(mean = x[index(leaf)], precision = 1/0.1)
    end
end

mod = treemodel(root, 0.1)
z1 = allowmissing(z)
z1[(0.4 .< x .< 0.6) .& (0.4 .< y .< 0.6)] .= missing
z1[rand(1:length(z1), 2000)] .= missing


dat = (y = z1,)
res = inference(model = mod, data = dat)

plot(mean.(res.posteriors[:x]), ribbon=2std.(res.posteriors[:x]))
plot!(data.(allcells(root)))

leaf_indices = index.(allleaves(root))
plot(mean.(res.posteriors[:x][leaf_indices]),
    ribbon=2std.(res.posteriors[:x][leaf_indices]))
plot!(data.(allleaves(root)))

pdata = scatter(x, y, zcolor=replace(z1, missing => NaN), markerstrokewidth=0, shape=:square, aspect_ratio=:equal,
    markersize=3, clims=(-0.4, 0.4))
psmooth = scatter(x, y, zcolor=mean.(res.posteriors[:x][leaf_indices]),
    markerstrokewidth=0, shape=:square, aspect_ratio=:equal,
    markersize=3)
perror = scatter(x, y, zcolor=std.(res.posteriors[:x][leaf_indices]),
    markerstrokewidth=0, shape=:square, aspect_ratio=:equal,
    markersize=3, c=:viridis)
plot(pdata, psmooth, perror, legend=false, markersize=1.3, size=(800, 800))
