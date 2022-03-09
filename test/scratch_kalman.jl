using MultiResolutionFilters
using RegionTrees
using DataFrames, CSV
using LinearAlgebra
using Distributions
using StaticArrays
using StatsPlots
using Optim, ForwardDiff

data = CSV.read("test/test_data.csv", DataFrame)
# data = data[sample(1:nrow(data), 200), :]
locations = [SVector(row...) for row in eachrow(data[:, [:x, :y]])]
x = [el for el in data.x]
y = [el for el in data.y]
observations = rand.(Poisson.(exp.(2.5data.z)))

plt = scatter(x, y, zcolor=observations, label="")

function obs_loglik(μ, observation)
    return sum(logpdf.(Poisson.(exp.(μ[1])), observation))
end

r = KalmanRefinery(I(2),
    0.009I(2),
    observations,
    locations,
    obs_loglik,
    MvNormal(zeros(2), 1.0),
    1
)

root = KalmanCellData(1:length(observations), r.state_prior)
tree = Cell(SVector(0.0, 0.0), SVector(1e3, 1e3), root)
adaptivesampling!(tree, r)

multiresolution_smooth!(r, tree)

cg = cgrad(:magma);
plt_post = plot(legend=nothing, xlabel="X", ylabel="Y");
zl = [-1, 1]
for leaf in allleaves(tree)
    xl, yl = leaf.boundary.origin
    w, h = leaf.boundary.widths
    μ = mean(leaf.data.state)[1]
    c = (μ - zl[1]) / (zl[2] - zl[1])
    # nobs(leaf.data) > 0 ? linecolor = :black : linecolor = get(cg, c)
    linecolor = get(cg, c)
    plot!(plt_post, Shape(xl .+ [0,w,w,0], yl .+ [0,0,h,h]), fill=get(cg, c),
        linecolor=linecolor)
end
scatter!(plt_post, x, y, color=:black, markersize=2);

cg_err = cgrad(:viridis);
zl_err = [0.5, 1.]
plt_err = plot(legend=nothing, xlabel="X", ylabel="Y");
for leaf in allleaves(tree)
    xl, yl = leaf.boundary.origin
    w, h = leaf.boundary.widths
    σ = cov(leaf.data.state)[1, 1]
    c = (σ - zl_err[1]) / (zl_err[2] - zl_err[1])
    # nobs(leaf.data) > 0 ? linecolor = :black : linecolor = get(cg, c)
    linecolor = get(cg_err, c)
    plot!(plt_err, Shape(xl .+ [0,w,w,0], yl .+ [0,0,h,h]), fill=get(cg_err, c),
        linecolor=linecolor)
end
scatter!(plt_err, x, y, color=:black, markersize=2);
plot(plt_post, plt_err)