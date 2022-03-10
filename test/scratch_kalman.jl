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

function obs_loglik(μ, observations)
    return sum(logpdf(Poisson(exp(μ[1])), obs) for obs in observations)
end
obs_loglik([1, 2], [0, 1, 1, 3, 2])

r = KalmanRefinery(I(2),
    0.9I(2),
    observations,
    locations,
    obs_loglik,
    MvNormal(zeros(2), 1.0),
    400
)

root = KalmanCellData(1:length(observations), r.state_prior)
tree = Cell(SVector(0.0, 0.0), SVector(1e3, 1e3), root)
adaptivesampling!(tree, r)

multiresolution_smooth!(r, tree)

heatmap(mean_array(tree, 1))
heatmap(cov_array(tree, 1), c=:viridis)
