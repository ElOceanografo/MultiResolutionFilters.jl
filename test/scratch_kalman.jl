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
observations = rand.(Poisson.(exp.(3data.z)))

plt = scatter(x, y, zcolor=observations, label="")

function obs_loglik(μ, observations)
    return sum(logpdf(Poisson(exp.(dot([1.0, 0.0], μ))), obs) for obs in observations)
end
obs_loglik([1, 2], [0, 1, 1, 3, 2])

r = KalmanRefinery(
    a -> I(2),
    a -> 0.5e-6 * a * I(2),
    observations,
    locations,
    obs_loglik,
    MvNormal(zeros(2), 1.0),
    250
)

root = KalmanCellData(1:length(observations), r.state_prior)
tree = Cell(SVector(0.0, 0.0), SVector(1e3, 1e3), root)
adaptivesampling!(tree, r)

simulate_prior!(r, tree)
heatmap(value_array(tree, 1))
set_priors!(r, tree, tree.data.state)

observe_data!(r, tree)
plot(heatmap(mean_array(tree, 1), yflip=true),
    heatmap(sqrt.(cov_array(tree, 1)), yflip=true, c=:viridis))


plt0 = plot(Normal(mean(parent)[1], sqrt(cov(parent)[1, 1])), label="Parent");
for (i, child) in enumerate(children)
    x_child = child.data.state
    μ_pred, P_pred, Pinv_pred = predict_upscale(r, parent, child)
    println(Pinv_pred)
    plot!(plt0, Normal(mean(x_child)[1], sqrt(cov(x_child)[1, 1])), 
        label="Child $(i)", color=i+1) 
    plot!(plt0, Normal(μ_pred[1], sqrt(P_pred[1, 1])), 
        label="", color=i+1, linestyle=:dash) 
end
plt0


merge_child_states!(r, parent)
plt1 = plot(Normal(mean(parent)[1], sqrt(cov(parent)[1, 1])), label="Parent");
Pt = zeros(2,2)
for (i, child) in enumerate(children)
    μ, P, Pinv = predict_upscale(r, parent, child)
    Pt += inv(P)
    plot!(plt1, Normal(μ[1], sqrt(P[1, 1])), 
        label="Child $(i)") 
end
plt1

plt2 = plot(Normal(mean(parent)[1], sqrt(cov(parent)[1, 1])), label="Parent");
for (i, child) in enumerate(children)
    state_smoothed = downscale(r, parent, child)
    plot!(plt2, Normal(mean(state_smoothed)[1], sqrt(cov(state_smoothed)[1, 1])), 
        label="Child $(i)") 
end
plot(plt0, plt1, plt2, layout=(3,1), xlims=(-3, 3.5))


@time multiresolution_smooth!(r, tree)
plot(heatmap(mean_array(tree, 1), yflip=true),
    heatmap(sqrt.(cov_array(tree, 1)), yflip=true, c=:viridis))

simulate_posterior!(r, tree)
heatmap(value_array(tree, 1), yflip=true)