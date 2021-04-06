using ImageFiltering
using RegionTrees
using StaticArrays
using Statistics, StatsBase
using Distributions
using StatsPlots


struct ParticleRefinery <: AbstractRefinery
    downscale         # function downscale(state, child_area) -> state_new
    observe           # function observe(state) -> simulated observation
    loglikelihood     # Function loglikelihood(state, observations)
    needs_refinement  # function needs_refinement(cell) -> Bool
end

struct CellData
    locations
    observations
    state_particles
    state_weights
end

StatsBase.nobs(cd::CellData) = length(cd.observations)


function RegionTrees.needs_refinement(r::ParticleRefinery, cell)
    # d = length(cell.boundary.origin)
    # dims = Tuple(2 for _ in 1:d)
    # idxs = CartesianIndices(dims)
    # npoints = zeros(Int, length(idxs))
    # for (i, ci) in enumerate(idxs)
    #     boundary = child_boundary(cell, ci.I)
    #     npoints[i] = sum([inrect(xi, boundary) for xi in eachcol(cell.data.x)])
    # end
    # return all(npoints .> r.n)
    # return size(cell.data.x, 2) > r.min_points
    r.needs_refinement(cell)
end

function inrect(x::AbstractVector, hr::HyperRectangle)
    return all(hr.origin .< x .< (hr.origin .+ hr.widths))
end

function calculate_weights(states, observations, loglikelihood)
    ll = [loglikelihood(s, observations) for s in states]
    w = exp.(ll .- maximum(ll))
    return w / sum(w)
end

area(hr::HyperRectangle) = prod(hr.widths)

function RegionTrees.refine_data(r::ParticleRefinery, cell::Cell, indices)
    boundary = child_boundary(cell, indices)
    x, z, states, w, n, nparticles = cell.data
    ii = [inrect(xi, boundary) for xi in eachcol(x)]
    new_x = x[:, ii]
    new_observations = z[ii]
    if sum(ii) > 0
        new_w = calculate_weights(states, new_observations, r.loglikelihood)
    else
        new_w = w
    end
    new_state = sample(states, weights(new_w), nparticles, replace=true) .+
        1.0randn(nparticles)
    return (x=new_x, z=new_observations, state=new_state, w=new_w, n=size(new_x, 2),
        nparticles=nparticles)
end


Z = imfilter(randn(1000, 1000), Kernel.gaussian(25)) .+ 0.02randn(1000, 1000)
Z = Z .+ (1:1000) ./ 2e3
heatmap(Z)

n = 1_000
x = [100randn(2, n) .+ 300  150randn(2, n) .+ 700]
x = filter(xi -> all(1 .< xi .<= 1000), collect(eachcol(x)))
z = [Z[round(Int, xi[2]), round(Int, xi[1])] for xi in x]
# x = hcat(x...)
x = SVector{2}.(x)
scatter(first.(x), last.(x), zcolor=z, markerstrokewidth=0, legend=false)

nparticles = 1000
state_particles = 10*randn(nparticles)
state_weights = ones(nparticles) / nparticles


x0 = [0., 0.]
width = [1000., 1000]

root_data = CellData(x, z, state_particles, state_weights)
root = Cell(SVector(x0...), SVector(width...), root_data)



loglik(state, obs) = loglikelihood(Normal(state, 1.0), obs)
predict_downscale(state, child_area) = state + 0.1randn()
observe(state) = state + randn()
needsrefinement(cell) = nobs(cell) > 3

r = ParticleRefinery(predict_downscale, observe, loglik, needsrefinement)

adaptivesampling!(root, r)

leaf = collect(allleaves(root))[3]

plt = plot(legend=nothing);
for leaf in allleaves(root)
    x, y = leaf.boundary.origin
    w, h = leaf.boundary.widths
    plot!(plt, Shape(x .+ [0,w,w,0], y .+ [0,0,h,h]))
end
plt


plt = plot(legend=false)
for leaf in allleaves(root)
    density!(plt, leaf.data.state)
end
plt


μs = [mean(leaf.data.state) for leaf in allleaves(root)]
histogram(μs)
zl = extrema(μs)
cg = cgrad(:magma);
plt = plot(legend=nothing);
for leaf in allleaves(root)
    # if leaf.data.n > 0
        x, y = leaf.boundary.origin
        w, h = leaf.boundary.widths
        μ = mean(leaf.data.state)
        c = (μ - zl[1]) / (zl[2] - zl[1])
        plot!(plt, Shape(x .+ [0,w,w,0], y .+ [0,0,h,h]), fill=get(cg, c),
            linecolor=get(cg, c))
    # end
end
plt
scatter(plt, x[1, :], x[2, :], color=:black, size=2)
