using MultiResolutionFilters
using ImageFiltering
using RegionTrees
using StaticArrays
using Statistics, StatsBase
using Distributions
using StatsPlots
using Random

Random.seed!(1)
Z = 1.5imfilter(randn(500, 500), Kernel.gaussian(20)) #.+ 0.02randn(500, 500)
Z = Z .+ (1:500) ./ 2e3

n = 1_000
x = [60randn(2, n) .+ 150  75randn(2, n) .+ 350]
x = filter(xi -> all(1 .< xi .<= 500), collect(eachcol(x)))
z = [Z[round(Int, xi[2]), round(Int, xi[1])] for xi in x]
z = z + 0.05randn(length(z))
# x = hcat(x...)
x = SVector{2}.(x)

zl = extrema(Z)
plt_truth = heatmap(Z, clim=zl, legend=false, xlabel="X", ylabel="Y")
scatter!(plt_truth, first.(x), last.(x), color=:black, markersize=2, label="")

scatter(first.(x), last.(x), zcolor=z, markerstrokewidth=0, legend=false, xlabel="X", ylabel="Y")

nparticles = 2000
state_particles = randn(nparticles)
state_weights = ones(nparticles) / nparticles
x0 = [0., 0.]
width = [500., 500]

dx = 500 ./ 2.0.^(0:10)
dA = dx.^2
plot(dx)
plot(dA)

loglik(state, obs) = loglikelihood(Normal(state, 0.1), obs)
predict_downscale(state, child_area) = state + 0.5randn()
observe(state) = state + randn()
needsrefinement(cell) = area(cell.boundary) > dA[6]
jitter(state) = state #+ 0.05randn()

r = ParticleRefinery(predict_downscale, observe, loglik, needsrefinement)
mrf = MultiResolutionPF(x, z, r, state_particles)
adaptive_filter!(mrf)
tree = mrf.tree


p0 = plot(legend=false)
v = hcat(collect(vertices(tree.boundary))...)
plot!(p0, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]])
c1 = children(tree)
p1 = plot(legend=false)
for c in c1
    v = hcat(collect(vertices(c.boundary))...)
    plot!(p1, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], color=2)
end
p1
p2 = plot(legend=false)
c2 = reduce(vcat, [children(c) for c in c1])
for c in c2
    v = hcat(collect(vertices(c.boundary))...)
    plot!(p2, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], color=3)
end
c3 = reduce(vcat, [children(c) for c in c2])
p3 = plot(legend=false)
for c in c3
    v = hcat(collect(vertices(c.boundary))...)
    plot!(p3, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], color=4)
end
plot(p0, p1, p2, p3)


μs = [mean(leaf.data.state_particles) for leaf in allleaves(tree)]
histogram(μs)

cg = cgrad(:magma);
plt_post = plot(legend=nothing, xlabel="X", ylabel="Y");
for leaf in allleaves(tree)
    xl, yl = leaf.boundary.origin
    w, h = leaf.boundary.widths
    μ = mean(leaf.data.state_particles)
    c = (μ - zl[1]) / (zl[2] - zl[1])
    # nobs(leaf.data) > 0 ? linecolor = :black : linecolor = get(cg, c)
    linecolor = get(cg, c)
    plot!(plt_post, Shape(xl .+ [0,w,w,0], yl .+ [0,0,h,h]), fill=get(cg, c),
        linecolor=linecolor)
end
plt_post
scatter!(plt_post, first.(x), last.(x), color=:black, markersize=2)

cg = cgrad(:viridis);
cl = extrema([std(leaf.data.state_particles) for leaf in allleaves(tree)])
plt_std = plot(legend=nothing, xlabel="X", ylabel="Y");
for leaf in allleaves(tree)
    # if leaf.data.n > 0
        xl, yl = leaf.boundary.origin
        w, h = leaf.boundary.widths
        σ = std(leaf.data.state_particles)
        c = (σ - cl[1]) / (cl[2] - cl[1])
        plot!(plt_std, Shape(xl .+ [0,w,w,0], yl .+ [0,0,h,h]), fill=get(cg, c),
            linecolor=get(cg, c))
    # end
end
plt_std

plot(plt_truth, plt_post, size=(800, 500))

B = [sum(area(leaf.boundary) * sample(leaf.data.state_particles)
    for leaf in allleaves(tree)) for i in 1:1000]

histogram(B)

dA = 1.0
vline!([sum(Z*dA)])


ii = sortperm([leaf.boundary.origin for leaf in allleaves(tree)])
heatmap(reshape(μs[ii], 32, 32))
