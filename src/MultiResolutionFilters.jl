module MultiResolutionFilters

using RegionTrees
using Statistics, StatsBase

export ParticleRefinery,
    CellData,
    nobs,
    needs_refinement,
    inrect,
    calculate_weights,
    area,
    particlefilter,
    refine_data


struct ParticleRefinery <: AbstractRefinery
    downscale         # function downscale(state, child_area) -> state_new
    observe           # function observe(state) -> simulated observation
    loglikelihood     # Function loglikelihood(state, observations)
    needs_refinement  # function needs_refinement(cell) -> Bool
    jitter            # function jitter(state) -> state + noise
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

function particlefilter(r, state_particles, observations)
    nparticles = length(state_particles)
    w = calculate_weights(state_particles, observations, r.loglikelihood)
    new_states = sample(r.jitter.(state_particles), weights(w), nparticles)
    # new_states = r.jitter.(new_states)
    return new_states, w
end

function RegionTrees.refine_data(r::ParticleRefinery, cell::Cell, indices)
    boundary = child_boundary(cell, indices)
    data = cell.data
    state_pred = r.downscale.(data.state_particles, area(boundary))

    ii = [inrect(x, boundary) for x in data.locations]
    child_locations = data.locations[ii]
    child_observations = data.observations[ii]
    if sum(ii) > 0
        state_filt, weights_filt = particlefilter(r, state_pred, child_observations)
    else
        state_filt = state_pred
        weights_filt = data.state_weights
    end
    return CellData(child_locations, child_observations, state_filt, weights_filt)
    # return (x=new_x, z=new_observations, state=new_state, w=new_w, n=size(new_x, 2),
    #     nparticles=nparticles)
end


end
