module MultiResolutionFilters

using RegionTrees
using StaticArrays
using Statistics, StatsBase

export ParticleRefinery,
    CellData,
    MultiResolutionPF,
    nobs,
    needs_refinement,
    inrect,
    calculate_weights,
    area,
    particlefilter,
    refine_data,
    adaptive_filter!


struct ParticleRefinery <: AbstractRefinery
    downscale::Function         # function downscale(state, child_area) -> state_new
    observe::Function           # function observe(state) -> simulated observation
    loglikelihood::Function     # Function loglikelihood(state, observations)
    needs_refinement::Function  # function needs_refinement(cell) -> Bool
    jitter::Function            # function jitter(state) -> state + noise
end

function ParticleRefinery(downscale::Function, observe::Function, loglikelihood::Function,
        needs_refinement::Function, jitter=x -> x)
    return ParticleRefinery(downscale, observe, loglikelihood, needs_refinement, jitter)
end

struct CellData
    locations::AbstractVector
    observations::AbstractVector
    state_particles::AbstractVector
    state_weights::AbstractVector
end

struct MultiResolutionPF
    locations::AbstractVector
    observations::AbstractVector
    refinery::AbstractRefinery
    tree::Cell
end

function MultiResolutionPF(locations::AbstractVector, observations::AbstractVector,
        refinery::AbstractRefinery, state_init::AbstractVector)
    origin = SVector(reduce((x1, x2) -> min.(x1, x2), locations)...)
    extreme = SVector(reduce((x1, x2) -> max.(x1, x2), locations)...)
    extent = extreme - origin
    nparticles = length(state_init)
    data = CellData(locations, observations, state_init, ones(nparticles)/nparticles)
    tree = Cell(origin, extent, data)
    return MultiResolutionPF(locations, observations, refinery, tree)
end

StatsBase.nobs(cd::CellData) = length(cd.observations)

function RegionTrees.needs_refinement(r::ParticleRefinery, cell)
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
    return new_states, w
end

function RegionTrees.refine_data(r::ParticleRefinery, cell::Cell, indices)
    boundary = child_boundary(cell, indices)
    data = cell.data
    state_pred = r.downscale.(data.state_particles, area(boundary))

    ii = findall(x -> inrect(x, boundary), data.locations)
    child_locations = @view data.locations[ii]
    child_observations = @view data.observations[ii]
    if length(ii) > 0
        state_filt, weights_filt = particlefilter(r, state_pred, child_observations)
    else
        state_filt = state_pred
        weights_filt = data.state_weights
    end
    return CellData(child_locations, child_observations, state_filt, weights_filt)
end

function adaptive_filter!(mrf::MultiResolutionPF)
    adaptivesampling!(mrf.tree, mrf.refinery)
end


end
