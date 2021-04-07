using MultiResolutionFilters
using Test
using DataFrames, CSV
using Distributions
using StaticArrays

@testset "MultiResolutionFilters.jl" begin
    data = CSV.read("test_data.csv", DataFrame)
    locations = [SVector(row...) for row in eachrow(data[:, [:x, :y]])]
    observations = data.z
    nparticles = 1000
    state_particles = randn(nparticles)

    loglik(state, obs) = loglikelihood(Normal(state, 0.1), obs)
    predict_downscale(state, child_area) = state + 0.5randn()
    observe(state) = state + randn()
    needsrefinement(cell) = area(cell.boundary) > 250#dA[6]
    jitter(state) = state

    refinery = ParticleRefinery(predict_downscale, observe, loglik, needsrefinement, jitter)
    mrf = MultiResolutionPF(locations, observations, refinery, state_particles)
    adaptive_filter!(mrf)

    @test 1 == 1
end
