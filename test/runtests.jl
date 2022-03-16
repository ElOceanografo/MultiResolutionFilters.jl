using MultiResolutionFilters
using Test
using DataFrames, CSV
using Distributions
using StaticArrays
using LinearAlgebra
using RegionTrees

function obs_loglik(μ, observation)
    return sum(logpdf.(Poisson.(exp.(μ[1])), observation))
end

function newtree()
    data = CSV.read("test_data.csv", DataFrame)
    locations = [SVector(row...) for row in eachrow(data[:, [:x, :y]])]
    observations = rand.(Poisson.(exp.(2.5data.z)))
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
    return r, tree
end

@testset "MultiResolutionFilters.jl" begin
    r, tree = newtree()
    
    nleaves = length(collect(allleaves(tree)))
    ii = findall(has_observation, collect(allleaves(tree)))
    
    @test length(ii) < nleaves

    i = first(ii)
    leaf = collect(allleaves(tree))[i]
    state0 = deepcopy(leaf.data.state)
    observation = r.observations[leaf.data.ii_data]
    observe!(leaf, r.observations[only(leaf.data.ii_data)], r.obs_loglik)

    @test mean(state0) != mean(leaf.data.state)
    @test diag(cov(state0)) != diag(cov(leaf.data.state))

    observe_data!(r, tree)
    @test all(isfiltered.(collect(allleaves(tree))))

    multiresolution_smooth!(r, tree)
end
