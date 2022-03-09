using MultiResolutionFilters
using Test
using DataFrames, CSV
using Distributions
using StaticArrays
using LinearAlgebra
using RegionTrees

@testset "MultiResolutionFilters.jl" begin
    data = CSV.read("test_data.csv", DataFrame)
    locations = [SVector(row...) for row in eachrow(data[:, [:x, :y]])]

    x = [el for el in data.x]
    y = [el for el in data.y]
    observations = rand.(Poisson.(exp.(2.5data.z)))
   
    function obs_loglik(μ, observation)
        return sum(logpdf.(Poisson.(exp.(μ[1])), observation))
    end

    r = KalmanRefinery(I(2),
        0.009I(2),
        observations,
        locations,
        obs_loglik,
        MvNormal(zeros(2), 1.0),
        1)

    root = KalmanCellData(1:length(observations), r.state_prior)
    tree = Cell(SVector(0.0, 0.0), SVector(1e3, 1e3), root)
    adaptivesampling!(tree, r)
    nleaves = length(collect(allleaves(tree)))
    ii = findall(has_observation, collect(allleaves(tree)))
    
    @test length(ii) < nleaves

    i = first(ii)
    leaf = collect(allleaves(tree))[i]
    state0 = deepcopy(leaf.data.state)
    observation = r.observations[leaf.data.ii_data]
    observe!(leaf, r.observations[only(leaf.data.ii_data)], r.obs_loglik)

    @test mean(state0) != mean(leaf.data.state)
    @test cov(state0) != cov(leaf.data.state)

    observe_data!(r, tree)

    children = [leaf for leaf in allleaves(tree) if isfiltered(leaf)]
    parents = setdiff(collect(allcells(tree)), children)
    nparents = length(parents)

    @test length(children) >= length(ii)

    ii = ready_to_merge.(parents)
    cells_to_merge = parents[ii]
    for cell in cells_to_merge
        merge_child_states!(r, cell)
    end
    parents = parents[.! ii]

    @test length(parents) < nparents

    filter_upscale!(r, tree)
    @test all(isfiltered.(allcells(tree)))

    smooth_downscale!(r, tree, tree.data.state)
    @test all(issmoothed.(allcells(tree)))


end
