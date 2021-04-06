using MultiResolutionFilters
using Test
using DataFrames, CSV
using StaticArrays

@testset "MultiResolutionFilters.jl" begin
    data = CSV.read("test_data.csv", DataFrame)
    locations = [SVector(row...) for row in eachrow(data[:, [:x, :y]])]
    observations = data.z
    nparticles = 1000

    x0 = [0.0, 0.0]
    w = [1e3, 1e3]

    cd = CellData(1:length(locations),
        randn(nparticles),
        ones(nparticles)/nparticles)

    mrf = MultiResolutionPF(

    )
end
