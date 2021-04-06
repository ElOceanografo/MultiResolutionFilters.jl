module MultiResolutionFilters

using RegionTrees

export MultiResolutionPF, CellData

struct CellData
    observation_idx
    state_particles
    state_weights
end

struct MultiResolutionPF
    root_cell
    data
    predict # (state, child) -> state
    loglik  # (state, observations) -> loglik
    refinery
end


#=
Data needed in each cell:
- locations
- observations
- state: each cell has a single state estimate (i.e., collection of particles)

=#


end
