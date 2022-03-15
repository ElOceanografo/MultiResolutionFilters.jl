import Statistics

struct KalmanRefinery <: StateSpaceRefinery
    A
    B
    observations
    locations
    obs_loglik
    state_prior
    min_area
end

mutable struct KalmanCellData
    ii_data
    prior
    state
    value
    filtered
    smoothed
end

function KalmanCellData(ii_data, prior) 
    value = fill(NaN, size(mean(prior)))
    state = deepcopy(prior)
    return KalmanCellData(ii_data, prior, state, value, false, false)
end

data_indices(cd::KalmanCellData) = cd.ii_data
has_observation(cell::Cell) = length(data_indices(cell.data)) >= 1
area(cell::Cell) = area(cell.boundary)
isfiltered(cell::Cell) = cell.data.filtered
issmoothed(cell::Cell) = cell.data.smoothed
state(cell::Cell) = cell.data.state
prior(cell::Cell) = cell.data.prior
Statistics.mean(cell::Cell) = mean(state(cell))
Statistics.cov(cell::Cell) = cov(state(cell))
value(cell::Cell) = cell.data.value

function RegionTrees.needs_refinement(r::KalmanRefinery, cell)
    # return length(data_indices(cell.data)) > r.min_points
    return area(cell.boundary) > r.min_area
end

function RegionTrees.refine_data(r::KalmanRefinery, cell::Cell, indices)
    boundary = child_boundary(cell, indices)
    ii = data_indices(cell.data)
    jj = filter(i -> inrect(r.locations[i], boundary), ii)
    return KalmanCellData(jj, r.state_prior)
end

function set_priors!(r, cell, cell_prior)
    cell.data.prior = cell_prior
    if children(cell) != nothing
        for child in children(cell)
            a = area(child)
            A = r.A(a)
            B = r.B(a)
            μ = A * mean(cell_prior)
            P = A * cov(cell_prior) * A' + B
            child_prior = MvNormal(μ, P)
            @assert all(diag(cov(child_prior)) .> diag(cov(cell_prior)))
            set_priors!(r, child, child_prior)
        end
    end
end

function observe!(leaf, observations, obs_loglik)
    f(μ) = -obs_loglik(μ, observations) - logpdf(leaf.data.state, μ)
    opt = optimize(f, zeros(length(leaf.data.state)))
    μ = opt.minimizer
    H = ForwardDiff.hessian(f, μ)
    P = inv(H)
    leaf.data.state = MvNormal(μ, P)
    return -opt.minimum
end

function observe_data!(r, tree)
    ll = 0.0
    for leaf in allleaves(tree)
        if has_observation(leaf)
            observations = r.observations[leaf.data.ii_data]
            ll += observe!(leaf, observations, r.obs_loglik)
        end
        leaf.data.filtered = true # even if no data, get it ready to merge upwards
    end
    return ll
end

function predict_upscale(r, parent, child)
    A = r.A(area(child))
    P_parent = cov(prior(parent))
    P_child = cov(prior(child))
    P_child_inv = inv(P_child)
    F = P_parent * A' * P_child_inv
    U = P_parent - P_parent * A' * P_child_inv * A * P_parent
    μ = F * mean(child)
    P = F * cov(child) * F' + U
    return (μ = μ, P = P, Pinv = inv(P))
end

function merge_child_states!(r, parent)
    predictions = [predict_upscale(r, parent, child) for child in children(parent)]
    Pinv_prior = inv(cov(prior(parent)))
    Pt = Symmetric(inv(Pinv_prior + sum((pred.Pinv - Pinv_prior) for pred in predictions)))
    μt = Pt * sum(pred.Pinv * pred.μ for pred in predictions)
    parent.data.state = MvNormal(μt, Pt)
    parent.data.filtered = true
end

function ready_to_merge(parent)
    return all(isfiltered.(children(parent)))
end

function filter_upscale!(r, tree)
    children = [leaf for leaf in allleaves(tree) if isfiltered(leaf)]
    parents = setdiff(collect(allcells(tree)), children)
    while length(parents) > 0
        ii = ready_to_merge.(parents)
        cells_to_merge = parents[ii]
        for cell in cells_to_merge
            merge_child_states!(r, cell)
        end
        parents = parents[.! ii]
    end
    tree.data.smoothed = true
end

function downscale(r, parent, child)
    # This is assuming that the prediction from the child up-scale to 
    # its parent is the "filtered" estimate, which gets "updated" 
    # when the other children are merged. Need to confirm that's right
    a = area(child.boundary)
    A = r.A(a)
    B = r.B(a)
    P_parent = cov(prior(parent))
    P_child = cov(prior(child))
    P_child_inv = inv(P_child)
    F = P_parent * A' * P_child_inv
    pred_up = predict_upscale(r, parent, child)
    J = cov(child) * F' * pred_up.Pinv
    μ_smoothed = mean(child) + J * (mean(parent) - pred_up.μ)
    P_smoothed = cov(child) + J * (cov(parent) - pred_up.P)
    return MvNormal(μ_smoothed, P_smoothed)
end

function smooth_downscale!(r, tree, x_parent_filtered)
    if children(tree) == nothing
        return
    else
        for child in children(tree)
            x_child = deepcopy(child.data.state)
            x_parent = tree.data.state
            child.data.state = downscale(r, tree, child)
            child.data.smoothed = true
            smooth_downscale!(r, child, x_child)
        end
    end
end

# default method assuming whole tree has been filtered and we're starting at the root node
smooth_downscale!(r, tree) = smooth_downscale!(r, tree, tree.data.state)

function multiresolution_smooth!(r, tree)
    observe_data!(r, tree)
    filter_upscale!(r, tree)
    smooth_downscale!(r, tree)
end

function stat_array(tree, stat, i)
    leaves = collect(allleaves(tree))
    nleaves = length(leaves)
    d = length(first(leaves).boundary.origin)
    nside = Int(nleaves^(1/d))
    Z = zeros(fill(nside, d)...)
    for j in 1:nleaves
        ic = morton2cartesian(j)
        Z[ic...] = stat(leaves[j])[i]
    end
    return reverse(Z, dims=2)
end

mean_array(tree, i) = stat_array(tree, mean, i)
cov_array(tree, i) = stat_array(tree, cov, i)
value_array(tree, i) = stat_array(tree, value, i)


function simulate_prior!(r, cell, parent_value)
    cell.data.value .= parent_value .+ rand(prior(cell))
    if children(cell) != nothing
        for child in children(cell)
            simulate_prior!(r, child, cell.data.value)
        end
    end
end

function simulate_prior!(r, tree)
    tree.data.value .= mean(prior(tree))
    simulate_prior!(r, tree, tree.data.value)    
end

function simulate_posterior!(r, tree)
    for leaf in allleaves(tree)
        leaf.data.value .= rand(state(leaf))
    end
end