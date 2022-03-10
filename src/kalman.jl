
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
    state
    filtered
    smoothed
end
KalmanCellData(ii_data, state) = KalmanCellData(ii_data,  state, false, false)

data_indices(cd::KalmanCellData) = cd.ii_data
has_observation(cell::Cell) = length(data_indices(cell.data)) >= 1
isfiltered(cell::Cell) = cell.data.filtered
issmoothed(cell::Cell) = cell.data.smoothed

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

function predict_upscale(r, μt, Pt)
    A = r.A
    B = r.B
    Ft = inv(A) * (I - B * B' * inv(Pt))
    μt_pred = Ft * μt
    Qt = I - B' * Pt * B
    𝒬t = inv(A) * B * Qt * B' * inv(A)
    Pt_pred = Ft * Pt * Ft' + 𝒬t
    return μt_pred, Pt_pred
end

function predict_upscale(r, leaf)
    μt = mean(leaf.data.state)
    Pt = cov(leaf.data.state)
    μt_pred, Pt_pred = predict_upscale(r, μt, Pt)
    return (μ = μt_pred, P = Pt_pred, Pinv = inv(Pt_pred))
end

function merge_child_states!(r, parent)
    predictions = [predict_upscale(r, child) for child in children(parent)]
    Pt = Symmetric(inv(sum(pred.Pinv for pred in predictions)))
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

function downscale(r, x_parent, x_parent_filtered, x_child)
    F = inv(r.A) * (I - r.B * r.B' * inv(cov(x_child)))
    J = cov(x_child) * F' * inv(cov(x_parent_filtered))
    μ_smoothed = mean(x_child) + J * (mean(x_parent) - mean(x_parent_filtered))
    P_smoothed = cov(x_child) + J * (cov(x_parent) - cov(x_parent_filtered)) * J'
    return MvNormal(μ_smoothed, P_smoothed)
end

function smooth_downscale!(r, tree, x_parent_filtered)
    if children(tree) == nothing
        return
    else
        for child in children(tree)
            x_child = deepcopy(child.data.state)
            x_parent = tree.data.state
            child.data.state = downscale(r, x_parent, x_parent_filtered, x_child)
            child.data.smoothed = true
            smooth_downscale!(r, child, x_child)
        end
    end
end

# default method assuming whole tree has been filtered and we're starting at the root node
smooth_downscale!(r, tree) = smooth_downscale!(r, tree, r.state_prior)#tree.data.state)

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
        Z[ic...] = stat(leaves[j].data.state)[i]
    end
    return reverse(Z, dims=2)
end

mean_array(tree, i) = stat_array(tree, mean, i)
cov_array(tree, i) = stat_array(tree, cov, i)