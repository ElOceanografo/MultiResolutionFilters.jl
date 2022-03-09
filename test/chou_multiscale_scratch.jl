using Distributions
using LinearAlgebra
using Optim
using ForwardDiff
using Random
using Plots

Random.seed!(1)
n = 5
xtα = MvNormal(randn(n), diagm((0.5 .+ rand(n)).^2))
xtβ = MvNormal(randn(n), diagm((0.5 .+ rand(n)).^2))

B = diagm(rand(n))
A = I(n)

Ftα = inv(A) * (I - B * B' * inv(cov(xtα)))
xt_pred_α = Ftα * mean(xtα)
Qtα = I - B' * cov(xtα) * B
𝒬tα = inv(A) * B * Qtα * B' * inv(A)
Pt_pred_α = Ftα * cov(xtα) * Ftα' + 𝒬tα

Ftβ = inv(A) * (I - B * B' * inv(cov(xtβ)))
xt_pred_β = Ftβ * mean(xtβ)
Qtβ = I - B' * cov(xtβ) * B
𝒬tβ = inv(A) * B * Qtβ * B' * inv(A)
Pt_pred_β = Ftβ * cov(xtβ) * Ftβ' + 𝒬tβ

Pt_prior = diagm(fill(10.0, n))

Pt_pred = inv(inv(Pt_pred_α) + inv(Pt_pred_β) + inv(Pt_prior))
xt_pred = Pt_pred * (inv(Pt_pred_α) * xt_pred_α + inv(Pt_pred_β) * xt_pred_β)

plot(xt_pred_α, ribbon=diag(Pt_pred_α), label="xt_pred_α")
plot!(xt_pred_β, ribbon=diag(Pt_pred_β), label="xt_pred_β")
plot!(xt_pred, ribbon=diag(Pt_pred), label="Fusion")


function loglik(x, y, x_pred, P_pred)
    ll = logpdf(MvNormal(x_pred, P_pred), x)
    return ll + sum(logpdf.(Poisson.(exp.(x)), y))
end

yt = rand.(Poisson.(exp.(xt_pred .+ 0.5randn(n))))

loglik(randn(n), yt, xt_pred, Pt_pred)
f = x -> -loglik(x, yt, xt_pred, Pt_pred)
opt = optimize(f, xt_pred)
xt_filter = opt.minimizer
H = ForwardDiff.hessian(f, xt_filter)
Pt_filter = inv(H)
plot!(xt_filter, ribbon=diag(Pt_filter), label="Filtered")

Pt_smooth = Pt_filter # assume this is at top of the tree

Jtα = cov(xtα) * Ftα' * inv(Pt_smooth)
xtα_smooth = mean(xtα) + Jtα * xt_filter
Ptα_smooth = cov(xtα) + Jtα * (Pt_smooth - Pt_filter)*Jtα'
plot(mean(xtα), ribbon=diag(cov(xtα)), label="Filtered")
plot!(xtα_smooth, ribbon=diag(Ptα_smooth), label="Smoothed")

Jtβ = cov(xtβ) * Ftβ' * inv(Pt_smooth)
xtβ_smooth = mean(xtβ) + Jtβ * xt_filter
Ptβ_smooth = cov(xtβ) + Jtβ * (Pt_smooth - Pt_filter)*Jtβ'
plot(mean(xtβ), ribbon=diag(cov(xtβ)), label="Filtered")
plot!(xtβ_smooth, ribbon=diag(Ptβ_smooth), label="Smoothed")
