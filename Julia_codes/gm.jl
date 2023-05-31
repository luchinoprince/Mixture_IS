using LinearAlgebra
using Random, Distributions


function lmodel_i(varl, theta, X, y, i)
    μ = X[i:i+1, begin:end]*theta
    d = Normal(μ, varl)
    return logpdf(μ, y[i])
end

function cond_theta(sigma::Diagonal{Float64, Vector{Float64}}, varl, y, X)
    ## We suppose y to be a vector [n,1]
    Σ = Hermitian(inv(1/varl.* X'*X + inv(sigma)));  ## We round off numerical errors
    aux = 1/varl.* X'*y 
    μ = Σ * aux
    #print(size(Σ))
    #return Σ
    return MultivariateNormal(μ, Σ)
end
## We do even much faster by computing all the predictives together, would need to work out the algebra of it.

function predictive_i(sigma::Diagonal{Float64, Vector{Float64}}, varl, y, X, index)
    ## This function computes p(y_i|y_{-i}), actually it returns the base and the exponent to get them so we don't have overflow and underflow
    ## To get the pdf from the output you have to do exp(coeff)^exponent
    X_index = vcat(X[1:index-1, :], X[index+1:end, :])
    y_index = vcat(y[1:index-1], y[index+1:end])
    cond_dist = cond_theta(sigma, varl, y_index, X_index)
    aux = cond_dist.μ
    aux = reshape(aux, (size(X, 2), 1))
    mu1 = X[index:index, :] * aux
    sigma1 = X[index:index, :] * cond_dist.Σ * X[index:index, :]'
    #print(sigma1)
    # ps = pdf(MvNormal(mu1, sigma1 + model.var), y[index])
    coefficient = log(1 / sqrt(2π * (sigma1[1,1] + varl)))
    exponent = (-0.5 / (sigma1[1,1] + varl)) .* (y[index] .- mu1).^2
    return coefficient[1,1]+exponent[1,1]
    #return [coefficient[1, 1], exponent[1, 1]]
end

function probs(sigma::Diagonal{Float64, Vector{Float64}}, varl, y, X)
    n,p=size(X)
    #println(n,p)
    ## This code works just for isotropic prior variances!
    cov = X*sigma*X' + Diagonal(varl.*ones(n))
    exp = -sum(y'*inv(cov)*y)/2
    coeff = -log((2π)^(n/2)*sqrt(det(cov)))
    #return [coeff, exp]
    return coeff+exp
end

