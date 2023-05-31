using LinearAlgebra
using Random, Distributions, LogExpFunctions, ParetoSmooth, ProgressMeter, PyCall
include("gm.jl")

attempts=10000
ns = [50, 100, 150]
ps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3]
couples = []
for n in ns
    for p in ps
        push!(couples, (Int(n), Int(n*p)))
    end
end

#BLAS.set_num_threads(1);
#nt_blas = BLAS.get_num_threads();
#println("I am using $nt_blas threads for BLAS\n")
@static if VERSION < v"1.7.0-DEV.610"
    ## function borrowed from stdlib/Distributed test file
    function get_num_blas_threads()::Int
        blas = LinearAlgebra.BLAS.vendor()
        # Wrap in a try to catch unsupported blas versions
        try
            if blas == :openblas
                return ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())
            elseif blas == :openblas64
                return ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
            elseif blas == :mkl
                return ccall((:MKL_Get_Max_Num_Threads, Base.libblas_name), Cint, ())
            end

            # OSX BLAS looks at an environment variable
            if Sys.isapple()
                return tryparse(Cint, get(ENV, "VECLIB_MAXIMUM_THREADS", "1"))
            end
        catch
        end

        return Int(get(ENV, "OMP_NUM_THREADS", Sys.CPU_THREADS))
    end
else
    get_num_blas_threads() = BLAS.get_num_threads()
end

# create a single-threaded-BLAS scope with do..end
function disable_blas_threads(f)
    old_num_threads = get_num_blas_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1)
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(old_num_threads)
    end
end


estimates_mix_nd = Dict()
estimates_post_nd = Dict()
estimates_bronze_nd = Dict()
estimates_psis_nd = Dict()
correct_values_nd = Dict()
nt =Threads.nthreads()
#println("n is $n")
println("I am using $nt threads\n")
for couple in couples
    n=couple[1]
    d=couple[2]
    #print("n$n\n")
    estimators_mix = zeros(attempts, n)
    estimators_post = zeros(attempts, n)
    estimators_bronze = zeros(attempts, n)
    estimators_psis = zeros(attempts, n)
    correct_values = zeros(attempts, n)
    disable_blas_threads() do
        if (n>=100 && d>=n)
            ## Here theading actually speeds things up
            @Threads.threads for attempt=1:attempts
            #@inbounds for attempt=1:attempts
            #@showprogress 1 "Processing n$n, d$d" for attempt in my_range
                #print("Processing n:$n, d:$d at attempt $attempt \r")
                println("Processing n:$n, d:$d at attempt $attempt out of $attempts, multithreading\n")
                sigma = 100/(d+1) * Diagonal(1*ones(d+1))
                varl = 1.0
                X1 = ones(n,1)
                X = rand(n,d)
                X = hcat(X1, X)

                theta=rand(MultivariateNormal(zeros(d+1), sigma))
                mean_t = X*theta
                y = rand(MultivariateNormal(mean_t, Diagonal(varl*ones(n))));
                full = probs(sigma,varl, y, X)
                correct_value = [predictive_i(sigma, varl, y, X, k) for k=1:n]
                #leave_one_outs_p = [ full-predictive_i(sigma, varl, y, X,index) for index=1:n];
                leave_one_outs_p = full .- correct_value
                minim = minimum(leave_one_outs_p)
                normalizer = sum(ℯ.^(leave_one_outs_p .- minim))
                weights = ℯ.^(leave_one_outs_p .- minim)/normalizer

                iterations=2000
                res=rand(Multinomial(iterations, weights));
                thetas_mixture = zeros((size(X)[2],iterations))
                index=1
                @inbounds for j=1:n
                    @views X_index =  vcat(X[1:j-1, :], X[j+1:end, :])
                    @views y_index =  vcat(y[1:j-1], y[j+1:end])
                    #X_index = view(X, [i != j for i in 1:n], :)
                    #y_index = view(y, [i!=j for i in 1:n], :)
                    loo = cond_theta(sigma, varl, y_index, X_index)
                    if res[j] > 0
                        if res[j] == 1
                            thetas_mixture[begin:end, index]=rand(loo, res[j])
                            index+=res[j]
                        else
                            thetas_mixture[begin:end, index:index+res[j]-1]=rand(loo, res[j])
                            index+=res[j]
                        end
                    end
                end

                posterior=cond_theta(sigma, varl, y, X)
                thetas_posterior=rand(posterior, iterations);

                l_lik =  -1/(2*varl).*(y .- X*thetas_posterior).^2 .- 0.5*log(2*π*varl)


                ## COMMON PART FOR ALL I OF THE EQUAL-WEIGHT AND MIXTURE ESTIMATES
                means= X*thetas_mixture
                exponents_num_mix = 0.5 .* (y .- means).^2
                common_mixture=logsumexp(exponents_num_mix,dims=1)

                numerator_mix = logsumexp(-1*common_mixture)

                #################################################################################################

                #estimates_posterior = iterations./sum(w_posterior, dims=2)/sqrt(2*π)
                estimate_posterior = log(iterations) .- logsumexp(-l_lik, dims=2) 


                ## MIXTURE ESTIMATES
                denominator_exponents_mix = exponents_num_mix' .- common_mixture'
                denominator_mixture = logsumexp(denominator_exponents_mix,dims=1)

                lw = ones(n, iterations, 1)
                lw[begin:end, begin:end, :] = l_lik
                estimate_mix_nd = numerator_mix .- denominator_mixture .- 0.5*log(2π)

                res = ParetoSmooth.psis_loo(lw, source="other");
                posterior_bronze = cond_theta(sigma, (n-1)/n*varl, y, X)
                thetas_bronze = rand(posterior_bronze, iterations)
                l_lik_bronze = -1*(1/(2*varl)*(y .- X*thetas_bronze).^2 .+ 0.5*log(2*π*varl))
                common_bronze = 1/n .* sum(l_lik_bronze, dims=1)
                l_ws_bronze = common_bronze .- l_lik_bronze
                lppd_bronze = logsumexp(common_bronze) .- logsumexp(l_ws_bronze,dims=2)


                estimators_mix[attempt, begin:end] = estimate_mix_nd
                estimators_post[attempt, begin:end] = estimate_posterior
                estimators_bronze[attempt, begin:end] = lppd_bronze
                estimators_psis[attempt, begin:end] = res.pointwise[begin:end, 1]
                correct_values[attempt, begin:end] = correct_value
                if attempt == attempts
                    key = "(n:$n, d:$d)"
                    estimates_mix_nd[key]=estimators_mix
                    estimates_post_nd[key]=estimators_post
                    estimates_bronze_nd[key]=estimators_bronze
                    estimates_psis_nd[key]=estimators_psis
                    correct_values_nd[key]= correct_values
                end
            end
        else
            @inbounds for attempt=1:attempts
            #@inbounds for attempt=1:attempts
            #@showprogress 1 "Processing n$n, d$d" for attempt in my_range
                #print("Processing n:$n, d:$d at attempt $attempt \r")
                println("Processing n:$n, d:$d at attempt $attempt out of $attempts, no multithread\n")
                sigma = 100/(d+1) * Diagonal(1*ones(d+1))
                varl = 1.0
                X1 = ones(n,1)
                X = rand(n,d)
                X = hcat(X1, X)

                theta=rand(MultivariateNormal(zeros(d+1), sigma))
                mean_t = X*theta
                y = rand(MultivariateNormal(mean_t, Diagonal(varl*ones(n))));
                full = probs(sigma,varl, y, X)
                correct_value = [predictive_i(sigma, varl, y, X, k) for k=1:n]
                #leave_one_outs_p = [ full-predictive_i(sigma, varl, y, X,index) for index=1:n];
                leave_one_outs_p = full .- correct_value
                minim = minimum(leave_one_outs_p)
                normalizer = sum(ℯ.^(leave_one_outs_p .- minim))
                weights = ℯ.^(leave_one_outs_p .- minim)/normalizer

                iterations=2000
                res=rand(Multinomial(iterations, weights));
                thetas_mixture = zeros((size(X)[2],iterations))
                index=1
                @inbounds for j=1:n
                    @views X_index =  vcat(X[1:j-1, :], X[j+1:end, :])
                    @views y_index =  vcat(y[1:j-1], y[j+1:end])
                    #X_index = view(X, [i != j for i in 1:n], :)
                    #y_index = view(y, [i!=j for i in 1:n], :)
                    loo = cond_theta(sigma, varl, y_index, X_index)
                    if res[j] > 0
                        if res[j] == 1
                            thetas_mixture[begin:end, index]=rand(loo, res[j])
                            index+=res[j]
                        else
                            thetas_mixture[begin:end, index:index+res[j]-1]=rand(loo, res[j])
                            index+=res[j]
                        end
                    end
                end

                posterior=cond_theta(sigma, varl, y, X)
                thetas_posterior=rand(posterior, iterations);

                l_lik =  -1/(2*varl).*(y .- X*thetas_posterior).^2 .- 0.5*log(2*π*varl)


                ## COMMON PART FOR ALL I OF THE EQUAL-WEIGHT AND MIXTURE ESTIMATES
                means= X*thetas_mixture
                exponents_num_mix = 0.5 .* (y .- means).^2
                common_mixture=logsumexp(exponents_num_mix,dims=1)

                numerator_mix = logsumexp(-1*common_mixture)

                #################################################################################################

                #estimates_posterior = iterations./sum(w_posterior, dims=2)/sqrt(2*π)
                estimate_posterior = log(iterations) .- logsumexp(-l_lik, dims=2) 


                ## MIXTURE ESTIMATES
                denominator_exponents_mix = exponents_num_mix' .- common_mixture'
                denominator_mixture = logsumexp(denominator_exponents_mix,dims=1)

                lw = ones(n, iterations, 1)
                lw[begin:end, begin:end, :] = l_lik
                estimate_mix_nd = numerator_mix .- denominator_mixture .- 0.5*log(2π)

                res = ParetoSmooth.psis_loo(lw, source="other");
                posterior_bronze = cond_theta(sigma, (n-1)/n*varl, y, X)
                thetas_bronze = rand(posterior_bronze, iterations)
                l_lik_bronze = -1*(1/(2*varl)*(y .- X*thetas_bronze).^2 .+ 0.5*log(2*π*varl))
                common_bronze = 1/n .* sum(l_lik_bronze, dims=1)
                l_ws_bronze = common_bronze .- l_lik_bronze
                lppd_bronze = logsumexp(common_bronze) .- logsumexp(l_ws_bronze,dims=2)


                estimators_mix[attempt, begin:end] = estimate_mix_nd
                estimators_post[attempt, begin:end] = estimate_posterior
                estimators_bronze[attempt, begin:end] = lppd_bronze
                estimators_psis[attempt, begin:end] = res.pointwise[begin:end, 1]
                correct_values[attempt, begin:end] = correct_value
                if attempt == attempts
                    key = "(n:$n, d:$d)"
                    estimates_mix_nd[key]=estimators_mix
                    estimates_post_nd[key]=estimators_post
                    estimates_bronze_nd[key]=estimators_bronze
                    estimates_psis_nd[key]=estimators_psis
                    correct_values_nd[key]= correct_values
                end
            end
        end
    end
end

println("Now saving results")
py"""
import pickle
"""

bytes_obj_psis = py"pickle.dumps"([estimates_psis_nd], protocol=2)
bytes_obj_mix = py"pickle.dumps"([estimates_mix_nd], protocol=2)
bytes_obj_post = py"pickle.dumps"([estimates_post_nd], protocol=2)
bytes_obj_bronze = py"pickle.dumps"([estimates_bronze_nd], protocol=2)
bytes_obj_cv = py"pickle.dumps"([correct_values_nd], protocol=2)


open("estimates_psis_hyperD_nd_2k_rand_10000att_p.pkl", "w") do f
    write(f, bytes_obj_psis)
end

open("estimates_mixture_hyperD_nd_2k_rand_10000att_p.pkl", "w") do f
    write(f, bytes_obj_mix)
end

open("estimates_posterior_hyperD_nd_2k_rand_10000att_p.pkl", "w") do f
    write(f, bytes_obj_post)
end

open("estimates_bronze_hyperD_nd_2k_rand_10000att_p.pkl", "w") do f
    write(f, bytes_obj_bronze)
end

open("correct_values_hyperD_nd_2k_rand_10000att_p.pkl", "w") do f
    write(f, bytes_obj_cv)
end
