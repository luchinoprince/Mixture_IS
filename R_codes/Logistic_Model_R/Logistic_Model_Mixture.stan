data {
    int n;
    int k;
    int<lower=0, upper=1> y[n];
    matrix [n,k] X;
    real prior_scale;
    
}

parameters {
    vector[k] beta;
}

model{
    vector[n] means=X*beta;
    vector[n] log_lik;
    for (index in 1:n){
        log_lik[index]= bernoulli_logit_lpmf(y[index] | means[index]);
    }
    target += double_exponential_lpdf(beta | 0, prior_scale);
    target += sum(log_lik);
    target += log_sum_exp(-log_lik);
}

generated quantities {
  vector[n] means=X*beta;
  vector[n] log_lik;
  for (nn in 1:n)
    log_lik[nn] = bernoulli_logit_lpmf(y[nn] | means[nn]);
    
}


