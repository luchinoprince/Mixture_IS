data {
    int <lower=0> n;
    int <lower=0> k;
    int <lower=0, upper=1> y[n];
    matrix [n,k] X;
    real <lower=0> prior_scale;
    
}

parameters {
    vector[k] beta;
}

model{
    vector[n] means=X*beta;
    target += double_exponential_lpdf(beta | 0, prior_scale);
    target += bernoulli_logit_lpmf(y | means);    
}

generated quantities {
  vector[n] means=X*beta;
  vector[n] log_lik;
  for (nn in 1:n)
    log_lik[nn] = bernoulli_logit_lpmf(y[nn] | means[nn]);
    
}

