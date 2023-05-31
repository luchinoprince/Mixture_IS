library(rstan)
library(matrixStats)
rstan_options(auto_write = FALSE)
options(mc.cores = 1)


#### Setting random seed
nrep <- 1
seed <- 1
batch <- 1

SEED = seed + nrep*(batch - 1)
set.seed(SEED)

## Dataset selection
data = read.csv("./../../Data/Voice_preprocessed.csv")
y <- data$y
X <- data[2:length(data)]
## The datasets are saved such that the first colunns is always the response variable


n = dim(X)[1]
k = dim(X)[2]
prior_scale = sqrt(50/k)

### Parameters for PyStan
control = list(max_treedepth=10)
n_iter = 2000
## Sampling from the posterior
stanmodel_post = stan_model(file='Logistic_Model_Stan.stan')
standata = list(n = n, k = k, X = as.matrix(X), y = c(y), prior_scale = prior_scale)
fit_post <- sampling(stanmodel_post, data = standata, chains = 4, iter = n_iter, control = control, pars="log_lik")

## Create the classical estimator
log_lik_post = extract(fit_post)$log_lik
lppd_post = log(dim(log_lik_post)[1])-rowLogSumExps(-1*t(log_lik_post))


## Sampling from the mixture
stanmodel_mix = stan_model(file='Logistic_Model_Mixture.stan')
fit_mix <- sampling(stanmodel_mix, data = standata, chains = 4, iter = n_iter,control = control, pars="log_lik")

## Create mixture estimator
log_lik_mix = extract(fit_mix)$log_lik
l_common_mix = rowLogSumExps(-1*log_lik_mix)
log_weights = -1*log_lik_mix - l_common_mix
lppd_mix = logSumExp(-l_common_mix) - rowLogSumExps(t(log_weights))

## Compare results
plot(lppd_post,lppd_mix)
