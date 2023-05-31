In this tutorial, we wil do a step-by-step implementation of both the classical and our novel estimation procedures on the Logistic model, but clear reference will be made to how this can be extended to general bayesian models. We suppose that in the notebook one as imported the following libraries:
```
import pystan
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
```
From here on we will indicate with **X** the __[n, p]__ matrix of covariates, with **beta** the __[p,1]__ column vector of parameters, with **Y** the __[n, 1]__ columns vector of response variables and with **scaled_var** the prior's isotropic variance. 
Firstly, we will show how to generate the forward model block in the PyStan model specification, and hence how to obtain the samples from the posterior, which can be used to implement the classical estimator. 

This implementation is meant to be very pedagogical, as such we will define every quantity separetly for clarity. It might be that in practical applications, especially in high-dimensional frameworks, such an implementation will give numerical overflow of some of the quantities involved and is also computationally inefficient. For a numerical stable and efficient, but less clear, implementation refer to the other two notebooks and the libraries imported there.

To implement the forward model one can simply write:
```
model
{
    vector[n] means=X*beta;
    target += double_exponential_lpdf(beta | 0, scaled_var);
    target += bernoulli_logit_lpmf(y | means);

}
```
Once the model specification in **PyStan** is completed with the other model blocks, one can create the model and obtain the samples, which we suppose are saved in the vector _betas_posterior_. Now we can generate the classical estimator by running the following code in **Python**:
```
lik_post = np.array([bernoulli.pmf(y[i], 1/(1+np.exp(-1*np.dot(X[i,::],betas_posterior)))) for i in range(n)]);
weights_post = 1/lik_post;
lppd_post = np.log(np.sum(lik_post * weights_post, axis=1)/np.sum(weights_post, axis=1))
```

Secondly we have to have to understand how to write our mixture in PyStan.
We know that, for a general bayesian model, we can write our mixture proposal as: <img src="https://render.githubusercontent.com/render/math?math=q_{mix}(\theta)\propto p(y|\theta)p(\theta)(\sum_{i=0}^n p(y_i|\theta)^{-1})">. As we know, in order to sample from a distribution in [**PyStan**](https://mc-stan.org/docs/2_25/stan-users-guide/user-defined-probability-functions.html) it suffices to know it up to a normalising constant. We hence just have to encode the above distriubition in the **_model_** code block trough specific incrementations of the __target__ variable, and we are done. As for our example case, this means:
```
model
{
    vector[n] means=X*beta;
    vector[n] contributions;
    for (index in 1:n)
    {
        contributions[index]= -1*(bernoulli_logit_lpmf(y[index] | means[index]));
    }

    target += double_exponential_lpdf(beta | 0, scaled_var);
    target += bernoulli_logit_lpmf(y | means);
    target += log_sum_exp(contributions);
}
```
**NB**: These lines of code have to be inserted in a classicial model specification of **PyStan**, they are not to be run in a Notebook cell. We do not do so for brevity of the tutorial. For a full model specification refer again to the library __models.py__.

Compiling the model and running it will give us the samples from our desired mixture distribution. Notice that the only difference with the classical model is that we have to generate the auxiliary vector _contributions_ and then increase the _target_ variable by the _log_sum_exp_ of such a vector. Notice also that if one here has a different prior/model specification, one would just have to substitute the __double_exponential_lpdf__ and __bernoulli_logit_lpmf__ calls with the desired ones.


At this point, once we have these samples, we have just to produce the estimates.
Suppose we have that these samples are contained in the vector `betas_mixture`, then we have just to compute:
```
## n is the number of samples of your dataset
lik_mix = np.array([bernoulli.pmf(y[i], 1/(1+np.exp(-1*np.dot(X[i,::],betas_mixture)))) for i in range(n)])
common_mix = 1/np.sum(1/lik_mix, axis=0)

##Here we leverage broadcasting
weights = 1/lik_mix * common_mix
lppd_mixture = np.log(np.sum(lik_mix * weights, axis=1)/np.sum(weights,axis=1))
```
Notice again that if one was interested in computing the estimator for a different model, one would just have to substitute the __bernoulli.pmf__ call with the desidered one.

***IMPLEMENTATION HINTS***: As we said at the beginning of the tutorial, this tutorial is meant to be very pedagogical, and as such it defines all the quantities of of interest separatly for clarity. This can cause one to incur in numerical overflow in some settings, especially high-dimensional ones. To solve them one might want to pass directly to the logarithms of some of the quantities involved and then leverage the __logsumexp__ function from __scipy__. Etherwise one can stick with the original quantities, but it has to carefully manage the exponents of the quantitites involved and perform the proper gropuping/simplifications by hand. To see a way to do so one can have a look at the classes implemented in the library **models.py**.




***EFFICIENCY TIPS***: To increse the sampling speed of the mixture we noticed that it is actually better to compute the _log-pdf_ of the mixture inside a used defined function in the __function__ code block, and then just increase the _target_ variable by that quantity, hence:
```
functions
{
    real mixture_lpmf(int[] y, matrix X, int n, int k, vector beta, real scaled_var)
    {
    vector[n] contributions;
    vector[n] means;
    means = X*beta;

    for (index in 1:n)
    {
        contributions[index]= -1*(bernoulli_logit_lpmf(y[index] | means[index]));
    }
    return bernoulli_logit_lpmf(y | means) + double_exponential_lpdf(beta | 0, scaled_var) + log_sum_exp(contributions);
    }
}
```
```
model
{
    target += mixture_lpmf(y|X,n,k,beta,scaled_var);
}
```



