import pandas as pd
import numpy as np
from scipy.stats import laplace,bernoulli
from scipy.special import logsumexp

## this is not a conjugate model remember
class Model:
    def __init__(self,µ, b):
        '''µ and b are goint to be the prior's hyperparameters that specify our logistic model.'''
        self.µ=µ
        self.b=b
        return
    
    
    def prior(self, beta):
        '''This function computes the pdf of the prior at a given value beta'''
        partial=1
        for bj in beta:
            partial=laplace.pdf(bj,loc=self.µ,scale=self.b)*partial
        return partial


    
    def lmodel2_i(self, beta, y, X, i):
        '''This function computes the logarithm of the probability mass function(pmf) at the i-th point for a given value of beta'''
        ps=(1/(1+np.exp(-1*np.dot(X[i,::],beta))))
        py=bernoulli.logpmf(y[i],ps)
        return py

    def lmodel_i(self, beta, y, X, i):
        '''This function computes the logarithm of the probability mass function(pmf) at the i-th point for a given value of beta'''
        mu=np.dot(X[i,::], beta)
        zeros = np.zeros(mu.shape)
        return mu*y[i] - logsumexp(np.array([zeros, mu]), axis=0)
    
    def lmodel(self, beta, y, X):
        mu=np.dot(X, beta)
        zeros = np.zeros(mu.shape)
        return mu*y.reshape((len(y),1)) - logsumexp(np.array([zeros, mu]), axis=0)

    def model_i(self, beta, y, X, i):
        '''This function probability mass function(pmf) at the i-th point for a given value of beta'''
        ps=(1/(1+np.exp(-1*np.dot(X[i,::],beta))))
        return bernoulli.pmf(y[i],ps)



def posterior():
    model_posterior="""
    data
    {
        int n;
        int k;
        int<lower=0, upper=1> y[n];
        matrix [n,k] X;
        real scaled_var;
        
    }

    parameters
    {
        vector[k] beta;
    }


    model
    {
        beta ~ double_exponential(0,scaled_var);
        y ~ bernoulli_logit(X*beta);
        
    }
    """
    return model_posterior

def mixture():
    model_mixture="""
    functions
    {
        real mixture_lpmf(int[] y, matrix X, int n, int k, vector beta, real scaled_var)
        {
        real log_full_model= 0.0;
        real log_prior = 0.0;
        vector[n] contributions;
        vector[n] means;
        means = X*beta;
        
        for (index in 1:n)
        {
            contributions[index]= -1*(bernoulli_logit_lpmf(y[index] | means[index]));
        }

        log_full_model = bernoulli_logit_lpmf(y | means);
        log_prior = double_exponential_lpdf(beta | 0, scaled_var);
        return (log_sum_exp(contributions) + log_prior + log_full_model);
        }
    }

    data
    {
        int n;
        int k;
        int<lower=0, upper=1> y[n];
        matrix [n,k] X;
        real scaled_var;
        
    }

    parameters
    {
        vector[k] beta;
    }

    model
    {
        target += mixture_lpmf(y|X,n,k,beta,scaled_var);
    }

    """
    return model_mixture
    





