## We want to try to implement in the more efficient way suggested by CAICAI. Everything that it is NOT model specific will be 
## putted in a super-class (parent-class) with respect to the models themselves. While everything which is model specific will be made a method of the model themselves

import pandas as pd
import numpy as np
from scipy.stats import laplace, bernoulli, multivariate_normal
from scipy.special import logsumexp


class Is_estimators:
    def __init__(self, model):
        self.model=model

    def mixture(self, samples_mix):
        log_lik = self.model.lmodel(samples_mix).transpose()
        lcommon_mix = logsumexp(-log_lik, axis=1)
        log_weights = -log_lik.transpose() - lcommon_mix.reshape((1,len(lcommon_mix)))
        lppd_mix = logsumexp(-lcommon_mix) - logsumexp(log_weights, axis=1)
        return lppd_mix
    
    def posterior(self, samples_post):
        log_lik = self.model.lmodel(samples_post).transpose()
        n_samples = samples_post.shape[1]
        lppd_post = np.log(n_samples) - logsumexp(-log_lik, axis=0)
        return lppd_post

    def PSIS(self, samples_post):
        log_lik = self.model.lmodel(samples_post).transpose()
        loo, lppd_psis, k_psis=psis.psisloo(log_lik)
        return lppd_psis


## Now i create the model class, this will encompass both the logistic and gaussian model
class Model:
    def __init__(self, y, X):
        ##theser are common to all models....
        self.y = y
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
 

class Gaussian_Model(Model):
    def __init__(self, y, X, theta_0, sigma_0, var):
        super().__init__(y, X)
        self.theta_0 = theta_0        ## prior mean
        self.sigma = sigma_0          ## prior covariance
        self.var = var                ## Gaussian model variance


    def model_i(self, theta, i):
        """ This function calculates the pdf of the Gaussian model for a given parameter values theta
        and at a given observation y[i]"""
        mean = np.dot(self.X[i,::], theta).reshape(1,)
        return multivariate_normal(mean = mean, cov = self.var).pdf(self.y[i])

    def model(self,theta):
        """ This function calculates the pdf of the Gaussian model for a given parameter values theta
        and at a given observation y[i]"""
        mean=np.dot(self.X, theta)
        return multivariate_normal(mean = mean.flatten(),cov = self.var*np.eye((self.n))).pdf(self.y)

    def lmodel_i(self, theta, i):
        """This function calculation the log-density of the observation y[i] for the model with parameter
        values equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta. """
        return -0.5*np.log(2*np.pi*self.var)-0.5/self.var*((self.y[i]-np.dot(self.X[i,::], theta))**2)
    

    def lmodel(self, theta):
        """This function calculates the log-density of the full sample for the model with parameter values 
        equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta."""
        return -1/2*np.log(2*np.pi*self.var)-0.5/self.var*((self.y.reshape((self.n,1))-np.dot(self.X, theta))**2)

    def cond_theta(self):
        """This function is part of the Gibbs step, samples theta from the full posterior"""
        variance = np.matrix(1/self.var*np.dot(self.X.transpose(), self.X)+np.matrix(self.sigma).I).I
        first=(np.matrix(1/self.var * np.dot(self.X.transpose(),self.X))+np.matrix(self.sigma).I).I
        second=np.matrix((1/self.var)*np.dot(self.X.transpose(),self.y).reshape((self.p,1))+np.dot(np.matrix(self.sigma).I, self.theta_0))
        mean=np.dot(first, second)
        x=np.array([mean[j,0] for j in range(mean.size)])
        mean=x
        
        return multivariate_normal(mean=mean, cov=variance)

    def prior(self, theta):
        pp=multivariate_normal(mean=self.theta_0.reshape(self.p), cov=self.sigma).pdf(theta)
        return pp

    def log_predictive_i(self, index):
        '''This function computes the leave-one-out log-predictive probability for the index-th observation. It is implemented to work also in very high-dimensional
        setting.'''
        X_index=np.concatenate([self.X[0:index], self.X[index+1:]])
        y_index=np.concatenate([self.y[0:index], self.y[index+1:]])
        model_i = Gaussian_Model(y_index, X_index, self.theta_0, self.sigma, self.var)
        aux=np.matrix(model_i.cond_theta().mean.reshape((self.p,1)))
        mu1=self.X[index:index+1,::]*aux
        sigma1=(self.X[index:index+1,::])*(np.matrix(model_i.cond_theta().cov))*(self.X[index:index+1,::].transpose())
        coefficient = np.log(1/(np.sqrt(2*np.pi*(sigma1 + self.var))))
        exponent = (-0.5/(sigma1+self.var)) * (self.y[index]-mu1)**2
                
        return coefficient[0,0]+exponent[0,0]

    def log_p_loo_i(self, index):
        '''This function computes the leave-one-out marginal log-probability for the full sample other than index-th observation. It is implemented to work also in very high-dimensional setting.'''
        coefficient = 0; exponent = 0;
        X_index=np.concatenate([self.X[0:index], self.X[index+1:]])
        y_index=np.concatenate([self.y[0:index], self.y[index+1:]])
        #model_i = Gaussian_Model(y_index, X_index, self.theta_0, self.sigma, self.var)
        for j in range(0, self.n-1):
            y1=y_index[j+1:]
            X1=X_index[j+1:,::]
            model_res = Gaussian_Model(y1, X1, self.theta_0, self.sigma, self.var)
            if len(y1)>0:
                aux=np.matrix(model_res.cond_theta().mean.reshape((self.p,1)))
                mu1=X_index[j:j+1,::]*aux
                sigma1=(X_index[j:j+1,::])*(np.matrix(model_res.cond_theta().cov))*(X_index[j:j+1,::].transpose())
                coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma1+self.var))))
                exponent += (-0.5/(sigma1+self.var)) * (y_index[j]-mu1)**2
            else:
                aux=X_index[j:j+1,::]*np.matrix(self.theta_0)
                sigma=(X_index[j:j+1,::])*(np.matrix(self.sigma))*(X_index[j:j+1,::].transpose()) + self.var
                coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma))))
                exponent += (-0.5/(sigma)) * (y_index[j]-aux)**2

        return (coefficient+exponent)[0,0]


    def log_p_full(self):
        """This function computes the marginal log-probability for the full sample. It is designed to work also in high-D setting"""
        coefficient = 0; exponent = 0;
        for j in range(0,self.n):
            y1=self.y[j+1:]
            X1=self.X[j+1:,::]
            if len(y1)>0:
                model_res = Gaussian_Model(y1, X1, self.theta_0, self.sigma, self.var)
                aux=np.matrix( (model_res.cond_theta().mean.reshape((self.p,1))) )
                mu1=self.X[j:j+1,::]*aux
                sigma1=(self.X[j:j+1,::])*(np.matrix(model_res.cond_theta().cov))*(self.X[j:j+1,::].transpose())
                coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma1 + self.var))))
                exponent = exponent + (-0.5/(sigma1+self.var))*(self.y[j]-mu1)**2
            else:
                aux=self.X[j:j+1,::]*np.matrix(self.theta_0)
                sigma=(self.X[j:j+1,::])*(np.matrix(self.sigma))*(self.X[j:j+1,::].transpose()) + self.var
                coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma))))
                exponent = exponent + (-0.5/(sigma))*(self.y[j]-aux)**2
                
        return coefficient[0,0]+exponent[0,0]


class Logistic_Model(Model):
    def __init__(self, y, X, mu, b):
        super().__init__(y, X)
        self.mu = mu                  
        self.b = b              ## location and scale of the double Exponential prior distribution

    def prior(self, beta):
        '''This function computes the pdf of the prior at a given value beta'''
        partial=1
        for bj in b:
            partial=laplace.pdf(bj,loc=self.mu,scale=self.b)*partial
        return partial

    def model_i(self, beta, i):
        np.exp(self.lmodel_i(self, beta, i))

    def model(beta):
        np.exp(self.lmodel(self, beta))


    def lmodel_i(self, beta, i):
        '''This function computes the logarithm of the probability mass function(pmf) at the i-th point for a given value of beta'''
        mu=np.dot(self.X[i,::], beta)
        zeros = np.zeros(mu.shape)
        return mu*self.y[i] - logsumexp(np.array([zeros, mu]), axis=0)

    def lmodel(self, beta):
        mu=np.dot(self.X, beta)
        zeros = np.zeros(mu.shape)
        return mu*self.y.reshape((self.n,1)) - logsumexp(np.array([zeros, mu]), axis=0)




## These are just the strings to create the model block of PyStan
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