from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import laplace, bernoulli, multivariate_normal
from scipy.special import logsumexp
import sys
sys.path.insert(1, "./../PSIS/py/")      ##########Put here path to your PSIS folder ###########
import psis


class Is_estimators:
    def __init__(self, model):
        self.model=model

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.__model = value
        return 
        

    def mixture(self, samples_mix):
        """This function computes the estimates of the different lppd terms given by the mixture estimator.
        Care should be taken that the function supposes that samples_mix has dimensions equal to: [p, n_samples]"""
        log_lik = self.model.lmodel(samples_mix).transpose()
        lcommon_mix = logsumexp(-log_lik, axis=1)
        log_weights = -log_lik.transpose() - lcommon_mix.reshape((1,len(lcommon_mix)))
        lppd_mix = logsumexp(-lcommon_mix) - logsumexp(log_weights, axis=1)
        return lppd_mix
    
    def posterior(self, samples_post):
        """This function computes the estimates of the different lppd terms given by the posterior estimator.
        Care should be taken that the function supposes that samples_post has dimensions equal to: [p, n_samples]"""
        log_lik = self.model.lmodel(samples_post).transpose()
        n_samples = samples_post.shape[1]
        lppd_post = np.log(n_samples) - logsumexp(-log_lik, axis=0)
        return lppd_post

    def PSIS(self, samples_post):
        """This function computes the estimates of the different lppd terms given by the PSIS estimator.
        Care should be taken that the function supposes that samples_post has dimensions equal to: [p, n_samples]"""
        log_lik = self.model.lmodel(samples_post).transpose()
        loo, lppd_psis, k_psis=psis.psisloo(log_lik)
        return lppd_psis


## Now i create the model class, this will encompass both the logistic and gaussian model
class Model(ABC):
    def __init__(self, y, X):
        ##theser are common to all models....
        self.y = y
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]

    ## A decorator is a function that receives in input a function, and returns a function
    @abstractmethod
    def lmodel(self, theta):
        pass
    
    @abstractmethod
    def lmodel_i(self, theta, i):
        pass

    ##This parent module is more an interface, it will put every function that should be implemented in the submodule
    #def lmodel(self, theta):
    #    raise NotImplementedError("In order to implement any of the estimators, you have to implement this function in it's specific model-dependent subclass. It should return the [n,1] vector of log-probabilities at the points given by y. Possibly try  to implement them in a numerically-stable way to avoid overflow during the estimation")
    

class Gaussian_Model(Model):
    def __init__(self, y, X, theta_0, sigma_0, var):
        super().__init__(y, X)
        self.theta_0 = theta_0        ## prior mean
        self.sigma = sigma_0          ## prior covariance
        self.var = var                ## Gaussian model variance

    #def __str__(self):
    #    return """Bayesian regression model with known model variance.\n\t\t y[i]|X[i],\u03B8 ~ N(X[i]*\u03B8, \u03C3^2)
    #    \n\t\t\t \u03B8 ~ N(\u03B8_0, \u03A3_0),\n where \u03B8_0 is model.theta_0, \u03A3_0 is model.sigma_0 and is model.var."""

    @property
    def theta_0(self):
        return self.__theta_0

    @theta_0.setter
    def theta_0(self, value):
        self.__theta_0 = value
        return
    
    @property
    def sigma_0(self):
        return self.__sigma_0
    
    @sigma_0.setter
    def sigma_0(self, value):
        if (np.linalg.eigvals(value) <= 0).any():
            raise ValueError("The Gaussian's prior covariance matrix should be positive definite")
        else:
            self.__sigma_0=value

    @property
    def var(self):
        return self.__var
    
    @var.setter
    def var(self, value):
        if value <= 0:
            raise ValueError("The model's variance in the Gaussian model has to be positive")
        else:
            self.__var = value


    def model_i(self, theta, i):
        """ This function calculates the pdf of the Gaussian model for a given parameter value theta
        and at a given observation y[i]"""
        mean = np.dot(self.X[i,::], theta).reshape(1,)
        return multivariate_normal(mean = mean, cov = self.var).pdf(self.y[i])

    def model(self,theta):
        """ This function calculates the pdfs of the Gaussian model for a given parameter value theta
        at the different values given by the observations y. Notice that it is not calculating the pdf
        of the full sample y, rather the n pdfs given by the different observations"""
        #mean=np.dot(self.X, theta)
        #return multivariate_normal(mean = mean.flatten(),cov = self.var*np.eye((self.n))).pdf(self.y)
        return np.exp(self.lmodel(theta))

    def lmodel_i(self, theta, i):
        """This function calculates the log-density of the observation y[i] for the model with parameter
        values equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector theta. """
        return -0.5*np.log(2*np.pi*self.var)-0.5/self.var*((self.y[i]-np.dot(self.X[i,::], theta))**2)
    

    def lmodel(self, theta):
        """This function calculates the log-densities of the full sample for the model with parameter values 
        equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta."""
        if theta.ndim == 1:
            theta=theta.reshape((self.p, 1))
        
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
        """This function return the value of the prior at a given value of the parameter theta"""
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
        sigma1=(self.X[index:index+1,::])*(np.matrix(model_i.cond_theta().cov_object.covariance))*(self.X[index:index+1,::].transpose())
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
                sigma1=(X_index[j:j+1,::])*(np.matrix(model_res.cond_theta().cov_object.covariance))*(X_index[j:j+1,::].transpose())
                coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma1+self.var))))
                exponent += (-0.5/(sigma1+self.var)) * (y_index[j]-mu1)**2
            else:
                aux=X_index[j:j+1,::]*np.matrix(self.theta_0)
                sigma=(X_index[j:j+1,::])*(np.matrix(self.sigma))*(X_index[j:j+1,::].transpose()) + self.var
                coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma))))
                exponent += (-0.5/(sigma)) * (y_index[j]-aux)**2

        return (coefficient+exponent)[0,0]


    #def log_p_full(self):
    #    """This function computes the marginal log-probability for the full sample. It is designed to work also in high-D setting"""
    #    coefficient = 0; exponent = 0;
    #    for j in range(0,self.n):
    #        y1=self.y[j+1:]
    #        X1=self.X[j+1:,::]
    #        if len(y1)>0:
    #            model_res = Gaussian_Model(y1, X1, self.theta_0, self.sigma, self.var)
    #            aux=np.matrix( (model_res.cond_theta().mean.reshape((self.p,1))) )
    #            mu1=self.X[j:j+1,::]*aux
    #            sigma1=(self.X[j:j+1,::])*(np.matrix(model_res.cond_theta().cov_object.covariance))*(self.X[j:j+1,::].transpose())
    #            coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma1 + self.var))))
    #            exponent = exponent + (-0.5/(sigma1+self.var))*(self.y[j]-mu1)**2
    #        else:
    #            aux=self.X[j:j+1,::]*np.matrix(self.theta_0)
    #            sigma=(self.X[j:j+1,::])*(np.matrix(self.sigma))*(self.X[j:j+1,::].transpose()) + self.var
    #            coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma))))
    #            exponent = exponent + (-0.5/(sigma))*(self.y[j]-aux)**2
    #            
    #    return coefficient[0,0]+exponent[0,0]

    def correct_values(self):
        """This function returns the correct values for the different lppd terms"""
        return np.array([self.log_predictive_i(k) for k in range(self.n)])

    def log_p_full(self):
        """This function computes the marginal log-probability for the full sample. It is designed to work also in high-D setting"""
        cov = np.dot(np.dot(self.X, self.sigma), np.transpose(self.X)) + self.var*np.eye(self.n)
        exp = -np.dot(np.dot(np.transpose(self.y), np.linalg.inv(cov)), self.y)/2
        coeff = -np.log((2*np.pi)**(self.n/2)*np.sqrt(np.linalg.det(cov)))
        #return [coeff, exp]
        return coeff+exp



class Logistic_Model(Model):
    def __init__(self, y, X, mu, beta):
        super().__init__(y, X)
        self.mu = mu                  ## location and scale of the double Exponential prior distribution
        self.beta = beta       
                                       
    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        self.__mu = value

    @property
    def beta(self):
        return self.__beta
    
    @beta.setter
    def beta(self, value):
        if value <= 0:
            raise ValueError("The prior's Laplace scale parameter has to be positive")
        else:
            self.__beta = value
    
    def prior(self, beta):
        '''This function computes the pdf of the prior at a given value beta'''
        partial=1
        for bj in b:
            partial=laplace.pdf(bj,loc=self.mu,scale=self.beta)*partial
        return partial

    def model_i(self, beta, i):
        """ This function calculates the pmf of the Logistic model for a given parameter value beta
        and at a given observation y[i]"""
        return np.exp(self.lmodel_i(beta, i))

    def model(beta):
        """ This function calculates the pmfs of the Logistic model for a given parameter value beta
        at the different values given by the observations y. Notice that it is not calculating the pmf
        of the full sample y, rather the n pmfs given by the different observations"""
        return np.exp(self.lmodel(beta))


    def lmodel_i(self, beta, i):
        '''This function computes the logarithm of the probability mass function(pmf) at the i-th point for a given value of beta'''
        mu=np.dot(self.X[i,::], beta)
        zeros = np.zeros(mu.shape)
        return mu*self.y[i] - logsumexp(np.array([zeros, mu]), axis=0)

    def lmodel(self, beta):
        """This function calculates the logarithms of the pmfs for the Logistic model with parameter values 
        equal to beta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta."""
        mu=np.dot(self.X, beta)
        zeros = np.zeros(mu.shape)
        return mu*self.y.reshape((self.n,1)) - logsumexp(np.array([zeros, mu]), axis=0)

    def str_post(self):
        """This function creates the string with which we create the model in PyStan, and hence sample from the posterior distribution"""
        model_posterior="""
        data
        {
            int n;
            int k;
            int<lower=0, upper=1> y[n];
            matrix [n,k] X;
            real prior_scale;
            
        }

        parameters
        {
            vector[k] beta;
        }


        model
        {
            beta ~ double_exponential(0, prior_scale);
            y ~ bernoulli_logit(X*beta);
            
        }
        """
        return model_posterior

    def str_mix(self):
        """This function creates the string with which we can sample from our mixture distribution in PyStan"""
        model_mixture="""
        functions
        {
            real mixture_lpmf(int[] y, matrix X, int n, int k, vector beta, real prior_scale)
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
            log_prior = double_exponential_lpdf(beta | 0, prior_scale);
            return (log_sum_exp(contributions) + log_prior + log_full_model);
            }
        }

        data
        {
            int n;
            int k;
            int<lower=0, upper=1> y[n];
            matrix [n,k] X;
            real prior_scale;
            
        }

        parameters
        {
            vector[k] beta;
        }

        model
        {
            target += mixture_lpmf(y|X,n,k,beta, prior_scale);
        }

        """
        return model_mixture

