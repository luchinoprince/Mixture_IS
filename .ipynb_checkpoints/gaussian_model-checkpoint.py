import pandas as pd
import numpy as np
from scipy.stats import invwishart, invgamma, multivariate_normal
from scipy.special import logsumexp



class Model:
    def __init__(self, theta_0, sigma, var):
        '''Theta_0 is going ot be the prior's variance, sigma the prior's isotropic variance and var the linear model's variance'''
        self.theta_0 = theta_0         ##parameter of the Normal
        self.sigma = sigma  
        self.var= var

    
    def model_i(self,theta,y,X,i):
        """ This function calculates the pdf of the Gaussian model for a given parameter values theta
        and at a given observation y[i]"""
        mean=np.dot(X[i,::],theta).reshape(1,)
        #print("mean:",mean)
        ##here i is for the ith observation
        return MultivariateGaussian(mean,self.var).pdf(y[i])

    
    def lmodel_i(self, theta, X, y, i):
        """This function calculation the log-density of the observation y[i] for the model with parameter
        values equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta. """
        
        return -0.5*np.log(2*np.pi*model.var)-0.5/model.var*((y[i]-np.dot(X[i,::], theta))**2)
    
    def lmodel(self, theta, var, X, y):
        """This function calculates the log-density of the full sample for the model with parameter values 
        equal to theta. Note that this function requires that the X matrix has already the column
        of ones to account for the intercept, which will be the first element of the vector beta."""
        n=X.shape[0]
        return -n/2*np.log(2*np.pi*model.var)-0.5/model.var*(np.sum((y-np.dot(X, theta))**2))
        
    
    def cond_theta(self, y, X): ##here X is the vector of covariates
        """This function is part of the Gibbs step, samples theta from the full posterior"""
        variance = np.matrix(1/self.var*np.dot(X.transpose(), X)+np.matrix(self.sigma).I).I
        first=(np.matrix(1/self.var * np.dot(X.transpose(),X))+np.matrix(self.sigma).I).I
        second=np.matrix((1/self.var)*np.dot(X.transpose(),y).reshape((X.shape[1],1))+np.dot(np.matrix(self.sigma).I, self.theta_0))
        mean=np.dot(first, second)
        x=np.array([mean[j,0] for j in range(mean.size)])
        mean=x
        
        return MultivariateGaussian(mean, variance)



    def prior(self, theta):
        pp=1
        pp=multivariate_normal(mean=model.theta_0.reshape(len(model.theta_0)), cov=self.sigma).pdf(theta)*pp
        return pp
    
    

#########################################################################################################################
#########################################################################################################################
        

    
    
class MultivariateGaussian:
    def __init__(self, mean, cov_matrix):
        self.mean = mean
        self.cov_matrix = cov_matrix
    
    def log_p(self, x):
        return multivariate_normal.logpdf(x=x, mean=self.mean, cov=self.cov_matrix)
    
    def pdf(self,x):
        return multivariate_normal.pdf(x=x, mean=self.mean,cov=self.cov_matrix)
    
    def sample(self, size=1):
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov_matrix, size=size)

    
    def cdf(self,x):
        return multivariate_normal.cdf(x=x, mean=self.mean,cov=self.cov_matrix)

 

def log_predictive_i(model,y,X,index):
    '''This function computes the leave-one-out log-predictive probability for the index-th observation. It is implemented to work also in very high-dimensional
    setting.'''
    X_index=np.concatenate([X[0:index], X[index+1:]])
    y_index=np.concatenate([y[0:index], y[index+1:]])
    aux=np.matrix(model.cond_theta(y_index,X_index).mean.reshape((X.shape[1],1)))
    mu1=X[index:index+1,::]*aux
    sigma1=(X[index:index+1,::])*(model.cond_theta(y_index,X_index).cov_matrix)*(X[index:index+1,::].transpose())
    coefficient = np.log(1/(np.sqrt(2*np.pi*(sigma1 + model.var))))
    exponent = (-0.5/(sigma1+model.var)) * (y[index]-mu1)**2
            
    return coefficient[0,0]+exponent[0,0]

def log_p_loo_i(model,y,X,index):
    '''This function computes the leave-one-out marginal log-probability for the full sample other than index-th observation. It is implemented to work also in very high-dimensional setting.'''
    coefficient = 0; exponent = 0;
    X_index=np.concatenate([X[0:index], X[index+1:]])
    y_index=np.concatenate([y[0:index], y[index+1:]])
    for j in range(0,len(y_index)):
        y1=y_index[j+1:]
        X1=X_index[j+1:,::]
        if len(y1)>0:
            aux=np.matrix(model.cond_theta(y1,X1).mean.reshape((X.shape[1],1)))
            mu1=X_index[j:j+1,::]*aux
            sigma1=(X_index[j:j+1,::])*(model.cond_theta(y1,X1).cov_matrix)*(X_index[j:j+1,::].transpose())
            coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma1+model.var))))
            exponent += (-0.5/(sigma1+model.var)) * (y_index[j]-mu1)**2
        else:
            aux=X_index[j:j+1,::]*np.matrix(model.theta_0)
            sigma=(X_index[j:j+1,::])*(np.matrix(model.sigma))*(X_index[j:j+1,::].transpose())+model.var
            coefficient += np.log(1/(np.sqrt(2*np.pi*(sigma))))
            exponent += (-0.5/(sigma)) * (y_index[j]-aux)**2
    return (coefficient+exponent)[0,0]


def log_p_full(model,y,X):
    """This function computes the marginal log-probability for the full sample. It is designed to work also in high-D setting"""
    coefficient = 0; exponent = 0;
    for j in range(0,len(y)):
        y1=y[j+1:]
        X1=X[j+1:,::]
        if len(y1)>0:
            aux=np.matrix( (model.cond_theta(y1,X1).mean.reshape((X.shape[1],1))) )
            mu1=X[j:j+1,::]*aux
            sigma1=(X[j:j+1,::])*(model.cond_theta(y1,X1).cov_matrix)*(X[j:j+1,::].transpose())
            coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma1+model.var))))
            exponent = exponent + (-0.5/(sigma1+model.var))*(y[j]-mu1)**2
        else:
            aux=X[j:j+1,::]*np.matrix(model.theta_0)
            sigma=(X[j:j+1,::])*(np.matrix(model.sigma))*(X[j:j+1,::].transpose())+model.var
            coefficient = coefficient + np.log(1/(np.sqrt(2*np.pi*(sigma))))
            exponent = exponent + (-0.5/(sigma))*(y[j]-aux)**2
            
    return coefficient[0,0]+exponent[0,0]


def posterior_estimates(model, thetas_posterior, y, X):
    """This function calculates the posterior estimates of lppd given by the sample thetas_posterior. The function calculates analytically the log weights to ensure one does not encounter numerical overflow. It then leverages the logsumexp function from scipy to deal with sum of the logarithms. """
    n=X.shape[0]; iterations=thetas_posterior.shape[1];
    lw_posterior = 1/(2*model.var)*(y.reshape((n,1))-np.dot(X, thetas_posterior))**2
    pred_posterior = np.log(iterations) -  0.5*np.log(2*np.pi*model.var) - logsumexp(lw_posterior, axis=1)
    return pred_posterior

def mixture_estimates(model, thetas_mixture, y, X):
    """This function calculates the mixture estimates of lppd given by the sample thetas_mixture. The function might
    seem cumbersome, but it is written to ensure numerical stability also in high-dimensional settings"""
    ##Numerators 
    n=X.shape[0]; iterations=thetas_mixture.shape[1];
    means= np.dot(X,thetas_mixture)
    exponents_num_mix = 0.5/model.var * (y.reshape((1,n)) - means.transpose())**2 
    common_mixture=logsumexp(exponents_num_mix, axis=1)
    numerator_mix = logsumexp(-1*common_mixture)
    
    #Denominator
    denominator_exponents_mix = exponents_num_mix.transpose() - common_mixture.reshape((iterations,1)).transpose()
    denominator_mix = logsumexp(denominator_exponents_mix,axis=1)
    
    #Estimator 
    lppd_mixture = (numerator_mix-denominator_mix) - 0.5*np.log(2*np.pi*model.var)
    return lppd_mixture



    





