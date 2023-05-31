import pandas as pd
import numpy as np


burn=50000
w_posterior=(pd.read_csv("w_posterior_logistic.csv",header=None).values)[::,burn:]
w_mixture=(pd.read_csv("w_mixture_logistic.csv",header=None).values)[::,burn:]
w_mixture_ew=(pd.read_csv("w_mixture_ew_logistic.csv",header=None)).values[::,burn:]





iterations=w_mixture.shape[1]
print("iterations:",iterations)
n=w_mixture.shape[0]
print("n:",n)
#w_posterior=w_posterior/(w_posterior.mean(axis=1).reshape((n,1)))
#w_mixture=w_mixture/(w_mixture.mean(axis=1).reshape((n,1)))
#w_mixture_ew=w_mixture_ew/(w_mixture_ew.mean(axis=1).reshape((n,1)))

#tells me how often i recalculate the shape parameter
freq=2000
attempts=np.int(iterations/freq)
ks_mixture=np.zeros((n,attempts-1))
ks_posterior=np.zeros((n,attempts-1))
ks_ew=np.zeros((n,attempts-1))

for j in range(n):
    #print("we are at j:",j)
    #check=1
    for s in range(1,attempts):
        print("i am at  "+str(100*s/attempts)+"%"+"and j:"+str(j), end="\r")
        ##mixture
        critical=w_mixture[j, 0:s*freq]
        aux=critical.copy()
        aux=np.sort(aux)

        x=aux[np.int(np.floor(0.9*len(aux)))::] - aux[np.int(np.floor(0.9*len(aux)))]
        n=len(x)
        #np.savetxt(f"/home/3005788/{t}_threshold_tails.csv", x, delimiter=",")
        m=20+np.floor(np.sqrt(n))

        ##fq = first quartile

        fq=x[np.int(np.floor(n/4+0.5))-1]
        theta=np.array([1/x[n-1] + (1-np.sqrt(m/(j-0.5)))/(3*fq) for j in range(1, np.int(m+1))])
            
        k=np.array([-1*np.mean((np.log(1-theta_j*x))) for theta_j in theta])
        l=n*(np.log(theta/k)+k-1)
        ##look better at this quantity, could be messing everything around!!!!!!!!!
        w=np.array([1/np.sum(np.exp(l-l_i)) for l_i in l])
        theta_new=np.sum(theta*w)
        k_new=-1*np.mean((np.log(1-theta_new*x)))
        sigma_new=k_new/theta_new
        
        #if k_new!=np.inf and check:
            #print("good, k:",-k_new)
            #check=0
        ks_mixture[j,s-1]=-1*k_new



        ##posterior
        critical=w_posterior[j, 0:s*freq]
        aux=critical.copy()
        aux=np.sort(aux)

        x=aux[np.int(np.floor(0.9*len(aux)))::] - aux[np.int(np.floor(0.9*len(aux)))]
        n=len(x)
        #np.savetxt(f"/home/3005788/{t}_threshold_tails.csv", x, delimiter=",")
        m=20+np.floor(np.sqrt(n))

        ##fq = first quartile

        fq=x[np.int(np.floor(n/4+0.5))-1]
        theta=np.array([1/x[n-1] + (1-np.sqrt(m/(j-0.5)))/(3*fq) for j in range(1, np.int(m+1))])
            
        k=np.array([-1*np.mean((np.log(1-theta_j*x))) for theta_j in theta])
        l=n*(np.log(theta/k)+k-1)
        w=np.array([1/np.sum(np.exp(l-l_i)) for l_i in l])
        theta_new=np.sum(theta*w)
        k_new=-1*np.mean((np.log(1-theta_new*x)))
        sigma_new=k_new/theta_new
        ks_posterior[j,s-1]=-1*k_new


        ########################################################
        #############ew mixture
        critical=w_mixture_ew[j, 0:s*freq]
        aux=critical.copy()
        aux=np.sort(aux)

        x=aux[np.int(np.floor(0.9*len(aux)))::] - aux[np.int(np.floor(0.9*len(aux)))]
        n=len(x)
        #np.savetxt(f"/home/3005788/{t}_threshold_tails.csv", x, delimiter=",")
        m=20+np.floor(np.sqrt(n))

        ##fq = first quartile

        fq=x[np.int(np.floor(n/4+0.5))-1]
        theta=np.array([1/x[n-1] + (1-np.sqrt(m/(j-0.5)))/(3*fq) for j in range(1, np.int(m+1))])
            
        k=np.array([-1*np.mean((np.log(1-theta_j*x))) for theta_j in theta])
        l=n*(np.log(theta/k)+k-1)
        w=np.array([1/np.sum(np.exp(l-l_i)) for l_i in l])
        theta_new=np.sum(theta*w)
        k_new=-1*np.mean((np.log(1-theta_new*x)))
        sigma_new=k_new/theta_new
        ks_ew[j,s-1]=-1*k_new


np.savetxt("/home/3005788/thesis/experiment_logistic/k_mixture.csv", ks_mixture, delimiter=",")
np.savetxt("/home/3005788/thesis/experiment_logistic/k_mixture_ew.csv", ks_ew, delimiter=",")
np.savetxt("/home/3005788/thesis/experiment_logistic/k_posterior.csv", ks_posterior, delimiter=",")