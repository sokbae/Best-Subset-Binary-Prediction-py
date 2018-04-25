#!/usr/bin/env python
"""
Reproduction of the simulation results of the performance of the best
subset maximum score approach of Chen and Lee (2017).
"""

import argparse
from . import *

def simulation_data(n,beta,sigma,typ):
    """ Generates data from a normal distribution.

    Args:
        n (int): Number of observations.
        beta (float): The upper limit of the range to generate, from 0 to `n` - 1.
        sigma (float): The upper limit of the range to generate, from 0 to `n` - 1.
        typ (int): 1 for heteroskedasticity

    Yields:
        y: outcome
        datax: covariates
    """
    np.random.seed(1)
    k = len(beta)
    reg = np.zeros((n,k-1))
    mn = multivariate_normal(mean = np.zeros(k-1), cov = sigma)
    reg = mn.rvs(size=n)
    datax = np.insert(reg,1,1,axis =1)
    if typ == 1: # heteroskedasticity
        sigmaW = reg[:,1] + reg[:,2]
        e = np.array([0.25*(np.ones(n) + 2*np.power(sigmaW,2) + np.power(sigmaW,4))*np.random.normal(0,1,n)]).T
    else: # homoskedasticity
        e=0.25*np.random.normal(0,1,n)
    y=((datax@beta)>=e).astype(int)
    return y, datax

def simulation(warm_start=0,tau=1.5,mio=1,q=1,N=100,N_val=5000,R=10,p=10,typ=1):
    """ Simulates best subset binary prediction on simulated data.

    Args:
        warm_start (int,optional): Set to 1 for warm start strategy. Default = 0.
        tau (float,optional): Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
        mio (int,optional): Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
        q (int, optional):  Value of the variable selection bound. Default = 1.
        N (int, optional): Size of training sample. Default = 100.
        N_val (int, optional): Size of validation sample. Default = 5000.
        R (int, optional): Simulation repetitions. Default = 10.
        typ (int, optional): Set to 1 for heteroskedastic error design. Set to 0 for homoskedastic error design. Default = 1.
    """

    # Focus Covariates
    beta0=1
    if (typ==1):
        beta_s = -1.5
    else:
        beta_s = -0.35

    beta = np.concatenate(([[beta0]], [[0]], [[beta_s]], (np.zeros(p-1)).reshape(-1,1)),axis=0)

    K=beta.shape[0]
    bhat=np.zeros((K-1,R))

    rho=0.25
    sigma=np.ones(K-1)
    for i in range (K-2):
        sigma[i+1]=rho**(i+1)
    sigma=toeplitz(sigma)

    gap=np.zeros((R,1)) # MIO gap
    rtime=np.zeros((R,1)) # MIO running time
    ncount=np.zeros((R,1)) # MIO node count
    score=np.zeros((R,1)) # MIO score

    DGP_score=np.zeros((R,1)) # in-sample score at the DGP parameter vector
    val_score=np.zeros((R,1)) # in-sample score at the estimated parameter vector
    DGP_score_test=np.zeros((R,1)) # out-of-sample score at the DGP parameter vector
    val_score_test=np.zeros((R,1)) # out-of-sample score at the estimated parameter vector

    bnd = np.concatenate(((-10*np.ones(11)).reshape(-1,1),(10*np.ones(11)).reshape(-1,1)),axis=1)
    bnd_h = np.zeros((bhat.shape[0],2,R))

    maxT=0

    if (q>=1 and p>N):
        tol=min(0.5*sqrt((1+q)*log(p)*N),0.05*N)  #early stopping rule
    else:
        tol=0

    for i in range(R):
        print(i+1)
        y,datax = simulation_data(N,beta,sigma,typ)

        try:
            # warm start MIO
            if (warm_start == 1):
                bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = warm_start_max_score(y,datax[:,0:2],datax[:,2:],beta0,q,maxT,tol,bnd,mio,tau)
            #cold start MIO
            else:
                bhat[:,i],score[i],gap[i],rtime[i],ncount[i] = max_score_constr_fn(y,datax[:,0:2],datax[:,2:],beta0,q,maxT,tol,bnd,mio)
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        DGP_score[i] = np.mean(y == ((datax@beta)>=0).T)
        val_score[i] = np.mean(y == (datax@np.concatenate((np.array([beta0]),bhat[:,i]),axis=0)>=0).T)

        if (N_val > 0):
            y_val,datax_val = simulation_data(N_val,beta,sigma,typ)
            DGP_score_test[i] = np.mean((y_val == ((datax_val@beta)>=0).T))
            val_score_test[i] = np.mean(y_val == ((datax_val@np.concatenate((np.array([beta0]),bhat[:,i]),axis=0)>=0)).T)
    print(np.mean(np.concatenate((val_score,DGP_score,val_score_test,DGP_score_test),axis=1),axis=0))
    print(np.mean(np.concatenate((val_score/DGP_score,val_score_test/DGP_score_test),axis=1),axis=0))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Input parameters. If no inputs, default parameters are used.')
    parser.add_argument('warmstart', type=int, nargs='?', default =0,help='set this variable to 1 for using the warm start strategy')
    parser.add_argument('tau', type=float, nargs='?', default =1.5,help='the tuning parameter to construct the refined bound used in the warm start approach')
    parser.add_argument('mio', type=int, nargs='?', default =1,help='set to 1 for using Method 1 for the MIO formulation, to 2 for using Method 2')
    parser.add_argument('q', type=int, nargs='?', default =1, help ='value of variable selection bound')
    parser.add_argument('N', type=int, nargs='?', default =100, help ='size of the training sample')
    parser.add_argument('Nval', type=int, nargs='?', default =5000, help ='size of the validation sample')
    parser.add_argument('R', type=int, nargs='?', default =10, help ='simulation repetitions')
    parser.add_argument('p', type=int, nargs='?', default =10, help ='auxiliary covariates dimension')
    parser.add_argument('type', type=int, nargs='?', default =1, help ='set to 1 for heteroskedastic error design, to 2 for homeskedastic')

    args = parser.parse_args()

    simulation(warm_start=args.warmstart,tau=args.tau,mio=args.mio,q=args.q,N=args.N,N_val=args.Nval,R=args.R,p=args.p,typ=args.type)
