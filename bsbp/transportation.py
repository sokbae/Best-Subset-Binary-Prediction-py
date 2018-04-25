#!/usr/bin/env python
"""
Reproduction of the empirical results concerning an empirical application
in the prediction of transportation mode choice using the best subset
maximum score approach of Chen and Lee (2017).
The work-trip mode choice dataset of Horowitz (1993) is included in the package.
"""

import argparse
import pkg_resources
from . import *

def transportation(warm_start=1,tau=1.5,mio=1,q=1,series_exp=0,b=10):
    """ Best subset maximum score approach on the work-trip mode choice dataset of Horowitz.

    Args:
        warm_start (int,optional): Set to 1 for warm start strategy. Default = 1.
        tau (float,optional): Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
        mio (int,optional): Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
        q (int, optional):  Value of the variable selection bound. Default = 1.
        series_exp (int, optional): Set to 1 to use quadratic expansion terms as covariates. Default = 0.
        b (int, optional): Bound value. Default = 1.
    """

    beta0=1
    data = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data_horowitz.csv'), delimiter=',')
    print('Estimation based on full sample')
    Y_tr=data[:,0]
    temp=data[:,1:]

    n_tr=len(Y_tr)
    # [DCOST CARS DOVTT DIVTT]
    x_std = (temp-np.matlib.repmat(np.mean(temp,axis=0),n_tr,1))/np.matlib.repmat(np.std(temp,axis=0),n_tr,1)
    x_foc = np.concatenate((np.array([x_std[:,0]]).T,(np.ones(n_tr)).reshape(-1,1)),axis=1)  # [DCOST Intercept]

    if series_exp == 1:
        z2 = np.array([x_std[:,1]]).T
        z3 = np.array([x_std[:,2]]).T
        z4 = np.array([x_std[:,3]]).T
        x_aux1 = x_std[:,1:] # linear terms
        x_aux2 = np.concatenate((z2*z3,z3*z4,z2*z4),axis=1)
        x_aux3 = np.concatenate((z2*z2,z3*z3,z4*z4),axis=1)
        x_aux = np.concatenate((x_aux1,x_aux2,x_aux3),axis=1)
    else:
        x_aux = x_std[:,1:] # [CARS DOVTT DIVTT]

    k=x_foc.shape[1]
    d=x_aux.shape[1]

    # set the initial parameter bounds
    bnd= np.concatenate((-b*(np.ones(k-1+d)).reshape(-1,1), b*(np.ones(k-1+d)).reshape(-1,1)),axis=1)

    # set the tolerance level value
    tol = floor(sqrt(np.log(n_tr)*n_tr)/2)
    print('Tolerance level: ', tol/n_tr)

    time_limit = 86400 # set the MIO solver time limit
    # warm start MIO
    if warm_start == 1:
        bhat, score, gap, rtime, ncount  = warm_start_max_score(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio,tau)
    # cold start MIO
    else:
        bhat, score, gap, rtime, ncount  = max_score_constr_fn(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio)

    print('parameter estimates: ', bhat)
    print('avg_score gap time node_count:', score, ' , ', gap, ' , ', rtime, ' , ', ncount)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Input parameters. If no inputs, default parameters are used.')
    parser.add_argument('warmstart', type=int, nargs='?', default =0,help='set this variable to 1 for using the warm start strategy')
    parser.add_argument('tau', type=float, nargs='?', default =1.5,help='the tuning parameter to construct the refined bound used in the warm start approach')
    parser.add_argument('mio', type=int, nargs='?', default =1,help='set to 1 for using Method 1 for the MIO formulation, to 2 for using Method 2')
    parser.add_argument('q', type=int, nargs='?', default =1, help ='value of variable selection bound')
    parser.add_argument('seriesexp', type=int, nargs='?', default =0, help ='set to 1 for using quadratic expansion terms as covariates')
    parser.add_argument('b', type=int, nargs='?', default =10, help ='bound value')

    args = parser.parse_args()

    # load data
    transportation(warm_start = args.warmstart,tau=args.tau,mio=args.mio,q=args.q,series_exp=args.seriesexp,b=args.b)
