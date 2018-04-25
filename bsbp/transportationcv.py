#!/usr/bin/env python
"""
Reproduction of the empirical results concerning an empirical application
in the prediction of transportation mode choice using the best subset
maximum score cross validation approach of Chen and Lee (2017).
The work-trip mode choice dataset of Horowitz (1993) is included in the package.
"""

import argparse
from . import *

def transportationcv(warm_start = 1,tau=1.5,mio=1,q=1,series_exp=1,b=10,time_limit=8640000):
    """ Best subset maximum score approach with cross validation on the work-trip mode choice dataset of Horowitz.

    Args:
        warm_start (int,optional): Set to 1 for warm start strategy. Default = 1.
        tau (float,optional): Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
        mio (int,optional): Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
        q (int, optional):  Value of the variable selection bound. Default = 1.
        series_exp (int, optional): Set to 1 to use quadratic expansion terms as covariates. Default = 0.
        b (int, optional): Bound value. Default = 1.
        time_limit (int, optional): MIO solver time limit. Default = 8640000.
    """

    data = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data_horowitz.csv'), delimiter=',')
    tr_ind = np.genfromtxt(pkg_resources.resource_filename(__name__, 'tr_ind.csv'), delimiter=',')
    test_ind = np.genfromtxt(pkg_resources.resource_filename(__name__, 'test_ind.csv'), delimiter=',')

    beta0 = 1
    fold=tr_ind.shape[1]
    bhat=np.zeros((10,fold))

    score=np.zeros((fold,1))
    gap=np.zeros((fold,1))
    in_score=np.zeros((fold,1))
    rtime=np.zeros((fold,1))
    ncount=np.zeros((fold,1))
    p_ratio=np.zeros((fold,1))

    for i in range (fold):

        data_tr=data[tr_ind[:,i],:]
        data_v=data[test_ind[:,i],:]

        print('Estimation based on training sample at fold: ',i+1)
        Y_tr=data_tr[:,0]
        temp=data_tr[:,1:]

        n_tr=len(Y_tr)

        # [DCOST CARS DOVTT DIVTT]
        x_std = (temp-np.matlib.repmat(np.mean(temp,axis=0),n_tr,1))/np.matlib.repmat(np.std(temp,axis=0),n_tr,1)
        x_foc = np.concatenate((np.array([x_std[:,0]]).T,np.ones((n_tr,1))),axis=1)  # [DCOST Intercept]

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

        bnd= np.concatenate((-b*np.ones((k-1+d,1)),b*np.ones((k-1+d,1))),axis=1) # set the initial parameter bounds
        tol = floor(sqrt(log(n_tr)*n_tr)/2)
        print('tolerance level: ',tol)

        if warm_start == 1: # warm start MIO
            bhat[:,i],score[i],gap[i],rtime[i],ncount[i] = warm_start_max_score(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio,tau);
        else: # cold start MIO
            bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = max_score_constr_fn(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio);

        print('coefficient values: ')
        print('bhat \n', bhat[:,i])
        print('gurobi score: ',score[i])
        print('gurobi absolute gap: ',gap[i])
        print('gurobi running time: ',rtime[i])
        print('gurobi node count: ',ncount[i])
        in_score[i]=sum(Y_tr==(np.concatenate((x_foc,x_aux),axis=1)@np.insert(bhat[:,i], 0, 1)>0))
        print('in-sample score: ',in_score[i])

        # validation sample

        Y_val=data_v[:,0]
        print()
        n_val=len(Y_val)
        temp=data_v[:,1:]
        x_std = (temp-np.matlib.repmat(np.mean(temp,axis=0),n_val,1))/np.matlib.repmat(np.std(temp,axis=0),n_val,1)

        if series_exp == 1:
            z2 = np.array([x_std[:,1]]).T
            z3 = np.array([x_std[:,2]]).T
            z4 = np.array([x_std[:,3]]).T
            x_aux1 = x_std[:,1:] # linear terms
            x_aux2 = np.concatenate((z2*z3,z3*z4,z2*z4),axis=1)
            x_aux3 = np.concatenate((z2*z2,z3*z3,z4*z4),axis=1)
            x_aux = np.concatenate((x_aux1,x_aux2,x_aux3),axis=1)
            x_v = np.concatenate((np.array([x_std[:,0]]).T,np.ones((n_val,1)),x_aux),axis=1)
        else:
            x_v = np.concatenate((np.array([x_std[:,0]]).T,np.ones((n_val,1)),x_std[:,1:]),axis=1)

        y_hat=((x_v@np.insert(bhat[:,i], 0, 1))>0)
        p_ratio[i]=np.mean(Y_val==y_hat)
        print('out-of-sample performance: ',p_ratio[i])

    print('Average coefficient vector: ',np.mean(bhat,axis=1))
    print('Average score: ',np.mean(score))
    print('Average gap: ',np.mean(gap))
    print('Average running time: ',np.mean(rtime))
    print('Average node count: ',np.mean(ncount))
    print('Average in-sample score: ',np.mean(in_score))
    print('average out-of-sample performance: ',np.mean(p_ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input parameters. If no inputs, default parameters are used.')
    parser.add_argument('warmstart', type=int, nargs='?', default =1,help='set this variable to 1 for using the warm start strategy')
    parser.add_argument('tau', type=float, nargs='?', default =1.5,help='the tuning parameter to construct the refined bound used in the warm start approach')
    parser.add_argument('mio', type=int, nargs='?', default =1,help='set to 1 for using Method 1 for the MIO formulation, to 2 for using Method 2')
    parser.add_argument('q', type=int, nargs='?', default =1, help ='value of variable selection bound')
    parser.add_argument('seriesexp', type=int, nargs='?', default =1, help ='set to 1 for using quadratic expansion terms as covariates')
    parser.add_argument('b', type=int, nargs='?', default =10, help ='bound value')
    parser.add_argument('timelimit', type=int, nargs='?', default =8640000, help ='set the MIO solver time limit')

    args = parser.parse_args()

    transportationcv(warm_start=args.warmstart,tau = args.tau,mio = args.mio,q = args.q,series_exp = args.seriesexp,b=args.b, time_limit = args.timelimit)
