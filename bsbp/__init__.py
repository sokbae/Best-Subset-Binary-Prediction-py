from gurobipy import *
from math import floor, log, sqrt
import numpy as np
import numpy.matlib as npm
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
from scipy.stats import norm
import copy

def miobnd_fn(x,beta0,bnd):
    n=np.shape(x)[0]
    k=np.shape(x)[1]

    # Create new model
    m = Model("mip1")

    # Set parameters
    tol=float("1e-6")

    m.setParam("outputflag", 0)
    m.setParam("OptimalityTol",tol)
    m.setParam("FeasibilityTol",tol)
    m.setParam("IntFeasTol",tol)

    v = [0]*2
    t = [0]*(k+1)

    value=np.zeros(n);

    # Create variables
    for i in range(0,k-1):
        t[i] = m.addVar(lb=bnd[i,0], ub=bnd[i,1], name='t%d' % i)

    for i in range(n):
        alpha =  beta0*x[i,0]

        #set objective
        objExpr=LinExpr()
        for j in range(1,k):
            objExpr.add(t[j-1],x[i,j])
        objExpr.addConstant(alpha)
        m.setObjective(objExpr, GRB.MAXIMIZE)
        m.update()
        try:
            expr = LinExpr()
            for j in range(1,k):
                expr.add(t[j-1],x[i,j])
            m.addConstr(expr, GRB.GREATER_EQUAL, -alpha)
            m.update()
            m.optimize()
            if i == 0:
                m.write('miobnd.lp')
            v[0] = m.objVal
            m.remove(m.getConstrs()[0])
            m.update()
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        #add objective
        objExpr=LinExpr()
        for j in range(1,k):
            objExpr.add(t[j-1],-x[i,j])
        objExpr.addConstant(-alpha)
        m.setObjective(objExpr, GRB.MAXIMIZE)
        m.update()

        try:
            expr = LinExpr()
            for j in range(1,k):
                expr.add(t[j-1],-x[i,j])
            m.addConstr(expr, GRB.GREATER_EQUAL, alpha)
            m.update()
            m.optimize()
            v[1] = m.objVal
            m.remove(m.getConstrs()[0])
            m.update()
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        value[i]=np.max(v)
    return value

def max_score_constr_fn(y,x_foc,x_aux,beta0,q,T,abgap,bnd,mio):
    N=len(y)
    k=np.shape(x_foc)[1]-1
    d=np.shape(x_aux)[1]
    bhat=np.zeros(k+d)
    gap=0
    rtime=0

    miobnd=miobnd_fn(np.concatenate((x_foc,x_aux),axis=1),beta0,bnd)

    # Create new model
    m = Model("mip1")

    # Set parameters
    tol=float("1e-6");

    m.setParam("outputflag", 0)
    m.setParam("OptimalityTol",tol)
    m.setParam("FeasibilityTol",tol)
    m.setParam("IntFeasTol",tol)

    t = [0]*(N+k+2*d)
    # Create variables
    for i in range(0,N):
        t[i] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='t%d' % i)
    for i in range(N,N+k+d):
        t[i] = m.addVar(lb=bnd[i-N,0], ub=bnd[i-N,1], name='t%d' % i)
    for i in range(N+k+d,N+k+2*d):
        t[i] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='t%d' % i)
    m.update()
    if T > 0:
        m.setParam("timeLimit", T)

    if abgap > 0:
        m.setParam("MIPGapAbs", abgap)

    ztemp1=np.zeros((N,d))
    ztemp2=np.zeros((2*d+1,N))
    htemp=np.concatenate((np.identity(d),-np.identity(d),np.array([np.zeros(d)])),axis=0)
    etemp=np.concatenate((-np.diag(bnd[k:k+d,1]),np.diag(bnd[k:k+d,0]),np.array([np.ones(d)])),axis=0) # check indices
    mtemp1=np.concatenate((ztemp2, np.zeros((2*d+1,k)),htemp, etemp),axis=1)
    mtemp2=(np.append(np.zeros(2*d),q)).reshape(-1,1)

    if mio == 1: #method 1 formulation
        #set objective
        objExpr=LinExpr()
        for j in range(1,N+1):
            objExpr.add(t[j-1],2*y[j-1]-1)
        objExpr.addConstant(np.sum(1-y))
        m.setObjective(objExpr, GRB.MAXIMIZE)
        m.update()

        miobnd_bar = miobnd+tol
        mtemp3=np.concatenate((np.diag(-miobnd_bar),x_foc[:,1:k+1], x_aux, ztemp1), axis=1)
        #constraints
        x_com = np.concatenate((np.diag(miobnd),-x_foc[:,1:k+1],-x_aux,ztemp1),axis=1)
        V = np.shape(x_com)[0]
        W = np.shape(x_com)[1]

        for i in range(V):
            expr = LinExpr()
            for j in range(W):
                expr.add(t[j],x_com[i,j])
            m.addConstr(expr, GRB.LESS_EQUAL, np.array(miobnd*(1-tol)+beta0*x_foc[:,0])[i])
        #mtemp3
        V = np.shape(mtemp3)[0]
        for i in range(V):
            expr = LinExpr()
            for j in range(1,W):
                expr.add(t[j-1],mtemp3[i,j-1])
            m.addConstr(expr, GRB.LESS_EQUAL, np.array(-tol*miobnd_bar-beta0*x_foc[:,0])[i])
        #mtemp1
        V = np.shape(mtemp1)[0]
        for i in range(V):
            expr = LinExpr()
            for j in range(W):
                expr.add(t[j],mtemp1[i,j])
            m.addConstr(expr, GRB.LESS_EQUAL, mtemp2[i])
        m.update()

    else: #Method 2 formulation
        #set objective
        objExpr=LinExpr()
        for j in range(N):
            objExpr.add(t[j],1)
        objExpr.addConstant(np.sum(1-y))
        m.setObjective(objExpr, GRB.MAXIMIZE)

        temp2=(1-2*y)

        #constraints
        x_com = np.concatenate((np.diag(miobnd), (np.matlib.repmat(temp2,k,1)).T*x_foc[:,1:k+1],(np.matlib.repmat(temp2,d,1)).T*x_aux, ztemp1),axis=1)
        V = np.shape(x_com)[0]
        W = np.shape(x_com)[1]

        for i in range(V):
            expr = LinExpr()
            for j in range(W):
                expr.add(t[j],x_com[i,j])
            m.addConstr(expr, GRB.LESS_EQUAL, np.array(miobnd*(1-tol)-(beta0*temp2*x_foc[:,0]))[i])
        #mtemp1
        V = np.shape(mtemp1)[0]
        for i in range(V):
            expr = LinExpr()
            for j in range(W):
                expr.add(t[j],mtemp1[i,j])
            m.addConstr(expr, GRB.LESS_EQUAL, mtemp2[i])
        m.update()

    try:
        m.write('maxscore1.lp')
        m.optimize()
        bhat = []
        for j in range(N,N+k+d):
            bhat.append(m.getVars()[j].x)
        score=m.objVal
        gap=m.objBound-score
        rtime=m.runTime
        ncount=m.nodeCount
        print('Optimization returned status: {}'.format(m.status))

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    return bhat,score,gap,rtime,ncount

def logit(y,x):
    cnv = 0
    b0 = np.zeros(x.shape[1])

    while (cnv==0):
        ind = x@b0
        P = np.exp(ind)/(1+np.exp(ind))
        grd = sum((np.matlib.repmat(y-P,x.shape[1],1)).T*x)
        hes = -x.T@((np.matlib.repmat(P*(1-P),x.shape[1],1)).T*x)

        b1 = b0 - np.linalg.inv(hes)@grd.T

        dev = max(abs(b1-b0))

        if (dev < 1e-8):
            cnv = 1
        b0 = b1
    b=b1
    return b

def get_bnd(y,x,beta0,bnd):
    k=x.shape[1]-1

    p_hat = 1/(1+np.exp(-x@logit(y,x)))
    constr=np.multiply((np.matlib.repmat(p_hat-0.5,x.shape[1],1)).T,x)
    bound=copy.copy(bnd)

    # Create new model
    m = Model("mip1")

    # Set parameters
    tol=float("1e-6")

    m.setParam("outputflag", 0)
    m.setParam("OptimalityTol",tol)
    m.setParam("FeasibilityTol",tol)
    m.setParam("IntFeasTol",tol)

    # Create variables
    t = [0]*k
    for i in range(k):
        t[i] = m.addVar(lb=bound[i,0], ub=bound[i,1], name='t%d' % i)
    m.update()

    #Constraints
    for i in range(x.shape[0]):
            expr = LinExpr()
            for j in range(k):
                expr.add(t[j],constr[i,j+1])
            m.addConstr(expr,GRB.GREATER_EQUAL,-constr[i,0]*beta0)

    for i in range(k):
        # Objective
        objExpr=LinExpr()
        objExpr.add(t[i],1)
        m.setObjective(objExpr, GRB.MINIMIZE)
        m.update()
        m.write('miobnd_{0}_min.lp'.format(i))
        try:
            m.optimize()
            bound[i,0]=m.objVal
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        # Objective
        m.setObjective(objExpr, GRB.MAXIMIZE)

        for j in range(k):
            m.getVars()[j].lb = bound[j,0]
        m.update()
        m.write('miobnd_{0}_max.lp'.format(i))

        try:
            m.optimize()
            bound[i,1]=m.objVal
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        for j in range(k):
            m.getVars()[j].ub = bound[j,1]
        m.update()
    return bound

def warm_start_max_score(y,x_foc,x_aux,beta0,q,T,tol,bnd,mio,tau):
    bnd_h = get_bnd(y,np.concatenate((x_foc,x_aux),axis=1),beta0,bnd)
    print(bnd_h)
    bnd_abs = np.array([tau*np.amax(np.absolute(bnd_h),1)])
    print(bnd_abs)
    bnd0 = np.concatenate(((np.maximum(-bnd_abs,bnd[:,0])).T,(np.minimum(bnd_abs,bnd[:,1])).T),axis =1)
    print(bnd0)
    bhat, score, gap, rtime, ncount  = max_score_constr_fn(y,x_foc,x_aux,beta0,q,T,tol,bnd0,mio)
    return bhat,score,gap,rtime,ncount

def cv_best_subset_maximum_score(tr_ind,test_ind,data,focus_ind,aux_ind,beta0,q_range,T,tol,bnd,mio):
    q_num = len(q_range)
    fold=tr_ind.shape[1]
    score=np.zeros((fold,q_num))
    gap=np.zeros((fold,q_num))
    rtime=np.zeros((fold,q_num))
    ncount=np.zeros((fold,q_num))
    bhat=np.zeros((len(focus_ind)+len(aux_ind)-1,fold,q_num))
    val_score=np.zeros((q_num,1))

    for q in range(q_num):
        for i in range(fold):
            print('(q,fold) : ',q_range(q),' ',i+1)
            y=data[tr_ind[:,i],0]
            datax=data[tr_ind[:,i],1:]

            bhat[:,i,q],score[i,q],gap[i,q],rtime[i,q],ncount[i,q] = max_score_constr_fn(y,datax[:,focus_ind],datax[:,aux_ind],beta0,q_range[q],T,tol[q],bnd,mio)
            y_v=data[test_ind[:,i],0]
            datax_v=data[test_ind[:,i],1:end]
            val_score[q] = val_score[q]+ np.mean(y_v == ((datax_v@np.insert(bhat[:,i,q], 0, 1))>=0))
        val_score[q]=val_score[q]/fold
    best_q = np.max(val_score)
    return best_q,bhat,score,gap,rtime,ncount
