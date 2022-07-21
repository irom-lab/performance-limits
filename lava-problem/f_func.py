import numpy as np
import cvxpy as cvx

def Kullback_Leibler(x):
    y = -cvx.entr(x) #x*log(x)
    return y

def Negative_Log(x):
    y = -cvx.log(x)
    return y

def Total_Variation(x):
    y = 0.5*cvx.abs(x-1)
    return y

def Jensen_Shannon(x):
    z = 0.5*(x+1)
    y = cvx.rel_entr(x,z)+cvx.log(1/z)
    return y