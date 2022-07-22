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

def Chi_Squared(x):
    y = cvx.square(x-1)
    return y

def Jensen_Shannon(x):
    z = 0.5*(x+1)
    y = cvx.rel_entr(x,z)+cvx.log(2)-cvx.log1p(x)
    return y

def Squared_Hellinger_Distance(x):
    y = 1-2*cvx.sqrt(x)+x
    return y

def Neyman_Chi_Squared(x):
    y = cvx.inv_pos(x)-1
    return y

# CVXPY Doesn't recognize following functions as convex
def Le_Cam_Distance(x):
    y = 1/(x+1)-1/2
    return y

