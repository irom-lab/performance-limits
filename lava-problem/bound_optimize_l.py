import numpy as np
from bound_rewards_l import *
from lava_problem import main as lava_problem
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import ray

def main(raw_args=None):
    n = 5 # n slices on interval [0,1]
    # s0 = [0]*(2*n) # set initial guesses to be 0
    s0_vals = [[-1]*5+[1]*5]*6+[[-10, -5, -3, -2, -1.25, -0.83, -0.6, -0.43, -0.35, -0.28]]*14

    # Setups for lava problem
    p_correct_vals = np.linspace(0.01, 0.99, 20) 
    reward_xs = np.array([0.1]) 
    reward_x = reward_xs[0]

    # Constraint: from 1 to 0 backwards, each slope has to be smaller
    # So elements in the slope vector need to be in decreasing order
    def con(svec):
        out = []
        for i in range(len(svec)-1):
            out.append(svec[i+1]-svec[i])
        return out
        
    cons = [{'type':'ineq','fun': con}]

    # Print optimization progress
    def callbackF(svec,status):
        print (svec,bounds(svec))

    def bounds(svec, p_correct = p_correct_vals[0]):
        nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
        bound_f_inverse = compute_bound(l,n,svec, nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
        return bound_f_inverse

    # def minimize(p_correct,svec):
    #     res = optimize.minimize(bounds,svec,method = 'trust-constr', constraints = cons, callback = callbackF, args = (p_correct), options={'disp':True})
    #     return[res.x, res.fun]
    
    @ray.remote
    def minimize(p_correct, s0):
        res = optimize.minimize(bounds,s0,method = 'trust-constr', constraints = cons, args = (p_correct),options = {'disp':True})
        return[res.x, res.fun]

    ''' Serial computing '''
    # tr = 20
    # opt_results = []
    # s0_guesses = [s0]
    # opt_results.append(minimize(p_correct_vals[0],s0)) # first iteration
    # s0_guesses.append(opt_results[0][0])

    # for i in range(1,tr):
    #     #opt_result = minimize(p_correct_vals[i],s0_guesses[-1])
    #     opt_results.append(opt_result)
    #     s0_guesses.append(opt_result[0])

    ''' Parallel computing '''
    ray.init()
    futures = [minimize.remote(p_correct_vals[i], s0_vals[i]) for i in range(20)]
    opt_results = ray.get(futures)
    
    vec = [opt_results[i][0] for i in range(20)]
    val = [opt_results[i][1] for i in range(20)]

    np.savez('results5_0921.npz',slopes=vec,bound_results=val)

    # Plot
    # Load x-axis and POMDP data
    opt_data = np.load("results/lava_problem_optimal_results.npz")
    opt_values = opt_data['opt_values']

    tight_data = np.load("tightest_bounds.npz")
    tight_values = tight_data['bounds']

    fig, ax = plt.subplots()
    ax.plot(p_correct_vals, opt_values, '*--', label='POMDP', linewidth=0.5)
    ax.plot(p_correct_vals, tight_values, 'o--', label='Tightest', linewidth=1)
    ax.plot(p_correct_vals, val,'o--', label='PL', linewidth=1)

    plt.xlabel('$p_{correct}$', fontsize=15)
    plt.ylabel('Cumulative reward', fontsize=15)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylim([0, 5.01])
    plt.savefig('PL_n=5_0921.svg', dpi=200)

#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  
