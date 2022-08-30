import numpy as np
from bound_rewards_l import *
from lava_problem import main as lava_problem
import scipy.optimize as optimize
from dask.distributed import Client
import matplotlib.pyplot as plt
import IPython as ipy

def main(raw_args=None):
    n = 2 # 50 slices on interval [0,1]
    svec = np.zeros(n) # set initial guesses to be 0


    # Setups for lava problem
    p_correct_vals = np.linspace(0.01, 0.99, 20) 
    reward_xs = np.array([0.1]) 
    reward_x = reward_xs[0]

    # Constraint: from 1 to 0 backwards, each slope has to be smaller
    # So elements in the slope vector need to be in decreasing order
    def con(svec):
        out = []
        for i in range(len(svec)-1):
            out.append(svec[i]-svec[i+1])
        return out
        
    cons = [{'type':'ineq','fun': con}]

    # Print optimization progress
    def callbackF(svec,status):
        print (svec,bounds(svec))

    def bounds(svec, p_correct = p_correct_vals[0]):
        nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
        bound_f_inverse = compute_bound(l,n,svec, nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
        return bound_f_inverse

    def minimize(p_correct):
        res = optimize.minimize(bounds,svec,method = 'trust-constr', constraints = cons, callback = callbackF, args = (p_correct), options={'disp':True})
        return[res.x, res.fun]
    
    client = Client(n_workers = 4)
    args = [(i) for i in p_correct_vals]
    futures = []
    for p in args:
        future = client.submit(minimize,p)
        futures.append(future)
    results = client.gather(futures)
    vals = [results[i,1] for i in range(20)]
    np.savetxt('results.txt',results)
    client.close()
    # Plot
    # Load x-axis and POMDP data
    opt_data = np.load("results/lava_problem_optimal_results.npz")
    p_correct_vals = opt_data['p_correct_vals']
    opt_values = opt_data['opt_values']

    fig, ax = plt.subplots()
    ax.plot(p_correct_vals, opt_values, '*--', label='POMDP', linewidth=0.5)
    ax.plot(p_correct_vals,vals,'o--', label='Tightest Bound', linewidth=1)

    plt.xlabel('$p_{correct}$', fontsize=15)
    plt.ylabel('Cumulative reward', fontsize=15)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylim([0, 5.01])
    plt.savefig('PL_n=5.svg', dpi=200)
    plt.show() 

#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  