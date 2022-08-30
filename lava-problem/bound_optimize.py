import numpy as np
from bound_rewards_g import compute_bound as compute_bound
from f_func import *
from lava_problem import main as lava_problem
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import IPython as ipy

def main(raw_args=None):


    # Combined function
    def g(x,c,func_list):
        # c and func_list should have matching dimensions
        y = 0
        for i in range (len(func_list)):
            y += c[i]*eval(func_list[i])(x)
        return y

    # List of functions to find the optimized combinations of
    func_list = [ 'Kullback_Leibler', 'Total_Variation','Chi_Squared','Negative_Log']

    # Initial guesses of function coefficients: g(x) = c.dot f(x)
    cx = np.tile(np.array([0,1,0,0]),(7,1))
    cy = np.tile(np.array([1e-5,1e-5,1e-5,1]),(13,1))
    c0 = np.concatenate((cx,cy))

    # Constraints
    def con(c_vec):
        return np.linalg.norm(c_vec)-1 #norm = 1
    def con1(c_vec):
        return list(c_vec-1e-5) #coefficients > 0 (important to meet DCP requirements)
    def con2(c_vec):
        return list(1-c_vec) #coefficients <= 1
    cons = [{'type':'ineq','fun': con1},{'type':'eq','fun':con},{'type':'ineq','fun': con2}]

    # Print optimization progress
    def callbackF(Xi):
        print ('{0: 3.2f}{1: 3.2f}{2: 3.2f}{3: 3.2f}{4: 3.6f}'.format(Xi[0],Xi[1],Xi[2],Xi[3],bounds(Xi)))

    # Different probabilities of sensor reporting correct state
    p_correct_vals = np.linspace(0.01, 0.99, 20) 
    # Different rewards for being in a non-lava/goal state
    reward_x = np.array([0.1])[0]

    # Loop
    results=[]
    coefs = []
    vals = []
    for ind in list(range(20)):
        p_correct = p_correct_vals[ind]
        def bounds(c_vec):
            nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
            bound_f_inverse = compute_bound(g,c_vec,func_list, nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
            return bound_f_inverse

        res = optimize.minimize(bounds,c0[ind], constraints = cons, callback=callbackF, options={'disp':True})
        results.append(res)
        coefs.append(res.x)
        vals.append(res.fun)

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
    plt.savefig('Tight_Bounds_vs_KL_5.svg', dpi=200)
    plt.show() 

#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  