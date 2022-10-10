import numpy as np
from bound_rewards import compute_bound
from bound_rewards_Fano import compute_Fano
from lava_one_step import main as lava_problem
import matplotlib.pyplot as plt

def main(raw_args=None):
    # One-step Lava Problem: Our bounds
    p_correct_vals = np.linspace(0.01, 0.99, 20) 
    reward_xs = np.array([0.1]) 
    reward_x = reward_xs[0]
    bounds = []
    for p_correct in p_correct_vals:
        nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
        bound_inverse = compute_bound(nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
        bounds.append(bound_inverse)

    # One-step Lava Problem: Fano bounds
    bounds_Fano = []
    for p_correct in p_correct_vals:
        nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
        Fano_b = compute_Fano(nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
        bounds_Fano.append(Fano_b)

    # One-step Lava Problem: POMDP
    # Brute-force computation
    r_right = np.array([0.1,1,1,0.0])
    r_left = np.array([0.1,0.1,1,1,0])

    def V1_b_cond_zi(p_c, i):
        V_right = (r_right[i-1]*p_c+np.sum(np.delete(r_right,i-1))*(1-p_c)/4)*p0[0]
        V_left = (r_left[i-1]*p_c+np.sum(np.delete(r_left,i-1))*(1-p_c)/4)*p0[0]
        return max(V_right, V_left)

    opt_values = []
    for p_correct in p_correct_vals:
        Vc = 0
        for i in range(5):
            V1_i = V1_b_cond_zi(p_correct,i)
            Vc += V1_i
        # print(Vc)
        opt_values.append(Vc)

    # Save results
    np.savez('results/compare_Fano.npz', p_correct_vals = p_correct_vals, our_bounds = bounds, Fano_bounds = bounds_Fano, POMDP_bounds = opt_values)

    # Plot results
    plt.plot(p_correct_vals, bounds, 'o--', label='Our Bounds', linewidth = 1)
    plt.plot(p_correct_vals, bounds_Fano, 'o--', label='Fano Bounds', linewidth = 1)
    plt.plot(p_correct_vals, opt_values, '*--', label='POMDP', linewidth = 0.5)
    plt.xlabel('$p_{correct}$', fontsize=15)
    plt.ylabel('One-step reward', fontsize=15)
    plt.legend(fontsize=12)
    plt.savefig('plots/compare_Fano.svg',dpi=200) 

#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  