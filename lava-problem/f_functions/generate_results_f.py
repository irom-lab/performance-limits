"""
Code for sweeping over different sensor noise parameters and generating results.
Example usage: 
python generate_results.py --problem lava_problem
OR 
python generate_results.py --problem two_lavas_problem
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import IPython as ipy
from bound_rewards_f import compute_bound as compute_bound
from f_func import *
import time
import os
import mosek

def main(raw_args=None):

	# Parse problem arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--problem", type=str, default="lava_problem", help="choose problem: lava_problem or two_lavas_problem (default: lava_problem)")
	parser.add_argument("--function", type=str, default="Kullback_Leibler", help="choose function: see list of available functions in README (default: KL-Divergence)")
	parser.add_argument("--POMDP", type=str, default="N", help="Optimal value? Y/N (default: N)")


	args = parser.parse_args(raw_args)
	problem = args.problem
	func = args.function
	opt_or_not = args.POMDP

	##########################################################################
	# Load code associated with chosen problem
	if (problem == 'lava_problem'):
		from lava_problem import main as lava_problem
		from lava_problem_pomdp import main as lava_problem_pomdp
	elif (problem == 'two_lavas_problem'):
		from two_lavas_problem import main as lava_problem
		from two_lavas_problem_pomdp import main as lava_problem_pomdp
	else:
		raise Exception("Problem not recognized.")
	##########################################################################

	# Different probabilities of sensor reporting correct state
	p_correct_vals = np.linspace(0.01, 0.99, 20) 

	if opt_or_not == 'N':
		# Parse function arguments
		f = eval(func)

		# Different rewards for being in a non-lava/goal state
		reward_xs = np.array([0.1]) 

		# Figure stuff
		fig, ax = plt.subplots()
		colors = cm.jet(np.linspace(1, 0, len(reward_xs)))
		for reward_x, color in zip(reward_xs, colors):

			bounds = []
			bounds_DV = []
			bounds_pinsker = []
			bounds_f_inverse = []
			opt_values = []

			for p_correct in p_correct_vals:

				################################################################################
				# Lava problem
				nx, nu, ny, T, p0, px_x, py_x, R, R0_expected = lava_problem(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
				################################################################################


				################################################################################
				# Compute f-inverse bound
				# t_start = time.time()
				bound_f_inverse = compute_bound(f, nx, nu, ny, T, p0, px_x, py_x, R, R0_expected) 
				# t_end = time.time()
				bounds_f_inverse.append(bound_f_inverse)
				print("Bound: ", bound_f_inverse)
				# print("Bound computation time: ", t_end - t_start)
				################################################################################

				# Save results
				path = 'results'
				filename = problem+"_"+func+"_results_f.npz"
				np.savez(os.path.join(path,filename), reward_xs=reward_xs, p_correct_vals=p_correct_vals, bounds=bounds_f_inverse)

	elif opt_or_not == 'Y':
		# Different rewards for being in a non-lava/goal state
		reward_xs = np.array([0.1]) 

		# Figure stuff
		fig, ax = plt.subplots()
		colors = cm.jet(np.linspace(1, 0, len(reward_xs)))
		for reward_x, color in zip(reward_xs, colors):

			bounds = []
			bounds_DV = []
			bounds_pinsker = []
			bounds_f_inverse = []
			opt_values = []
			for p_correct in p_correct_vals:
				###############################################################################
				# Compute optimal value from POMDP solution
				# t_start = time.time()
				opt_value = lava_problem_pomdp(['--p_correct', str(p_correct), '--reward_x', str(reward_x)])
				# t_end = time.time()
				opt_values.append(opt_value)
				print("POMDP: ", opt_value)
				# print("POMDP computation time: ", t_end - t_start)
				###############################################################################

				# Save results
				path = 'results'
				filename = problem+"_optimal_results.npz",
				np.savez(problem+"_optimal_results.npz", reward_xs=reward_xs, p_correct_vals=p_correct_vals, opt_values=opt_values)

	else:
		raise Exception('Choice not recognized, compute optimal values?')
	print(" ")





#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  