'''
Main version of code for computing bound on expected rewards for double integrator system. 
'''


import numpy as np
import scipy.optimize as optimize
import IPython as ipy
import cvxpy as cvx
import argparse	

import gym
import gym_ball_catching
import scipy.stats as stats

def kl_inverse(q, c):
    '''
    Compute kl inverse using Relative Entropy Programming
    Solves: sup p in [0,1] with KL(p,q) <= c
    '''   
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(p_bernoulli,q_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cvx.MOSEK) # solver=cvx.MOSEK or solver=cvx.SCS

    if prob.status not in ["optimal"]: # ["infeasible", "unbounded"]:
    	print("Optimization error in KL inverse computation.")
    	ipy.embed()
    
    return p_bernoulli.value[0] 


def bound_scaled(ab, I, R_plus_R_next): 
	'''
	Compute bound with scaling of rewards.
	'''

	# Get a and b
	a = ab[0]
	b = ab[1]

	# Compute R_u: a*R_plus_R_next-b
	R_u = a*R_plus_R_next-b # Expected reward-to-go of each action (scaled to be in [0,1])

	# Take maximum
	Rt_perp = np.max(R_u)
	Rt_perp = min(1-1e-4, max(1e-4, Rt_perp))

	# Compute bound
	bound = kl_inverse(Rt_perp, I)

	# Scale it back
	bound = (1/a)*(b+bound)

	return bound

def bound_opt_scaling(I, R_plus_R_next, min_R, max_R, initial_guess):
	'''
	Compute optimal scaling for bound. 
	'''

	# Constraints
	A = np.array([[min_R,-1], [max_R,-1],[1,0]])
	lb = np.array([1e-5, 1e-5, 1e-3])
	ub = np.array([1.0-1e-5, 1.0-1e-5, np.inf])
	cons = optimize.LinearConstraint(A, lb, ub)

	res = optimize.minimize(bound_scaled, initial_guess, args=(I, R_plus_R_next), constraints=cons)

	if (not res.success): # If optimizer failed, try different optimizer (slower, but sometimes better)
		res = optimize.minimize(bound_scaled, initial_guess, args=(I, R_plus_R_next), constraints=cons, method='trust-constr')

	if (not res.success):
		print("Optimization error.")
		ipy.embed()

	bound = res.fun

	return bound

def kl_inverse_bound(I, R_plus_R_next, px_u_seq): 
	'''Version of bound with no scaling'''

	R_u = px_u_seq @ (R_plus_R_next) # Expected reward-to-go of each action (scaled to be in [0,1])
	Rt_perp = np.max(R_u)
	Rt_perp = min(1-1e-4, max(1e-4, Rt_perp))

	# Compute bound
	bound = kl_inverse(Rt_perp, I)

	return bound

def compute_bound_H(H, env, u_seqs_all, R_bounds_all_prev):

	##############################################################################
	# Parameters
	nx = env.nx # Dimension of states
	ny = env.ny # Dimension of obs
	nu = env.nu # Number of control inputs
	TOL = 1e-3 # 1e-3

	##############################################################################
	# Initialize things
	R_bound_next = np.zeros(tuple(H*[nu])) # array with shape (nu, nu, ..., nu)
	R_bounds_all = H*[None]
	for t in np.flip(range(1,H)): # H-1,H-2,...,1
		R_bound = np.zeros(tuple(t*[nu]))
		for i in range(len(u_seqs_all[t-1])): # for all sequences of control inputs up to t-1
			
			##############################################################################
			# Get u sequence
			u_seq = u_seqs_all[t-1][i]

			##############################################################################
			# Set maximum and minimum rewards from this t onwards

			min_R = 0.0
			max_R = 1.0*(H-t)

			##############################################################################
			# Compute bound


			############################
			# Compute E[r_t(x_t, u)] for each u (where exp. is w.r.t. x_t~p0,u_{0:t-1})
			r_xu = np.zeros(nu)
			for u in range(nu):
				u_seq_w_u = u_seq + [u]
				mu_x, cov_x = env.state_distribution(u_seq_w_u)
				r_xu[u] = env.compute_expected_reward(mu_x, cov_x)

			# Now compute R + R_bound_next
			R_plus_R_next = r_xu+R_bound_next[tuple(u_seq)][:]


			############################
			# Now compute mutual information
			mu_x, cov_x = env.state_distribution(u_seq)
			I = env.compute_MI(mu_x, cov_x)

			# Scale rewards to be in [0,1]
			a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 

			ab = [a_scale, b_scale]

			# Compute bound
			bound = bound_scaled(ab, I, R_plus_R_next)

			# # Version where we optimize scaling
			# a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 
			# initial_guess = np.array([a_scale, b_scale])
			# bound = bound_opt_scaling(I, R_plus_R_next, min_R, max_R, initial_guess)

			# Tighten R_bound with maximum R given px_u_seq
			bound = min(max_R, bound)

			if t < (H-1): # tighten with bound from previous H
				bound = min(bound, R_bounds_all_prev[t][tuple(u_seq)] + 1)
					

			# Final bound for this u_seq
			R_bound[tuple(u_seq)] = bound

		# Set R_bound_next to R_bound
		R_bound_next = R_bound
		R_bounds_all[t] = R_bound


	##############################################################################
	# Now compute bound for t=0

	# Compute R + R_next
	r_xu = np.zeros(nu)
	for u in range(nu):
		mu_x, cov_x = env.state_distribution([u])
		r_xu[u] = env.compute_expected_reward(mu_x, cov_x)
	R_plus_R_next = r_xu+R_bound_next

	
	max_R = H
	min_R = 0.0

	# Compute mutual information
	I = env.compute_MI(env.x0_mu, env.x0_cov)

	# Scale rewards to be in [0,1], compute bound, and then rescale
	a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 

	ab = [a_scale, b_scale]
	# Compute bound
	bound = bound_scaled(ab, I, R_plus_R_next)


	# # Version where we optimize scaling 
	# a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 
	# initial_guess = np.array([a_scale, b_scale])
	# bound = bound_opt_scaling(I, R_plus_R_next, min_R, max_R, initial_guess)

	# Tighten bound with maximum R given px_0
	bound = min(max_R, bound)

	if (H > 1): # tighten with bound from previous H
		bound = min(bound, R_bounds_all_prev[0]+1)

	R_bounds_all[0] = bound

	return bound, R_bounds_all


def main(raw_args=None):
	'''
	Main function for computing upper bound on expected reward.
	'''

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--noise_scale", type=float, help="scaling of observation noise")

	args = parser.parse_args(raw_args)
	noise_scale = args.noise_scale

	##### Parameters #####
	env = gym.make("BallCatching-v1", noise_scale=noise_scale)
	nx = env.nx # Dimension of states
	ny = env.ny # Dimension of obs
	nu = env.nu # Number of control inputs

	T = env.T  # Time horizon

	# Switch off Kalman filtering (this is only used for control)
	env.kalman_filter = False

	##### Compute all sequences of inputs of length T #####
	u_seqs = [[u] for u in range(nu)] # Sequences of length 1 (list of lists)
	u_seqs_all = T*[None]
	u_seqs_all[0] = u_seqs


	for t in range(1,T): # 1,2,...,T-1
		print("t: ", t)
		new_seqs = []
		for us in u_seqs:
			for u in range(nu):
				u_seq = us+[u]
				new_seqs = new_seqs + [u_seq]

		u_seqs_all[t] = new_seqs
		u_seqs = new_seqs

	##### Compute bound ##########################################################
	bound_best = np.inf # Keeps track of upper bounds using each horizon
	R_bounds_all_prev = []
	for H in range(1,T+1): # for horizon H = 1,...T
		# Compute bound for different horizons. 
		bound, R_bounds_all_prev = compute_bound_H(H, env, u_seqs_all, R_bounds_all_prev)
		bound = bound + (T-H) # Add possible rewards from H to T
		
		R0_expected = env.compute_expected_reward([env.x0_mu], env.x0_cov)
		bound = bound + R0_expected # Add in expected reward from initial state
	
		if bound < bound_best:
			bound_best = bound

		print("H: ", H)
		print("bound: ", bound_best)
	##############################################################################


	return bound_best

