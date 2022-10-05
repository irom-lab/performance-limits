'''
Code for computing bound on expected rewards. Implements scaling based on rewards (not
expected rewards). Computes bound for each horizon and takes best bound. 
'''

import numpy as np
import scipy.optimize as optimize
import IPython as ipy
import cvxpy as cvx
import argparse	


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
    prob.solve(verbose=False, solver=cvx.SCS) # solver=cvx.MOSEK
    
    return p_bernoulli.value[0] 

def bound_scaled(ab, I, R_plus_R_next, px_u_seq): 
	'''
	Compute bound with scaling of rewards.
	'''

	# Get a and b
	a = ab[0]
	b = ab[1]

	R_u = px_u_seq @ (a*R_plus_R_next-b) # Expected reward-to-go of each action (scaled to be in [0,1])
	Rt_perp = np.max(R_u)
	Rt_perp = min(1-1e-4, max(1e-4, Rt_perp))

	# Compute bound
	bound = kl_inverse(Rt_perp, I)

	# Scale it back
	bound = (1/a)*(b+bound)

	return bound

def bound_opt_scaling(I, R_plus_R_next, min_R, max_R, px_u_seq, initial_guess):
	'''
	Compute optimal scaling for bound. 
	'''

	# Constraints
	A = np.array([[min_R,-1], [max_R,-1],[1,0]])
	lb = np.array([1e-5, 1e-5, 1e-3])
	ub = np.array([1.0-1e-5, 1.0-1e-5, np.inf])
	cons = optimize.LinearConstraint(A, lb, ub)

	res = optimize.minimize(bound_scaled, initial_guess, args=(I, R_plus_R_next, px_u_seq), constraints=cons)

	if (not res.success): # If optimizer failed, try different optimizer (slower, but sometimes better)
		res = optimize.minimize(bound_scaled, initial_guess, args=(I, R_plus_R_next, px_u_seq), constraints=cons, method='trust-constr')

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

def compute_bound_H(nx, nu, ny, H, p0, px, px_x, py_x, R, u_seqs_all):
	'''
	Compute bound up to horizon H.
	'''

	##############################################################################
	# Now calculate the bound itself 
	R_bound_next = np.zeros(tuple(H*[nu])) # array with shape (nu, nu, ..., nu)
	for t in np.flip(range(1,H)): # H-1,H-2,...,1
		R_bound = np.zeros(tuple(t*[nu]))
		for i in range(len(u_seqs_all[t-1])): # for all sequences of control inputs with length t-1
			
			##############################################################################
			# Compute R_plus_R_next
			u_seq = u_seqs_all[t-1][i]
			px_u_seq = px[t][tuple(u_seq)][:]
			R_plus_R_next = R+R_bound_next[tuple(u_seq)][:]


			##############################################################################
			# Now compute mutual information
			pyx_u_seq = py_x*px_u_seq[None,:]  # pyx(i,j) is probability of measurement i and state j (given u_seq)
			py_u_seq = np.sum(pyx_u_seq, 1)

			I = 0.0
			for ii in range(0,ny):
				for jj in range(0,nx):
					if (np.abs(pyx_u_seq[ii,jj]) > 1e-5):
						I = I + pyx_u_seq[ii,jj]*np.log(pyx_u_seq[ii,jj]/(py_u_seq[ii]*px_u_seq[jj]))


			##############################################################################
			# Set maximum and minimum rewards from this t onwards

			max_R = np.max(R)*(H-t)
			min_R = np.min(R)*(H-t)

			##############################################################################

			if (np.abs(min_R - max_R) < 1e-5):
				# If min and max reward are the same, no need to compute bound
				bound = max_R
			else:
				##############################################################################
				# Compute bound

				# Scale rewards to be in [0,1]
				a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 
				ab = [a_scale, b_scale]
				# Compute bound
				bound = bound_scaled(ab, I, R_plus_R_next, px_u_seq)

				# # Version with optimal scaling
				# initial_guess = np.array([a_scale, b_scale])
				# bound = bound_opt_scaling(I, R_plus_R_next, min_R, max_R, px_u_seq, initial_guess)

				# Tighten R_bound with maximum R given px_u_seq
				bound = min(max_R, bound)


			# Final bound for this u_seq
			R_bound[tuple(u_seq)] = bound

		# Set R_bound_next to R_bound
		R_bound_next = R_bound


	##############################################################################
	# Now compute bound for t=0

	# Compute mutual information
	px_0 = p0 # Probability of x_0
	pyx_0 = py_x*px_0[None,:]  # pyx(i,j) is probability of measurement i and state j 
	py_0 = np.sum(pyx_0, 1)

	I = 0.0
	for ii in range(0,ny):
		for jj in range(0,nx):
			if (np.abs(pyx_0[ii,jj]) > 1e-5):
				I = I + pyx_0[ii,jj]*np.log(pyx_0[ii,jj]/(py_0[ii]*px_0[jj]))

	##############################################################################
	# Compute R + R_next
	R_plus_R_next = R+R_bound_next

	# Set max and min rewards
		
	max_R = np.max(R)*H
	min_R = np.min(R)*H

	if (np.abs(min_R - max_R) < 1e-5):
		bound = max_R
	else:

		# Scale rewards to be in [0,1], compute bound, and then rescale
		a_scale = 1/(max_R-min_R); b_scale = a_scale*min_R 
		ab = [a_scale, b_scale]
		# Compute bound
		bound = bound_scaled(ab, I, R_plus_R_next, px_0)

		# # Version with optimal scaling
		# initial_guess = np.array([a_scale, b_scale])
		# bound = bound_opt_scaling(I, R_plus_R_next, min_R, max_R, px_0, initial_guess)

		# Tighten bound with maximum R given px_0
		bound = min(max_R, bound)

	return bound

def compute_bound(nx, nu, ny, T, p0, px_x, py_x, R, R0_expected):
	'''
	Main function for computing upper bound on expected reward.
	'''

	##############################################################################
	# First compute probabilities p(x_t|u_0, u_0,...,u_{t-1})
	px = T*[None]
	px[0] = p0
	u_seqs = [[u] for u in range(nu)] # Sequences of length 1 (list of lists)
	u_seqs_all = T*[None]
	u_seqs_all[0] = u_seqs
	for t in range(1,T): # 1,2,...,T-1
		px[t] = np.zeros(tuple(t*[nu]+[nx]))
		new_seqs = []
		for us in u_seqs:
			if t == 1:
				px[t][tuple(us)][:] = np.sum(px_x[:,:,us[-1]]*px[t-1][None,:],1)
			else:
				px[t][tuple(us)][:] = np.sum(px_x[:,:,us[-1]]*px[t-1][tuple(us[0:-1])][:][None,:],1)
			for u in range(nu):
				new_seqs = new_seqs + [us+[u]]
		u_seqs_all[t] = new_seqs
		u_seqs = new_seqs

	##############################################################################

	##### Compute bound ##########################################################
	bound_best = np.inf # Keeps track of upper bounds using each horizon
	for H in range(1,T+1): # for horizon H = 1,...T
		# Compute bound for different horizons. Use bound from H to get estimate for bound with H+1.
		bound = compute_bound_H(nx, nu, ny, H, p0, px, px_x, py_x, R, u_seqs_all)
		bound = bound + (T-H) # Add possible rewards from H to T
		bound = bound + R0_expected # Add in expected reward from initial state
	
		if bound < bound_best:
			bound_best = bound

	##############################################################################

	return bound_best