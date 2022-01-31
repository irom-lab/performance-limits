"""
Lava Problem

#####################
#   #   # G #   # L #
#####################

"""

import numpy as np
import IPython as ipy
import argparse

def main(raw_args=None):

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--p_correct", type=float, default=0.2, help="probability of correct measurement (default: 0.2)")
	parser.add_argument("--reward_x", type=float, default=0.5, help="reward of being in non-goal/lava state (default: 0.5)")


	args = parser.parse_args(raw_args)
	p_correct = args.p_correct
	reward_x = args.reward_x


	################## Define states and actions ##################
	nx = 5 # Number of states
	nu = 2 # Number of control inputs
	xs = np.arange(0,nx) # States
	us = np.array([-1, +1]) # Actions
	T = 5-1 # 5 # Time horizon


	################## Define reward matrix #########################
	R = np.zeros((nx,nu)) # r_t(x,u)
	lava = nx-1 # Lava state (right most state)
	goal = 2  # Goal state
	r_xs = reward_x*np.ones(nx) # Costs of being at any given (non-lava/goal) state
	r_xs[lava] = 0.000001 # Reward of being in lava (right most state)
	r_xs[goal] = 0.999999 # Reward of being in goal

	# Define initial distribution over states 
	p0 = np.ones(nx)
	p0[lava] = 0.0 # You cannot start in the lava
	p0 = p0/sum(p0)

	# r_t[x,u]
	for x in range(nx):
		for u in range(nu):
			if x == goal:
				x_next = x # goal is absorbing
			elif x == lava:
				x_next = x # lava is absorbing
			elif u == 0: # left
				x_next = max(x-1,0)
			elif u == 1: # right
				x_next = min(x+1,nx-1)

			# Assign reward
			R[x,u] = r_xs[x_next]

	# Now, initial expected reward
	R0_expected = r_xs @ p0


	########## Define sensor model ###############################
	ny = nx # Number of sensor measurements
	py_x = np.zeros((ny, nx))
	# p_correct = argument 1/ny # Probability that sensor reports correct state 
	# Conditional probabilities: py_x[i,j] is probability of measurement i given state j
	for i in range(ny):
		for j in range(nx):
			if (i == j):
				py_x[i,j] = p_correct
			else:
				py_x[i,j] = (1-p_correct)/(ny-1)


	########## Define dynamics model #############################
	px_x = np.zeros((nx,nx,nu))
	# Conditional probabilities: px_x[i,j,k] is probability of x_t+1 = i given x_t = j, u_t = k
	for i in range(nx):
		for j in range(nx):
			if (j==goal and i == goal): # the goal is absorbing
				px_x[i,j,0] = 1.0
				px_x[i,j,1] = 1.0
				continue
			if (j==lava and i == lava): # the lava is absorbing
				px_x[i,j,0] = 1.0
				px_x[i,j,1] = 1.0
				continue

			if (max(j-1,0)==i and (j is not goal) and (j is not lava)): # i is left of j
				px_x[i,j,0] = 1.0
			if (min(j+1,nx-1)==i and (j is not goal) and (j is not lava)): # i is right of j
				px_x[i,j,1] = 1.0

	return nx, nu, ny, T, p0, px_x, py_x, R, R0_expected



#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  
