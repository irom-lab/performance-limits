import numpy as np
import scipy.optimize as optimize
import IPython as ipy
import cvxpy as cvx
import argparse	

import gym
import gym_ball_catching
import scipy.stats as stats

from multiprocessing import Pool

def run_trial(inputs):

	# Unpack inputs 
	noise_scale = inputs[0][0]
	u_seqs_all = inputs[1][0]

	##### Environment #####
	env = gym.make("BallCatching-v1", noise_scale=noise_scale)
	nx = env.nx # Dimension of states
	ny = env.ny # Dimension of obs
	nu = env.nu # Number of control inputs
	T = env.T

	# Reset environment
	obs = env.reset()

	# Add initial expected reward (to match theoretical formulation)
	reward_episode = env.R0_expected # Initial reward env.R0_expected to be consistent with theoretical formulation in paper

	# Run controller
	for t in range(T):
		
		# Compute control input
		u_seqs_t = u_seqs_all[T-t-1]
		action = env.MPC(u_seqs_t)

		# Update state and get new observation
		obs, reward, _, _ = env.step(action)
		reward_episode += reward

	return reward_episode



def main(raw_args=None):

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--noise_scale", type=float, default=1.0, help="Scale of measurement noise (default: 1.0)")
	parser.add_argument("--num_parallel", type=int, default=16, help="Parallel threads (default: 16)")
	parser.add_argument("--num_runs", type=int, default=100, help="Number of rollouts (default: 100)")


	args = parser.parse_args(raw_args)
	noise_scale = args.noise_scale
	num_parallel = args.num_parallel
	num_runs = args.num_runs

	##### Environment #####
	env = gym.make("BallCatching-v1")
	nx = env.nx # Dimension of states
	ny = env.ny # Dimension of obs
	nu = env.nu # Number of control inputs
	T = env.T

	##### Compute all sequences of inputs of length T #####
	u_seqs = [[u] for u in range(nu)] # Sequences of length 1 (list of lists)
	u_seqs_all = T*[None]
	u_seqs_all[0] = u_seqs

	for t in range(1,T): # 1,2,...,T-1
		# print("t: ", t)
		new_seqs = []
		for us in u_seqs:
			for u in range(nu):
				u_seq = us+[u]
				new_seqs = new_seqs + [u_seq]

		u_seqs_all[t] = new_seqs
		u_seqs = new_seqs

	##### Run trials #####
	print("Running MPC with different initial conditions...")
	with Pool(num_parallel) as p:
		# Pack inputs to use with map
		inputs = num_runs*[[[noise_scale], [u_seqs_all]]]
		rewards_all = p.map(run_trial, inputs)

	print("Done.")
	print("Mean reward: ", np.mean(rewards_all))

	rewards_mean = np.mean(rewards_all)

	return rewards_mean


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 