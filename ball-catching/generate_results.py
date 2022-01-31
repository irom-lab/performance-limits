import numpy as np
import IPython as ipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from bound_rewards import main as bound_rewards
import gym
import gym_ball_catching
from run_MPC import main as run_MPC


#########################
# Noise scales to generate results for
noise_scales = [0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] 

# Number of seeds for MPC
num_seeds = 5

#########################
# Initialize results vectors
bounds = [None]*len(noise_scales)
MPC_rewards_all = [None]*len(noise_scales)
MPC_reward_means = [None]*len(noise_scales)
MPC_reward_std_devs = [None]*len(noise_scales)

for i in range(len(noise_scales)):

	# Noise scale
	noise_scale = noise_scales[i]
	print("Generating results for noise scale: ", noise_scale)

	# Compute bound
	print("Computing bound...")
	bound = bound_rewards(['--noise_scale', str(noise_scale)])
	bounds[i] = bound
	print("Done.")

	# Run MPC
	print("Running MPC...")
	MPC_rewards_all[i] = [None]*num_seeds
	for seed in range(num_seeds):
		print("seed: ", seed)
		rewards_mean = run_MPC(['--noise_scale', str(noise_scale), '--num_runs', str(100), '--num_parallel', str(16)])
		MPC_rewards_all[i][seed] = rewards_mean
	
	MPC_reward_means[i] = np.mean(MPC_rewards_all[i])
	MPC_reward_std_devs[i] = np.std(MPC_rewards_all[i])
	print("Done.")



# Save results
np.savez("ball_catching_results.npz", noise_scales=noise_scales, bounds=bounds, MPC_reward_means=MPC_reward_means, MPC_reward_std_devs=MPC_reward_std_devs, MPC_rewards_all=MPC_rewards_all)