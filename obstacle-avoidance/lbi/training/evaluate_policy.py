import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython as ipy

from lbi.sensing.lidar import *
from lbi.envs.envs import *
import lbi.geom as geom
from lbi.primitives.utils import *
import matplotlib.pyplot as plt
from lbi.sensing.types import state_to_workspace
from lbi.training.policy_networks import MLPSoftmax
import json
import argparse	


def main(raw_args=None):


	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")

	args = parser.parse_args(raw_args)
	verbose = args.verbose

	###################################################################
	# Load params from json file
	with open("params.json", "r") as read_file:
	    params = json.load(read_file)


	# Define workspace
	workspace_x_lims = params["workspace_x_lims"]
	workspace_y_lims = params["workspace_y_lims"]

	# Robot state
	robot_location = params["robot_location"]
	robot_state = geom.Point(robot_location[0], robot_location[1])

	# Limits for obstacles centers
	obs_x_lims = params["obs_x_lims"]
	obs_y_lims = params["obs_y_lims"]
	num_obs = params["num_obs"]
	obs_radius = params["obs_radius"]

	# Define sensor
	num_rays = params["num_rays"]
	lidar_angle_range = params["lidar_angle_range"]
	lidar_noise_std = params["lidar_noise_std"]
	lidar_failure_rate = params["lidar_failure_rate"]
	lidar_max_distance = params["lidar_max_distance"]

	sensor = lidar(num_rays, lidar_angle_range,
	                local_noise_std=lidar_noise_std,
	                failure_rate=lidar_failure_rate,
	                max_distance=lidar_max_distance)

	# Load motion primitives
	prim_lib = np.load('lbi/primitives/primitive_library.npy')
	num_primitives = np.shape(prim_lib)[1]

	# Seed
	seed = np.random.randint(1000) # Randomize
	key = jax.random.PRNGKey(seed)
	key, subkey = jax.random.split(key)
	##################################################

	# Load policy
	num_in = num_rays
	num_out = num_primitives
	model = MLPSoftmax(num_in, num_out)

	model.load_state_dict(torch.load("lbi/training/models/trained_model"))
	model.eval()

	num_test_envs = 5000
	rewards_all = np.zeros(num_test_envs)

	# Seed
	seed = np.random.randint(1000) # Randomize
	key = jax.random.PRNGKey(seed)
	subkeys = jax.random.split(key, num_test_envs)

	for i in range(num_test_envs):

		if verbose and ((i+1) % 200 == 0):
			print("Test env: ", i+1, " of ", num_test_envs)
		

		# Generate random environment
		# env = generate_random_env(obs_x_lims, obs_y_lims, num_obs)
		env = random_traversable_env(obs_x_lims, obs_y_lims, num_obs, obs_radius, prim_lib)
		workspace = state_to_workspace(env, obs_radius, workspace_x_lims, workspace_y_lims)

		# Generate LIDAR measurement
		lidar_dists = sample_lidar(subkeys[i], robot_state, workspace, sensor)
		lidar_dists = torch.Tensor(np.array(lidar_dists))
		lidar_dists = lidar_dists.reshape([1,num_rays])

		# Run policy and get primitive
		scores = model(lidar_dists)
		pred_primitive_ind = scores.argmax().item()
		pred_primitive = prim_lib[:,pred_primitive_ind,:]

		# Check collision
		collision = check_primitive_collision(env, obs_radius, pred_primitive)

		# Reward
		reward = 1 - collision 

		# Store
		rewards_all[i] = reward

	mean_rewards = np.mean(rewards_all)

	if verbose:
		print(" ")
		print("Mean test reward: ", mean_rewards)

	return mean_rewards


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 




