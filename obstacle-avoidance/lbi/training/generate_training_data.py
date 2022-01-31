'''
Code for generating training data for policy training.
'''

from lbi.sensing.lidar import *
from lbi.envs.envs import *
import lbi.geom as geom
from lbi.primitives.utils import *
import numpy as np
import matplotlib.pyplot as plt
import IPython as ipy
import cvxpy as cvx
import argparse
import jax.random as rnd
from lbi.sensing.types import state_to_workspace
import json

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
    ##################################################


    # Number of training environments
    num_training_envs = 5000 # 5000

    # Seed
    seed = np.random.randint(1000) # Randomize
    key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, num_training_envs)

    # This will store inputs (i.e., LIDAR measurements)
    lidar_all_training = np.zeros((num_training_envs, num_rays))

    # This will store labels
    labels_all_training = np.zeros((num_training_envs, num_primitives))

    for i in range(num_training_envs):

        if verbose and ((i+1) % 200 == 0):
            print("Training env: ", i+1, " of ", num_training_envs)

        # Generate random environment
        # env = generate_random_env(obs_x_lims, obs_y_lims, num_obs)
        env = random_traversable_env(obs_x_lims, obs_y_lims, num_obs, obs_radius, prim_lib)
        workspace = state_to_workspace(env, obs_radius, workspace_x_lims, workspace_y_lims)

        # Generate LIDAR measurement
        lidar_dists = sample_lidar(subkeys[i], robot_state, workspace, sensor)

        # Store in training data
        lidar_all_training[i,:] = lidar_dists

        # # Generate label -- option 1
        # # Label is one-hot encoding of best primitive (i.e., one that maximizes distance from obstacles)
        # best_primitive_ind = get_best_primitive(env, prim_lib)
        # label_one_hot = np.zeros(num_primitives)
        # label_one_hot[best_primitive_ind] = 1.0
        # labels_all_training[i, :] = label_one_hot

        # Generate label -- option 2 (seems to work best)
        # Label is softmax of distances of each primitive
        dists = get_primitive_dists(env, prim_lib)
        label = scipy.special.softmax(dists)
        labels_all_training[i,:] = label

        # # # Generate label -- option 3
        # # # Label is hot-encoding of collision-free primitives
        # label_free = get_free_primitives(env, obs_radius, prim_lib)
        # labels_all_training[i,:] = label_free


    # # Normalize labels to be in [0,1]
    # labels_all_training = labels_all_training - np.min(labels_all_training)
    # labels_all_training = labels_all_training/np.max(labels_all_training)

    if verbose:
        print("Done generating data.")

    # Save data
    np.savez('lbi/training/training_data.npz',
             lidar_all_training=lidar_all_training,
             labels_all_training=labels_all_training)


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 

