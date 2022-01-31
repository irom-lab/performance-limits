from lbi.envs.envs import *
from lbi.primitives.utils import *
from lbi import geom, sensing as sense, bound
from lbi.sensing.lidar import *
from lbi.sensing.types import state_to_workspace

import jax
import numpy as np
import matplotlib.pyplot as plt
import IPython as ipy
import cvxpy as cvx
import argparse	
import json
import time
import sys
import tqdm
from multiprocessing import Pool


def kl_inverse(q, c):
    '''
    Compute (left) kl inverse using Relative Entropy Programming
    Solves: sup p in [0,1] with KL(p,q) <= c
    '''   
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(p_bernoulli,q_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cvx.SCS) # solver=cvx.MOSEK
    
    return p_bernoulli.value[0] 

def kl_right_inverse(p, c):
    '''
    Compute (right) kl inverse using Relative Entropy Programming
    Solves: sup q in [0,1] with KL(p,q) <= c
    '''   

    q_bernoulli = cvx.Variable(2)

    p_bernoulli = np.array([p,1-p])

    constraints = [c >= cvx.sum(cvx.kl_div(p_bernoulli,q_bernoulli)), 0 <= q_bernoulli[0], q_bernoulli[0] <= 1, q_bernoulli[1] == 1.0-q_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(q_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cvx.SCS) # solver=cvx.MOSEK
    
    return q_bernoulli.value[0] 

def estimate_open_loop_reward(primitive, envs, obs_radius):

    # Rewards for each env
    num_envs = np.shape(envs)[0]
    rewards = np.zeros(num_envs)

    for i in range(num_envs):

        # Get env
        env = envs[i,:,:]

        # Check collision
        collision = check_primitive_collision(env, obs_radius, primitive)

        # Assign reward
        reward = 1 - collision
        rewards[i] = reward

    # Compute mean reward across environments for this primitive
    mean_reward = np.mean(rewards)

    return mean_reward


def log_output_given_state(scan, conf, env, aabb, radius, sensor) -> float:
    w = sense.state_to_workspace(env, radius, aabb.xlim, aabb.ylim)

    return sense.lidar_logpdf(scan, conf, w, sensor)

def hoeffding_bound_MI(I_est, delta_mi, num_samples, num_batches, sensor):
    '''
    Use Hoeffding's inequality to upper bound MI with probability >= 1 - delta_mi.
    '''

    # Get bounds on random variable in MI estimation
    mi_rand_var_low, mi_rand_var_high = bound.estimate_mi_bounds(sensor, num_samples)

    # Scale to be within [0,1]
    I_est_scaled = (I_est-mi_rand_var_low)/(mi_rand_var_high-mi_rand_var_low) 
    
    # Compute Hoeffding bound
    I_bound_scaled = kl_right_inverse(I_est_scaled, (1/num_batches)*np.log(2/delta_mi))
    
    # Unscale
    I_bound = I_bound_scaled*(mi_rand_var_high-mi_rand_var_low) + mi_rand_var_low

    return I_bound

def hoeffding_bound_R0(R0, delta_R0, num_envs_R0_estimation, num_primitives):
    '''
    Use Hoeffding's inequality to upper bound R0 with probability >= 1 - delta_R0.
    '''

    # Divide delta by num_primitives to use union bound
    delta = delta_R0/num_primitives

    R0_bound = kl_right_inverse(R0, (1/num_envs_R0_estimation)*np.log(2/delta))

    return R0_bound

def main(raw_args=None):


    ###################################################################
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")
    parser.add_argument("--compare_bounds", type=int, default=1, help="Compute Fano and Pinsker bounds too (default: 1 (False)")
    parser.add_argument("--use_hoeffding", type=int, default=0, help="Use concentration inequality for bounds (default: 0 (False)")
    parser.add_argument("--delta_mi", type=float, default=0.04, help="Probability of error in MI estimation (default: 0.04")
    parser.add_argument("--delta_R0", type=float, default=0.01, help="Probability of error in R0 estimation (default: 0.01")
    parser.add_argument("--num_samples_MI", type=int, default=1000, help="Number of samples per batch in MI estimation (default: 1000")
    parser.add_argument("--num_batches_MI", type=int, default=20000, help="Number of batches for MI estimation (default: 20000")
    parser.add_argument("--num_envs_R0_estimation", type=int, default=20000, help="Number of envs for estimating R0 (default: 20000")
    parser.add_argument("--num_cores", type=int, default=4, help="Number of cores for CPU parallelization (default: 4")


    args = parser.parse_args(raw_args)
    verbose = args.verbose
    compare_bounds = args.compare_bounds
    use_hoeffding = args.use_hoeffding
    delta_mi = args.delta_mi
    delta_R0 = args.delta_R0
    num_samples_MI = args.num_samples_MI
    num_batches_MI = args.num_batches_MI
    num_envs_R0_estimation = args.num_envs_R0_estimation
    num_cores = args.num_cores



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

    sensor = sense.lidar(num_rays, lidar_angle_range,
                        local_noise_std=lidar_noise_std,
                        failure_rate=lidar_failure_rate,
                        max_distance=lidar_max_distance)

    # Load motion primitives
    prim_lib = np.load('lbi/primitives/primitive_library.npy')
    num_primitives = np.shape(prim_lib)[1]
    ##################################################


    ##################################################
    # Estimate best open-loop reward
    open_loop_rewards = np.zeros(num_primitives)

    # Pack inputs to use with map for generating envs
    inputs_to_gen_envs = num_envs_R0_estimation*[[[obs_x_lims], [obs_y_lims], [num_obs], [obs_radius], [prim_lib] ]]
       
    # Generate environment samples
    with Pool(num_cores) as p:
        envs = p.map(random_traversable_env_pool, inputs_to_gen_envs)     

    envs = np.reshape(envs, (num_envs_R0_estimation, 2, num_obs))

    for i in range(num_primitives):

        # Primitive i
    	primitive = prim_lib[:,i,:]

        # Estimate open-loop reward for this primitive
    	open_loop_rewards[i] = estimate_open_loop_reward(primitive, envs, obs_radius) # prim_lib, obs_x_lims, obs_y_lims, num_obs, obs_radius)

    # Best open-loop reward
    R0 = np.max(open_loop_rewards)

    if verbose:
        print("Best open-loop reward: ", R0)
    ##################################################

    ##################################################
    # Estimate mutual information
    num_samples = num_samples_MI # Number of samples per batch
    if use_hoeffding:
        num_batches = num_batches_MI # Number of batches for concentration inequality
    else:
        num_batches = 1 # Just use one batch

    # Will store estimates for every batch
    I_est_batches = np.zeros(num_batches) # Will store MI estimates for each batch

    aabb = geom.aabb(workspace_x_lims, workspace_y_lims)
    logs = jax.jit(lambda y, x: log_output_given_state(y, robot_state, x, aabb, obs_radius, sensor))

    # Pack inputs to use with map
    inputs_to_gen_envs = num_samples*[[[obs_x_lims], [obs_y_lims], [num_obs], [obs_radius], [prim_lib] ]]
            
    for batch in tqdm.tqdm(range(num_batches), file=sys.stdout): # range(num_batches):

        # Generate environment samples
        with Pool(num_cores) as p:
            env_samples = p.map(random_traversable_env_pool, inputs_to_gen_envs)

        env_samples = np.reshape(env_samples, (num_samples, 2, num_obs))
        env_samples = jnp.array(env_samples)

        # Seed
        seed = np.random.randint(100000) # Randomize
        key = jax.random.PRNGKey(seed)
        subkeys = jax.random.split(key, num_samples)

        # Generate LIDAR samples
        lidar_samples = jax.vmap(lambda env, subkey: sample_lidar(subkey, robot_state, sense.state_to_workspace(env, obs_radius, workspace_x_lims, workspace_y_lims), sensor))(
            env_samples, subkeys)

        # Compute MI estimate for this batch
        I_est_batch = bound.estimate_mi(env_samples, lidar_samples, logs, num_samples)
        I_est_batches[batch] = I_est_batch

    I_est = np.mean(I_est_batches)

    if verbose:
        print("Estimated MI: ", I_est)
    ##################################################

    ##################################################
    # Compute KL-inverse bound
    bound_fano = np.inf
    bound_pinsker = np.inf
    if np.isnan(I_est):
        bound_kl_inv = 1.0
        bound_fano = 1.0
        bound_pinsker = 1.0
    else:

        if use_hoeffding:

            # Use_concentration inequality for R0
            R0_bound = hoeffding_bound_R0(R0, delta_R0, num_envs_R0_estimation, num_primitives)

            # Use concentration inequality for MI
            I_bound = hoeffding_bound_MI(I_est, delta_mi, num_samples, num_batches, sensor)
        else:
            R0_bound = R0
            I_bound = I_est

        # Compute final bound on expected reward
        bound_kl_inv = kl_inverse(R0_bound, I_bound)
        bound_kl_inv = min(1.0, bound_kl_inv)

    if verbose:
        print("bound: ", bound_kl_inv)
    ##################################################

    ##################################################
    # Compute other bounds
    if compare_bounds and (not np.isnan(I_est)): 

        # Fano bound
        bound_fano = (I_bound + np.log(2-R0_bound))/np.log(1/R0_bound)
        bound_fano = min(1.0, bound_fano)

        # Pinsker bound
        bound_pinsker = R0_bound + np.sqrt(I_bound/max(np.log(1/R0_bound), 2))
        bound_pinsker = min(1.0, bound_pinsker)


    ##################################################

    return bound_kl_inv, bound_fano, bound_pinsker



#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 

