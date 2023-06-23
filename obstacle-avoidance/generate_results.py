import json
import numpy as np
from lbi.bound.compute_bound import main as compute_bound
from lbi.training.generate_training_data import main as generate_training_data
from lbi.training.train_policy import main as train_policy
from lbi.training.evaluate_policy import main as evaluate_policy
from lbi.training.evaluate_planner_policy import main as evaluate_planner_policy
import IPython as ipy
import time


########## Define default parameters ##########
params = {
"workspace_x_lims": [-1.0, 1.0],
"workspace_y_lims": [-0.1, 1.2], # Note, robot should not be on boundary of workspace 
"robot_location": [0.0, 0.0],
"obs_x_lims": [-1.0, 1.0],
"obs_y_lims": [0.9, 1.1],
"num_obs": 6,
"obs_radius": 0.3,
"num_rays": 10,
"lidar_angle_range": [45*np.pi/180, 135*np.pi/180], # 90 degrees is y-axis
"lidar_noise_std": 0.3, 
"lidar_failure_rate": 0.05, 
"lidar_max_distance": 1.5
}


########## Settings ##########
verbose = 0 # print less: 0 or more: 1
compare_bounds = 1 # Compare with other bounds (Fano and Pinsker)
use_hoeffding = 0
delta_mi = 0.04
delta_R0 = 0.01

# Number of seeds for training/testing
num_seeds = 1

########## Noise scales ##########
noise_scales = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5] 


########## Arrays for results ##########
bounds = np.zeros(len(noise_scales))
bounds_fano = np.inf*np.ones(len(noise_scales))
bounds_pinsker = np.inf*np.ones(len(noise_scales))
test_rewards_learning_all = [None]*len(noise_scales) # np.zeros(len(noise_scales))
test_rewards_planning_all = np.zeros(len(noise_scales))

for i in range(len(noise_scales)):

    print("Noise scale: ", noise_scales[i])

    # LIDAR noise std. dev.
    noise_scale = noise_scales[i]
    params["lidar_noise_std"] = noise_scale

    # Write params to json file
    with open("params.json", "w") as write_file:
        json.dump(params, write_file)

    # Compute bound
    # t_start = time.time()
    print("Computing bound...")
    if compare_bounds:
        bound, bound_fano, bound_pinsker = compute_bound(['--verbose', str(verbose), '--compare_bounds', str(compare_bounds), '--use_hoeffding', str(use_hoeffding), '--delta_mi', str(delta_mi), '--delta_R0', str(delta_R0), '--num_batches_MI', str(1)])
        bounds[i] = bound
        bounds_fano[i] = bound_fano
        bounds_pinsker[i] = bound_pinsker

        print("Bound: ", bound)
        print("Bound (Fano): ", bound_fano)
        print("Bound (Pinsker): ", bound_pinsker)
    else:
        bound, _, _ = compute_bound(['--verbose', str(verbose), '--compare_bounds', str(compare_bounds), '--use_hoeffding', str(use_hoeffding), '--delta_mi', str(delta_mi), '--delta_R0', str(delta_R0)])
        bounds[i] = bound
		
    print("Bound: ", bound)

	# Print computation time
	# print("Bound computation time: ", time.time()-t_start)

    #######################################################
    # Planning-based approach: choose primitive with largest minimum distance to detected points
    test_rewards_planning_all[i] = evaluate_planner_policy(['--verbose', str(verbose)])
    print("Mean test reward (planning)", test_rewards_planning_all[i])


    #######################################################
    # Learning-based approach
    test_rewards_learning_all[i] = [None]*num_seeds

    for seed in range(num_seeds):
        print("seed: ", seed)

        # Generate training data and train policy
        print("Generating training data...")
        generate_training_data(['--verbose', str(verbose)])

        print("Training policy...")
        train_policy(['--verbose', str(verbose)])

        # Evaluate policy
        print("Evaluating policy...")
        mean_test_reward = evaluate_policy(['--verbose', str(verbose)])
        print("Mean test reward (learning): ", mean_test_reward)

        test_rewards_learning_all[i][seed] = mean_test_reward

    print(" ")


# Save results
np.savez("results_noise.npz", noise_scales=noise_scales, bounds=bounds, bounds_fano=bounds_fano, bounds_pinsker=bounds_pinsker, test_rewards_learning_all=test_rewards_learning_all, test_rewards_planning_all=test_rewards_planning_all)


