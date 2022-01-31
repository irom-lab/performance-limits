import numpy as np
import IPython as ipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym
import gym_ball_catching
# from train_rnn_policy import main as train_rnn_policy


################## Load data ###############################
data = np.load("ball_catching_results.npz")
noise_scales = data['noise_scales']
bounds = data['bounds']
MPC_reward_means = data['MPC_reward_means']
MPC_reward_std_devs = data['MPC_reward_std_devs']

################## Make plot ###############################
env = gym.make("BallCatching-v1")
plt.plot(noise_scales, bounds, 'o--', color='#780e0e', label='Upper Bound', linewidth=1)
plt.errorbar(noise_scales, MPC_reward_means, MPC_reward_std_devs, color="#007FFF", fmt=".--", ecolor="#007FFF", capsize=4, linewidth=0.5, label="MPC + Kalman Filter")
plt.ylim([0, env.T+1.5])

plt.xlabel('Noise scale ($\eta$)', fontsize=15)
plt.ylabel('Cumulative reward', fontsize=15)
plt.legend(fontsize=12, loc='lower right')
plt.show()



