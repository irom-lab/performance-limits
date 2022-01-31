import numpy as np
import IPython as ipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


################## Load data ###############################
data = np.load("results.npz")
noise_scales = data['noise_scales']
bounds = data['bounds']
bounds_fano = data['bounds_fano']
bounds_pinsker = data['bounds_pinsker']
test_rewards_all = data['test_rewards_all']

test_rewards_max = np.zeros(len(noise_scales))
test_rewards_mean = np.zeros(len(noise_scales))
test_rewards_std_dev = np.zeros(len(noise_scales))
for i in range(len(noise_scales)):
	test_rewards_i = test_rewards_all[i]
	test_rewards_max[i] = np.max(test_rewards_i)
	test_rewards_mean[i] = np.mean(test_rewards_i)
	test_rewards_std_dev[i] = np.std(test_rewards_i)

################## Make plot ###############################
# plt.scatter(noise_scales, test_rewards_max, color="#00B000", label="Learned policy")
# plt.plot(noise_scales, bounds_fano, '*--', color='#fcba03', label='Upper Bound (Fano)', linewidth=1)
# plt.plot(noise_scales, bounds_pinsker, '.--', color='#fc7b03', label='Upper Bound (Pinsker)', linewidth=1)
plt.plot(noise_scales, bounds, 'o--', color='#780e0e', label='Upper Bound (KL inverse)', linewidth=1)
plt.errorbar(noise_scales, test_rewards_mean, test_rewards_std_dev, color="#007FFF", fmt=".--", ecolor="#007FFF", capsize=4, linewidth=0.5, label="Learned policy")
plt.ylim([0, 1.1])

plt.xlabel('Noise std. dev. ($\eta$)', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend(fontsize=12, loc='lower right')
plt.show()