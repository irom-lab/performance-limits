import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bound_rewards_l import l

def main():

	# Load data
	data = np.load("results5_0913.npz")
	p_correct_vals = np.load("lava_problem_optimal_results.npz")['p_correct_vals']
	slopes = data['slopes']
	bounds = data['bound_results']
	n = int(np.shape(slopes)[1]/2)

	k = 1000
	xt = np.linspace(0,2,k)
		
	fig, ax = plt.subplots(5,4)
	for i in range(5):
		for j in range(4):
			svec = slopes[i*4+j]
			yt = [l(x,n,svec).value for x in xt]
			ax[i,j].plot(xt,yt)
			p = str(i*4+j)
			ax[i,j].set_title(p)
			ax[i,j].set_xlabel('x')
			ax[i,j].set_ylabel('f(x)')

	plt.savefig('PL_n=5_func.svg', dpi=200)

#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  