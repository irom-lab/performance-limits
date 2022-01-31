import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import IPython as ipy



def main(raw_args=None):

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--problem", type=str, default="lava_problem", help="choose problem: lava_problem or two_lavas_problem (default: lava_problem)")

	args = parser.parse_args(raw_args)
	problem = args.problem


	# Load data
	data = np.load(problem+"_results.npz")
	p_correct_vals = data['p_correct_vals']
	bounds = data['bounds']
	opt_values = data['opt_values']


	# Plot
	fig, ax = plt.subplots()
	ax.plot(p_correct_vals, bounds, 'o--', color='#780e0e', label='Upper Bound', linewidth=1)
	ax.plot(p_correct_vals, opt_values, '*--', color='#007FFF', label='POMDP', linewidth=0.5)

	plt.xlabel('$p_{correct}$', fontsize=15)
	plt.ylabel('Cumulative reward', fontsize=15)
	plt.legend(fontsize=12, loc='center right')
	plt.ylim([0, 5.01])
	plt.show() 


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  