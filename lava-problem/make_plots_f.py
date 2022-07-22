import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import IPython as ipy
import os


def main(raw_args=None):

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--problem", type=str, default="lava_problem", help="choose problem: lava_problem or two_lavas_problem (default: lava_problem)")

	args = parser.parse_args(raw_args)
	problem = args.problem

	# Load all npz filenames
	fname = os.listdir()
	npz_name = []
	for i in fname:
		if i[-1] == 'z':
			npz_name.append(i)
	
	# Load data
	data = np.load(npz_name[0])
	p_correct_vals = data['p_correct_vals']
	opt_values = data['opt_values']

	# Plot
	fig, ax = plt.subplots()
	ax.plot(p_correct_vals, opt_values, '*--', label='POMDP', linewidth=0.5)

	for j in npz_name:
		data = np.load(j)
		bounds = data['bounds']
		ax.plot(p_correct_vals, bounds, 'o--', label=j[13:-14], linewidth=1)

	plt.xlabel('$p_{correct}$', fontsize=15)
	plt.ylabel('Cumulative reward', fontsize=15)
	plt.legend(fontsize=12, loc='lower right')
	plt.ylim([0, 5.01])
	plt.show() 


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  