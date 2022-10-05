# Lava problem

## Packages

We utilize the ```pomdp_py``` package for comparing the bounds we compute with the optimal POMDP solution:

https://h2r.github.io/pomdp-py/html/

This can be installed via pip:
```
pip install pomdp-py
```

## Computing bounds

Run the following to generate bounds and POMDP values:
```
python generate_results.py
```

Run the following to plot the results:
```
python make_plots.py
```

We also provide code for a different version of the lava problem with two lavas (one on either end of the environment). Use the option "--problem two_lavas_problem" in the scripts above to generate and plot results for this version. 

## Using f-divergennce
Run the following to generate bounds and POMDP values:
```
python generate_results_f.py --function f
```
List of available functions:
- Kullback_Leibler
- Negative_Log
- Total_Variation
- Chi_Squared
- Jensen_Shannon
- Squared_Hellinger_Distance
- Neyman_Chi_Squared

To run all functions and make plots, use
```
./run.sh
```

## Finding the tightest bounds
To use a piecewise linear function $f$ to find the tightest upper bounds on performance, run
```
python bound_optimize_l.py
```
(On server, `conda deactivate` and use `python3`)

This takes very long to run. For a new try, change 
- $n$
- $s_0$ 
- `savenpz` file name
- `savefig` file name
in code for now.

To plot the optimized piecewise-linear functions, run
```
python make_plots_l.py
```
TODO: clean up this plot