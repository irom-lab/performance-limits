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
List of functions:
Kullback_Leibler
Negative_Log
Total Variation

