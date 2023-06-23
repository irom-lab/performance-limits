# Obstacle avoidance with noisy depth sensor

## Installation notes

We provide yml files for creating conda environments (with or without cuda). In addition, we utilize PyTorch for training neural network policies; install via:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
We used PyBullet to make the anchor figure (Fig.1); install via:
```
pip install pybullet
```
This is only used for creating 3D visualizations.


## Generate and plot all results

To generate all results, run:
```
python generate_results.py
```

Note: By default, the code above will use concentration inequalities to obtain high-confidence upper bounds. This requires a large number of samples. To speed things up (without obtaining high-confidence bounds), set ```use_hoeffding = 0``` in generate_results.py, and pass the option ```num_batches_MI = 1``` to compute_bound.py. 

To make plots, run:
```
python make_plots.py
```


