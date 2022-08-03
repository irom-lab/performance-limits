#!/bin/bash

func_list=(
    Kullback_Leibler
    Negative_Log
    Total_Variation
    Chi_Squared
    Jensen_Shannon
    Squared_Hellinger_Distance
    Neyman_Chi_Squared
)

for func in "${func_list[@]}"
do
    echo "${func}"
    python generate_results_f.py --function ${func}
done

python generate_results_f.py --POMDP Y

python make_plots_f.py
