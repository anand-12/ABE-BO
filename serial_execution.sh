#!/bin/bash

# Function to run command and print output
run_and_print() {
    echo "Running $1"
    start_time=$(date +%s)
    time python "$@"
    end_time=$(date +%s)
    echo "Execution time: $((end_time - start_time)) seconds"
    echo -e "\n----------------------------------------\n"
}


# Run each script and print the output
run_and_print ABEBO_NW_std.py --experiments 25 --seed 0 --acquisition LogEI LogPI UCB_0.1 --kernel Matern52 --function Branin --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 25 --seed 0 --acquisition LogEI LogPI UCB_0.9 --kernel Matern52 --function Branin --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 25 --seed 0 --acquisition LogEI UCB_0.9 UCB_0.1 --kernel Matern52 --function Branin --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 25 --seed 0 --acquisition UCB_0.5 UCB_0.9 UCB_0.1 --kernel Matern52 --function Branin --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 25 --seed 0 --acquisition UCB_0.999 UCB_0.001 --kernel Matern52 --function Branin --dim 2 --use_abe 
echo "Execution complete."
