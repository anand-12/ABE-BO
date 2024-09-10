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
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Branin --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Branin --dim 2 --acq_weight bandit
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Branin --dim 2 --acq_weight random
run_and_print ABEBO_NW_std.py --experiments 20 --seed 42 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Hartmann --dim 6 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 20 --seed 42 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Hartmann --dim 6 --acq_weight bandit
run_and_print ABEBO_NW_std.py --experiments 20 --seed 42 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Hartmann --dim 6 --acq_weight random
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function DropWave --dim 2 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function DropWave --dim 2 --acq_weight bandit
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function DropWave --dim 2 --acq_weight random
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Cosine8 --dim 8 --use_abe 
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Cosine8 --dim 8 --acq_weight bandit
run_and_print ABEBO_NW_std.py --experiments 20 --seed 0 --acquisition LogEI LogPI UCB_0.1 UCB_0.3 UCB_0.7 UCB_0.9 --kernel Matern52 --function Cosine8 --dim 8 --acq_weight random

run_and_print baseline.py --experiments 20 --seed 0 --acquisition LogEI --kernel Matern52 --function Branin --dim 2
run_and_print baseline.py --experiments 20 --seed 42 --acquisition LogEI --kernel Matern52 --function Hartmann --dim 6
run_and_print baseline.py --experiments 20 --seed 0 --acquisition LogEI --kernel Matern52 --function DropWave --dim 2
run_and_print baseline.py --experiments 20 --seed 0 --acquisition LogEI --kernel Matern52 --function Cosine8 --dim 8
echo "Execution complete."
