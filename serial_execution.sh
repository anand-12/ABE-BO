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

run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Griewank --dim 5 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Griewank --dim 5 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function DropWave --dim 2 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function DropWave --dim 2 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function DixonPrice --dim 3 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function DixonPrice --dim 3 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Cosine8 --dim 8 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Cosine8 --dim 8 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Bukin --dim 2 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Branin --dim 2 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Branin --dim 2 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Beale --dim 2 --use_abe
run_and_print test.py --experiments 10 --seed 0 --acquisition LogEI LogPI UCB --kernel Matern52 --function Beale --dim 2 --use_abe --use_least_risk
run_and_print test.py --experiments 10 --seed 42 --acquisition LogEI LogPI UCB --kernel Matern52 --function Ackley --dim 4 --use_abe
run_and_print test.py --experiments 10 --seed 42 --acquisition LogEI LogPI UCB --kernel Matern52 --function Ackley --dim 4 --use_abe --use_least_risk


echo "Execution complete."