#!/bin/bash

# the seeds for our experiments
seeds=(0)
runs=(0 1 2 3 4)
# experiments with pytorch
library="pytorch"

# mode (train|test)
mode="train"

echo "TRAINING TESTS - 1 - HYPERPARAMETERS"

echo "First set of hyperparameters from config_agent_1_1.json"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_hypertests_config_1_1_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=false --prioritizedExpReplay=false --configAgent="configs/config_agent_1_1.json"
    done
done

echo "Second set of hyperparameters from config_agent_1_2.json"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_hypertests_config_1_2_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=false --prioritizedExpReplay=false --configAgent="configs/config_agent_1_2.json"
    done
done