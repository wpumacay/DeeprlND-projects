#!/bin/bash

# the seeds for our experiments
seeds=(0)
runs=(0 1 2 3 4)
# experiments with pytorch
library="pytorch"

# mode (train|test)
mode="train"

echo "TRAINING TESTS - 3 - ALMOST-NO-EXPLORATION"

echo "First configuraton: config_agent_3_1.json, with almost no exploration"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_exploration_config_3_1_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=false --prioritizedExpReplay=false --configAgent="configs/config_agent_3_1.json"
    done
done

echo "First configuraton: config_agent_3_2.json, with almost no exploration, but all improvements enabled"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_exploration_config_3_2_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=true --prioritizedExpReplay=true --configAgent="configs/config_agent_3_2.json"
    done
done