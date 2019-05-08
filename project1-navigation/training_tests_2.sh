#!/bin/bash

# the seeds for our experiments
seeds=(0)
runs=(0 1 2 3 4)
# experiments with pytorch
library="pytorch"

# mode (train|test)
mode="train"

echo "TRAINING TESTS - 2 - IMPROVEMENTS"

echo "First configuration: config_agent_2_1.json, DDQN enabled, PER disabled"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_improvements_config_2_1_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=true --prioritizedExpReplay=false --configAgent="configs/config_agent_2_1.json"
    done
done

echo "Second configuration: config_agent_2_2.json, DDQN disabled, PER enabled"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_improvements_config_2_2_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=false --prioritizedExpReplay=true --configAgent="configs/config_agent_2_2.json"
    done
done

echo "Third configuration: config_agent_2_3.json, DDQN enabled, PER enabled"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_improvements_config_2_3_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=true --prioritizedExpReplay=true --configAgent="configs/config_agent_2_3.json"
    done
done