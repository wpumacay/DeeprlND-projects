#!/bin/bash

# the seeds for our experiments
seeds=(0 1 2)
runs=(0 1 2 3 4)
# experiments with pytorch
library="tensorflow"

# mode (train|test)
mode="train"

echo "TRAINING TESTS - 0 - SUBMISSION"

echo "Set of hyperparameters from config_submission.json"

for seed in "${seeds[@]}"
do
    for run in "${runs[@]}"
    do
        echo "Running experiment> run=${run}, seed=${seed}, library=${library}"
        python trainer.py "${mode}" --library="${library}" --sessionId="banana_submission_run_${run}_${library}_seed_${seed}" --seed="${seed}" --ddqn=false --prioritizedExpReplay=false --configAgent="configs/config_submission.json"
    done
done