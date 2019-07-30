#!/bin/bash

# the seeds for our experiments
seeds=(0 1 2 3 4 5)

echo "TRAINING TESTS: RUN OVER VARIOUS SEEDS"

for seed in "${seeds[@]}"
do
    echo "Running experiment> seed=${seed}"
    ## python maddpg_tennis.py train --sessionId="session_submission_seed_${seed}" --seed="${seed}"
    python maddpg_tennis_original.py train --sessionId="session_submission_original_seed_${seed}" --seed="${seed}"
done