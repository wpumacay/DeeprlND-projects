#!/bin/bash

# the seeds for our experiments
seeds=(0 1 2 3 4 5)
config="./configs/ddpg_reacher_multi_seeds.gin"

echo "TRAINING TESTS - 0 - RUN OVER VARIOUS SEEDS"

echo "Gin-config used: ${config}"

for seed in "${seeds[@]}"
do
    echo "Running experiment> seed=${seed}, config=${config}"
    python trainer.py train --config="${config}" --seed="${seed}"
done