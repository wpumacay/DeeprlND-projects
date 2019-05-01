#!/bin/bash

# the seeds for our experiments
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# experiments with pytorch
library="pytorch"

for seed in "${seeds[@]}"
do
    echo "Running experiment with seed: ${seed}, library: ${library}"
    python trainer.py train --library="${library}" --sessionId="banana_simple_pth_${seed}" --seed="${seed}"
done

# experiments with tensorflow
library="tensorflow"

for seed in "${seeds[@]}"
do
    echo "Running experiment with seed: ${seed}, library: ${library}"
    python trainer.py train --library="${library}" --sessionId="banana_simple_tf_${seed}" --seed="${seed}"
done