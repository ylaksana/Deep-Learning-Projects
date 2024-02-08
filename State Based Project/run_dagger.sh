#!/bin/bash

# Number of epochs
num_epochs=20

# Run simulations
for (( epoch=0; epoch<num_epochs; epoch++ )); do
    echo "Running dagger epoch: $epoch"
    python3 -m state_agent.train_state_agent dagger_final_1 dagger_final_2 --log-dir logs
    echo "Done with dagger epoch: $epoch"
done


