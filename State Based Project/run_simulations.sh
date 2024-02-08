#!/laksa/bin/bash

# Define the agents
agents=("yoshua_agent" "yann_agent" "jurgen_agent" "geoffrey_agent" "image_jurgen_agent")

# Output directory
output_dir="imitation_data/"

# Create the directory if it does not exist
mkdir -p "$output_dir"

# Number of epochs
num_epochs=10

# Get the number of agents
num_agents=${#agents[@]}
index_offset=0 # set to the last index of the previous run if needed
index=0
# Run simulations
for (( epoch=0; epoch<num_epochs; epoch++ )); do
    for (( i=0; i<num_agents; i++ )); do
        for (( j=0; j<num_agents; j++ )); do
            agent1=${agents[$i]}
            agent2=${agents[$j]}
            filename="${output_dir}/$((index + index_offset))_${agent1}_vs_${agent2}.pkl"
            echo "Running simulation: $agent1 vs $agent2"
            python3 -m tournament.runner "$agent1" "$agent2" -s "$filename"
            ((index++))
        done
    done
    echo "Done with epoch: $epoch"
done

