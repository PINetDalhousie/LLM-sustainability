#!/bin/bash
# This script is used to measure the energy consumption of chain summarization with varying output token lengths over vLLM.
# Command: ./scripts/vllm_varying_output_limit.sh (considering main project directory is /workspace/vllm_experiment in the magical_kilby container)
# Author: Monzurul
# Date: 2025-06-13

# Array of output token lengths to test
token_limits=(25 50 75 100)

# Number of runs per experiment
num_runs=10

# Base directory for logs
base_log_dir="log/chain_summarization/output_limit_experiments_with_default_attn"
# # remove any existing log directory (intermediate step)
# if [ -d "$base_log_dir" ]; then 
#     echo "Removing existing log directory: $base_log_dir"
#     rm -rf "$base_log_dir"
# fi
# # Create base log directory
# mkdir -p "$base_log_dir"

# Function to kill all related processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "chain_summarization.py" || true
    # pkill -f "/root/.local/miniconda3/envs/vllm_experiment_e" || true
    # pkill -u "$(whoami)" python

    sleep 2
}

# Function to extract time and energy from the output
extract_metrics() {
    local output_file=$1
    local metrics_file=$2
    
    # Extract the line containing "Total workflow consumed"
    grep "Chain summarization application consumed Time:" "$output_file" | tail -n 1 > "$metrics_file"
    
    # If no metrics found, write a message to the file
    if [ ! -s "$metrics_file" ]; then
        echo "No energy metrics found in output" > "$metrics_file"
    fi
}

# Ensure cleanup runs even if script is interrupted
# trap cleanup EXIT

# Create or append to a summary file for all experiments
summary_file="$base_log_dir/experiment_summary.txt"
# Add a separator line if the file already exists
if [ -f "$summary_file" ]; then
    echo -e "\n\n==================================================" >> "$summary_file"
    echo "New Experiment Run - $(date)" >> "$summary_file"
    echo "==================================================" >> "$summary_file"
    echo "" >> "$summary_file"
else
    echo "Chain Summarization Experiments Summary" > "$summary_file"
    echo "======================================" >> "$summary_file"
    date >> "$summary_file"
    echo "" >> "$summary_file"
fi

for token_limit in "${token_limits[@]}"; do
    echo "Starting experiments with token_limit = $token_limit"
    echo "Token Limit: $token_limit" >> "$summary_file"
    echo "Running $num_runs times..." >> "$summary_file"
    echo "" >> "$summary_file"
    
    for run in $(seq 1 $num_runs); do
        echo "Run $run of $num_runs for token_limit = $token_limit"
        
        # Create log directory for this experiment run
        log_dir="$base_log_dir/token_limit_${token_limit}/run_${run}"
        mkdir -p "$log_dir"
        
        # # Ensure no previous servers are running
        # cleanup
        
        # # Start vLLM server 
        # echo "Starting vllm server..."
        vllm serve  meta-llama/Llama-2-7b --api-key token123 2>&1 | tee "$log_dir/vllm_server.log" &

        # # Wait for core server to start
        # sleep 120

        # Run the chain summarization directly in foreground
        python3 examples/chain_summarization.py --token_limit "$token_limit" > "$log_dir/experiment_output.log" 2>&1

        # Save the exit code
        echo $? > "$log_dir/exit_code"
        
        # Display the output for monitoring
        cat "$log_dir/experiment_output.log"
        
        # Check if the process completed successfully
        if [ "$(cat "$log_dir/exit_code")" -ne 0 ]; then
            echo "Error: Chain summarization process failed with exit code $(cat "$log_dir/exit_code")"
            echo "Check logs at $log_dir/experiment_output.log"
        fi
        
        # Extract metrics
        extract_metrics "$log_dir/experiment_output.log" "$log_dir/metrics.txt"
        
        # Add to summary
        echo "Run $run:" >> "$summary_file"
        cat "$log_dir/metrics.txt" >> "$summary_file"
        echo "" >> "$summary_file"

        # Cleanup before next run
        cleanup
        
        # Wait before starting next run
        sleep 10
        
        echo "Completed run $run for token_limit = $token_limit"
        echo "----------------------------------------"
    done
    echo "----------------------------------------" >> "$summary_file"
    echo "" >> "$summary_file"
done

echo "All experiments completed!"
echo "Summary available at: $summary_file"
echo "Individual metrics files are available in $base_log_dir/"