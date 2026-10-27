#!/bin/bash
# This script tests the Bing Copilot application with certain number of concurrent users
# and varying batch sizes 
# Author: Monzurul
# Date: 2025-08-06
# Command: ./scripts/bing_copilot/batch_size_experiment.sh

# Discarding stderr to avoid cluttering the output
exec 2>/dev/null

# Fixed number of concurrent users
num_users=8 

# Number of runs per experiment
num_runs=10

# Maximum wait time for processes to complete (in seconds)
max_wait_time=300

# Batch sizes to test
batch_sizes=(1 2 4 8 16) 

# Array to store process IDs
declare -a pids

# Base directory for logs
base_log_dir="log/bing_copilot/batch_exp_users_$num_users"
# remove any existing log directory (intermediate step)
if [ -d "$base_log_dir" ]; then 
    echo "Removing existing log directory: $base_log_dir"
    rm -rf "$base_log_dir"
fi
# Create base log directory
mkdir -p "$base_log_dir"

# Function to cleanup processes
cleanup() {
    echo "Cleaning up processes..."
    
    # Kill all background processes
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null; then
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # Kill any remaining processes
    pkill -f "bingCopilot.py" 2>/dev/null
    pkill -f "zeus-monitor" 2>/dev/null
    pkill -f "/root/.local/miniconda3/envs/vllm_experiment_e" || true
    pkill -u "$(whoami)" python
    
    # Reset array
    pids=()
    
    # Wait for processes to terminate
    sleep 5
}

# Function to extract time and energy from the output
extract_metrics() {
    local log_dir=$1
    local metrics_file=$2

    : > "$metrics_file"            # truncate output
    shopt -s nullglob

    # Process in numeric user order for stable output
    for user in $(seq 1 "$num_users"); do
        local f="$log_dir/user_${user}_output.log"
        if [[ -f "$f" ]]; then
            # Pull matching lines and prefix with "User N:"
            grep "Application consumed Time:" "$f" \
              | sed -e "s/^/User ${user}: /" >> "$metrics_file"
        fi
    done

    # If no matches at all, write a message
    if [ ! -s "$metrics_file" ]; then
        echo "No energy metrics found in output" > "$metrics_file"
    fi
}

# Create summary file
summary_file="$base_log_dir/experiment_summary.txt"
mkdir -p "$base_log_dir"
echo "Bing Copilot Batch Size Experiments" > "$summary_file"
echo "=============================================" >> "$summary_file"
date >> "$summary_file"
echo "" >> "$summary_file"
echo "Fixed Concurrent Users: $num_users" >> "$summary_file"
echo "Batch Sizes Tested: ${batch_sizes[*]}" >> "$summary_file"
echo "----------------------------------------" >> "$summary_file"

# Trap Ctrl+C and call cleanup
trap cleanup EXIT

# Load user queries
query_file="data/user_queries.txt"
if [ ! -f "$query_file" ]; then
    echo "Error: Query file not found: $query_file"
    exit 1
fi

# Count number of queries
num_queries=$(wc -l < "$query_file")
echo "Found $num_queries queries in $query_file"

# Main experiment loop for each batch size
for batch_size in "${batch_sizes[@]}"; do
    echo "----------------------------------------" >> "$summary_file"
    echo "Testing with batch size: $batch_size" >> "$summary_file"
    echo "----------------------------------------" >> "$summary_file"
    
    for ((run=1; run<=$num_runs; run++)); do
        echo "Starting run $run with batch size $batch_size and $num_users concurrent users..."
        
        # Create log directory for this run
        log_dir="$base_log_dir/batch${batch_size}_${num_users}users_run_$run"
        mkdir -p "$log_dir"
        
        # Ensure no previous servers are running
        cleanup
        
        # Start vLLM server for Llama-2 7B model
        echo "Starting vllm server..."        
        vllm serve meta-llama/Llama-2-7b --seed 0 --num-gpu-blocks-override 2000 --max-num-seqs $batch_size --max-num-batched-tokens 2560 --max-seq-len-to-capture 8192 --no-enable-chunked-prefill --tokenizer hf-internal-testing/llama-tokenizer --api-key token123 2>&1 | tee "$log_dir/server_output.log" &

        # Wait for vLLM server to start
        sleep 120
        
        # Run concurrent Bing Copilot instances
        echo "Running $num_users concurrent Bing Copilot instances..."
        
        # Array to store user process IDs
        declare -a user_pids
        
        # Start users with queries from the file
        for ((user=1; user<=$num_users; user++)); do
            # Get a query from the file (cycling through if needed)
            query_line=$(( (user - 1) % num_queries + 1 ))
            query=$(sed "${query_line}q;d" "$query_file")
            
            # Escape quotes in the query to avoid shell interpretation issues
            escaped_query=$(echo "$query" | sed 's/"/\\"/g')
            
            # user_output_log="$log_dir/experiment_output.log"
            # echo "User $user output:" >> "$user_output_log"
            # Run Bing Copilot with the query (pass directly as argument, not with --query flag)
            python3 examples/bingCopilot.py --user_query "$escaped_query" > "$log_dir/user_${user}_output.log" 2>&1 &
            # python3 examples/bingCopilot.py "$escaped_query" >> "$user_output_log" 2>&1 &
            user_pid=$!
            user_pids+=($user_pid)
            pids+=($user_pid)
            
            echo "Started user $user with query: ${query:0:50}... (PID: $user_pid)"
            
            # Small delay between user launches to avoid connection issues
            sleep 0.5
        done
        
        # Wait for all processes to complete or timeout
        echo "Waiting for all processes to complete (timeout: $max_wait_time seconds)..."
        start_time=$(date +%s)
        all_done=false
        
        while ! $all_done; do
            all_done=true
            
            # Check if any processes are still running
            for pid in "${user_pids[@]}"; do
                if ps -p $pid > /dev/null; then
                    all_done=false
                    break
                fi
            done
            
            # Check if we've exceeded the timeout
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))
            
            if [ $elapsed_time -gt $max_wait_time ]; then
                echo "Timeout reached ($max_wait_time seconds). Terminating remaining processes..."
                break
            fi
            
            # Wait a bit before checking again
            sleep 1
        done
        
        # Kill any remaining processes
        for pid in "${user_pids[@]}"; do
            if ps -p $pid > /dev/null; then
                echo "Killing process $pid that didn't complete within the timeout..."
                kill -9 $pid 2>/dev/null
            fi
        done
        
        # Save the exit code
        echo $? > "$log_dir/exit_code"
        
        # # Display the output for monitoring
        # cat "$user_output_log"
        
        # # Check if the process completed successfully
        # if [ "$(cat "$log_dir/exit_code")" -ne 0 ]; then
        #     echo "Error: application failed with exit code $(cat "$log_dir/exit_code")"
        #     echo "Check logs at $user_output_log for details."
        # fi
        
        # Extract metrics
        # extract_metrics "$user_output_log" "$log_dir/metrics.txt"
        extract_metrics "$log_dir" "$log_dir/metrics.txt"
        
        # Add to summary
        echo "Run $run:" >> "$summary_file"
        cat "$log_dir/metrics.txt" >> "$summary_file"
        echo "" >> "$summary_file"
        
        # Cleanup after each run
        cleanup

        # Wait before starting next run
        sleep 5
        
        echo "Completed run $run with batch size $batch_size"
        echo "----------------------------------------"
    done
done

echo "All experiments completed!"
echo "Summary available at: $summary_file"
echo "Individual metrics files are available in $base_log_dir/"

# Final cleanup
cleanup
