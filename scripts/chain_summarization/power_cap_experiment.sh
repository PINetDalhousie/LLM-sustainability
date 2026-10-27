#!/bin/bash
# This script tests the Chain Summarization application with different GPU power caps for a vLLM setup
# Author: Monzurul
# Date: 2025-07-02
# commands to run: bash ./scripts/power_cap_experiment.sh

# Power caps to test (in watts)
power_caps=(150 175 200 225 250)
# Number of runs per experiment
num_runs=10

# Base directory for logs
base_log_dir="log/power_cap_experiment/chain_summarization"
# remove any existing log directory (intermediate step)
if [ -d "$base_log_dir" ]; then 
    echo "Removing existing log directory: $base_log_dir"
    rm -rf "$base_log_dir"
fi
# Create base log directory
mkdir -p "$base_log_dir"

# Maximum wait time for processes to complete (in seconds)
max_wait_time=300

# Flag to control whether to run all power caps or just one for testing
run_all_power_caps=true

# Array to store process IDs
declare -a pids

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
    pkill -f "vllm_experiment_env" 2>/dev/null
    pkill -f "chain_summarization.py" 2>/dev/null
    pkill -f "python" 2>/dev/null

    # kill any remaining GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk '{print $1}' | xargs -r kill -9
    # kill process running on port 9000
    # lsof -i :9000 | awk 'NR!=1 {print $2}' | xargs -r kill -9
    
    # Reset array
    pids=()
    
    # Wait for processes to terminate
    sleep 5
}

# Trap Ctrl+C and call cleanup
trap cleanup INT

# Main experiment loop for each power cap
for power_cap in "${power_caps[@]}"; do
    echo "==========================================="
    echo "Starting experiments with power cap: $power_cap W"
    echo "==========================================="
    
    for ((run=1; run<=$num_runs; run++)); do
        echo "Starting run $run with power cap $power_cap W..."
        
        # Create log directory for this run
        log_dir="$base_log_dir/powerCap${power_cap}_run_$run"
        mkdir -p "$log_dir"
        
        # Ensure no previous servers are running
        cleanup
        
        # Set GPU power cap
        echo "Setting GPU power cap to $power_cap W..."
        nvidia-smi -pl $power_cap

        # Start vLLM server
        vllm serve meta-llama/Llama-2-7b --seed 0 --num-gpu-blocks-override 2000 --max-num-seqs 256 --max-num-batched-tokens 2560 --max-seq-len-to-capture 8192 --no-enable-chunked-prefill --tokenizer hf-internal-testing/llama-tokenizer --api-key token123 2>&1 | tee "$log_dir/server_output.log" &
        server_pid=$!
        pids+=($server_pid)

        # Wait for vllm server to start
        sleep 120
        # Check if core server is running
        if ! pgrep -f "vllm_experiment_env" > /dev/null; then
            echo "Error: vLLM server failed to start. Check logs at $log_dir/server_output.log"
            continue
        fi
        
        # Array to store user process IDs
        declare -a user_pids

        # Start the Chain Summarization application
        echo "Starting Chain Summarization application..."
        python3 examples/chain_summarization.py > "$log_dir/output.log" 2>&1 &

        # Save the exit code
        echo $? > "$log_dir/exit_code"
        
        # Display the output for monitoring
        cat "$log_dir/output.log"

        user_pid=$!
        user_pids+=($user_pid)
        pids+=($user_pid)

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
        
        # Wait a moment to ensure all logs are flushed
        sleep 5
        
        echo "Completed run $run with power cap $power_cap W"
        echo "----------------------------------------"
        
        # Cleanup after each run
        cleanup
        
        # Break out of loop if we're only running one power cap for testing
        if [ "$run_all_power_caps" = false ]; then
            break 2  # Break out of both loops
        fi
    done
done

# Reset power cap to default value
echo "Resetting GPU power cap to default (250W)..."
nvidia-smi -pl 250

echo "Experiment completed successfully."

# Final cleanup
cleanup
