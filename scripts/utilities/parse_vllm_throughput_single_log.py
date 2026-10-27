#!/usr/bin/env python3
"""
Script to parse vLLM server logs and calculate mean metrics for throughput analysis.
Extracts: Avg prompt throughput, Avg generation throughput, GPU KV cache usage, Prefix cache hit rate
Command: python scripts/parse_vllm_throughput_single_log.py <server_log_file>
"""

import re
import sys
from statistics import mean, stdev
from pathlib import Path


def parse_vllm_server_log(log_file_path):
    """
    Parse vLLM server log file to extract engine metrics.
    
    Args:
        log_file_path: Path to the server log file
        
    Returns:
        dict: Contains lists of extracted metrics
    """
    
    # Regex pattern to match the engine log lines
    engine_pattern = r'Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: \d+ reqs, Waiting: \d+ reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%'
    
    metrics_data = {
        'prompt_throughput': [],
        'generation_throughput': [],
        'gpu_kv_cache_usage': [],
        'prefix_cache_hit_rate': []
    }
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(engine_pattern, line)
                if match:
                    prompt_throughput = float(match.group(1))
                    generation_throughput = float(match.group(2))
                    gpu_kv_cache = float(match.group(3))
                    prefix_cache_hit = float(match.group(4))
                    
                    metrics_data['prompt_throughput'].append(prompt_throughput)
                    metrics_data['generation_throughput'].append(generation_throughput)
                    metrics_data['gpu_kv_cache_usage'].append(gpu_kv_cache)
                    metrics_data['prefix_cache_hit_rate'].append(prefix_cache_hit)
    
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found!")
        return None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None
    
    return metrics_data


def calculate_metrics_stats(metrics_data, skip_first_last=True):
    """
    Calculate mean statistics for the extracted metrics.
    
    Args:
        metrics_data: Dictionary containing lists of metrics
        skip_first_last: Whether to skip first and last records to avoid startup/shutdown impact
        
    Returns:
        dict: Summary statistics for each metric
    """
    
    if not metrics_data:
        return None
    
    # Check if we have any data
    total_records = len(metrics_data['prompt_throughput'])
    if total_records == 0:
        print("No engine metrics found in the log file!")
        return None
    
    print(f"Found {total_records} engine metric records")
    
    # Skip first and last records if requested and we have enough data
    if skip_first_last and total_records > 2:
        filtered_data = {}
        for key, values in metrics_data.items():
            filtered_data[key] = values[1:-1]  # Skip first and last
        print(f"Using {len(filtered_data['prompt_throughput'])} records (skipped first and last)")
    else:
        filtered_data = metrics_data
        print(f"Using all {total_records} records")
    
    # Calculate statistics
    stats = {}
    for metric_name, values in filtered_data.items():
        if values:  # Make sure we have data
            stats[metric_name] = {
                'mean': mean(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            
            # Calculate standard deviation if we have more than 1 value
            if len(values) > 1:
                stats[metric_name]['std'] = stdev(values)
            else:
                stats[metric_name]['std'] = 0.0
        else:
            stats[metric_name] = {
                'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
            }
    
    return stats


def print_summary(stats):
    """Print a formatted summary of the statistics."""
    
    if not stats:
        print("No statistics to display.")
        return
    
    print("\\n" + "="*60)
    print("vLLM Server Performance Metrics Summary")
    print("="*60)
    
    # Define metric display names and units
    metric_info = {
        'prompt_throughput': ('Avg Prompt Throughput', 'tokens/s'),
        'generation_throughput': ('Avg Generation Throughput', 'tokens/s'),
        'gpu_kv_cache_usage': ('GPU KV Cache Usage', '%'),
        'prefix_cache_hit_rate': ('Prefix Cache Hit Rate', '%')
    }
    
    for metric_key, (display_name, unit) in metric_info.items():
        if metric_key in stats:
            s = stats[metric_key]
            print(f"\\n{display_name}:")
            print(f"  Mean:  {s['mean']:.2f} {unit}")
            print(f"  Min:   {s['min']:.2f} {unit}")
            print(f"  Max:   {s['max']:.2f} {unit}")
            print(f"  Std:   {s['std']:.2f} {unit}")
            print(f"  Count: {s['count']} samples")
    
    print("\\n" + "="*60)


def main():
    """Main function to parse command line arguments and run the analysis."""
    
    if len(sys.argv) != 2:
        print("Usage: python parse_vllm_throughput.py <server_log_file>")
        print("\\nExample:")
        print("  python parse_vllm_throughput.py server_output.log")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    # Check if file exists
    if not Path(log_file_path).exists():
        print(f"Error: File '{log_file_path}' does not exist!")
        sys.exit(1)
    
    print(f"Parsing vLLM server log: {log_file_path}")
    
    # Parse the log file
    metrics_data = parse_vllm_server_log(log_file_path)
    
    if metrics_data is None:
        sys.exit(1)
    
    # Calculate statistics (skip first and last records)
    stats = calculate_metrics_stats(metrics_data, skip_first_last=False)
    
    if stats is None:
        sys.exit(1)
    
    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
