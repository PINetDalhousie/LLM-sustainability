#!/usr/bin/env python3
# Utility script to analyze vicuna log files for prefill and generation metrics
# Author: Monzurul

import os
import re
import sys
import argparse
import json

def extract_prefill_metrics_from_engine_log(engine_log_file):
    """Extract prefill metrics from engine log file"""
    if not os.path.exists(engine_log_file):
        return {
            'total_prefill_tokens': 0,
            'total_prefill_time': 0.0,
            'prefill_throughput': 0.0
        }
    
    with open(engine_log_file, 'r') as f:
        log_content = f.read()
    
    # Parse prefill tokens using logic similar to the awk command
    lines = log_content.split('\n')
    total_prefill_tokens = 0
    current_tokens = 0
    
    for line in lines:
        # Look for "Schedule 1 jobs" with "cur_num_batched_tokens="
        if "Schedule 1 jobs" in line and "cur_num_batched_tokens=" in line:
            # Extract tokens from cur_num_batched_tokens=X,
            parts = line.split("cur_num_batched_tokens=")
            if len(parts) > 1:
                val_part = parts[1].split(",")[0]
                try:
                    current_tokens = int(val_part)
                except ValueError:
                    current_tokens = 0
        
        # Look for "1 Fills" and add the current tokens
        elif "1 Fills" in line and current_tokens > 0:
            total_prefill_tokens += current_tokens
            current_tokens = 0
    
    # Parse prefill time from "Job Fill* finished. Latency:" pattern (time is in ms)
    prefill_latency_pattern = r"Job Fill.*finished\. Latency: ([\d.]+) ms"
    total_prefill_time = 0.0
    
    latency_matches = re.findall(prefill_latency_pattern, log_content)
    for match in latency_matches:
        latency_ms = float(match)  # Time is in milliseconds
        total_prefill_time += latency_ms / 1000.0  # Convert ms to seconds
    
    # Calculate prefill throughput (tokens/second)
    prefill_throughput = total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0.0
    
    return {
        'total_prefill_tokens': total_prefill_tokens,
        'total_prefill_time': total_prefill_time,
        'prefill_throughput': prefill_throughput
    }

def extract_generation_metrics_from_engine_log(engine_log_file):
    """Extract generation metrics from engine log file"""
    if not os.path.exists(engine_log_file):
        return {
            'total_generated_tokens': 0,
            'total_generation_time': 0.0,
            'generation_throughput': 0.0
        }
    
    with open(engine_log_file, 'r') as f:
        log_content = f.read()
    
    # Parse number of generated tokens from "X Generations). Total Time used:" pattern
    token_pattern = r"(\d+) Generations\)\. Total Time used:"
    total_generated_tokens = 0
    
    token_matches = re.findall(token_pattern, log_content)
    for match in token_matches:
        tokens = int(match)
        total_generated_tokens += tokens
    
    # Parse generation time from "Job Generate* finished. Latency:" pattern (time is in ms)
    latency_pattern = r"Job Generate.*finished\. Latency: ([\d.]+) ms"
    total_generation_time = 0.0
    
    latency_matches = re.findall(latency_pattern, log_content)
    for match in latency_matches:
        latency_ms = float(match)  # Time is in milliseconds
        total_generation_time += latency_ms / 1000.0  # Convert ms to seconds
    
    # Calculate throughput (tokens/second)
    generation_throughput = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0.0
    
    return {
        'total_generated_tokens': total_generated_tokens,
        'total_generation_time': total_generation_time,
        'generation_throughput': generation_throughput
    }

def analyze_vicuna_log(log_file_path, output_format='text'):
    """Analyze a vicuna log file and return prefill and generation metrics"""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return None
    
    try:
        # Extract prefill metrics
        prefill_metrics = extract_prefill_metrics_from_engine_log(log_file_path)
        
        # Extract generation metrics
        generation_metrics = extract_generation_metrics_from_engine_log(log_file_path)
        
        # Combine all metrics
        all_metrics = {
            'file_path': log_file_path,
            'prefill_metrics': prefill_metrics,
            'generation_metrics': generation_metrics,
            'summary': {
                'total_tokens': prefill_metrics['total_prefill_tokens'] + generation_metrics['total_generated_tokens'],
                'total_time': prefill_metrics['total_prefill_time'] + generation_metrics['total_generation_time'],
                'overall_throughput': (prefill_metrics['total_prefill_tokens'] + generation_metrics['total_generated_tokens']) / 
                                    (prefill_metrics['total_prefill_time'] + generation_metrics['total_generation_time']) 
                                    if (prefill_metrics['total_prefill_time'] + generation_metrics['total_generation_time']) > 0 else 0.0
            }
        }
        
        return all_metrics
        
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        return None

def print_metrics(metrics, output_format='text'):
    """Print metrics in the specified format"""
    if metrics is None:
        return
    
    if output_format == 'json':
        print(json.dumps(metrics, indent=2))
    else:
        print("VICUNA LOG ANALYSIS RESULTS")
        print("=" * 50)
        print(f"File: {metrics['file_path']}")
        print()
        
        print("PREFILL METRICS:")
        print(f"  Tokens: {metrics['prefill_metrics']['total_prefill_tokens']}")
        print(f"  Time: {metrics['prefill_metrics']['total_prefill_time']:.3f} seconds")
        print(f"  Throughput: {metrics['prefill_metrics']['prefill_throughput']:.2f} tokens/s")
        print()
        
        print("GENERATION METRICS:")
        print(f"  Tokens: {metrics['generation_metrics']['total_generated_tokens']}")
        print(f"  Time: {metrics['generation_metrics']['total_generation_time']:.3f} seconds")
        print(f"  Throughput: {metrics['generation_metrics']['generation_throughput']:.2f} tokens/s")
        print()
        
        print("SUMMARY:")
        print(f"  Total Tokens: {metrics['summary']['total_tokens']}")
        print(f"  Total Time: {metrics['summary']['total_time']:.3f} seconds")
        print(f"  Overall Throughput: {metrics['summary']['overall_throughput']:.2f} tokens/s")
        print()
        
        # Additional breakdown
        if metrics['summary']['total_tokens'] > 0:
            prefill_pct = (metrics['prefill_metrics']['total_prefill_tokens'] / metrics['summary']['total_tokens']) * 100
            generation_pct = (metrics['generation_metrics']['total_generated_tokens'] / metrics['summary']['total_tokens']) * 100
            print("TOKEN BREAKDOWN:")
            print(f"  Prefill: {prefill_pct:.1f}% ({metrics['prefill_metrics']['total_prefill_tokens']} tokens)")
            print(f"  Generation: {generation_pct:.1f}% ({metrics['generation_metrics']['total_generated_tokens']} tokens)")
            print()
        
        if metrics['summary']['total_time'] > 0:
            prefill_time_pct = (metrics['prefill_metrics']['total_prefill_time'] / metrics['summary']['total_time']) * 100
            generation_time_pct = (metrics['generation_metrics']['total_generation_time'] / metrics['summary']['total_time']) * 100
            print("TIME BREAKDOWN:")
            print(f"  Prefill: {prefill_time_pct:.1f}% ({metrics['prefill_metrics']['total_prefill_time']:.3f} seconds)")
            print(f"  Generation: {generation_time_pct:.1f}% ({metrics['generation_metrics']['total_generation_time']:.3f} seconds)")

def main():
    parser = argparse.ArgumentParser(description='Analyze vicuna log files for prefill and generation metrics')
    parser.add_argument('log_file', help='Path to the vicuna log file (e.g., engine_1_vicuna_7b.log)')
    parser.add_argument('--format', choices=['text', 'json'], default='text', 
                       help='Output format (default: text)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Analyze the log file
    metrics = analyze_vicuna_log(args.log_file, args.format)
    
    if metrics is None:
        sys.exit(1)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'json':
                json.dump(metrics, f, indent=2)
            else:
                # Redirect stdout to file for text output
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                print_metrics(metrics, args.format)
                sys.stdout = old_stdout
        print(f"Results saved to: {args.output}")
    else:
        print_metrics(metrics, args.format)

if __name__ == "__main__":
    main()
