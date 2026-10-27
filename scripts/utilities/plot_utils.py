"""
Utility functions for parsing experiment logs and creating comparison plots
with confidence intervals.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
plt.style.use("scripts/paper.mplstyle")


def parse_parrot_log(filepath, experiment_type='chunk_size'):
    """
    Parse Parrot experiment log file.
    
    Args:
        filepath: Path to the log file
        experiment_type: 'chunk_size' or 'output_limit'
    
    Returns:
        tuple: (results_dict, raw_data_dict)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Determine pattern based on experiment type
    if experiment_type == 'chunk_size':
        param_pattern = r"Chunk size: (\d+)"
        section_prefix = "Chunk size: "
    else:  # output_limit
        param_pattern = r"Token Limit: (\d+)"
        section_prefix = "Token Limit: "
    
    param_values = re.findall(param_pattern, content)
    results = {}
    raw_data = {}
    
    for param_value in param_values:
        # Section for this parameter value
        section_pattern = section_prefix + param_value + r"\nRunning \d+ times...\n\n(.*?)(?=----------------------------------------|$)"
        section_match = re.search(section_pattern, content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            # Parse individual runs
            run_pattern = (
                r"Total workflow consumed [\d.]+ s and Measurement\(time=([\d.]+), "
                r"gpu_energy=\{0: ([\d.]+)\}, "
                r"cpu_energy=\{0: ([-\d.]+), 1: ([-\d.]+)\}, "
                r"dram_energy=\{0: ([-\d.]+), 1: ([-\d.]+)\}, soc_energy=None\) J"
            )
            runs = re.findall(run_pattern, section)
            if runs:
                times, gpus, cpus, drams = [], [], [], []
                run_details = []
                
                for i, run in enumerate(runs):
                    # time, gpu, cpu0, cpu1, dram0, dram1
                    t = float(run[0])
                    gpu = float(run[1])
                    cpu_total = float(run[2]) + float(run[3])
                    dram_total = float(run[4]) + float(run[5])
                    
                    # Store raw data for analysis
                    run_detail = {
                        'run': i + 1,
                        'time': t,
                        'gpu': gpu,
                        'cpu': cpu_total,
                        'dram': dram_total,
                        'total': gpu + cpu_total + dram_total
                    }
                    run_details.append(run_detail)
                    
                    # Filter out negative energy values
                    times.append(t)  # Time is always positive
                    if gpu >= 0:
                        gpus.append(gpu)
                    if cpu_total >= 0:
                        cpus.append(cpu_total)
                    if dram_total >= 0:
                        drams.append(dram_total)
                
                # Calculate means only from valid (non-negative) values
                raw_data[int(param_value)] = run_details
                results[int(param_value)] = {
                    'time': np.mean(times) if times else 0,
                    'gpu': np.mean(gpus) if gpus else 0,
                    'cpu': np.mean(cpus) if cpus else 0,
                    'dram': np.mean(drams) if drams else 0,
                    'total': (np.mean(gpus) if gpus else 0) + (np.mean(cpus) if cpus else 0) + (np.mean(drams) if drams else 0),
                    'valid_counts': {
                        'time': len(times),
                        'gpu': len(gpus),
                        'cpu': len(cpus),
                        'dram': len(drams)
                    }
                }
    
    return results, raw_data


def parse_vllm_log(filepath, experiment_type='chunk_size'):
    """
    Parse vLLM experiment log file.
    
    Args:
        filepath: Path to the log file
        experiment_type: 'chunk_size' or 'output_limit'
    
    Returns:
        tuple: (results_dict, raw_data_dict)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Determine pattern based on experiment type
    if experiment_type == 'chunk_size':
        param_pattern = r"Chunk size: (\d+)"
        section_prefix = "Chunk size: "
    else:  # output_limit
        param_pattern = r"Token Limit: (\d+)"
        section_prefix = "Token Limit: "
    
    param_values = re.findall(param_pattern, content)
    results = {}
    raw_data = {}
    
    for param_value in param_values:
        section_pattern = section_prefix + param_value + r"\nRunning \d+ times...\n\n(.*?)(?=----------------------------------------|$)"
        section_match = re.search(section_pattern, content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            run_pattern = (
                r"Chain summarization application consumed Time: ([\d.]+) s, CPU energy: ([\d.]+) J, GPU energy: ([\d.]+) J, DRAM energy: ([\d.]+) J & Total energy: ([\d.]+) J during inference\."
            )
            runs = re.findall(run_pattern, section)
            if runs:
                times, cpus, gpus, drams, totals = [], [], [], [], []
                run_details = []
                
                for i, run in enumerate(runs):
                    t = float(run[0])
                    cpu = float(run[1])
                    gpu = float(run[2])
                    dram = float(run[3])
                    total = float(run[4])
                    
                    # Store raw data for analysis
                    run_detail = {
                        'run': i + 1,
                        'time': t,
                        'cpu': cpu,
                        'gpu': gpu,
                        'dram': dram,
                        'total': total
                    }
                    run_details.append(run_detail)
                    
                    # Filter out negative energy values
                    times.append(t)  # Time is always positive
                    if cpu >= 0:
                        cpus.append(cpu)
                    if gpu >= 0:
                        gpus.append(gpu)
                    if dram >= 0:
                        drams.append(dram)
                    if total >= 0:
                        totals.append(total)
                
                # Calculate means only from valid (non-negative) values
                raw_data[int(param_value)] = run_details
                results[int(param_value)] = {
                    'time': np.mean(times) if times else 0,
                    'gpu': np.mean(gpus) if gpus else 0,
                    'cpu': np.mean(cpus) if cpus else 0,
                    'dram': np.mean(drams) if drams else 0,
                    'total': np.mean(totals) if totals else 0,
                    'valid_counts': {
                        'time': len(times),
                        'cpu': len(cpus),
                        'gpu': len(gpus),
                        'dram': len(drams),
                        'total': len(totals)
                    }
                }
    
    return results, raw_data


def save_parsed_data(parrot_results, parrot_raw, vllm_results, vllm_raw, filepath):
    """Save parsed data to JSON file for reuse"""
    data = {
        'parrot': {
            'results': parrot_results,
            'raw_data': parrot_raw
        },
        'vllm': {
            'results': vllm_results,
            'raw_data': vllm_raw
        }
    }
    
    # Convert numpy types to regular Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    data = convert_numpy(data)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Parsed data saved to {filepath}")


def load_parsed_data(filepath):
    """Load parsed data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded parsed data from {filepath}")
        return (data['parrot']['results'], data['parrot']['raw_data'], 
                data['vllm']['results'], data['vllm']['raw_data'])
    except FileNotFoundError:
        print(f"No cached data found at {filepath}")
        return None, None, None, None


def print_filtering_summary(parrot_results, vllm_results, experiment_type='chunk_size'):
    """Print summary of filtering negative values"""
    param_name = "Chunk Size" if experiment_type == 'chunk_size' else "Token Limit"
    
    print("\n=== Filtering Summary ===")
    for param_value in sorted(parrot_results.keys()):
        parrot_counts = parrot_results[param_value]['valid_counts']
        vllm_counts = vllm_results[param_value]['valid_counts']
        
        print(f"\n{param_name} {param_value}:")
        print(f"  Parrot - Valid runs: GPU:{parrot_counts['gpu']}/3, CPU:{parrot_counts['cpu']}/3, DRAM:{parrot_counts['dram']}/3")
        print(f"  vLLM   - Valid runs: GPU:{vllm_counts['gpu']}/3, CPU:{vllm_counts['cpu']}/3, DRAM:{vllm_counts['dram']}/3")


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a list of values"""
    if len(data) <= 1:
        return 0
    n = len(data)
    m = np.mean(data)
    s = np.std(data, ddof=1)
    t = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t * (s / np.sqrt(n))
    return margin


def plot_comparison_with_ci(parrot, vllm, parrot_raw, vllm_raw, metric, ylabel, filename, experiment_type='chunk_size'):
    """
    Create comparison plot with confidence intervals.
    
    Args:
        parrot: Parrot results dictionary
        vllm: vLLM results dictionary
        parrot_raw: Parrot raw data dictionary
        vllm_raw: vLLM raw data dictionary
        metric: Metric to plot ('cpu', 'gpu', 'dram', 'total', 'time')
        ylabel: Y-axis label
        filename: Output filename
        experiment_type: 'chunk_size' or 'output_limit'
    """
    param_values = sorted(set(parrot.keys()) & set(vllm.keys()), key=int)
    
    parrot_vals = []
    vllm_vals = []
    parrot_errors = []
    vllm_errors = []
    
    for pv in param_values:
        parrot_vals.append(parrot[pv][metric])
        vllm_vals.append(vllm[pv][metric])
        
        # Extract raw values for this metric and parameter value
        parrot_raw_vals = [run[metric] for run in parrot_raw[pv] if run[metric] >= 0]
        vllm_raw_vals = [run[metric] for run in vllm_raw[pv] if run[metric] >= 0]
        
        # Calculate 95% confidence intervals
        parrot_errors.append(calculate_confidence_interval(parrot_raw_vals))
        vllm_errors.append(calculate_confidence_interval(vllm_raw_vals))
    
    x = np.arange(len(param_values))
    width = 0.35

    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, parrot_vals, width, label='Parrot', color='orange', 
            capsize=5) # yerr=parrot_errors,
    plt.bar(x + width/2, vllm_vals, width, label='vLLM', color='navy', 
            capsize=5) # yerr=vllm_errors,

    # Set appropriate x-axis label based on experiment type
    xlabel = 'Chunk Size' if experiment_type == 'chunk_size' else 'Output Token Limit'
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    # plt.title(f'Parrot vs vLLM: {ylabel} Comparison (95% CI)', fontsize=15)
    plt.xticks(x, [str(pv) for pv in param_values])
        # Limit y-axis ticks to maximum 6 with nice spacing
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_experiment_plots(parrot_log, vllm_log, cached_data_file, plot_directory, experiment_type='chunk_size'):
    """
    Main function to create all experiment plots.
    
    Args:
        parrot_log: Path to Parrot log file
        vllm_log: Path to vLLM log file
        cached_data_file: Path to cached data JSON file
        plot_directory: Directory to save plots
        experiment_type: 'chunk_size' or 'output_limit'
    """
    # Try to load cached data first
    parrot_data, parrot_raw, vllm_data, vllm_raw = load_parsed_data(cached_data_file)
    
    if parrot_data is None or vllm_data is None:
        # Parse logs if no cached data available
        print("Parsing log files...")
        parrot_data, parrot_raw = parse_parrot_log(parrot_log, experiment_type)
        vllm_data, vllm_raw = parse_vllm_log(vllm_log, experiment_type)
        
        # Save parsed data for future use
        os.makedirs(os.path.dirname(cached_data_file), exist_ok=True)
        save_parsed_data(parrot_data, parrot_raw, vllm_data, vllm_raw, cached_data_file)
    
    # Print filtering summary
    print_filtering_summary(parrot_data, vllm_data, experiment_type)

    # Create plots directory
    os.makedirs(plot_directory, exist_ok=True)
    
    # Generate all comparison plots
    metrics = [
        ('cpu', 'Workflow Mean CPU Energy (J)'),
        ('gpu', 'Workflow Mean GPU Energy (J)'),
        ('dram', 'Workflow Mean DRAM Energy (J)'),
        ('total', 'Workflow Mean Total Energy (J)'),
        ('time', 'Workflow Mean Latency (s)')
    ]
    
    for metric, ylabel in metrics:
        filename = f"{plot_directory}/{metric}_energy_comparison.pdf" if metric != 'time' else f"{plot_directory}/latency_comparison.pdf"
        plot_comparison_with_ci(parrot_data, vllm_data, parrot_raw, vllm_raw, metric, ylabel, filename, experiment_type)
    
    print(f"\nPlots saved to {plot_directory}")
