# Characterizing Performance–Energy Trade-offs of Large Language Models in Multi-Request Workflows

**Md. Monzurul Amin Ifath** and **Israat Haque**  
Dalhousie University, Canada

*Proc. ACM Meas. Anal. Comput. Syst., Vol. 10, No. 1, Article 7 (March 2026)*  
DOI: [10.1145/3788089](https://doi.org/10.1145/3788089)

---

## Overview

This repository contains the workload implementations, experiment scripts, and result plots for the paper above — the **first systematic characterization of performance–energy trade-offs in multi-request LLM inference**.

Modern LLM applications (document summarization pipelines, search copilots, multi-agent programming frameworks) chain multiple LLM calls together. Unlike single-request benchmarks, these workflows introduce inter-request dependencies, heterogeneous prompt structures, and critical-path latency accumulation that reshape how energy knobs behave. This work characterizes those effects empirically on a controlled NVIDIA A100 testbed using two state-of-the-art serving systems — **vLLM** and **Parrot**.

---

## Key Findings

- **Batch size** is the most impactful energy knob, but its effect is highly workload-dependent. It reduces energy per token by up to **2.6×** for LLM-powered search (via prefix-cache reuse) while *increasing* energy for sequential chain summarization (sequential dependencies prevent batching gains).
- **Output length** induces near-linear energy scaling with limited efficiency improvements.
- **GPU power capping** provides modest but predictable savings (~21% for Parrot, ~12.6% for vLLM at 150 W) at the cost of higher latency.
- **vLLM** (PagedAttention + continuous batching + chunked prefill) sustains higher GPU utilization (mean 90.1% vs. 63.3% for Parrot) and achieves up to **28% lower total energy** on the most energy-intensive workload.
- **Parrot**'s workflow-aware scheduling yields relative gains only under stringent power constraints.

---

## Workloads

| Workload | Pattern | LLM Calls per Instance |
|---|---|---|
| Document Chain Summarization | Sequential pipeline | 2–40 |
| LLM-powered Search | Interactive copilot (shared system prompt) | 2–8 |
| Multi-Agent Coding | DAG (architect → engineer → reviewer → reviser) | 15–20 |
| Composite | Mixed summarization + search (concurrent) | variable |

All workloads follow the [LangChain](https://github.com/langchain-ai/langchain) orchestration paradigm.

---

## Experimental Setup

| Component | Specification |
|---|---|
| Server | Supermicro SYS-740GP-TRT |
| CPU | 2× Intel Xeon 4310 (12-core, 2.1 GHz) |
| GPU | NVIDIA Ampere A100 (40 GB HBM2) |
| RAM | 128 GB DDR4 2933 MHz |
| Software | CUDA 12.6, cuDNN 9.1.0, Ubuntu 24.04.2 LTS |
| Model | Llama-2-7B (decoder-only, 4096-token context) |
| Serving systems | vLLM v0.9.1 (V1 engine), Parrot (OSDI'24) |
| Energy measurement | [Zeus](https://github.com/ml-energy/zeus) (CPU via RAPL, GPU via NVML, DRAM) |

Each experiment is repeated **10 times**; results report the mean with 95% confidence intervals.

---

## Repository Structure

```
.
├── examples/                        # Workload implementations
│   ├── chain_summarization.py       # Document chain summarization (LangChain + vLLM)
│   ├── bingCopilot.py               # LLM-powered search / copilot workload
│   └── agentic_coding.py            # Multi-agent coding (architect → engineer → reviewer → reviser)
│
├── scripts/                         # Experiment automation scripts
│   ├── chain_summarization/
│   │   ├── varying_batch_size.sh        # Sweep max-num-seqs (batch size)
│   │   ├── vllm_varying_chunk_size.sh   # Sweep input chunk/token length
│   │   ├── vllm_varying_output_limit.sh # Sweep max output tokens
│   │   └── power_cap_experiment.sh      # Sweep GPU power cap (nvidia-smi -pl)
│   ├── bing_copilot/
│   │   ├── batch_size_experiment.sh
│   │   └── power_cap_experiment.sh
│   ├── agentic_coding/
│   │   ├── batch_size_experiment.sh
│   │   └── power_cap_experiment.sh
│   └── utilities/
│       ├── parse_vllm_throughput_single_log.py
│       ├── parse_parrot_throughput_single_log.py
│       └── plot_utils.py
│
└── plots/                           # Generated result figures (PDF)
    ├── RQ2_energy_per_token_vs_batch_size.pdf
    ├── RQ2_prefix_cache_boxplot.pdf
    ├── RQ3_compare_GPU_utilization.pdf
    ├── chain_summarization/
    │   ├── batch_size_exp/
    │   ├── chunk_size_exp/
    │   ├── output_limit_exp/
    │   └── power_cap_exp/
    └── LLM_powered_search/
        └── dual_axes_plots/
            ├── batch_size_plots/
            ├── output_limit_plots/
            └── power_cap_plots/
```

---

## Dependencies

```bash
pip install vllm==0.9.1
pip install zeus
pip install langchain langchain-openai langchain-community
pip install openai tiktoken
```

Parrot (OSDI'24) must be installed separately from the [official repository](https://github.com/microsoft/ParrotServe).

Both serving systems are run in isolated Docker containers during experiments.

---

## Running Experiments

### 1. Start the vLLM server

```bash
vllm serve meta-llama/Llama-2-7b \
    --seed 0 \
    --max-num-seqs <batch_size> \
    --max-num-batched-tokens 2560 \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --api-key token123
```

### 2. Run a workload example

```bash
# Document chain summarization
python3 examples/chain_summarization.py --chunk_size 2000

# LLM-powered search (copilot)
python3 examples/bingCopilot.py "What are the latest trends in renewable energy?"

# Multi-agent coding
python3 examples/agentic_coding.py --max_tokens 1000
```

### 3. Run a full experiment sweep

Each script in `scripts/` starts and stops vLLM automatically, sweeps the target knob, and saves per-run logs under `log/`.

```bash
# Batch size sweep for chain summarization
bash scripts/chain_summarization/varying_batch_size.sh

# GPU power cap sweep for chain summarization
bash scripts/chain_summarization/power_cap_experiment.sh

# Batch size sweep for LLM-powered search
bash scripts/bing_copilot/batch_size_experiment.sh

# Batch size sweep for multi-agent coding
bash scripts/agentic_coding/batch_size_experiment.sh
```

Experiment logs are saved under `log/<workload>/<knob>_<config>_run_<n>/`.

---

## Research Questions

| RQ | Question | Key Result |
|---|---|---|
| **RQ1** | Which energy knob most significantly influences performance and efficiency? | Batch size dominates; output length scales linearly; power capping is a fine-tuning lever |
| **RQ2** | How does batch size shape trade-offs across all four workloads? | Workload-dependent: up to 2.6× improvement for search, degradation for summarization |
| **RQ3** | How do vLLM and Parrot compare on the most energy-intensive workload? | vLLM: up to 28% lower energy; Parrot: modest gains only under strict power caps |

---

## Citation

If you use this code or findings, please cite:

```bibtex
@article{ifath2026characterizing,
  author    = {Md. Monzurul Amin Ifath and Israat Haque},
  title     = {Characterizing Performance--Energy Trade-offs of Large Language Models
               in Multi-Request Workflows},
  journal   = {Proc. ACM Meas. Anal. Comput. Syst.},
  volume    = {10},
  number    = {1},
  articleno = {7},
  year      = {2026},
  month     = {March},
  pages     = {26},
  doi       = {10.1145/3788089}
}
```

---

## Acknowledgments

We thank the anonymous SIGMETRICS reviewers, Nael Abu-Ghazaleh (UC Riverside), and Mosharaf Chowdhury (University of Michigan) for their feedback. GPU resources were generously provided by Gabriel Spadon De Souza (Dalhousie University) and the PINet and MAPS Labs at Dalhousie University.
