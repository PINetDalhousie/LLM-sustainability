# Workload Implementations

This directory contains the four representative multi-request LLM workloads evaluated in the paper. Each workload captures a distinct dependency structure and interaction pattern common in modern LLM-based applications.

---

## Workload Overview

### (a) Document Chain Summarization — `chain_summarization.py`

A sequential pipeline for summarizing long documents that exceed a model's context window. The document is split into semantically coherent chunks; the first chunk produces an initial summary, and each subsequent chunk refines it through a refinement loop. Because the output of each LLM call is fed as input to the next, requests are strictly ordered and cannot be batched together.

```
Document
  │
  ├─► Chunk 1 ──► Summary₁
  │                   │
  ├─► Chunk 2 ──► Summary₂
  │                   │
  ├─► Chunk N ──►  Final Summary
```

arization.png![Document Chain Summarization Workflow](../assets/chain-summarization-workflow.pdf)

**Key characteristics:**
- Decode-heavy; each LLM call must complete before the next begins
- Sequential dependencies eliminate batching opportunities
- Input length directly amplifies prefill computation
- Evaluated on long-form academic papers from the [Arxiv dataset](https://arxiv.org)

**Usage:**
```bash
python3 examples/chain_summarization.py --chunk_size 2000
```

---

### (b) LLM-powered Search — `bingCopilot.py`

Models a productivity copilot (e.g., Bing Chat, Google Gemini) that attaches a large static system prompt (~3000 tokens encoding role specifications, safety guidelines, and few-shot exemplars) to every user query. Each user session triggers 2–8 dependent LLM calls. The shared system prompt is preserved across all requests in a session, enabling high prefix-cache reuse when requests are batched.

```
User Input
    │
    ▼
Input Reformulation
    │
    ▼
Answer Validation
    │
    ▼
Safety Filtering
    │
    ▼
Final Response
```

![LLM-powered Search Workflow](../assets/bing-copilot-workflow.pdf)

**Key characteristics:**
- Prefill-heavy due to the long static system prompt
- High prefix-cache reuse across concurrent queries → batching is highly effective
- 64 diverse queries (AI, IoT, renewable energy, biology) appended to the shared prompt
- Responses capped at 800 output tokens

**Usage:**
```bash
python3 examples/bingCopilot.py "What are recent advances in renewable energy?"
# or interactive mode
python3 examples/bingCopilot.py
```

---

### (c) Multi-Agent Coding — `agentic_coding.py`

A collaborative software-development workflow following the [MetaGPT](https://github.com/FoundationAgents/MetaGPT) paradigm. The task (building a CLI snake game) is decomposed across four specialized roles whose outputs flow through a directed acyclic graph (DAG): the architect designs the system and APIs; three parallel engineers implement the files; a reviewer provides quality feedback; three parallel revisers apply the review.

```
          Task
           │
       Architect
           │
   ┌───────┼───────┐
Engineer Engineer Engineer
   └───────┼───────┘
           │
        Reviewer
           │
   ┌───────┼───────┐
 Reviser Reviser Reviser
   └───────┼───────┘
           │
         Output
```
![Multi-Agent Coding Workflow](../assets/multi-agent-coding-workflow.pdf)

**Key characteristics:**
- DAG execution: downstream agents cannot proceed until upstream agents complete
- Heterogeneous roles have structurally distinct prompts with minimal shared prefixes → poor KV-cache reuse at small batch sizes
- 15–20 LLM calls per workflow instance
- Token budgets: ~1000 tokens for coding roles, ~500 for review/revision

**Usage:**
```bash
python3 examples/agentic_coding.py --max_tokens 1000
```

---

### (d) Composite Workload

The composite workload runs the Document Chain Summarization and LLM-powered Search pipelines **concurrently** on the same serving-system deployment. It mimics a multi-tenant datacenter scenario where diverse applications share a GPU pool, exposing interference effects between a decode-heavy sequential pipeline and a prefill-heavy interactive copilot.

**Key characteristics:**
- Competing demands on batching, KV-cache reuse, and scheduling fairness
- Prefix-heavy search queries overlap with stateful sequential summarization stages
- Implemented by launching both `chain_summarization.py` and `bingCopilot.py` concurrently against the same vLLM/Parrot server

---

## Energy Measurement

All workloads use [Zeus](https://github.com/ml-energy/zeus) for fine-grained component-level energy measurement:

```python
from zeus.monitor import ZeusMonitor

zeus_monitor = ZeusMonitor()
zeus_monitor.begin_window("workflow_name")

# ... run workload ...

mes = zeus_monitor.end_window("workflow_name")
print(f"Time: {mes.time:.2f} s")
print(f"CPU energy:  {sum(mes.cpu_energy.values()):.2f} J")
print(f"GPU energy:  {sum(mes.gpu_energy.values()):.2f} J")
print(f"DRAM energy: {sum(mes.dram_energy.values()):.2f} J")
```

Zeus samples at 1-second granularity using RAPL (CPU/DRAM) and NVML (GPU), providing per-component breakdowns without hardware modifications.