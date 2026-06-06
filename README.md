# dynamic-agent-harness

A collection of production-ready multi-agent orchestration patterns in Python.

Provider-agnostic via [LiteLLM](https://docs.litellm.ai) — swap one model string to switch between Azure OpenAI, AWS Bedrock, Google Vertex AI, Ollama (local), or plain OpenAI. No other code changes needed.

---

## Why a Harness?

Single agents fail quietly on complex tasks in three ways:

| Failure Mode | What Happens |
|---|---|
| **Agentic Laziness** | Stops early, declares partial work complete |
| **Self-Preferential Bias** | Can't objectively review its own output |
| **Goal Drift** | Requirements erode across many turns and compactions |

A harness solves all three by coordinating multiple isolated agents with independent context windows and focused goals.

---

## Patterns

| Pattern | File | Use When |
|---|---|---|
| Fan-Out and Synthesize | `patterns/fan_out.py` | Task is too large for one context; parallel processing helps |
| Adversarial Verification | `patterns/adversarial.py` | Output quality must be independently validated |
| Loop Until Done | `patterns/loop.py` | Amount of work is unknown upfront |
| Model Routing | `patterns/routing.py` | Mixed-complexity workload; want to minimise cost |
| Tournament (Best-of-N) | `patterns/tournament.py` | Need the best output, not just a good one |

---

## Installation

```bash
git clone https://github.com/Ronakchhabra/dynamic-agent-harness.git
cd dynamic-agent-harness
pip install litellm
```

Set credentials via environment variables — LiteLLM picks them up automatically:

```bash
# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"

# Google Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"

# OpenAI
export OPENAI_API_KEY="..."
```

---

## Quick Start

### Fan-Out and Synthesize

```python
import asyncio
from harness.patterns.fan_out import run_fan_out_harness

modules = ["auth.py", "payments.py", "api_gateway.py"]
subtasks = [
    {
        "system": "You are a senior security engineer.",
        "user": f"Security review this module: {m}",
    }
    for m in modules
]

result = asyncio.run(run_fan_out_harness(
    goal="Find all security vulnerabilities",
    subtasks=subtasks,
))
print(result)
```

### Adversarial Verification

```python
from harness.patterns.adversarial import run_adversarial_harness

RUBRIC = "Must include line numbers, severity classification, and a concrete fix for each issue."

results = asyncio.run(run_adversarial_harness(subtasks=subtasks, rubric=RUBRIC))
```

### Tournament

```python
from harness.patterns.tournament import run_tournament

winner = asyncio.run(run_tournament(
    system="You are a senior API designer.",
    user="Design a REST endpoint for paginated audit logs filtered by user ID.",
    rubric="RESTful, self-explanatory, consistent with Google/Microsoft conventions.",
    variants=[
        "strict REST purist",
        "developer experience focus",
        "enterprise API standards",
    ],
))
print(winner)
```

### Full Migration Pipeline (composed example)

```python
from harness.migration_harness import run_migration

report = asyncio.run(run_migration(
    files=["src/auth.py", "src/payments.py", "src/api.py"],
    spec="Rename UserModel to AccountModel across all files...",
))
print(report)
```

---

## Project Structure

```
harness/
├── llm_client.py           # Universal LLM caller (LiteLLM)
├── patterns/
│   ├── fan_out.py          # Pattern 1: Fan-out + synthesise
│   ├── adversarial.py      # Pattern 2: Worker + verifier pairs
│   ├── loop.py             # Pattern 3: Loop until done
│   ├── routing.py          # Pattern 4: Classify → route to model tier
│   └── tournament.py       # Pattern 5: Best-of-N with pairwise judging
└── migration_harness.py    # Composed real-world example
```

---

## Model Strings by Provider

```python
# Azure OpenAI
FAST_MODEL   = "azure/gpt-4o-mini"
STRONG_MODEL = "azure/gpt-4o"

# AWS Bedrock
FAST_MODEL   = "bedrock/amazon.nova-lite-v1:0"
STRONG_MODEL = "bedrock/amazon.nova-pro-v1:0"

# Google Vertex AI
FAST_MODEL   = "vertex_ai/gemini-2.0-flash"
STRONG_MODEL = "vertex_ai/gemini-2.5-pro"

# Ollama (fully local/offline)
FAST_MODEL   = "ollama/qwen2.5:7b"
STRONG_MODEL = "ollama/qwen2.5:32b"

# OpenAI
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"
```

---

## Key Design Principles

| Principle | Why It Matters |
|---|---|
| Isolated context per agent | No cross-contamination between parallel workers |
| Separate generation from evaluation | Verifiers must not know they reviewed AI output |
| Explicit stop conditions | Never let the agent decide when it is done |
| Cheap model for classification | 60–90% cost reduction on mixed-complexity workloads |
| Strong model for synthesis only | Merging N outputs needs more reasoning — justify the cost |
| Always cap max iterations | `loop_until_done` with no ceiling is a runaway process |
| Pass failure context on retry | Blind retries almost never improve output |

---

## When NOT to Use a Harness

- Routine tasks that fit in one context window
- When latency matters more than thoroughness
- When token cost cannot absorb 3–10x overhead
- When a good single prompt already gets the job done

**Rule of thumb:** Use a harness when you need independence between generation and evaluation, parallelism across large workloads, or adversarial checking that can't come from the same context.

---

## Related
 
- 📖 Medium article: [Stop Giving One AI Agent All the Work — Build a Harness Instead](https://medium.com/@ronaksinghchhabra3/stop-giving-one-ai-agent-all-the-work-build-a-harness-instead-d876e91969fa)

---

*Built by Ronny | Associate Data Scientist at YASH Technologies*  
*GitHub: https://github.com/Ronakchhabra/dynamic-agent-harness*  
*LiteLLM docs: https://docs.litellm.ai*
