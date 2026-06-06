"""
patterns/fan_out.py
────────────────────
Pattern 1: Fan-Out and Synthesize

Split a large task into N parallel subtasks. Each subtask runs in its own
isolated context window. A synthesiser agent (using a stronger model) merges
all results into one coherent output.

                Task
                 ├── Worker 1 → result 1 ──┐
                 ├── Worker 2 → result 2 ──┤→ Synthesiser → Final output
                 └── Worker N → result N ──┘

Use when:
  - Task is too large for one context window
  - Parallel processing speeds things up
  - Independent perspectives improve quality (e.g. security reviews, research)
  - Each subtask benefits from a clean context (no cross-contamination)
"""

import asyncio
from harness.llm_client import call_llm

# ── Configure your models here ────────────────────────────────────────────────
FAST_MODEL   = "azure/gpt-4o-mini"   # Workers — cheap and parallel
STRONG_MODEL = "azure/gpt-4o"        # Synthesiser — needs stronger reasoning


# ── Core functions ─────────────────────────────────────────────────────────────

async def run_worker(task: dict) -> str:
    """
    Single isolated agent call.

    Each worker gets its own context — no shared state with other workers.
    This is what prevents cross-contamination between parallel tasks.

    task dict keys:
        system (str): Role/behaviour prompt for this worker.
        user   (str): The specific subtask this worker should handle.
        model  (str, optional): Override the default FAST_MODEL for this worker.
    """
    return await call_llm(
        system=task["system"],
        user=task["user"],
        model=task.get("model", FAST_MODEL),
    )


async def fan_out(subtasks: list[dict]) -> list[str]:
    """
    Spawn all workers in parallel using asyncio.gather().

    Results are returned in the same order as the input subtasks.
    All workers run concurrently — total time ≈ slowest single worker, not sum of all.
    """
    return await asyncio.gather(*[run_worker(t) for t in subtasks])


async def synthesise(results: list[str], goal: str) -> str:
    """
    Merge parallel outputs into one coherent result.

    Uses the stronger model — synthesis requires understanding N independent
    outputs and producing one deduplicated, coherent result. Worth the cost.
    """
    joined = "\n\n---\n\n".join(
        f"[Agent {i + 1}]\n{r}" for i, r in enumerate(results)
    )
    return await call_llm(
        system=(
            "You are a synthesis agent. You receive outputs from multiple independent agents "
            "who each worked on a portion of a larger task. Your job is to merge them into one "
            "clear, deduplicated result that fully addresses the original goal. "
            "Preserve all unique findings. Remove redundancies. Do not add new content."
        ),
        user=f"Original goal: {goal}\n\nAgent outputs:\n{joined}",
        model=STRONG_MODEL,
    )


async def run_fan_out_harness(goal: str, subtasks: list[dict]) -> str:
    """
    Full fan-out pipeline: spawn workers → synthesise results.

    Args:
        goal:      The overall objective — passed to the synthesiser for context.
        subtasks:  List of task dicts, each with 'system' and 'user' keys.

    Returns:
        Synthesised final output as a string.
    """
    print(f"[FanOut] Spawning {len(subtasks)} agents in parallel...")
    results = await fan_out(subtasks)
    print("[FanOut] All workers done. Synthesising...")
    output = await synthesise(results, goal)
    print("[FanOut] Complete.")
    return output


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modules = ["auth.py", "payments.py", "api_gateway.py", "user_service.py"]

    goal = "Find all security vulnerabilities across these modules"
    subtasks = [
        {
            "system": "You are a senior security engineer. Be precise. List only confirmed issues with severity.",
            "user": f"Security review this Python module: {m}",
        }
        for m in modules
    ]

    result = asyncio.run(run_fan_out_harness(goal=goal, subtasks=subtasks))
    print("\n── Final Output ──────────────────────────────")
    print(result)
