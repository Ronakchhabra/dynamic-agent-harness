"""
patterns/loop.py
─────────────────
Pattern 3: Loop Until Done

Keep spawning agents in a loop, passing accumulated history each time.
Stop when an external stop condition is satisfied — not when the agent
decides it is done. Always enforces a hard maximum iteration ceiling.

    history = []
    loop:
        result = agent(history)
        history.append(result)
        if stop_fn(result): break
        if iterations >= max: break

Use when:
  - The amount of work is unknown upfront (open-ended triage, research)
  - Each pass should build on the results of previous passes
  - You need a safety ceiling on runaway loops
  - Tasks like: bug triage, iterative refinement, backlog processing, monitoring
"""

import asyncio
from typing import Awaitable, Callable
from harness.llm_client import call_llm

# ── Configure your models here ────────────────────────────────────────────────
FAST_MODEL = "azure/gpt-4o-mini"


# ── Core function ──────────────────────────────────────────────────────────────

async def loop_until_done(
    task_fn: Callable[[list[str]], Awaitable[str]],
    stop_fn: Callable[[str], bool],
    max_iterations: int = 15,
) -> list[str]:
    """
    Repeatedly call task_fn until stop_fn returns True or max_iterations is reached.

    Design notes:
    - task_fn receives the full history of prior results so each pass has context.
    - stop_fn is defined externally — never let the agent decide when it is done.
    - max_iterations is a hard ceiling, not a guideline. Set it deliberately.

    Args:
        task_fn:        Async function that takes list[str] (history) and returns str.
        stop_fn:        Sync function that takes the latest result and returns bool.
        max_iterations: Hard ceiling on number of loop passes.

    Returns:
        List of all results produced across all iterations.
    """
    history: list[str] = []

    for i in range(max_iterations):
        print(f"[Loop] Iteration {i + 1}/{max_iterations}...")
        result = await task_fn(history)
        history.append(result)

        if stop_fn(result):
            print(f"[Loop] Stop condition met after {i + 1} iteration(s).")
            return history

    print(f"[Loop] Reached max iterations ({max_iterations}). Returning accumulated results.")
    return history


# ── Example 1: Bug triage until backlog is clear ───────────────────────────────

async def triage_agent(prior_results: list[str]) -> str:
    """Triage agent that builds on previous passes."""
    context = "\n\n---\n\n".join(prior_results) if prior_results else "No prior triage passes."
    return await call_llm(
        system=(
            "You are a bug triage agent. Your job is to classify and address critical bugs "
            "in the backlog. Work through them systematically. "
            "When ALL critical bugs have been addressed, end your response with exactly: TRIAGE COMPLETE"
        ),
        user=(
            f"Prior triage passes:\n{context}\n\n"
            "Continue triaging. Address the next batch of critical bugs. "
            "If all critical bugs are done, say TRIAGE COMPLETE."
        ),
        model=FAST_MODEL,
    )


def triage_done(result: str) -> bool:
    return "TRIAGE COMPLETE" in result.upper()


# ── Example 2: Research loop until no new findings ────────────────────────────

async def research_agent(prior_results: list[str]) -> str:
    """Research agent that digs deeper each pass."""
    context = "\n\n---\n\n".join(prior_results) if prior_results else "No prior research."
    return await call_llm(
        system=(
            "You are a research agent. Each pass, explore a different angle of the topic. "
            "Avoid repeating findings already covered. "
            "When you have exhausted all meaningful angles, end with: NO NEW FINDINGS"
        ),
        user=(
            f"Prior research:\n{context}\n\n"
            "What new angles or findings can you add? "
            "If nothing meaningful remains, say NO NEW FINDINGS."
        ),
        model=FAST_MODEL,
    )


def research_done(result: str) -> bool:
    return "NO NEW FINDINGS" in result.upper()


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Running: Bug Triage Loop ===")
    triage_results = asyncio.run(
        loop_until_done(
            task_fn=triage_agent,
            stop_fn=triage_done,
            max_iterations=10,
        )
    )
    print(f"\nCompleted triage in {len(triage_results)} pass(es).")
    print("\nFinal pass:\n", triage_results[-1])
