"""
patterns/adversarial.py
────────────────────────
Pattern 2: Adversarial Verification

Each worker's output is independently challenged by a separate verifier agent.
The verifier has no knowledge that it is reviewing AI-generated content.
Failed outputs are retried with the specific failure reason fed back in.

    Worker Agent ──→ output ──→ Verifier Agent ──→ pass ──→ return output
                                                 └──→ fail ──→ retry with reason

Use when:
  - Output quality must be independently validated (code review, compliance checks)
  - Self-preferential bias is a risk (same agent that generated can't verify)
  - You need structured QA — pass/fail with specific reasons, not vague feedback
  - Report fact-checking, security findings, migration correctness
"""

import asyncio
import json
from harness.llm_client import call_llm

# ── Configure your models here ────────────────────────────────────────────────
FAST_MODEL   = "azure/gpt-4o-mini"   # Workers
STRONG_MODEL = "azure/gpt-4o"        # Verifiers — verification needs stronger reasoning


# ── Core functions ─────────────────────────────────────────────────────────────

async def run_worker(task: dict) -> str:
    """Single worker agent call."""
    return await call_llm(
        system=task["system"],
        user=task["user"],
        model=task.get("model", FAST_MODEL),
    )


async def verify(output: str, rubric: str) -> dict:
    """
    Independent adversarial verifier.

    Critically: the verifier is NOT told it is reviewing AI output.
    It receives the output as if it were any document to evaluate.
    This prevents the verifier from being lenient toward AI-generated content.

    Args:
        output: The worker's output to evaluate.
        rubric: Criteria the output must satisfy to pass.

    Returns:
        dict with keys:
            "pass"   (bool): Whether the output satisfies the rubric.
            "issues" (str):  Description of problems found, or "none" if passed.
    """
    raw = await call_llm(
        system=(
            "You are a strict, independent evaluator. "
            "Your job is to assess whether the provided output meets the given rubric. "
            "Be rigorous. Do not give benefit of the doubt for ambiguous cases — flag them. "
            'Return ONLY valid JSON with no preamble: {"pass": true/false, "issues": "specific description or none"}'
        ),
        user=f"Rubric:\n{rubric}\n\nOutput to evaluate:\n{output}",
        model=STRONG_MODEL,
    )
    try:
        # Strip any accidental markdown fences before parsing
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"pass": False, "issues": f"Verifier returned unparseable response: {raw[:200]}"}


async def worker_with_verification(
    task: dict,
    rubric: str,
    max_retries: int = 2,
) -> str:
    """
    Run a worker, verify its output, retry with failure reason if it fails.

    Key design decision: failure reason is always fed back into the retry prompt.
    Blind retries (same prompt again) almost never produce different output.
    Telling the agent *why* it failed gives it specific information to act on.

    Args:
        task:        Task dict with 'system', 'user', and optional 'model'.
        rubric:      Pass/fail criteria for the verifier.
        max_retries: Number of additional attempts after the first failure.

    Returns:
        Best output produced (verified pass, or best-effort after max retries).
    """
    current_task = task.copy()

    for attempt in range(max_retries + 1):
        output = await run_worker(current_task)
        check  = await verify(output, rubric)

        if check["pass"]:
            if attempt > 0:
                print(f"[Verify] Passed on attempt {attempt + 1}.")
            return output

        print(f"[Verify] Attempt {attempt + 1} failed: {check['issues']}")

        if attempt < max_retries:
            # Feed failure reason into the next attempt's prompt
            current_task = {
                **current_task,
                "user": (
                    task["user"]
                    + f"\n\n--- Previous attempt was rejected ---\n"
                    + f"Reason: {check['issues']}\n"
                    + "Please fix these specific issues in your response."
                ),
            }

    print(f"[Verify] Returning best-effort output after {max_retries + 1} attempts.")
    return output


async def run_adversarial_harness(
    subtasks: list[dict],
    rubric: str,
    max_retries: int = 2,
) -> list[str]:
    """
    Run all subtasks in parallel, each with independent adversarial verification.

    Args:
        subtasks:    List of task dicts.
        rubric:      Shared pass/fail criteria applied to all outputs.
        max_retries: Retries per worker on failure.

    Returns:
        List of verified outputs in the same order as input subtasks.
    """
    print(f"[Adversarial] Running {len(subtasks)} workers with verification...")
    jobs = [worker_with_verification(t, rubric, max_retries) for t in subtasks]
    results = await asyncio.gather(*jobs)
    print("[Adversarial] Complete.")
    return list(results)


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RUBRIC = (
        "The code review must: "
        "(1) identify specific line numbers for each issue, "
        "(2) classify severity as Critical / High / Medium / Low, "
        "(3) suggest a concrete fix for each issue. "
        "Vague comments without specific locations or fixes do not pass."
    )

    subtasks = [
        {
            "system": "You are a senior Python engineer conducting a thorough code review.",
            "user": "Review auth.py for bugs, security issues, and code quality problems.",
        },
        {
            "system": "You are a senior Python engineer conducting a thorough code review.",
            "user": "Review payments.py for bugs, security issues, and code quality problems.",
        },
    ]

    results = asyncio.run(run_adversarial_harness(subtasks=subtasks, rubric=RUBRIC))

    for i, r in enumerate(results):
        print(f"\n── Review {i + 1} ──────────────────────────────")
        print(r)
