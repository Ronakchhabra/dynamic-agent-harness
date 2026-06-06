"""
patterns/routing.py
────────────────────
Pattern 4: Model Routing by Complexity

Use a cheap, fast classifier to assess task complexity, then route to the
appropriate model tier. This can reduce costs by 60–90% on mixed-complexity
workloads without sacrificing quality on tasks that need stronger models.

    Task ──→ Classifier (cheap model) ──→ "simple"  ──→ Fast model
                                       ├──→ "medium"  ──→ Balanced model
                                       └──→ "complex" ──→ Strong model

Use when:
  - You have a mixed-complexity workload (support tickets, bug reports, queries)
  - You want to minimise cost without a blanket "use cheap model everywhere"
  - Batch processing where most tasks are simple but some require deep reasoning
  - Any high-volume task that can be tiered by complexity
"""

import asyncio
from harness.llm_client import call_llm

# ── Model tiers — swap strings to change provider ─────────────────────────────
MODELS = {
    "simple":  "azure/gpt-4o-mini",   # Fast, cheap — classifiers, filters, simple Q&A
    "medium":  "azure/gpt-4o",         # Balanced — most production tasks
    "complex": "azure/gpt-4o",         # Synthesis, complex reasoning (swap to o1/o3 if needed)
}

# The classifier itself always uses the cheapest model
CLASSIFIER_MODEL = MODELS["simple"]


# ── Core functions ─────────────────────────────────────────────────────────────

async def classify_complexity(task: str) -> str:
    """
    Classify a task as 'simple', 'medium', or 'complex'.

    Uses the cheapest model — this call is meant to be extremely fast and cheap.
    The classifier's only job is to route; it does not attempt the task itself.

    Returns:
        One of: "simple", "medium", "complex"
    """
    result = await call_llm(
        system=(
            "You are a task complexity classifier. "
            "Classify the complexity of the given task based on reasoning depth required. "
            "simple  = straightforward lookup, formatting, or single-step answer. "
            "medium  = requires some reasoning, multi-step logic, or domain knowledge. "
            "complex = requires deep analysis, synthesis across many factors, or expert judgment. "
            'Reply with ONLY one word: "simple", "medium", or "complex". No explanation.'
        ),
        user=task,
        model=CLASSIFIER_MODEL,
        max_tokens=10,
        temperature=0.0,   # Deterministic — classification should not vary
    )
    word = result.strip().lower()
    # Fallback to "medium" if classifier returns something unexpected
    return word if word in MODELS else "medium"


async def routed_agent(system: str, user: str) -> str:
    """
    Classify the task, select the appropriate model tier, then run the agent.

    Args:
        system: The agent's role/behaviour prompt.
        user:   The task to perform.

    Returns:
        The agent's response using the routed model.
    """
    complexity = await classify_complexity(user)
    model = MODELS[complexity]
    print(f"[Router] Classified as '{complexity}' → {model}")
    return await call_llm(system=system, user=user, model=model)


async def routed_batch(tasks: list[dict]) -> list[str]:
    """
    Route and run a batch of tasks in parallel.

    Each task is classified independently, then all run concurrently
    using their individually selected models.

    Args:
        tasks: List of dicts with 'system' and 'user' keys.

    Returns:
        List of results in the same order as input tasks.
    """
    print(f"[Router] Classifying and routing {len(tasks)} tasks...")
    return await asyncio.gather(*[routed_agent(t["system"], t["user"]) for t in tasks])


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mix of simple, medium, and complex tasks — router decides which model handles each
    tasks = [
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of France?",                          # simple
        },
        {
            "system": "You are a Python expert.",
            "user": "Explain the difference between asyncio.gather and asyncio.wait.",  # medium
        },
        {
            "system": "You are a senior systems architect.",
            "user": (
                "We have a microservices system with 40 services communicating via REST. "
                "We're seeing cascading failures under load. Analyse the likely root causes, "
                "compare circuit breaker vs bulkhead vs retry strategies, and recommend "
                "a specific architecture with trade-offs explained."                        # complex
            ),
        },
    ]

    results = asyncio.run(routed_batch(tasks))

    for i, (task, result) in enumerate(zip(tasks, results)):
        print(f"\n── Task {i + 1} ──────────────────────────────")
        print(f"Q: {task['user'][:80]}...")
        print(f"A: {result[:300]}...")
