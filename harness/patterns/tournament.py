"""
patterns/tournament.py
───────────────────────
Pattern 5: Tournament (Best-of-N with Pairwise Judging)

N agents independently attempt the same task from different angles.
A judge agent compares candidates pairwise — comparative judgment is
more reliable than absolute scoring. Rounds continue until one winner remains.

    Candidates: [A, B, C, D]
    Round 1:    A vs B → winner_AB,  C vs D → winner_CD
    Round 2:    winner_AB vs winner_CD → Champion

Use when:
  - You need the best output, not just a good one
  - Taste-based tasks where quality matters: naming, design, strategy, writing
  - Comparing alternative implementations or approaches
  - Any task where "good enough" is not good enough

Why pairwise comparison vs absolute scoring:
  - Absolute scoring ("rate this 1-10") is inconsistent across calls
  - Pairwise ("which is better, A or B?") anchors on a concrete comparison
  - Pairwise is how human experts judge in practice (code review, design critique)
"""

import asyncio
from harness.llm_client import call_llm

# ── Configure your models here ────────────────────────────────────────────────
FAST_MODEL   = "azure/gpt-4o-mini"   # Candidate generators — run in parallel
STRONG_MODEL = "azure/gpt-4o"        # Judge — needs reliable comparative reasoning


# ── Core functions ─────────────────────────────────────────────────────────────

async def generate_candidate(
    system: str,
    user: str,
    variant_hint: str,
) -> str:
    """
    Generate one candidate solution from a specific angle.

    The variant_hint nudges each candidate toward a different approach,
    ensuring diversity across the candidate pool. Without this, candidates
    tend to converge on the same answer.

    Args:
        system:       Base agent role/behaviour.
        user:         The task to solve.
        variant_hint: The angle this candidate should approach from.
    """
    return await call_llm(
        system=system,
        user=f"{user}\n\nApproach this specifically from the angle of: {variant_hint}",
        model=FAST_MODEL,
    )


async def judge_pair(a: str, b: str, rubric: str) -> str:
    """
    Compare two candidates and return the better one.

    The judge is told only the rubric and the two candidates — no context
    about who produced them or which round this is. Clean comparison only.

    Returns:
        "A" or "B" — the label of the better candidate.
    """
    result = await call_llm(
        system=(
            "You are an impartial judge evaluating two candidate outputs. "
            "Compare them strictly against the rubric provided. "
            "Choose the candidate that better satisfies the rubric overall. "
            'Reply with ONLY the letter "A" or "B". No explanation.'
        ),
        user=(
            f"Rubric:\n{rubric}\n\n"
            f"[Candidate A]\n{a}\n\n"
            f"[Candidate B]\n{b}"
        ),
        model=STRONG_MODEL,
        max_tokens=5,
        temperature=0.0,   # Deterministic — judgment should not vary
    )
    return "A" if "A" in result.upper() else "B"


async def run_tournament(
    system: str,
    user: str,
    rubric: str,
    variants: list[str],
) -> str:
    """
    Run the full tournament: generate candidates → pairwise knockout → return winner.

    Args:
        system:   Base agent role/behaviour prompt.
        user:     The task all candidates will attempt.
        rubric:   Criteria the judge uses to evaluate candidates.
        variants: List of angle hints — one per candidate. len(variants) = N candidates.

    Returns:
        The winning candidate's output as a string.

    Example variants for an API naming task:
        ["strict REST purist", "developer experience focus", "enterprise API standards"]
    """
    print(f"[Tournament] Generating {len(variants)} candidates in parallel...")
    candidates = list(
        await asyncio.gather(*[generate_candidate(system, user, v) for v in variants])
    )

    round_num = 1
    while len(candidates) > 1:
        print(f"[Tournament] Round {round_num}: {len(candidates)} candidates remaining...")
        next_round = []

        # Pair up candidates; if odd number, last one gets a bye (advances automatically)
        for i in range(0, len(candidates) - 1, 2):
            winner_label = await judge_pair(candidates[i], candidates[i + 1], rubric)
            winner = candidates[i] if winner_label == "A" else candidates[i + 1]
            next_round.append(winner)

        if len(candidates) % 2 == 1:
            next_round.append(candidates[-1])   # Bye for odd-one-out
            print(f"[Tournament] Candidate {len(candidates)} advances via bye.")

        candidates = next_round
        round_num += 1

    print("[Tournament] Champion selected.")
    return candidates[0]


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Example 1: Pick the best REST API endpoint name
    print("=== Tournament: API Endpoint Naming ===\n")
    winner = asyncio.run(run_tournament(
        system="You are a senior API designer with 10 years of experience.",
        user="Design a REST endpoint for: retrieving paginated audit logs filtered by user ID and date range.",
        rubric=(
            "The endpoint must be: RESTful and noun-based, "
            "self-explanatory without documentation, "
            "consistent with Google and Microsoft API conventions, "
            "handle pagination and filtering cleanly in the URL design."
        ),
        variants=[
            "strict REST purist — resources and HTTP verbs only",
            "developer experience focus — optimise for readability and discoverability",
            "enterprise API standards — follow Google Cloud / Microsoft Azure API guidelines",
            "pragmatic engineer — balance purity with real-world implementation simplicity",
        ],
    ))
    print("\n── Winner ────────────────────────────────────")
    print(winner)

    # Example 2: Pick the best name for a CLI tool
    print("\n\n=== Tournament: CLI Tool Naming ===\n")
    cli_winner = asyncio.run(run_tournament(
        system="You are a developer tools expert focused on naming and branding.",
        user="Name a CLI tool that watches file changes and automatically runs tests.",
        rubric=(
            "The name must be: short (1-2 syllables preferred), "
            "memorable and pronounceable, "
            "not already taken by a major open-source project, "
            "evocative of speed, automation, or watching."
        ),
        variants=[
            "onomatopoeia and sound-based names",
            "action verb or imperative names",
            "portmanteau or compound word names",
        ],
    ))
    print("\n── Winner ────────────────────────────────────")
    print(cli_winner)
