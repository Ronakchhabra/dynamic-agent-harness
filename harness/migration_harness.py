"""
migration_harness.py
─────────────────────
Real-world example: Codebase Migration Harness

Chains three patterns together for a production-grade migration workflow:
  1. Routing     — classify each file's complexity, route to cheap or strong model
  2. Fan-Out     — migrate all files in parallel, one isolated agent per file
  3. Adversarial — independently verify each migration against a quality rubric
  4. Synthesise  — produce a consolidated migration report across all files

This is the shape of large-scale migrations like the Bun rewrite from Zig → Rust:
fan-out per module, adversarial review per change, report at the end.

Usage:
    python migration_harness.py

Or import and call run_migration() with your own file list and spec.
"""

import asyncio

from harness.patterns.fan_out     import synthesise
from harness.patterns.adversarial import worker_with_verification
from harness.patterns.routing     import classify_complexity
from harness.llm_client           import call_llm

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS = {
    "simple":  "azure/gpt-4o-mini",
    "medium":  "azure/gpt-4o",
    "complex": "azure/gpt-4o",
}

# Quality rubric applied to every migration — verifier checks each file against this
MIGRATION_RUBRIC = (
    "The migration output must: "
    "(1) apply all required changes from the migration spec completely, "
    "(2) not break existing function signatures or public interfaces unless the spec requires it, "
    "(3) preserve all existing comments and docstrings unless directly affected by the migration, "
    "(4) follow the existing code style and conventions of the file, "
    "(5) not introduce new bugs or logic errors. "
    "Partial migrations or migrations with unexplained changes do not pass."
)


# ── Harness ────────────────────────────────────────────────────────────────────

async def migrate_file(file_path: str, spec: str, complexity: str) -> dict:
    """
    Migrate a single file with adversarial verification.

    Returns a dict with file_path, success flag, and the migration output.
    """
    model = MODELS.get(complexity, MODELS["medium"])

    task = {
        "system": (
            "You are a senior software engineer performing a precise codebase migration. "
            "Apply only the changes specified. Do not refactor unrelated code. "
            "Output the complete migrated file content."
        ),
        "user": (
            f"Migration spec:\n{spec}\n\n"
            f"File to migrate: {file_path}\n\n"
            "Apply the migration spec to this file. Output the complete migrated file."
        ),
        "model": model,
    }

    print(f"  [Migrate] {file_path} (complexity={complexity}, model={model})")
    output = await worker_with_verification(task, MIGRATION_RUBRIC, max_retries=2)

    return {
        "file":   file_path,
        "model":  model,
        "output": output,
    }


async def run_migration(files: list[str], spec: str) -> str:
    """
    Full migration pipeline for a list of files.

    Args:
        files: List of file paths to migrate.
        spec:  The migration specification describing what changes to apply.

    Returns:
        A synthesised migration report summarising all changes made.
    """
    print(f"\n[Migration] Starting migration of {len(files)} file(s)...")
    print(f"[Migration] Spec preview: {spec[:120]}...\n")

    # Step 1: Classify complexity of each file in parallel
    print("[Migration] Step 1/3: Classifying file complexity...")
    complexities = await asyncio.gather(*[classify_complexity(f) for f in files])
    for f, c in zip(files, complexities):
        print(f"  {f} → {c}")

    # Step 2: Migrate all files in parallel with adversarial verification
    print("\n[Migration] Step 2/3: Migrating files with adversarial verification...")
    migration_results = await asyncio.gather(*[
        migrate_file(f, spec, c) for f, c in zip(files, complexities)
    ])

    # Step 3: Synthesise a migration report
    print("\n[Migration] Step 3/3: Synthesising migration report...")
    result_texts = [
        f"File: {r['file']} (model: {r['model']})\n\n{r['output']}"
        for r in migration_results
    ]
    report = await synthesise(
        results=result_texts,
        goal="Produce a clear migration report: what changed in each file, any issues found, and overall status.",
    )

    print("\n[Migration] Complete.")
    return report


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example migration spec: rename a class across a codebase
    SPEC = """
    Migration: Rename 'UserModel' to 'AccountModel' across the codebase.
    
    Changes required:
    - Rename the class definition from 'class UserModel' to 'class AccountModel'
    - Update all references to UserModel within the file to AccountModel
    - Update all import statements that reference UserModel
    - Update all type hints that reference UserModel
    - Update all docstrings that reference UserModel
    - Do NOT change any database table names or API field names
    - Do NOT change any tests — those will be handled separately
    """

    FILES = [
        "src/models/user_model.py",
        "src/services/auth_service.py",
        "src/api/user_router.py",
        "src/utils/validators.py",
    ]

    report = asyncio.run(run_migration(files=FILES, spec=SPEC))

    print("\n" + "=" * 60)
    print("MIGRATION REPORT")
    print("=" * 60)
    print(report)
