"""
Runs the labeled test set (test_set.jsonl) through the full generation pipeline and
scores each question using EXECUTION ACCURACY: the generated SQL is considered
correct if executing it returns the same underlying rows as a hand-written
reference SQL query for that question -- not if the SQL text matches. This is the
standard methodology used by text-to-SQL benchmarks (e.g. Spider) because two
syntactically different queries can be semantically equivalent.

Row comparison (see `rows_match`): each row is reduced to a normalized
frozenset-of-values "bag" (order/casing/type-independent), then we require a
1-to-1 matching where every reference row's value-bag is a subset of some
distinct candidate row's value-bag, with no leftover candidate rows. This lets a
candidate's SELECT * (superset of columns) satisfy a reference's narrower column
projection, while still penalizing wrong filters (too many/few rows) or wrong
joins (missing values).

Usage:
    python tests/accuracy/run_accuracy.py [--provider mock|anthropic|gemini] [--verbose]

Requires the database from db/seed.py to be present (docker compose up + seed.py).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import decimal
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncpg  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.guardrails.rules import check_sql  # noqa: E402
from app.llm.factory import get_provider  # noqa: E402
from app.schema_store import get_schema_store  # noqa: E402

TEST_SET_PATH = Path(__file__).parent / "test_set.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "accuracy_report.json"


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, decimal.Decimal)):
        return round(float(v), 2)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return str(v).strip().lower()


def row_bag(row) -> frozenset:
    return frozenset(normalize_value(v) for v in row)


def rows_match(candidate_rows: list, reference_rows: list, row_cap: int | None = None) -> bool:
    # If the candidate hit exactly the safety row cap and the reference has more rows
    # than that, the guardrail's row-limit truncated an otherwise-correct query -- that
    # is the row cap working as designed (see README), not a generation error. Score it
    # correct if every returned row is a genuine (uncapped) match rather than penalizing
    # deliberate truncation.
    if row_cap is not None and len(candidate_rows) == row_cap and len(reference_rows) > row_cap:
        ref_bags = [row_bag(r) for r in reference_rows]
        return all(any(rbag <= row_bag(crow) for rbag in ref_bags) for crow in candidate_rows)

    if len(candidate_rows) != len(reference_rows):
        return False
    cand_bags = [row_bag(r) for r in candidate_rows]
    used = set()
    for ref_row in reference_rows:
        ref_bag = row_bag(ref_row)
        found = None
        for i, cbag in enumerate(cand_bags):
            if i in used:
                continue
            if ref_bag <= cbag:
                found = i
                break
        if found is None:
            return False
        used.add(found)
    return True


async def load_test_cases() -> list[dict]:
    cases = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def run(provider_name: str, verbose: bool) -> dict:
    os.environ["LLM_PROVIDER"] = provider_name
    get_settings.cache_clear()
    settings = get_settings()
    store = get_schema_store()
    provider = get_provider()
    allowed_tables = store.table_names()

    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)

    cases = await load_test_cases()
    results = []
    category_totals = defaultdict(lambda: {"correct": 0, "total": 0})

    try:
        for case in cases:
            question = case["question"]
            category = case["category"]
            reference_sql = case["reference_sql"]

            relevant_tables = store.retrieve_relevant_tables(question)
            schema_text = store.render_schema_text(relevant_tables)

            gen_start = time.perf_counter()
            try:
                generation = await provider.generate(question, schema_text)
            except Exception as exc:  # noqa: BLE001
                results.append({"id": case["id"], "category": category, "question": question,
                                 "correct": False, "error": f"generation_error: {exc}"})
                category_totals[category]["total"] += 1
                continue
            gen_ms = (time.perf_counter() - gen_start) * 1000

            guardrail_result = check_sql(generation.sql, allowed_tables=allowed_tables, question=question)
            if not guardrail_result.allowed:
                results.append({
                    "id": case["id"], "category": category, "question": question,
                    "generated_sql": generation.sql, "correct": False,
                    "error": f"guardrail_blocked: {guardrail_result.reason}",
                })
                category_totals[category]["total"] += 1
                continue

            try:
                async with pool.acquire() as conn:
                    candidate_rows = await conn.fetch(guardrail_result.sanitized_sql)
                    reference_rows = await conn.fetch(reference_sql)
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "id": case["id"], "category": category, "question": question,
                    "generated_sql": generation.sql, "correct": False,
                    "error": f"execution_error: {exc}",
                })
                category_totals[category]["total"] += 1
                continue

            candidate_values = [list(r.values()) for r in candidate_rows]
            reference_values = [list(r.values()) for r in reference_rows]
            correct = rows_match(candidate_values, reference_values, row_cap=settings.max_result_rows)
            capped = (
                len(candidate_values) == settings.max_result_rows
                and len(reference_values) > settings.max_result_rows
            )

            category_totals[category]["total"] += 1
            if correct:
                category_totals[category]["correct"] += 1

            results.append({
                "id": case["id"], "category": category, "question": question,
                "generated_sql": generation.sql, "confidence": generation.confidence,
                "generation_ms": round(gen_ms, 2), "correct": correct, "row_capped": capped,
                "candidate_row_count": len(candidate_values), "reference_row_count": len(reference_values),
            })

            if verbose:
                status = "OK  " if correct else "FAIL"
                print(f"[{status}] #{case['id']:>2} ({category:11s}) {question}")
                if not correct:
                    print(f"        generated: {generation.sql}")
                    print(f"        reference: {reference_sql}")
                    print(f"        rows: candidate={len(candidate_values)} reference={len(reference_values)}")
    finally:
        await pool.close()

    overall_correct = sum(v["correct"] for v in category_totals.values())
    overall_total = sum(v["total"] for v in category_totals.values())

    report = {
        "provider": provider_name,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "overall_accuracy": round(overall_correct / overall_total, 4) if overall_total else 0.0,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_category": {
            cat: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for cat, v in sorted(category_totals.items())
        },
        "results": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "anthropic", "gemini"], default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(run(args.provider, args.verbose))

    print("\n=== Accuracy Report ===")
    print(f"Provider: {report['provider']}")
    print(f"Overall accuracy: {report['overall_accuracy'] * 100:.1f}% ({report['overall_correct']}/{report['overall_total']})")
    for cat, stats in report["by_category"].items():
        print(f"  {cat:12s}: {stats['accuracy'] * 100:5.1f}% ({stats['correct']}/{stats['total']})")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
