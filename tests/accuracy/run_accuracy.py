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
    python tests/accuracy/run_accuracy.py [--provider mock|anthropic|gemini|groq|local|routed] [--verbose]

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
from app.llm.router import cost_usd  # noqa: E402
from app.schema_store import get_schema_store  # noqa: E402

TEST_SET_PATH = Path(__file__).parent / os.environ.get("ACCURACY_SET", "test_set.jsonl")
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


def routing_summary(results: list[dict]) -> dict:
    """Where the answers came from (only meaningful for --provider routed) and what the model calls cost."""
    ok = [r for r in results if "generation_ms" in r]
    by_route: dict[str, dict] = {}
    for r in ok:
        d = by_route.setdefault(r.get("route") or "n/a", {"n": 0, "correct": 0, "ms": 0.0})
        d["n"] += 1
        d["correct"] += bool(r["correct"])
        d["ms"] += r["generation_ms"]
    model_calls = [r for r in ok if r.get("model") not in (None, "rules")]
    unpriced = [r for r in model_calls if r.get("cost_usd") is None]
    return {
        "by_route": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4), "mean_generation_ms": round(v["ms"] / v["n"], 1)}
                     for k, v in by_route.items()},
        "mean_generation_ms": round(sum(r["generation_ms"] for r in ok) / len(ok), 1) if ok else None,
        "input_tokens": sum(r.get("input_tokens") or 0 for r in ok),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in ok),
        "model_calls": len(model_calls),
        "model_calls_without_a_known_price": len(unpriced),  # cost below covers only the priced calls
        "cost_usd_list_price": round(sum(r["cost_usd"] or 0 for r in ok), 6),
    }


async def run(provider_name: str, verbose: bool) -> dict:
    os.environ["LLM_PROVIDER"] = provider_name
    get_settings.cache_clear()
    settings = get_settings()
    store = get_schema_store()
    provider = get_provider()
    allowed_tables = store.table_names()

    embedded_db = settings.database_url.startswith("sqlite")  # DATABASE_URL=sqlite:///:memory: runs the benchmark with no Postgres
    pool = None if embedded_db else await asyncpg.create_pool(settings.database_url.replace("postgresql+asyncpg://", "postgresql://"), min_size=2, max_size=5)

    async def fetch_rows(sql: str) -> list[list]:
        if embedded_db:
            from app import embedded

            return [list(row) for row in embedded.run(sql, 10**6, 30)[1]]
        async with pool.acquire() as conn:
            return [list(row.values()) for row in await conn.fetch(sql)]

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
                candidate_values = await fetch_rows(guardrail_result.sanitized_sql)
                reference_values = await fetch_rows(reference_sql)
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "id": case["id"], "category": category, "question": question,
                    "generated_sql": generation.sql, "correct": False,
                    "error": f"execution_error: {exc}",
                })
                category_totals[category]["total"] += 1
                continue

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
                "route": generation.route, "model": generation.model, "escalated": generation.escalated,
                "input_tokens": generation.input_tokens, "output_tokens": generation.output_tokens,
                "cost_usd": cost_usd(generation.model, generation.input_tokens, generation.output_tokens),
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
        if pool is not None:
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
        "routing": routing_summary(results),
        "results": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "anthropic", "gemini", "groq", "local", "routed"], default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(run(args.provider, args.verbose))

    print("\n=== Accuracy Report ===")
    print(f"Provider: {report['provider']}")
    print(f"Overall accuracy: {report['overall_accuracy'] * 100:.1f}% ({report['overall_correct']}/{report['overall_total']})")
    for cat, stats in report["by_category"].items():
        print(f"  {cat:12s}: {stats['accuracy'] * 100:5.1f}% ({stats['correct']}/{stats['total']})")

    # the mock provider keeps the canonical report name; real-LLM runs get their own file so they never overwrite it
    out_path = REPORT_PATH if args.provider == "mock" else REPORT_PATH.with_name(f"accuracy_report_{args.provider}.json")
    if TEST_SET_PATH.name != "test_set.jsonl":  # e.g. ACCURACY_SET=heldout_set.jsonl: never overwrite the main reports
        out_path = out_path.with_name(out_path.stem + "_" + TEST_SET_PATH.stem.replace("_set", "") + ".json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
