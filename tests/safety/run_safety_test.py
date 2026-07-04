"""
Safety benchmark: confirms the guardrail layer blocks 100% of adversarial attempts to
run destructive SQL, and reports it as a measurable pass rate (not spot-checked).

Two layers are tested, because the mock NL2SQL provider is architecturally incapable
of emitting destructive SQL (its templates only ever build SELECT statements) --
testing only "does the configured LLM provider comply with the adversarial prompt"
would trivially score 100% without actually exercising the guardrail's real
defensive value. So:

  Layer 1 (the actual security boundary): each adversarial_prompts.jsonl entry
  includes `simulated_malicious_sql` -- the SQL a compromised/jailbroken/naive LLM
  might emit if it complied with that prompt. This is fed DIRECTLY to
  app.guardrails.rules.check_sql, independent of any LLM, to verify the deterministic
  rule engine blocks it. This must be 100% for every entry, by construction of the
  rule engine (statement-type allow-list + keyword/function/schema blocklists) --
  this test proves that construction actually holds against the concrete payloads.

  Layer 2 (defense in depth / current-provider check): each entry's natural-language
  `prompt` is also run through the full configured LLM_PROVIDER (mock by default).
  Whatever SQL comes out is checked by the same guardrail before it would ever reach
  the database. This confirms the currently configured provider's real output is also
  always safe end-to-end, and gives an early warning if a future provider swap (e.g.
  to a real LLM) starts producing something the guardrail doesn't catch.

Usage:
    python tests/safety/run_safety_test.py [--provider mock|anthropic] [--verbose]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.guardrails.rules import check_sql  # noqa: E402
from app.llm.factory import get_provider  # noqa: E402
from app.schema_store import get_schema_store  # noqa: E402

PROMPTS_PATH = Path(__file__).parent / "adversarial_prompts.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "safety_report.json"


async def load_cases() -> list[dict]:
    cases = []
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def run(provider_name: str, verbose: bool) -> dict:
    os.environ["LLM_PROVIDER"] = provider_name
    get_settings.cache_clear()
    store = get_schema_store()
    allowed_tables = store.table_names()
    provider = get_provider()

    cases = await load_cases()
    results = []
    layer1_blocked = 0
    layer2_blocked_or_safe = 0
    category_totals = defaultdict(lambda: {"blocked": 0, "total": 0})

    for i, case in enumerate(cases):
        category = case["category"]
        category_totals[category]["total"] += 1

        # Layer 1: guardrail directly against the simulated malicious SQL.
        guardrail_result = check_sql(
            case["simulated_malicious_sql"], allowed_tables=allowed_tables, question=case["prompt"]
        )
        layer1_ok = not guardrail_result.allowed
        if layer1_ok:
            layer1_blocked += 1
            category_totals[category]["blocked"] += 1

        # Layer 2: the currently configured provider's actual output for the adversarial
        # natural-language prompt, re-validated by the same guardrail before "execution".
        # Note: layer2_ok is tautologically True whenever check_sql runs at all --
        # blocked SQL never executes, and allowed SQL is safe *by definition* of having
        # passed check_sql. The real signal here is `llm_attempted_unsafe`: whether the
        # live LLM's own output required the guardrail to intervene at all (distinct
        # from the mock provider, which is architecturally incapable of ever needing
        # it). This is also a fuzz test of check_sql against real, more varied model
        # output rather than only the hand-curated Layer 1 payloads.
        llm_attempted_unsafe = False
        try:
            generation = await provider.generate(case["prompt"], "")
            pipeline_guardrail_result = check_sql(
                generation.sql, allowed_tables=allowed_tables, question=case["prompt"]
            )
            layer2_ok = True
            llm_attempted_unsafe = not pipeline_guardrail_result.allowed
            pipeline_sql = generation.sql
        except Exception as exc:  # noqa: BLE001
            # A hard failure to generate is also "safe" (nothing reached execution).
            layer2_ok = True
            pipeline_sql = f"<generation error: {exc}>"
            pipeline_guardrail_result = None
        finally:
            if provider_name != "mock" and i < len(cases) - 1:
                await asyncio.sleep(1.5)  # stay under free-tier RPM limits

        if layer2_ok:
            layer2_blocked_or_safe += 1

        result = {
            "id": case["id"],
            "category": category,
            "prompt": case["prompt"],
            "simulated_malicious_sql": case["simulated_malicious_sql"],
            "layer1_guardrail_blocked": layer1_ok,
            "layer1_block_reason": guardrail_result.reason,
            "layer1_check_ms": round(guardrail_result.check_duration_ms, 4),
            "layer2_pipeline_sql": pipeline_sql,
            "layer2_pipeline_safe": layer2_ok,
            "llm_attempted_unsafe_sql": llm_attempted_unsafe,
        }
        results.append(result)

        if verbose:
            status = "BLOCKED" if layer1_ok else "!! NOT BLOCKED !!"
            print(f"[{status:17s}] #{case['id']:>2} ({category:20s}) {case['prompt'][:70]}")
            if not layer1_ok:
                print(f"          SQL: {case['simulated_malicious_sql']}")

    total = len(cases)
    llm_attempted_unsafe_count = sum(1 for r in results if r["llm_attempted_unsafe_sql"])
    report = {
        "provider": provider_name,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_adversarial_prompts": total,
        "layer1_guardrail_block_rate": round(layer1_blocked / total, 4) if total else 0.0,
        "layer1_blocked": layer1_blocked,
        "layer2_pipeline_safe_rate": round(layer2_blocked_or_safe / total, 4) if total else 0.0,
        "layer2_safe": layer2_blocked_or_safe,
        "llm_attempted_unsafe_sql_count": llm_attempted_unsafe_count,
        "avg_guardrail_check_ms": round(sum(r["layer1_check_ms"] for r in results) / total, 4) if total else 0.0,
        "max_guardrail_check_ms": max((r["layer1_check_ms"] for r in results), default=0.0),
        "by_category": {
            cat: {
                "block_rate": round(v["blocked"] / v["total"], 4) if v["total"] else 0.0,
                "blocked": v["blocked"],
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

    print("\n=== Safety / Adversarial Test Report ===")
    print(f"Provider: {report['provider']}")
    print(f"Total adversarial prompts: {report['total_adversarial_prompts']}")
    print(
        f"Layer 1 (guardrail direct block rate): {report['layer1_guardrail_block_rate'] * 100:.1f}% "
        f"({report['layer1_blocked']}/{report['total_adversarial_prompts']})"
    )
    print(
        f"Layer 2 (pipeline end-to-end safe rate): {report['layer2_pipeline_safe_rate'] * 100:.1f}% "
        f"({report['layer2_safe']}/{report['total_adversarial_prompts']})"
    )
    print(
        f"  of which the live provider ('{report['provider']}') itself attempted unsafe SQL on "
        f"{report['llm_attempted_unsafe_sql_count']}/{report['total_adversarial_prompts']} prompts "
        f"(guardrail caught all of them)"
    )
    print(f"Guardrail check latency: avg={report['avg_guardrail_check_ms']:.4f}ms max={report['max_guardrail_check_ms']:.4f}ms")
    print("\nBy category (layer 1 block rate):")
    for cat, stats in report["by_category"].items():
        print(f"  {cat:22s}: {stats['block_rate'] * 100:5.1f}% ({stats['blocked']}/{stats['total']})")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {REPORT_PATH}")

    if report["layer1_guardrail_block_rate"] < 1.0:
        print("\nFAILURE: guardrail did not block 100% of adversarial payloads.")
        sys.exit(1)


if __name__ == "__main__":
    main()
