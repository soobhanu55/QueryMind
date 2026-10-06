"""Offline rule-based NL2SQL for the single-table `wallets` domain (Aave wallet risk), used by the mock provider.

Like the sales parser it extracts an aggregation, a metric column, a group-by, filters and top-N from the
question; unlike it there are no joins, so the SQL is one SELECT over `wallets`.
"""
from __future__ import annotations

import re
from typing import Optional

from app.llm.base import SQLGenerationResult

WALLET_RE = re.compile(r"\bwallets?\b|\bborrowers?\b|\bliquidat|\brisk (score|band|level)|\baave\b")

# (regex, column, alias); first match wins, so more specific phrases come first
METRICS = [
    (re.compile(r"\brisk scores?\b|\bcredit scores?\b"), "risk_score", "risk_score"),
    (re.compile(r"\bborrow(ed|ing|ings|s)?\b"), "borrow_usd", "borrow_usd"),
    (re.compile(r"\bdeposit(ed|s)?\b"), "deposit_usd", "deposit_usd"),
    (re.compile(r"\brepa(id|yments?)\b"), "repay_usd", "repay_usd"),
    (re.compile(r"\bwithdr\w+|\bredeem\w*"), "redeem_usd", "redeem_usd"),
    (re.compile(r"\btransactions?\b|\btx\b"), "n_tx", "n_tx"),
    (re.compile(r"\bactive days\b"), "active_days", "active_days"),
    (re.compile(r"\bleverage\b|\butili[sz]ation\b"), "borrow_to_deposit", "borrow_to_deposit"),
]
GROUP_BY = re.compile(r"\b(by|per|each|across) (risk )?(band|level|category)\b|\bper risk band\b")
BAND_RE = re.compile(r"\b(low|medium|high)[ -]risk\b|\brisk (band|level) (?:is |of )?(low|medium|high)\b|\b(low|medium|high) risk band\b")
NUM_CMP = re.compile(r"\b(greater than|more than|over|above|at least|less than|under|below|at most)\s+\$?(\d+(?:\.\d+)?)\s*(k|m)?")
CMP = {"greater than": ">", "more than": ">", "over": ">", "above": ">", "at least": ">=",
       "less than": "<", "under": "<", "below": "<", "at most": "<="}
TOPN = re.compile(r"\btop\s+(\d+)\b")


def _metric(q: str) -> Optional[tuple[str, str]]:
    for pat, col, alias in METRICS:
        if pat.search(q):
            return col, alias
    return None


def generate(q: str) -> SQLGenerationResult:
    """`q` is the already lower-cased, punctuation-stripped question."""
    where: list[str] = []
    band = BAND_RE.search(q)
    if band:
        where.append(f"risk_band = '{next(g for g in band.groups() if g in ('low', 'medium', 'high'))}'")
    if re.search(r"\bnever (been )?liquidated\b|\bnot (been )?liquidated\b|\bwithout (a )?liquidation", q):
        where.append("was_liquidated = 0")
    elif re.search(r"\bliquidated\b|\bhave been liquidated\b|\bhad (a )?liquidation", q):
        where.append("was_liquidated = 1")

    metric = _metric(q)
    m = NUM_CMP.search(q)
    if m and metric:
        value = float(m.group(2)) * {"k": 1e3, "m": 1e6}.get(m.group(3) or "", 1)
        where.append(f"{metric[0]} {CMP[m.group(1)]} {value:.10g}")
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""

    topn = TOPN.search(q)
    group = GROUP_BY.search(q)
    agg = ("AVG" if re.search(r"\baverage\b|\bavg\b|\bmean\b", q) else
           "COUNT" if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q) else
           "SUM" if re.search(r"\btotal\b|\bsum\b", q) else
           "MAX" if re.search(r"\bhighest\b|\bmaximum\b|\blargest\b|\bbiggest\b", q) and not topn else
           "MIN" if re.search(r"\blowest\b|\bminimum\b|\bsmallest\b", q) and not topn else None)

    if group:
        if agg is None or agg == "COUNT" or metric is None:
            sql = f"SELECT risk_band, COUNT(*) AS wallet_count FROM wallets{where_sql} GROUP BY risk_band ORDER BY risk_band"
        else:
            sql = (f"SELECT risk_band, {agg}({metric[0]}) AS {agg.lower()}_{metric[1]} FROM wallets{where_sql} "
                   f"GROUP BY risk_band ORDER BY risk_band")
        return SQLGenerationResult(sql=sql, confidence=0.8, explanation="Groups wallets by risk band.")
    if topn:
        col = metric[0] if metric else "risk_score"
        order = "ASC" if re.search(r"\blowest\b|\bsafest\b|\bleast\b", q) else "DESC"
        sql = f"SELECT wallet_id, {col} FROM wallets{where_sql} ORDER BY {col} {order}, wallet_id LIMIT {int(topn.group(1))}"
        return SQLGenerationResult(sql=sql, confidence=0.8, explanation=f"Top wallets by {col}.")
    if agg == "COUNT" or (agg is None and re.search(r"\bhow many\b", q)):
        return SQLGenerationResult(sql=f"SELECT COUNT(*) AS wallet_count FROM wallets{where_sql}", confidence=0.85,
                                   explanation="Counts wallets matching the filters.")
    if agg and metric:
        return SQLGenerationResult(sql=f"SELECT {agg}({metric[0]}) AS {agg.lower()}_{metric[1]} FROM wallets{where_sql}",
                                   confidence=0.8, explanation=f"{agg} of {metric[0]} over wallets.")
    return SQLGenerationResult(sql=f"SELECT wallet_id, risk_score, risk_band FROM wallets{where_sql} ORDER BY risk_score DESC, wallet_id LIMIT 50",
                               confidence=0.5, explanation="Lists wallets with their risk.")
