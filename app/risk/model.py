"""Serves the wallet liquidation-risk model in pure Python (no pandas or sklearn at request time).

The model is a logistic regression exported to app/risk/model.json by ml/evaluate_wallet_risk.py: features are
log1p-transformed where listed, standardised, then combined linearly. The output is a relative risk, not a
calibrated probability of default (class-balanced training inflates it)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("model.json")


@lru_cache(maxsize=1)
def load_model() -> dict:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def risk_band(score: int) -> str:
    return "low" if score <= 333 else "medium" if score <= 666 else "high"


def score_wallet(features: dict[str, float], model: dict | None = None) -> dict:
    """features: every name in model["features"]. Returns risk_score 0-1000, risk_band and the probability."""
    m = model or load_model()
    z = m["intercept"]
    for i, name in enumerate(m["features"]):
        x = float(features[name])
        if name in m["log_features"]:
            x = math.log1p(max(x, 0.0))
        z += m["coef"][i] * (x - m["mean"][i]) / m["scale"][i]
    p = 1 / (1 + math.exp(-z))
    score = round(1000 * p)
    return {"risk_score": score, "risk_band": risk_band(score), "probability": round(p, 4)}
