"""Wallet liquidation-risk models on real Aave V2 (Polygon) data, evaluated forward in time.

    python ml/evaluate_wallet_risk.py /path/to/user-wallet-transactions.json

Train: features before 2021-06-01, label "liquidated in June". Test: features before 2021-07-01, label "liquidated
from July on" (the model never sees July or later). Writes docs/wallet_risk_eval.md, the serving model
app/risk/model.json and the per-wallet table db/wallets.csv loaded into Postgres.
The dataset (3,497 wallets, 100k transactions, a third-party Google Drive file) is not committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wallet_features import FEATURES, heuristic_score, liquidated_between, load_transactions, wallet_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
T1, T2 = pd.Timestamp("2021-06-01"), pd.Timestamp("2021-07-01")
LOG_FEATURES = [f for f in FEATURES if f.endswith("_usd") or f in ("n_tx", "n_borrows", "borrow_to_deposit")]


def transform(f: pd.DataFrame) -> np.ndarray:
    x = f[FEATURES].copy()
    x[LOG_FEATURES] = np.log1p(x[LOG_FEATURES].clip(lower=0))
    return x.to_numpy(dtype=float)


def walletguard_like(f: pd.DataFrame) -> pd.Series:
    """WalletGuard's formula without its interest term (the dataset has no interest paid per wallet):
    0.3 x normalised borrow + 0.4 x normalised utilisation (borrow / deposit), renormalised to 0-1000."""
    mm = lambda s: (s - s.min()) / (s.max() - s.min())
    return 1000 * (0.3 * mm(f["borrow_usd"]) + 0.4 * mm(f["borrow_to_deposit"])) / 0.7


def bootstrap(y: np.ndarray, s: np.ndarray, fn, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            vals.append(fn(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def lift_top_decile(y: np.ndarray, s: np.ndarray) -> float:
    top = np.argsort(-s, kind="stable")[: max(1, len(y) // 10)]
    return float(y[top].mean() / y.mean())


def main(path: str) -> None:
    tx = load_transactions(path)
    end = tx["ts"].max() + pd.Timedelta(seconds=1)
    f1, f2 = wallet_features(tx, T1), wallet_features(tx, T2)
    y1 = f1.index.isin(liquidated_between(tx, T1, T2)).astype(int)
    y2 = f2.index.isin(liquidated_between(tx, T2, end)).astype(int)

    scaler = StandardScaler().fit(transform(f1))
    x1, x2 = scaler.transform(transform(f1)), scaler.transform(transform(f2))
    models = {
        "logistic regression": LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
        "random forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0),
        "gradient boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0),
    }
    scores = {
        "random (base rate)": np.random.default_rng(0).random(len(f2)),
        "any prior liquidation": f2["prior_liquidations"].to_numpy() + 1e-3 * np.random.default_rng(1).random(len(f2)),
        "ChainScore heuristic (USD)": -heuristic_score(f2).to_numpy(),
        "WalletGuard-style (no interest)": walletguard_like(f2).to_numpy(),
    }
    for name, m in models.items():
        m.fit(x1, y1)
        scores[name] = m.predict_proba(x2)[:, 1]

    lines = [
        "# Wallet liquidation-risk evaluation\n",
        f"Aave V2 on Polygon, 2021-03-31 to 2021-09-02. Train: {len(f1)} borrowers with features before {T1.date()} "
        f"({y1.sum()} liquidated in June). Test: {len(f2)} borrowers with features before {T2.date()}, "
        f"{y2.sum()} liquidated from July on ({y2.mean():.1%} base rate). The test period is never seen in training. "
        "Only wallets with a borrow before the cutoff are scored (no debt, no liquidation).\n",
        f"| Scorer | ROC-AUC (95% CI) | PR-AUC (base rate {y2.mean():.3f}) | Lift, top 10% |", "|---|---|---|---|",
    ]
    for name, s in scores.items():
        lo, hi = bootstrap(y2, np.asarray(s), roc_auc_score)
        lines.append(f"| {name} | {roc_auc_score(y2, s):.2f} ({lo:.2f} to {hi:.2f}) | {average_precision_score(y2, s):.3f} "
                     f"| {lift_top_decile(y2, np.asarray(s)):.1f}x |")
    lines += ["", f"{y2.sum()} positives make every interval wide; differences of a few AUC points between the learned models are noise.",
              "The original ChainScore R-squared of 0.45 measured how well a forest recovers its own hand-made score; "
              "this table measures real liquidations."]

    # serving model: logistic regression refit on train + test windows (no labels beyond data end are used)
    lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
    xa = np.vstack([transform(f1), transform(f2)])
    sc = StandardScaler().fit(xa)
    lr.fit(sc.transform(xa), np.concatenate([y1, y2]))
    (ROOT / "app" / "risk").mkdir(exist_ok=True)
    (ROOT / "app" / "risk" / "model.json").write_text(json.dumps({
        "features": FEATURES, "log_features": LOG_FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "note": "logistic regression, class_weight=balanced: outputs a relative risk probability, not a calibrated default rate"},
        indent=1), encoding="utf-8")

    # per-wallet table for Postgres: all borrowers as of the last day, scored by the serving model
    fl = wallet_features(tx, end)
    table = fl.copy()
    table["risk_score"] = np.round(1000 * lr.predict_proba(sc.transform(transform(fl)))[:, 1]).astype(int)
    table["risk_band"] = pd.cut(table["risk_score"], [-1, 333, 666, 1000], labels=["low", "medium", "high"]).astype(str)
    table["was_liquidated"] = table.index.isin(liquidated_between(tx, tx["ts"].min())).astype(int)
    table = table.round(4).reset_index().rename(columns={"wallet": "wallet_id"})
    table.to_csv(ROOT / "db" / "wallets.csv", index=False)
    lines += ["", f"Serving model: logistic regression refit on both windows; `db/wallets.csv` holds {len(table)} wallets scored with it "
              "(an in-sample ranking for the demo database, not a held-out prediction)."]
    out = ROOT / "docs" / "wallet_risk_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
