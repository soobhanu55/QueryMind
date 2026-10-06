"""Point-in-time wallet features and liquidation labels from raw Aave V2 transactions.

Every feature uses only transactions strictly before `cutoff`; the label is "liquidated in [cutoff, horizon_end)".
Only wallets that borrowed before the cutoff are scored, because only a wallet with debt can be liquidated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DECIMALS = {"USDC": 6, "USDT": 6, "WBTC": 8}  # every other asset in the data has 18
STABLES = {"USDC", "USDT", "DAI"}
FEATURES = ["n_tx", "active_days", "span_days", "days_since_last_tx", "deposit_usd", "borrow_usd", "repay_usd",
            "redeem_usd", "borrow_to_deposit", "repay_to_borrow", "n_borrows", "n_assets", "stable_borrow_share",
            "prior_liquidations"]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Flat frame with columns wallet, ts, action, asset, usd (USD value of the amount at the price in the event)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    df = pd.DataFrame({
        "wallet": [r["userWallet"] for r in rows],
        "ts": pd.to_datetime([r["timestamp"] for r in rows], unit="s"),
        "action": [r["action"] for r in rows],
        "asset": [r["actionData"].get("assetSymbol", "") for r in rows],
        "amount": [float(r["actionData"].get("amount") or 0) for r in rows],
        "price": [float(r["actionData"].get("assetPriceUSD") or 0) for r in rows],
    })
    scale = 10.0 ** df["asset"].map(lambda a: DECIMALS.get(a, 18)).to_numpy()
    df["usd"] = df["amount"] / scale * df["price"]
    return df.drop(columns=["amount", "price"]).sort_values("ts").reset_index(drop=True)


def wallet_features(tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per wallet that borrowed before `cutoff`, indexed by wallet, using only earlier transactions."""
    past = tx[tx["ts"] < cutoff]
    borrowers = past.loc[past["action"] == "borrow", "wallet"].unique()
    past = past[past["wallet"].isin(borrowers)]
    g = past.groupby("wallet")
    out = pd.DataFrame({
        "n_tx": g.size(),
        "active_days": g["ts"].agg(lambda s: s.dt.date.nunique()),
        "span_days": (g["ts"].max() - g["ts"].min()).dt.total_seconds() / 86400,
        "days_since_last_tx": (cutoff - g["ts"].max()).dt.total_seconds() / 86400,
        "n_assets": g["asset"].nunique(),
    })
    usd = past.pivot_table(index="wallet", columns="action", values="usd", aggfunc="sum", fill_value=0)
    cnt = past.pivot_table(index="wallet", columns="action", values="usd", aggfunc="count", fill_value=0)
    for a in ("deposit", "borrow", "repay", "redeemunderlying", "liquidationcall"):
        if a not in usd:
            usd[a] = cnt[a] = 0.0
    out["deposit_usd"], out["borrow_usd"], out["repay_usd"] = usd["deposit"], usd["borrow"], usd["repay"]
    out["redeem_usd"] = usd["redeemunderlying"]
    out["n_borrows"], out["prior_liquidations"] = cnt["borrow"], cnt["liquidationcall"]
    out["borrow_to_deposit"] = out["borrow_usd"] / out["deposit_usd"].replace(0, np.nan)
    out["repay_to_borrow"] = out["repay_usd"] / out["borrow_usd"].replace(0, np.nan)
    b = past[past["action"] == "borrow"]
    stable = b.assign(s=b["asset"].isin(STABLES) * b["usd"]).groupby("wallet")["s"].sum()
    out["stable_borrow_share"] = (stable / b.groupby("wallet")["usd"].sum().replace(0, np.nan))
    out = out.reindex(columns=FEATURES)
    # ratios with no deposits or no borrow value: borrow_to_deposit capped, missing repay/stable share = 0
    out["borrow_to_deposit"] = out["borrow_to_deposit"].fillna(100.0).clip(upper=100.0)
    return out.fillna({"repay_to_borrow": 0.0, "stable_borrow_share": 0.0}).fillna(0.0)


def liquidated_between(tx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None) -> set[str]:
    liq = tx[(tx["action"] == "liquidationcall") & (tx["ts"] >= start)]
    if end is not None:
        liq = liq[liq["ts"] < end]
    return set(liq["wallet"])


def heuristic_score(f: pd.DataFrame) -> pd.Series:
    """ChainScore's scoring idea on dollar values (higher = safer): rewards deposits, repayments and activity,
    penalises borrowing and past liquidations. The original used raw token units, so a 18-decimal token swamped
    everything; USD values make the terms comparable. Weights are the original's, not tuned."""
    raw = (0.2 * f["deposit_usd"] + 0.2 * f["repay_usd"] - 0.3 * f["borrow_usd"]
           - 50 * f["prior_liquidations"] + 2 * f["active_days"])
    return 1000 * (raw - raw.min()) / (raw.max() - raw.min())
