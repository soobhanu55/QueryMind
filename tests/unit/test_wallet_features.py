import json

import pandas as pd
import pytest

pytest.importorskip("pandas")

from ml.wallet_features import liquidated_between, load_transactions, wallet_features  # noqa: E402

T = pd.Timestamp("2021-06-01")


def _tx(rows):
    return pd.DataFrame(rows, columns=["wallet", "ts", "action", "asset", "usd"]).assign(ts=lambda d: pd.to_datetime(d.ts))


BASE = [
    ("a", "2021-05-01", "deposit", "WETH", 1000.0),
    ("a", "2021-05-02", "borrow", "USDC", 400.0),
    ("a", "2021-05-10", "repay", "USDC", 100.0),
    ("b", "2021-05-03", "deposit", "DAI", 50.0),  # never borrowed: not scored
]


def test_only_wallets_that_borrowed_are_scored_and_ratios_are_computed():
    f = wallet_features(_tx(BASE), T)
    assert list(f.index) == ["a"]
    row = f.loc["a"]
    assert row["borrow_to_deposit"] == pytest.approx(0.4)
    assert row["repay_to_borrow"] == pytest.approx(0.25)
    assert row["stable_borrow_share"] == 1.0 and row["n_tx"] == 3


def test_transactions_at_or_after_the_cutoff_do_not_change_features():
    later = BASE + [("a", "2021-06-01", "borrow", "WETH", 9999.0), ("a", "2021-06-20", "liquidationcall", "", 0.0)]
    pd.testing.assert_frame_equal(wallet_features(_tx(BASE), T), wallet_features(_tx(later), T))


def test_liquidation_label_window():
    tx = _tx([("a", "2021-06-10", "liquidationcall", "", 0.0), ("b", "2021-07-10", "liquidationcall", "", 0.0)])
    assert liquidated_between(tx, T, pd.Timestamp("2021-07-01")) == {"a"}
    assert liquidated_between(tx, pd.Timestamp("2021-07-01")) == {"b"}


def test_usd_value_uses_token_decimals(tmp_path):
    rows = [
        {"userWallet": "a", "timestamp": 1620000000, "action": "deposit",
         "actionData": {"amount": "2000000000", "assetSymbol": "USDC", "assetPriceUSD": "1.0"}},
        {"userWallet": "a", "timestamp": 1620000001, "action": "deposit",
         "actionData": {"amount": "2000000000000000000", "assetSymbol": "WETH", "assetPriceUSD": "2500"}},
    ]
    path = tmp_path / "tx.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert list(load_transactions(path)["usd"]) == pytest.approx([2000.0, 5000.0])
