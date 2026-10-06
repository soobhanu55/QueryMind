# Wallet liquidation-risk evaluation

Aave V2 on Polygon, 2021-03-31 to 2021-09-02. Train: 657 borrowers with features before 2021-06-01 (15 liquidated in June). Test: 1247 borrowers with features before 2021-07-01, 36 liquidated from July on (2.9% base rate). The test period is never seen in training. Only wallets with a borrow before the cutoff are scored (no debt, no liquidation).

| Scorer | ROC-AUC (95% CI) | PR-AUC (base rate 0.029) | Lift, top 10% |
|---|---|---|---|
| random (base rate) | 0.52 (0.42 to 0.63) | 0.038 | 1.4x |
| any prior liquidation | 0.70 (0.59 to 0.81) | 0.124 | 4.7x |
| ChainScore heuristic (USD) | 0.55 (0.45 to 0.65) | 0.039 | 1.7x |
| WalletGuard-style (no interest) | 0.53 (0.45 to 0.62) | 0.032 | 0.8x |
| logistic regression | 0.71 (0.62 to 0.80) | 0.077 | 3.4x |
| random forest | 0.79 (0.70 to 0.87) | 0.136 | 4.5x |
| gradient boosting | 0.73 (0.63 to 0.82) | 0.139 | 3.6x |

36 positives make every interval wide; differences of a few AUC points between the learned models are noise.
The original ChainScore R-squared of 0.45 measured how well a forest recovers its own hand-made score; this table measures real liquidations.

Serving model: logistic regression refit on both windows; `db/wallets.csv` holds 1625 wallets scored with it (an in-sample ranking for the demo database, not a held-out prediction).
