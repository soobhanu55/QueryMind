import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.risk.model import score_wallet
from app.routers import score

app = FastAPI()
app.include_router(score.router)

WALLET = dict(n_tx=40, active_days=12, span_days=60.0, days_since_last_tx=5.0, deposit_usd=20000.0, borrow_usd=9000.0,
              repay_usd=2000.0, redeem_usd=1000.0, borrow_to_deposit=0.45, repay_to_borrow=0.22, n_borrows=6,
              n_assets=4, stable_borrow_share=0.8, prior_liquidations=0)


async def _post(body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post("/score", json=body)


async def test_score_returns_score_band_and_probability():
    r = await _post(WALLET)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1000 and body["risk_band"] in {"low", "medium", "high"}
    assert body["risk_score"] == round(1000 * body["probability"])


async def test_invalid_features_are_rejected():
    assert (await _post({**WALLET, "deposit_usd": -1})).status_code == 422
    assert (await _post({**WALLET, "stable_borrow_share": 1.5})).status_code == 422
    missing = {k: v for k, v in WALLET.items() if k != "n_tx"}
    assert (await _post(missing)).status_code == 422


def test_logistic_arithmetic_on_a_tiny_model():
    model = {"features": ["x"], "log_features": [], "mean": [1.0], "scale": [2.0], "coef": [2.0], "intercept": 0.0}
    assert score_wallet({"x": 1.0}, model)["risk_score"] == 500  # z = 0
    assert score_wallet({"x": 3.0}, model)["probability"] == pytest.approx(0.8808, abs=1e-4)  # z = 2 * (3-1)/2 = 2
