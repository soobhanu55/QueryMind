"""POST /score: wallet liquidation risk from on-chain activity features (see docs/wallet_risk_eval.md)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.risk.model import score_wallet

router = APIRouter()


class WalletFeatures(BaseModel):
    n_tx: int = Field(..., ge=0)
    active_days: int = Field(..., ge=0)
    span_days: float = Field(..., ge=0)
    days_since_last_tx: float = Field(..., ge=0)
    deposit_usd: float = Field(..., ge=0)
    borrow_usd: float = Field(..., ge=0)
    repay_usd: float = Field(..., ge=0)
    redeem_usd: float = Field(..., ge=0)
    borrow_to_deposit: float = Field(..., ge=0, le=100)
    repay_to_borrow: float = Field(..., ge=0)
    n_borrows: int = Field(..., ge=0)
    n_assets: int = Field(..., ge=0)
    stable_borrow_share: float = Field(..., ge=0, le=1)
    prior_liquidations: int = Field(..., ge=0)


class ScoreResponse(BaseModel):
    risk_score: int
    risk_band: str
    probability: float


@router.post("/score", response_model=ScoreResponse)
async def score(features: WalletFeatures) -> ScoreResponse:
    return ScoreResponse(**score_wallet(features.model_dump()))
