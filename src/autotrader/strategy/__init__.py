"""전략 팩토리."""
from __future__ import annotations

from autotrader.config import AppConfig
from autotrader.strategy.amr import AMRParams, AMRStrategy
from autotrader.strategy.base import Strategy, StrategyContext


def build_strategy(cfg: AppConfig, ml_blend: float | None = None) -> Strategy:
    s = cfg.strategy
    if s.name != "amr":
        raise ValueError(f"알 수 없는 전략: {s.name}")
    params = AMRParams(
        risk_assets=[a.symbol for a in cfg.universe.risk_assets],
        cash_asset=cfg.universe.cash_asset.symbol if cfg.universe.cash_asset else None,
        lookback_months=list(s.lookback_months), lookback_weights=s.lookback_weights, top_k=s.top_k,
        vol_window_months=s.vol_window_months, weighting=s.weighting, max_weight=s.max_weight,
        target_vol=s.target_vol, trend_sma_months=s.trend_sma_months, drawdown_brake=s.drawdown_brake,
        drawdown_brake_scale=s.drawdown_brake_scale,
        ml_blend=(cfg.ml.blend if cfg.ml.enabled else 0.0) if ml_blend is None else ml_blend,
    )
    return AMRStrategy(params)


__all__ = ["Strategy", "StrategyContext", "AMRStrategy", "AMRParams", "build_strategy"]
