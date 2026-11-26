"""추세필터 리스크패리티 + 변동성 타기팅(레버리지 허용).

  1) 자산별 위험(공분산)으로 ERC(동일위험기여) 비중 계산 — 상관이 낮은 원화 기준 자산(코스피·미국주식·금·미국채)
     의 분산효과를 최대한 활용
  2) 추세 필터(선택): 가격이 N개월 이동평균 아래인 자산은 비중을 trend_floor 배로 축소(완전 제외 아님 → 휩쏘 완화)
  3) 포트폴리오 예상변동성을 target_vol 에 맞추도록 전체 배율 조정(최대 max_leverage)
  4) 나머지는 현금성 자산
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autotrader.strategy.base import Strategy, StrategyContext
from autotrader.strategy.signals import BarSpec, cov_matrix, erc_weights, portfolio_vol


@dataclass
class RPParams:
    risk_assets: list[str]
    cash_asset: str | None = None
    vol_window_months: float = 12
    target_vol: float | None = 0.10
    max_leverage: float = 1.0
    trend_months: float | None = 10
    trend_floor: float = 0.0          # 추세 이탈 자산 비중 배율(0=제외, 0.5=절반)
    cov_shrink: float = 0.2


class TrendRiskParity(Strategy):
    name = "trend_rp"

    def __init__(self, p: RPParams):
        self.p = p

    def symbols(self) -> list[str]:
        return list(self.p.risk_assets) + ([self.p.cash_asset] if self.p.cash_asset else [])

    def warmup_bars(self, bars: BarSpec) -> int:
        return bars.bars(max(self.p.vol_window_months, self.p.trend_months or 0)) + 1

    def target_weights(self, prices: pd.DataFrame, ctx: StrategyContext) -> pd.Series:
        p, bars = self.p, ctx.bars
        out = pd.Series(0.0, index=self.symbols())
        px = prices[p.risk_assets].ffill()
        win = bars.bars(p.vol_window_months)
        valid = [s for s in p.risk_assets if px[s].iloc[-win - 1:].notna().all()]
        if not valid:
            if p.cash_asset:
                out[p.cash_asset] = 1.0
            return out
        cov = cov_matrix(px[valid], win, bars.periods_per_year, shrink=p.cov_shrink)
        w = erc_weights(cov)
        if p.trend_months:
            n = bars.bars(p.trend_months)
            up = px[valid].iloc[-1] > px[valid].iloc[-n:].mean()
            w = w.where(up, w * p.trend_floor)
            ctx.diagnostics["trend_up"] = up.to_dict()
        scale = 1.0
        if p.target_vol and w.sum() > 0:
            est = portfolio_vol(w, cov)
            scale = min(p.max_leverage / max(w.sum(), 1e-9), p.target_vol / est) if est > 0 else 1.0
            scale = min(scale, p.max_leverage / max(w.sum(), 1e-9))
            ctx.diagnostics["est_vol"] = round(est, 4)
        w = w * scale
        if w.sum() > p.max_leverage:
            w = w * p.max_leverage / w.sum()
        for s, v in w.items():
            out[s] = float(v)
        if p.cash_asset:
            out[p.cash_asset] = max(0.0, 1.0 - float(w.sum()))
        ctx.diagnostics["scale"] = round(scale, 3)
        return out
