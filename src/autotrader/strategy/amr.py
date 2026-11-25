"""AMR — Adaptive Momentum Risk-parity (적응형 모멘텀 리스크패리티). 기본 전략.

왜 이 전략인가 (실증 근거가 가장 두터운 요소만 결합, 파라미터 수 최소화)
  1) 시계열 모멘텀(절대 모멘텀): 자산이 현금보다 못하면 보유하지 않는다
     → 2000·2008·2022 같은 장기 하락장에서 손실을 잘라낸다 (Moskowitz·Ooi·Pedersen 2012)
  2) 횡단면 모멘텀(상대 모멘텀): 강한 자산 상위 K개만 담는다 (Antonacci 2014, Faber 2007)
  3) 역변동성(리스크패리티) 배분: 변동성이 큰 자산은 적게 → 위험을 고르게 분산
  4) 변동성 타기팅: 포트폴리오 예상 변동성이 목표(연 10%)를 넘으면 현금 비중을 늘린다
     (Moreira·Muir 2017). 레버리지는 쓰지 않는다(scale ≤ 1)
  5) 월 1회 리밸런싱 + 노트레이드 밴드 → 회전율·수수료 최소화

알고리즘 (신호 시점 t 의 종가까지만 사용)
  score_i  = mean_k( r_i(k개월) − r_cash(k개월) ),  k ∈ {3, 6, 12}
  후보     = score 상위 top_k (ML 오버레이가 켜지면 z(score)·z(ML) 혼합 점수로 순위만 조정)
  비중     = 후보들 간 역변동성(또는 ERC) 비중, 단 score ≤ 0 인 후보의 몫은 현금으로
  상한     = 종목당 max_weight, 초과분은 현금
  변동성   = 포트폴리오 예상변동성 > target_vol 이면 전체를 비례 축소
  나머지   = 현금성 ETF(예: CD금리 ETF)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.strategy.base import Strategy, StrategyContext
from autotrader.strategy.signals import (BarSpec, above_sma, cov_matrix, erc_weights, inverse_vol_weights,
                                         momentum_score, portfolio_vol, realized_vol, zscore)


@dataclass
class AMRParams:
    risk_assets: list[str]
    cash_asset: str | None = None
    lookback_months: list[float] = field(default_factory=lambda: [3, 6, 12])
    lookback_weights: list[float] | None = None
    top_k: int = 3
    vol_window_months: float = 3
    weighting: str = "inverse_vol"      # inverse_vol | erc | equal
    max_weight: float = 0.40
    target_vol: float | None = 0.10     # None 이면 변동성 타기팅 끔
    trend_sma_months: float | None = None  # 예: 10 → 10개월 이동평균 위일 때만 보유(추가 필터)
    drawdown_brake: float | None = None    # 예: 0.15 → 전략 낙폭 15% 초과 시 위험자산 절반
    drawdown_brake_scale: float = 0.5
    ml_blend: float = 0.0                  # 0~1. ML 점수를 순위에 반영하는 비율(검증 통과 시에만)
    cov_shrink: float = 0.10


class AMRStrategy(Strategy):
    name = "amr"

    def __init__(self, params: AMRParams):
        if params.top_k < 1:
            raise ValueError("top_k 는 1 이상")
        if params.weighting not in ("inverse_vol", "erc", "equal"):
            raise ValueError("weighting 은 inverse_vol | erc | equal")
        self.p = params

    def symbols(self) -> list[str]:
        s = list(self.p.risk_assets)
        if self.p.cash_asset:
            s.append(self.p.cash_asset)
        return s

    def warmup_bars(self, bars: BarSpec) -> int:
        return bars.bars(max(self.p.lookback_months)) + 1

    def target_weights(self, prices: pd.DataFrame, ctx: StrategyContext) -> pd.Series:
        p = self.p
        bars = ctx.bars
        out = pd.Series(0.0, index=self.symbols())
        risk = [s for s in p.risk_assets if s in prices.columns]
        px = prices[risk].ffill()
        cash_px = prices[p.cash_asset].ffill() if p.cash_asset and p.cash_asset in prices.columns else None

        lbs = [bars.bars(m) for m in p.lookback_months]
        mom = momentum_score(px, lbs, p.lookback_weights, cash=cash_px)
        vol_w = bars.bars(p.vol_window_months)
        # 변동성 추정 구간에 결측이 없는 자산만 유효
        valid = [s for s in risk if np.isfinite(mom.get(s, np.nan)) and px[s].iloc[-vol_w - 1:].notna().all()]
        diag = ctx.diagnostics
        diag["momentum"] = mom.round(4).to_dict()
        if not valid:
            diag["reason"] = "유효 자산 없음(이력 부족) → 전액 현금"
            return self._fill_cash(out)

        # ---- 순위 점수(ML 오버레이는 순위에만 영향, 진입 자격은 실제 모멘텀으로 판단)
        rank_score = mom[valid]
        if p.ml_blend > 0 and ctx.ml_scores is not None:
            ml = ctx.ml_scores.reindex(valid)
            if ml.notna().sum() >= 2:
                rank_score = (1 - p.ml_blend) * zscore(mom[valid]) + p.ml_blend * zscore(ml.fillna(ml.mean()))
                diag["ml_used"] = True
        candidates = list(rank_score.sort_values(ascending=False).index[: p.top_k])

        eligible = mom[candidates] > 0
        if p.trend_sma_months:
            eligible &= above_sma(px[candidates], bars.bars(p.trend_sma_months))

        # ---- 후보 간 위험배분 → 부적격 후보 몫은 현금
        cov = cov_matrix(px[candidates], vol_w, bars.periods_per_year, shrink=p.cov_shrink)
        if p.weighting == "equal" or len(candidates) == 1:
            base = pd.Series(1.0 / len(candidates), index=candidates)
        elif p.weighting == "erc" and np.isfinite(cov.values).all():
            base = erc_weights(cov)
        else:
            base = inverse_vol_weights(realized_vol(px[candidates], vol_w, bars.periods_per_year))
            base = base.reindex(candidates).fillna(0.0)
        w = base.where(eligible, 0.0)
        w = w.clip(upper=p.max_weight)

        # ---- 변동성 타기팅(레버리지 없음)
        scale = 1.0
        if p.target_vol and w.sum() > 0 and np.isfinite(cov.values).all():
            est = portfolio_vol(w, cov)
            if est > p.target_vol:
                scale = p.target_vol / est
            diag["est_vol"] = round(est, 4)
        # ---- 낙폭 브레이크(선택)
        if p.drawdown_brake and ctx.equity_drawdown <= -abs(p.drawdown_brake):
            scale *= p.drawdown_brake_scale
            diag["drawdown_brake"] = True
        w = w * scale

        for s, v in w.items():
            out[s] = float(v)
        diag.update({"candidates": candidates, "eligible": eligible.to_dict(), "scale": round(scale, 4)})
        return self._fill_cash(out)

    def _fill_cash(self, w: pd.Series) -> pd.Series:
        w = w.clip(lower=0.0)
        if self.p.cash_asset:
            w[self.p.cash_asset] = max(0.0, 1.0 - float(w.drop(self.p.cash_asset).sum()))
        total = float(w.sum())
        if total > 1.0 + 1e-9:
            w = w / total
        return w
