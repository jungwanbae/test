"""MSAP — Multi-Sleeve Adaptive Portfolio (멀티 슬리브 적응형 포트폴리오). 기본 실전 전략.

  슬리브 A(알파, 기본 70%): KRX 종목 머신러닝 랭킹(LightGBM, 26개 가격·거래 피처 + 시장국면 피처)
                          → 상위 15종목 동일가중, 순위 히스테리시스로 회전율 억제, 월 1회
  슬리브 B(분산, 기본 30%): 원화 기준 리스크패리티(ERC) — KODEX200·S&P500·금현물·미국채10년 ETF,
                          12개월 공분산, 목표변동성 10%, 나머지는 CD금리 ETF
  ⇒ 두 슬리브의 상관이 낮아(위기 때 원화 약세가 해외자산을 지지) 포트폴리오 낙폭이 크게 줄어든다.

검증(2001.02~2026.09, 실비용·다음날 시가 체결):
  MSAP-M(미너비니 −15% 손절, 기본): CAGR 18.1%, Sharpe 0.91, 일별 MDD −32.1%, 검증(2013~) 16.8%, 최근 10년 14.8%
  MSAP(손절 없음):                 CAGR 20.0%, Sharpe 0.98, 일별 MDD −41.1%, 검증(2013~) 16.6%, 최근 10년 13.9%
  — reports/minervini_report.md, reports/strategy_report.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from autotrader.equity.live import StockSleeveConfig
from autotrader.strategy.base import StrategyContext
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity
from autotrader.strategy.signals import MONTHLY

log = logging.getLogger(__name__)


@dataclass
class MSAPConfig:
    stock_weight: float = 0.70
    stock: StockSleeveConfig = field(default_factory=StockSleeveConfig)
    rp_target_vol: float = 0.10
    rp_vol_window_months: float = 12
    rp_max_leverage: float = 1.0


def etf_sleeve_weights(monthly_closes: pd.DataFrame, risk: list[str], cash: str | None, cfg: MSAPConfig,
                       diag: dict | None = None) -> pd.Series:
    strat = TrendRiskParity(RPParams(risk_assets=risk, cash_asset=cash, vol_window_months=cfg.rp_vol_window_months,
                                     target_vol=cfg.rp_target_vol, max_leverage=cfg.rp_max_leverage, trend_months=None))
    ctx = StrategyContext(bars=MONTHLY)
    w = strat.target_weights(monthly_closes, ctx)
    if diag is not None:
        diag["etf_sleeve"] = {**ctx.diagnostics, "weights": w.round(4).to_dict()}
    return w


def combine(stock_w: pd.Series, etf_w: pd.Series, stock_weight: float, previously_managed: list[str]) -> pd.Series:
    """두 슬리브를 합친 최종 목표비중. 이전에 전략이 보유했던 종목 중 제외된 것은 0(매도)으로 포함."""
    out = pd.concat([stock_w * stock_weight, etf_w * (1 - stock_weight)])
    out = out.groupby(level=0).sum()
    for s in previously_managed:
        if s not in out.index:
            out[s] = 0.0
    return out
