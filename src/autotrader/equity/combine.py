"""슬리브 결합 유틸: 월간 결합(성과표) · 일별 결합(일별 MDD)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.data.krw_assets import kospi200_daily, usdkrw_daily


def monthly_mix(stock_eq: pd.Series, etf_eq: pd.Series, w: float, end: str | None = None) -> pd.Series:
    ra = stock_eq.dropna().resample("ME").last().pct_change()
    r = pd.concat([ra, etf_eq.pct_change()], axis=1, sort=True).loc["2001-02":end].dropna()
    return (1 + w * r.iloc[:, 0] + (1 - w) * r.iloc[:, 1]).cumprod()


def daily_etf_returns(panel, px_m: pd.DataFrame) -> pd.DataFrame:
    """ETF 슬리브 자산의 일별 원화 수익률. 실제 일별: KOSPI200, S&P500(2022년까지)×환율. 나머지는 월수익 기하분할."""
    days = panel.dates[(panel.dates >= "1996-01-01")]
    out = pd.DataFrame(index=days)
    out["KOSPI200"] = kospi200_daily(panel).reindex(days).fillna(0.0)
    fx = usdkrw_daily().reindex(days).ffill()
    try:
        from skfolio.datasets import load_sp500_index
        spx_d = (load_sp500_index()["SP500"].astype(float).reindex(days).ffill() * fx).pct_change()
    except Exception:
        spx_d = pd.Series(np.nan, index=days)
    mret = px_m.pct_change()
    month = days.to_period("M")
    n = pd.Series(1, index=days).groupby(month).transform("count")
    for col in ("SPX_KRW", "GOLD_KRW", "UST10_KRW", "CASH_KRW"):
        m = mret[col]
        mr = pd.Series(m.values, index=m.index.to_period("M")).reindex(month).values
        out[col] = (1 + pd.Series(mr, index=days)) ** (1 / n) - 1
        if col == "SPX_KRW":
            real = spx_d.dropna()
            out.loc[real.index, col] = real
    return out.fillna(0.0)


def etf_daily_from_weights(dr: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    w = weights.reindex(columns=dr.columns).fillna(0.0)
    w.index = w.index.to_period("M")
    wd = w.reindex(dr.index.to_period("M") - 1)   # 전월말 비중으로 이번 달 운용
    wd.index = dr.index
    return (wd.fillna(0.0) * dr).sum(axis=1)


def daily_mix(stock_eq: pd.Series, etf_daily_ret: pd.Series, w: float, start: str = "2001-02-01",
              end: str | None = None) -> pd.Series:
    """월초에 w : 1-w 로 리밸런싱, 월중 드리프트 반영."""
    sd = stock_eq.pct_change()
    days = etf_daily_ret.index[(etf_daily_ret.index >= start) & ((etf_daily_ret.index <= end) if end else True)]
    sd = sd.reindex(days).fillna(0.0)
    er = etf_daily_ret.reindex(days).fillna(0.0)
    vals, eq, cur = [w, 1 - w], [], None
    for d, a, b in zip(days, sd.values, er.values):
        if d.to_period("M") != cur:
            tot = vals[0] + vals[1]
            vals, cur = [tot * w, tot * (1 - w)], d.to_period("M")
        vals = [vals[0] * (1 + a), vals[1] * (1 + b)]
        eq.append(vals[0] + vals[1])
    return pd.Series(eq, index=days)
