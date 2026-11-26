"""종목 전략 평가 유틸: 기간별 성과(IS/OOS), 규칙기반 합성점수."""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.equity.factors import STOCK_FEATURES

IS_END = "2012-12-31"
OOS_START = "2013-01-01"


def stats(equity: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    e = equity.loc[start:end].dropna()
    if len(e) < 12:
        return {}
    e = e / e.iloc[0]
    r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    ppy = 12 if pd.Series(e.index).diff().dt.days.median() >= 25 else 252
    vol = r.std() * np.sqrt(ppy)
    dd = (e / e.cummax() - 1).min()
    yearly = e.resample("YE").last().pct_change().dropna()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan,
            "MDD": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan, "WorstY": yearly.min() if len(yearly) else np.nan,
            "Years": round(yrs, 1)}


def period_table(curves: dict[str, pd.Series], start: str = "2001-01-01") -> pd.DataFrame:
    rows = {}
    for k, eq in curves.items():
        a, b, c = stats(eq, start, IS_END), stats(eq, OOS_START), stats(eq, start)
        rows[k] = {"IS_CAGR": a.get("CAGR"), "IS_Sharpe": a.get("Sharpe"), "IS_MDD": a.get("MDD"),
                   "OOS_CAGR": b.get("CAGR"), "OOS_Sharpe": b.get("Sharpe"), "OOS_MDD": b.get("MDD"),
                   "ALL_CAGR": c.get("CAGR"), "ALL_Sharpe": c.get("Sharpe"), "ALL_MDD": c.get("MDD"),
                   "ALL_WorstY": c.get("WorstY")}
    return pd.DataFrame(rows).T


def monthly_signal_dates(ds: pd.DataFrame) -> pd.DatetimeIndex:
    d = pd.Series(ds["date"].unique())
    return pd.DatetimeIndex(d.groupby(d.dt.to_period("M")).max().values)


def composite_score(ds: pd.DataFrame, is_end: str = IS_END) -> pd.Series:
    """IS 기간 월말 IC 의 부호·크기로 가중한 합성점수(OOS 정보 미사용)."""
    m = ds[(ds["date"] <= is_end)].dropna(subset=["target"])
    me = m.groupby(m["date"].dt.to_period("M"))["date"].transform("max")
    m = m[m["date"] == me]
    feats = [f for f in STOCK_FEATURES if f != "kospi"]
    ic = {f: m.groupby("date").apply(lambda g, f=f: g[f].corr(g["target"], method="spearman"),
                                     include_groups=False).mean() for f in feats}
    w = pd.Series(ic)
    w = w.where(w.abs() >= 0.02, 0.0)
    return (ds[feats] * w).sum(axis=1)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes_std: float, periods_per_year: int = 12) -> dict:
    """Deflated Sharpe Ratio(Bailey & López de Prado 2014): 여러 설정을 시험한 데 따른 '운 좋은 최댓값' 보정.

    SR0 = 시험 횟수 N 에서 귀무가설(진짜 SR=0)하에 기대되는 최대 SR. DSR = P(진짜 SR > SR0).
    """
    from math import exp, sqrt
    from statistics import NormalDist
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std()                      # 기간 단위
    g3 = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    euler = 0.5772156649
    nd = NormalDist()
    sd_trials = trial_sharpes_std / sqrt(periods_per_year)   # 연율 → 기간 단위
    sr0 = sd_trials * ((1 - euler) * nd.inv_cdf(1 - 1 / n_trials) + euler * nd.inv_cdf(1 - 1 / (n_trials * exp(1))))
    z = (sr - sr0) * sqrt(T - 1) / sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    return {"sharpe_annual": sr * sqrt(periods_per_year), "sr0_annual": sr0 * sqrt(periods_per_year),
            "dsr": nd.cdf(z), "n_trials": n_trials, "T": T}
