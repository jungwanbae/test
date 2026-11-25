"""성과 지표."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def performance(equity: pd.Series, periods_per_year: int, rf_annual: float = 0.0,
                turnover: pd.Series | None = None) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(periods_per_year)
    rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = rets - rf_p
    sharpe = ex.mean() / rets.std() * np.sqrt(periods_per_year) if rets.std() > 0 else np.nan
    downside = rets[rets < rf_p] - rf_p
    dd_dev = np.sqrt((downside ** 2).sum() / len(rets)) * np.sqrt(periods_per_year)
    sortino = ex.mean() * periods_per_year / dd_dev if dd_dev > 0 else np.nan
    dd = drawdown(eq)
    mdd = dd.min()
    # 최장 수중기간(고점 회복까지)
    underwater = dd < 0
    longest, cur = 0, 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    monthly = eq.resample("ME").last().pct_change().dropna()
    out = {
        "start": eq.index[0].date().isoformat(), "end": eq.index[-1].date().isoformat(),
        "years": round(years, 1), "final_multiple": round(eq.iloc[-1] / eq.iloc[0], 2),
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "longest_underwater_years": round(longest / periods_per_year, 1),
        "worst_year": yearly.min(), "best_year": yearly.max(),
        "pct_positive_months": float((monthly > 0).mean()) if len(monthly) else np.nan,
    }
    if turnover is not None and len(turnover):
        out["turnover_per_year"] = float(turnover.sum() / years) if years > 0 else np.nan
    return out


def yearly_returns(equity: pd.Series) -> pd.Series:
    ye = equity.resample("YE").last()
    r = ye.pct_change()
    r.iloc[0] = ye.iloc[0] / equity.iloc[0] - 1
    r.index = r.index.year
    return r


def format_stats(stats: dict) -> dict:
    pct = {"cagr", "vol", "max_drawdown", "worst_year", "best_year", "pct_positive_months"}
    out = {}
    for k, v in stats.items():
        if isinstance(v, float) and k in pct:
            out[k] = f"{v * 100:.2f}%"
        elif isinstance(v, float):
            out[k] = f"{v:.2f}"
        else:
            out[k] = v
    return out
