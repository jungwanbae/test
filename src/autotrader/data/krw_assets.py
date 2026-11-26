"""원화(KRW) 기준 멀티에셋 월간 총수익 지수(1996~현재) — 한국 투자자가 실제로 받는 수익.

  KOSPI200  : KRX 전종목 패널로 만든 시총 상위 200 가중지수(가격, 배당 제외 → 보수적)
  SPX_KRW   : S&P500 총수익(USD) × 원/달러 — 1990~2022 일별 지수 월말값 + 배당, 이후 Shiller 월간
  GOLD_KRW  : 금(USD, 월평균) × 원/달러
  UST10_KRW : 미국채 10년 파채권 합성 총수익 × 원/달러
  CASH_KRW  : 원화 단기금리 근사(한국은행 기준금리 연평균 + 0.3%p, 공개자료 기반 근사치)

환율은 FRED H.10(github.com/datasets/exchange-rates) 일별 원/달러의 월말값.
월평균 기반 시계열(SPX 2023~, 금, 금리)은 백테스트에서 1개월 체결지연으로 평활 편향을 제거한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.data.research import build_us_monthly, fetch

# 한국은행 기준금리 연평균 근사(%) — 현금성 자산 수익률 추정용(공개 이력 기반 근사, 정밀값 아님)
_BOK_BASE = {1996: 12.0, 1997: 13.0, 1998: 15.0, 1999: 4.8, 2000: 5.1, 2001: 4.7, 2002: 4.2, 2003: 3.9, 2004: 3.6,
             2005: 3.4, 2006: 4.3, 2007: 4.8, 2008: 4.7, 2009: 2.0, 2010: 2.2, 2011: 3.1, 2012: 3.0, 2013: 2.6,
             2014: 2.3, 2015: 1.6, 2016: 1.3, 2017: 1.3, 2018: 1.6, 2019: 1.6, 2020: 0.6, 2021: 0.7, 2022: 2.1,
             2023: 3.5, 2024: 3.4, 2025: 2.6, 2026: 2.5}

EXCHANGE_URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"


def usdkrw_daily(cache_dir: str | Path = "data/research") -> pd.Series:
    p = Path(cache_dir) / "fx_daily.csv"
    if not p.exists():
        import requests
        r = requests.get(EXCHANGE_URL, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        df[df["Country"] == "South Korea"][["Date", "Exchange rate"]].to_csv(p, index=False)
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Exchange rate"].astype(float)
    return s[s > 0].sort_index()


def kospi200_daily(panel) -> pd.Series:
    """시총 상위 200 종목 시총가중 일간수익률(전일 시총 기준)."""
    cap = panel.marcap.shift(1)
    top = cap.rank(axis=1, ascending=False) <= 200
    w = cap.where(top)
    r = panel.ret.where(top)
    ok = r.notna() & w.notna()
    return ((r.fillna(0) * w.where(ok, 0)).sum(axis=1) / w.where(ok, 0).sum(axis=1)).fillna(0.0)


def build_krw_monthly(panel, cache_dir: str | Path = "data/research") -> tuple[pd.DataFrame, dict]:
    us, meta = build_us_monthly(cache_dir)                    # SPX(월평균 TR), GOLD, UST10 (USD)
    me = lambda s: s.resample("ME").last()                    # noqa: E731
    fx = me(usdkrw_daily(cache_dir))
    k200 = me((1 + kospi200_daily(panel)).cumprod())

    # S&P500: 1990~2022 일별 지수 월말 + 배당수익(Shiller D/P 연율/12) → 이후 Shiller 월평균 TR 로 연결
    try:
        from skfolio.datasets import load_sp500_index
        spd = me(load_sp500_index()["SP500"].astype(float))
        fetch(cache_dir)
        sh = pd.read_csv(Path(cache_dir) / "sp500.csv", parse_dates=["Date"]).set_index("Date")
        dy = (sh["Dividend"] / sh["SP500"]).replace(0, np.nan).ffill()
        dy.index = dy.index.to_period("M").to_timestamp("M")
        tr = spd.pct_change() + dy.reindex(spd.index).ffill() / 12
        spx = (1 + tr.fillna(0)).cumprod()
        tail = us["SPX"].loc[spx.index[-1]:]
        spx = pd.concat([spx, spx.iloc[-1] * tail.iloc[1:] / tail.iloc[0]])
        meta["spx_source"] = "1990-2022 일별 월말(skfolio)+Shiller 배당, 2023~ Shiller 월평균 TR"
    except Exception:
        spx = us["SPX"]
        meta["spx_source"] = "Shiller 월평균 TR"
    df = pd.DataFrame({"KOSPI200": k200, "SPX_USD": spx, "GOLD_USD": us["GOLD"], "UST10_USD": us["UST10"],
                       "USDKRW": fx})
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.groupby(level=0).last().loc["1995-12":].ffill(limit=1)
    out = pd.DataFrame(index=df.index)
    out["KOSPI200"] = df["KOSPI200"]
    for k in ("SPX", "GOLD", "UST10"):
        out[f"{k}_KRW"] = df[f"{k}_USD"] * df["USDKRW"]
    cash_r = pd.Series([(_BOK_BASE.get(d.year, 2.5) + 0.3) / 100 / 12 for d in out.index], index=out.index)
    out["CASH_KRW"] = (1 + cash_r).cumprod()
    out = out.dropna()
    meta["usdkrw_last"] = float(df["USDKRW"].dropna().iloc[-1])
    meta["cash_note"] = "한국은행 기준금리 연평균+0.3%p 근사"
    return out / out.iloc[0] * 100, meta
