"""전략 토너먼트용 일별 데이터(원화 기준) — 미국 ETF·장기 합성지수·환율·VIX·단기금리·한국 지수/슬리브.

데이터 출처(모두 공개):
  * 미국 ETF 수정종가: github.com/Johnbrick123/sp500-data (prices.parquet, 1995~)
  * 장기 합성 총수익지수: github.com/mwilczynska/financial_datasets (금 LBMA·장기/중기/단기 국채·GSCI, 1970~)
  * 원/달러: FRED H.10(github.com/datasets/exchange-rates), VIX: github.com/datasets/finance-vix
  * 미국 단기금리: Fama-French RF(1926~2018, arch 패키지 내장) + 2019년 이후 3개월 T-bill 연평균 근사치
  * 한국: KRX 전종목 패널로 만든 KOSPI200 프록시, 종목 알파 슬리브 일별 자산곡선

레버리지 상품(일일 리셋): r_L = L·r − (L−1)·금리 − 보수  (미국 상품은 달러 금리, 한국 상품은 원화 금리)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

US_DIR = Path("data/us")
# 3개월 T-bill 연평균(%) — Fama-French RF 이후 구간 근사(공개 통계 기반 근사치)
_TBILL_AFTER_2018 = {2018: 1.95, 2019: 2.06, 2020: 0.37, 2021: 0.05, 2022: 2.02, 2023: 5.07, 2024: 4.97,
                     2025: 4.15, 2026: 3.70}
ETF_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "TIP", "SHY", "GLD", "AGG", "LQD", "SOXX",
               "VWO", "HYG", "XLK"]
MW_SERIES = {"GLD_L": "gold", "TLT_L": "long_term_us_treasury", "IEF_L": "intermediate_term_us_treasury",
             "SHY_L": "short_term_us_treasury", "DBC_L": "broad_commodities", "SPX_L": "us_large_cap_sp500"}


def _mw(name: str) -> pd.Series:
    d = pd.read_csv(US_DIR / f"{name}.csv", usecols=["Date", "Adj Close"], parse_dates=["Date"])
    s = d.dropna().set_index("Date")["Adj Close"].astype(float)
    return s[~s.index.duplicated()].sort_index()


def us_prices() -> pd.DataFrame:
    px = pd.read_parquet(US_DIR / "prices.parquet", columns=["date", "ticker", "adj_close"],
                         filters=[("ticker", "in", ETF_TICKERS)])
    w = px.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    w.index = pd.to_datetime(w.index).normalize()
    for k, f in MW_SERIES.items():
        if (US_DIR / f"{f}.csv").exists():
            w[k] = _mw(f).reindex(w.index.union(_mw(f).index)).ffill().reindex(w.index)
    return w


def tbill_daily(index: pd.DatetimeIndex) -> pd.Series:
    """미국 단기금리(연, 소수)."""
    try:
        import arch.data.frenchdata as fd
        rf = fd.load()["RF"] / 100  # 월간 수익률
        rf.index = pd.to_datetime(rf.index).to_period("M")
        ann = (1 + rf) ** 12 - 1
    except Exception:
        ann = pd.Series(dtype=float)
    out = pd.Series(np.nan, index=index)
    per = index.to_period("M")
    m = ann.reindex(per).values
    out[:] = m
    for y, v in _TBILL_AFTER_2018.items():
        mask = (index.year == y) & out.isna().values
        out[mask] = v / 100
    return out.ffill().fillna(0.03)


def kr_rate_daily(index: pd.DatetimeIndex) -> pd.Series:
    from autotrader.data.krw_assets import _BOK_BASE
    return pd.Series([(_BOK_BASE.get(d.year, 2.5) + 0.3) / 100 for d in index], index=index)


def usdkrw() -> pd.Series:
    from autotrader.data.krw_assets import usdkrw_daily
    return usdkrw_daily()


def vix() -> pd.Series:
    p = US_DIR / "vix.csv"
    v = pd.read_csv(p, parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
    return v.sort_index()


def build_returns(panel=None, stock_sleeve: pd.Series | None = None, start: str = "1999-01-01") -> tuple[pd.DataFrame, dict]:
    """원화 기준 일별 수익률 표(열 = 투자 가능 자산·합성 레버리지 상품·현금).

    반환: (returns, aux) — aux 에 원본 가격(USD)·VIX·금리 등 신호 계산용 시계열.
    """
    w = us_prices()
    w = w.loc[start:]
    days = w.index
    fx = usdkrw().reindex(days.union(usdkrw().index)).ffill().reindex(days)
    fx_r = fx.pct_change().fillna(0.0)
    rf_us = tbill_daily(days)
    rf_kr = kr_rate_daily(days)
    dt = pd.Series(days, index=days).diff().dt.days.fillna(1).clip(1, 5)
    r_usd = w.pct_change()
    # 짧은 이력 ETF 는 장기 합성지수로 앞부분을 채운다(신호·수익 모두)
    for etf, long in (("GLD", "GLD_L"), ("TLT", "TLT_L"), ("IEF", "IEF_L"), ("SHY", "SHY_L")):
        if long in r_usd:
            r_usd[etf] = r_usd[etf].where(w[etf].notna(), r_usd[long])
    r_usd["DBC"] = r_usd.get("DBC_L")
    r_usd["CASH_USD"] = (1 + rf_us) ** (dt / 365) - 1
    R = pd.DataFrame(index=days)
    for c in ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "TIP", "SHY", "GLD", "DBC", "AGG", "LQD",
              "SOXX", "CASH_USD"]:
        if c in r_usd:
            R[c] = (1 + r_usd[c]) * (1 + fx_r) - 1
    # 레버리지(일일 리셋, 달러 금리 + 보수)
    fee = {2: 0.0095, 3: 0.0095}
    for base in ("QQQ", "SPY", "GLD", "TLT", "SOXX"):
        for L in (2, 3):
            lev = L * r_usd[base] - (L - 1) * rf_us * dt / 360 - fee[L] * dt / 365
            R[f"{base}x{L}"] = (1 + lev) * (1 + fx_r) - 1
    R["CASH_KRW"] = (1 + rf_kr) ** (dt / 365) - 1
    if panel is not None:
        from autotrader.data.krw_assets import kospi200_daily
        # 한국 수익률을 미국 거래일 달력으로 옮길 때 누적지수로 맞춰 미국 휴장일의 한국 수익이 빠지지 않게 한다
        kidx = (1 + kospi200_daily(panel)).cumprod()
        k = kidx.reindex(days.union(kidx.index)).ffill().reindex(days).pct_change().fillna(0.0)
        R["K200"] = k
        R["K200x2"] = 2 * k - rf_kr * dt / 365 - 0.0064 * dt / 365
    if stock_sleeve is not None:
        st = stock_sleeve.dropna()
        R["KR_ALPHA"] = st.reindex(days.union(st.index)).ffill().reindex(days).pct_change().fillna(0.0)
    sig_cols = [c for c in ("SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "TIP", "SHY", "GLD", "DBC", "AGG",
                            "LQD", "SOXX") if c in r_usd]
    P_usd = (1 + r_usd[sig_cols]).cumprod()            # 신호용 달러 가격지수(상장 전 구간은 합성지수)
    P_usd = P_usd.where(r_usd[sig_cols].notna().cumsum() > 0)
    if "K200" in R:
        P_usd["K200"] = (1 + R["K200"]).cumprod()       # 한국 지수는 원화
    aux = {"px_usd": w, "P_sig": P_usd, "fx": fx, "rf_us": rf_us, "rf_kr": rf_kr,
           "vix": vix().reindex(days).ffill()}
    return R.fillna(0.0), aux


def price_index(R: pd.DataFrame) -> pd.DataFrame:
    return (1 + R).cumprod()


def build_long_usd() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """1970~현재 미국 장기 합성(달러 기준) — 꼬리위험 검증용. 열: SPY(=S&P500 TR)·TLT·IEF·SHY·GLD·DBC·CASH_USD·레버리지."""
    L = pd.DataFrame({k: _mw(f) for k, f in MW_SERIES.items() if (US_DIR / f"{f}.csv").exists()})
    L = L.dropna(how="all").sort_index().ffill()
    days = L.index
    rf = tbill_daily(days)
    dt = pd.Series(days, index=days).diff().dt.days.fillna(1).clip(1, 5)
    r = L.pct_change().fillna(0.0)
    R = pd.DataFrame(index=days)
    for k, v in {"SPY": "SPX_L", "TLT": "TLT_L", "IEF": "IEF_L", "SHY": "SHY_L", "GLD": "GLD_L", "DBC": "DBC_L"}.items():
        if v in r:
            R[k] = r[v]
    R["QQQ"] = R["SPY"]   # 장기 검증에서는 나스닥 대신 S&P500 으로 대체(나스닥100 은 1985년 이후)
    R["CASH_USD"] = (1 + rf) ** (dt / 365) - 1
    for base in ("SPY", "QQQ", "GLD", "TLT"):
        for Lv in (2, 3):
            R[f"{base}x{Lv}"] = Lv * R[base] - (Lv - 1) * rf * dt / 360 - 0.0095 * dt / 365
    R["CASH_KRW"] = R["CASH_USD"]
    P = (1 + R[["SPY", "QQQ", "TLT", "IEF", "SHY", "GLD", "DBC"]]).cumprod()
    return R, P, rf
