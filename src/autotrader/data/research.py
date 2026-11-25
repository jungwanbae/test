"""장기 검증용 실제 시장 데이터(월간, 1871~현재) — 공개 데이터셋(github.com/datasets).

  * S&P500: Shiller 데이터(월평균 지수 + 연환산 배당) → 배당 재투자 총수익 지수
            배당 데이터가 끊긴 이후 구간은 마지막 12개월 평균 배당수익률로 추정(estimated 표시)
  * 금: 월간 금 가격(USD). 1971년 브레턴우즈 체제 종료 전은 고정가격이라 검증은 1972년 이후 사용
  * 미국채 10년: 월간 금리로 '만기 10년 파(par) 채권' 총수익을 합성
            r_t = P(쿠폰 y_{t-1}, 금리 y_t, 잔존 10-1/12년) − 1 + y_{t-1}/12
  * 현금: 무료 공개 단기금리 시계열을 쓸 수 없어 0% 로 둔다(현금을 많이 들고 가는 전략에 불리 → 보수적)

국내 ETF 는 상장 기간이 짧아(대부분 2010년 이후) 수십 년 검증이 불가능하므로, 동일 자산군의
장기 데이터로 전략 '로직'을 검증하고, 실제 운용 종목은 `--source broker` 로 증권사 일봉을 받아 재검증한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

URLS = {
    "sp500": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "gold": "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
    "us10y": "https://raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv",
}

US_ASSETS = {"SPX": "미국주식(S&P500 총수익)", "GOLD": "금(USD)", "UST10": "미국채10년(합성 총수익)"}


def fetch(cache_dir: str | Path = "data/research", refresh: bool = False) -> dict[str, Path]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in URLS.items():
        p = cache / f"{key}.csv"
        if refresh or not p.exists():
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        out[key] = p
    return out


def par_bond_return(y_prev: float, y_now: float, maturity_years: float = 10.0, freq: int = 12) -> float:
    """연복리 가정의 파채권 1기간 총수익."""
    n = maturity_years - 1.0 / freq
    c = y_prev
    if y_now <= 1e-9:
        price = c * n + 1.0
    else:
        disc = (1 + y_now) ** (-n)
        price = c / y_now * (1 - disc) + disc
    return price - 1.0 + y_prev / freq


def build_us_monthly(cache_dir: str | Path = "data/research", start: str = "1871-01") -> tuple[pd.DataFrame, dict]:
    """반환: (월말 기준 총수익 지수 DataFrame[SPX, GOLD, UST10], 메타정보)."""
    paths = fetch(cache_dir)
    meta: dict = {}

    sp = pd.read_csv(paths["sp500"], parse_dates=["Date"])
    sp["m"] = sp["Date"].dt.to_period("M")
    sp = sp.set_index("m").sort_index()
    price = sp["SP500"].astype(float)
    div = sp["Dividend"].astype(float)
    known = div[div > 0]
    last_known = known.index.max()
    yld = (known / price.reindex(known.index)).iloc[-12:].mean()
    div_est = div.copy()
    est_mask = div_est.index > last_known
    div_est[est_mask] = price[est_mask] * yld
    meta["spx_dividend_estimated_from"] = str(last_known + 1)
    meta["spx_dividend_yield_assumed"] = round(float(yld), 4)
    spx_ret = (price + div_est / 12) / price.shift(1) - 1

    gold = pd.read_csv(paths["gold"])
    gold["m"] = pd.PeriodIndex(gold["Date"], freq="M")
    gold = gold.set_index("m")["Price"].astype(float).sort_index()
    gold_ret = gold / gold.shift(1) - 1

    y = pd.read_csv(paths["us10y"], parse_dates=["Date"])
    y["m"] = y["Date"].dt.to_period("M")
    y = y.set_index("m")["Rate"].astype(float).sort_index() / 100
    shiller_y = (sp["Long Interest Rate"].astype(float) / 100).replace(0, np.nan)
    y = y.combine_first(shiller_y).dropna()
    bond_ret = pd.Series([np.nan] + [par_bond_return(a, b) for a, b in zip(y.values[:-1], y.values[1:])],
                         index=y.index)

    rets = pd.concat({"SPX": spx_ret, "GOLD": gold_ret, "UST10": bond_ret}, axis=1)
    rets = rets.loc[pd.Period(start, "M"):]
    rets = rets.dropna(how="all")
    # 각 자산의 최초 관측 이후만 지수화(앞쪽 결측은 상장 전처럼 NaN 유지)
    idx = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for c in rets.columns:
        r = rets[c]
        first = r.first_valid_index()
        if first is None:
            continue
        seg = r.loc[first:].fillna(0.0)
        level = 100 * (1 + seg).cumprod()
        idx.loc[level.index, c] = level / (1 + seg.iloc[0])
    # 모든 자산이 관측되는 마지막 달까지만
    last_common = min(rets[c].last_valid_index() for c in rets.columns)
    idx = idx.loc[:last_common]
    idx.index = idx.index.to_timestamp(how="end").normalize()
    meta["last_month"] = str(last_common)
    return idx.astype(float), meta
