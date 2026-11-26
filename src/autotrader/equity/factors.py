"""종목 팩터(피처) 엔진 — 가격·거래대금·시가총액만으로 계산(재무데이터 불필요).

한국 시장에서 실증된 이상현상(anomaly) 위주로 구성:
  모멘텀(12-1, 6-1), 단기반전(1주·1개월), 저변동성, 복권효과(MAX), 52주 신고가 근접도,
  규모, 유동성(Amihud·회전율), 거래대금 추세, 베타·고유변동성, 왜도, 이동평균 이격, 상승일 비율
+ 시장 국면 피처(시장 추세·변동성·breadth) — 모델이 국면별로 다른 팩터를 쓰도록 허용

모든 피처는 t 일 종가까지의 정보만 사용한다. 타깃(미래수익)은 t+1 일 '시가'에 사서
t+1+H 일 시가에 파는 수익(실제 체결 시점과 동일)이다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from autotrader.equity.panel import Panel

log = logging.getLogger(__name__)

STOCK_FEATURES = [
    "r_1w", "r_1m", "r_3m", "mom_6_1", "mom_12_1", "vol_20", "vol_60", "vol_ratio", "max_21", "min_21",
    "hi_52", "lo_52", "size", "turnover", "amihud", "amt_trend", "beta", "idio_vol", "skew_60",
    "ma50", "ma200", "up_ratio", "gap_20", "price", "age", "kospi",
]
MARKET_FEATURES = ["m_r_1m", "m_r_3m", "m_vol_20", "m_ma200", "breadth"]


def sample_dates(index: pd.DatetimeIndex, freq: str = "W") -> pd.DatetimeIndex:
    """주기별 마지막 거래일(W 주간, M 월간, 2W 격주)."""
    s = pd.Series(index, index=index)
    if freq == "2W":
        wk = s.groupby(index.to_period("W")).max()
        return pd.DatetimeIndex(wk.values[::2])
    return pd.DatetimeIndex(s.groupby(index.to_period(freq)).max().values)


def _stock_features(ret: pd.DataFrame, gap: pd.DataFrame, close: pd.DataFrame, amt: pd.DataFrame,
                    cap: pd.DataFrame, mkt: pd.Series, dates: pd.DatetimeIndex, horizon: int) -> dict[str, pd.DataFrame]:
    """종목 묶음(열 일부)에 대한 피처 — 롤링 연산은 열별 독립이므로 묶음 처리 결과가 전체와 동일."""
    listed = ret.notna()
    L = np.log1p(ret.fillna(0.0)).cumsum().where(listed)
    G = np.log1p(gap.fillna(0.0))
    age = listed.cumsum()

    def at(df):
        return df.reindex(dates).astype(np.float32)

    f: dict[str, pd.DataFrame] = {}
    f["r_1w"] = at(L - L.shift(5))
    f["r_1m"] = at(L - L.shift(21))
    f["r_3m"] = at(L - L.shift(63))
    f["mom_6_1"] = at(L.shift(21) - L.shift(126))
    f["mom_12_1"] = at(L.shift(21) - L.shift(252))
    r_ = ret.where(listed)
    v20 = r_.rolling(20, min_periods=15).std()
    v60 = r_.rolling(60, min_periods=40).std()
    f["vol_20"], f["vol_60"] = at(v20), at(v60)
    f["vol_ratio"] = at(np.log((v20 / v60).clip(lower=1e-3)))
    del v20
    f["max_21"] = at(r_.rolling(21, min_periods=15).max())
    f["min_21"] = at(r_.rolling(21, min_periods=15).min())
    f["hi_52"] = at(L - L.rolling(252, min_periods=120).max())
    f["lo_52"] = at(L - L.rolling(252, min_periods=120).min())
    f["size"] = at(np.log(cap.clip(lower=1)))
    a20 = amt.rolling(20, min_periods=10).mean()
    a120 = amt.rolling(120, min_periods=60).mean()
    f["turnover"] = at(np.log((a20 / cap).clip(lower=1e-9)))
    f["amihud"] = at(np.log((r_.abs() / amt.where(amt > 0)).rolling(60, min_periods=30).mean().clip(lower=1e-15)))
    f["amt_trend"] = at(np.log((a20 / a120).clip(lower=1e-3)))
    del a20, a120
    m = pd.DataFrame(np.repeat(mkt.values[:, None], ret.shape[1], axis=1), index=ret.index,
                     columns=ret.columns).where(listed)
    mean_rm = (r_ * m).rolling(120, min_periods=60).mean()
    cov = mean_rm - r_.rolling(120, min_periods=60).mean() * m.rolling(120, min_periods=60).mean()
    beta = cov / m.rolling(120, min_periods=60).var(ddof=0)
    del mean_rm, cov
    f["beta"] = at(beta)
    f["idio_vol"] = at((r_ - beta * m).rolling(60, min_periods=40).std())
    del beta, m
    f["skew_60"] = at(r_.rolling(60, min_periods=40).skew())
    f["ma50"] = at(L - L.rolling(50, min_periods=40).mean())
    ma200 = L.rolling(200, min_periods=150).mean()
    f["ma200"] = at(L - ma200)
    f["_above200"] = at((L > ma200).astype(float).where(L.notna() & ma200.notna()))
    f["up_ratio"] = at((r_ > 0).astype(float).where(listed).rolling(20, min_periods=15).mean())
    f["gap_20"] = at(G.where(listed).rolling(20, min_periods=15).mean())
    f["price"] = at(np.log(close.clip(lower=1)))
    f["age"] = at(np.log1p(age))
    f["_age_days"] = at(age)
    f["_fwd"] = at((L.shift(-horizon) - L) - G.shift(-1) + G.shift(-horizon - 1))
    f["_fwd_1"] = at((L.shift(-1) - L) - G.shift(-1))
    return f


def build_dataset(p: Panel, dates: pd.DatetimeIndex, *, horizon: int = 21, universe_size: int = 500,
                  min_amount: float = 1e8, min_price: float = 1000.0, min_age: int = 252,
                  chunk: int = 400, universe_by: str = "marcap") -> pd.DataFrame:
    """(date, code) 롱 포맷 데이터셋: 피처(횡단면 순위 정규화) + 원값 일부 + 타깃. 메모리 절약을 위해 종목 묶음 처리."""
    mkt = p.market_return().astype(np.float64)
    cols = list(p.ret.columns)
    parts: dict[str, list[pd.DataFrame]] = {}
    for k in range(0, len(cols), chunk):
        cc = cols[k: k + chunk]
        f = _stock_features(p.ret[cc].astype(np.float64), p.gap[cc].astype(np.float64),
                            p.close[cc].astype(np.float64), p.amount[cc].astype(np.float64),
                            p.marcap[cc].astype(np.float64), mkt, dates, horizon)
        for name, v in f.items():
            parts.setdefault(name, []).append(v)
        log.info("피처 계산 %d/%d", min(k + chunk, len(cols)), len(cols))
    F = {k: pd.concat(v, axis=1) for k, v in parts.items()}
    kospi = (p.market.reindex(cols) == "KOSPI").astype(np.float32).values
    F["kospi"] = pd.DataFrame(np.repeat(kospi[None, :], len(dates), axis=0), index=dates, columns=cols)

    amt60 = p.amount.astype(np.float64).rolling(60, min_periods=30).mean().reindex(dates)
    Lm = np.log1p(mkt).cumsum()
    mf = pd.DataFrame({
        "m_r_1m": Lm - Lm.shift(21), "m_r_3m": Lm - Lm.shift(63),
        "m_vol_20": mkt.rolling(20).std() * np.sqrt(252), "m_ma200": Lm - Lm.rolling(200).mean(),
    }).reindex(dates)
    mf["breadth"] = F["_above200"].mean(axis=1)

    listed = p.ret.reindex(dates).notna()
    elig = (listed & (F["_age_days"] >= min_age) & (p.close.reindex(dates) >= min_price) & (amt60 >= min_amount)
            & (p.flag.reindex(dates).fillna(0) < 0.5) & (p.amount.reindex(dates) > 0))
    # 유니버스: 시가총액 상위(기본) — 거래대금 순위는 '관심 집중' 종목 편향(이후 저조)이 생김
    rank_base = p.marcap.reindex(dates) if universe_by == "marcap" else amt60
    elig &= rank_base.where(elig).rank(axis=1, ascending=False) <= universe_size

    rows = []
    for d in dates:
        e = elig.loc[d]
        codes = e.index[e.values]
        if len(codes) < 30:
            continue
        block = pd.DataFrame({k: F[k].loc[d, codes] for k in STOCK_FEATURES})
        raw = block[["vol_60", "size", "beta"]].add_prefix("raw_")
        ranked = block.rank(pct=True).sub(0.5).mul(2.0).fillna(0.0).astype(np.float32)  # [-1, 1]
        ranked["kospi"] = block["kospi"]
        out = pd.concat([ranked, raw], axis=1)
        out["fwd"] = F["_fwd"].loc[d, codes].values
        out["fwd_1"] = F["_fwd_1"].loc[d, codes].values
        out["amount60"] = amt60.loc[d, codes].values
        for k in MARKET_FEATURES:
            out[k] = np.float32(mf.loc[d, k])
        out["date"] = d
        out["code"] = codes
        rows.append(out.reset_index(drop=True))
    ds = pd.concat(rows, ignore_index=True)
    ds["target"] = ds.groupby("date")["fwd"].rank(pct=True).sub(0.5).mul(2.0)
    log.info("데이터셋 %d행, %d개 날짜", len(ds), ds["date"].nunique())
    return ds
