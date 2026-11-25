"""전략 신호 계산(순수 함수). 모든 함수는 '마지막 행 = 신호 시점'만 사용하며 미래 데이터를 보지 않는다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarSpec:
    """데이터 주기 정보. 전략 파라미터를 '개월' 단위로 쓰고 바(bar) 수로 변환한다."""

    bars_per_month: float
    periods_per_year: int

    def bars(self, months: float) -> int:
        return max(1, int(round(months * self.bars_per_month)))

    @classmethod
    def infer(cls, index: pd.Index) -> "BarSpec":
        if len(index) < 3:
            return cls(21, 252)
        gaps = pd.Series(pd.DatetimeIndex(index)).diff().dt.days.dropna()
        med = float(gaps.median())
        if med >= 25:
            return cls(1, 12)
        if med >= 5:
            return cls(52 / 12, 52)
        return cls(21, 252)


DAILY = BarSpec(21, 252)
MONTHLY = BarSpec(1, 12)


def period_return(prices: pd.DataFrame, bars: int) -> pd.Series:
    if len(prices) <= bars:
        return pd.Series(np.nan, index=prices.columns)
    last = prices.iloc[-1]
    past = prices.iloc[-1 - bars]
    return last / past - 1.0


def momentum_score(prices: pd.DataFrame, lookbacks: list[int], weights: list[float] | None = None,
                   cash: pd.Series | None = None) -> pd.Series:
    """여러 기간 수익률의 가중평균(현금 대비 초과수익). 기간이 부족한 자산은 NaN."""
    weights = list(weights) if weights is not None else [1.0] * len(lookbacks)
    if len(weights) != len(lookbacks):
        raise ValueError("lookbacks 와 weights 길이가 다릅니다")
    tot = pd.Series(0.0, index=prices.columns)
    wsum = sum(weights)
    for lb, w in zip(lookbacks, weights):
        r = period_return(prices, lb)
        if cash is not None and len(cash.dropna()) > lb:
            c = cash.dropna()
            r = r - (c.iloc[-1] / c.iloc[-1 - lb] - 1.0)
        tot = tot + w * r  # NaN 전파 → 이력 부족 자산은 NaN
    return tot / wsum


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def realized_vol(prices: pd.DataFrame, window: int, periods_per_year: int, halflife: float | None = None) -> pd.Series:
    r = log_returns(prices).iloc[-window:]
    if halflife:
        v = r.ewm(halflife=halflife).std().iloc[-1]
    else:
        v = r.std()
    return v * np.sqrt(periods_per_year)


def cov_matrix(prices: pd.DataFrame, window: int, periods_per_year: int, shrink: float = 0.1) -> pd.DataFrame:
    """연율화 공분산. 표본이 작을 때를 대비해 대각행렬 방향으로 shrink."""
    r = log_returns(prices).iloc[-window:].dropna(how="any")
    if len(r) < 3:
        n = prices.shape[1]
        return pd.DataFrame(np.eye(n) * np.nan, index=prices.columns, columns=prices.columns)
    c = r.cov() * periods_per_year
    d = np.diag(np.diag(c.values))
    return pd.DataFrame((1 - shrink) * c.values + shrink * d, index=c.index, columns=c.columns)


def inverse_vol_weights(vol: pd.Series) -> pd.Series:
    v = vol.replace(0, np.nan).dropna()
    if v.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / v
    return inv / inv.sum()


def erc_weights(cov: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-10) -> pd.Series:
    """동일위험기여(ERC) 가중치 — 순환좌표하강법(Griveau-Billion et al., 2013).

    min ½yᵀΣy − Σ bᵢ ln yᵢ 의 해 y 를 정규화하면 위험기여 wᵢ(Σw)ᵢ 가 bᵢ 에 비례한다(bᵢ=1/n).
    """
    S = cov.values
    n = S.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    b = np.full(n, 1.0 / n)
    y = 1.0 / np.sqrt(np.diag(S))
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            a = S[i, i]
            c = S[i].dot(y) - a * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * a * b[i])) / (2 * a)
        if np.max(np.abs(y - y_old)) < tol * max(1.0, np.max(np.abs(y))):
            break
    return pd.Series(y / y.sum(), index=cov.index)


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    idx = w.index.intersection(cov.index)
    if len(idx) == 0:
        return 0.0
    wv = w[idx].values
    return float(np.sqrt(max(wv @ cov.loc[idx, idx].values @ wv, 0.0)))


def above_sma(prices: pd.DataFrame, window: int) -> pd.Series:
    if len(prices) < window:
        return pd.Series(False, index=prices.columns)
    return prices.iloc[-1] > prices.iloc[-window:].mean()


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd
