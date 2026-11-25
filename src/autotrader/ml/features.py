"""ML 피처/타깃 생성. 모든 피처는 t 시점까지의 데이터만 사용(rolling/과거 shift)하고,
타깃만 미래(t→t+h) 수익률을 사용한다. 주기(일봉/월봉)에 무관하게 '개월' 단위로 정의한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.strategy.signals import BarSpec

FEATURES = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_ratio", "dist_sma10", "dd_12m", "xs_rank_6m", "ret_1bar"]


def _vol(lr: pd.DataFrame, bars: BarSpec, months: float) -> pd.DataFrame:
    w = max(3, bars.bars(months))
    return lr.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(bars.periods_per_year)


def make_features(prices: pd.DataFrame, bars: BarSpec) -> dict[str, pd.DataFrame]:
    p = prices.astype(float)
    lr = np.log(p / p.shift(1))
    vol3 = _vol(lr, bars, 3).clip(lower=1e-4)
    vol1 = _vol(lr, bars, 1)
    f: dict[str, pd.DataFrame] = {}
    for m in (1, 3, 6, 12):
        k = bars.bars(m)
        f[f"mom_{m}m"] = np.log(p / p.shift(k)) / (vol3 * np.sqrt(m / 12))  # 위험조정 모멘텀
    f["vol_ratio"] = np.log((vol1 / vol3).clip(lower=1e-3))
    f["dist_sma10"] = (p / p.rolling(bars.bars(10)).mean() - 1) / vol3
    f["dd_12m"] = (p / p.rolling(bars.bars(12), min_periods=2).max() - 1) / vol3
    f["xs_rank_6m"] = (f["mom_6m"].rank(axis=1, pct=True) - 0.5) * 2
    f["ret_1bar"] = lr / (vol3 / np.sqrt(bars.periods_per_year))
    return {k: v.clip(-5, 5) for k, v in f.items()}


def make_target(prices: pd.DataFrame, bars: BarSpec, horizon_months: float = 1,
                cash: pd.Series | None = None, gap: int = 1) -> pd.DataFrame:
    """변동성 조정 미래 초과수익 (t+gap → t+gap+h). 마지막 gap+h 바는 NaN(미래 미확정).

    gap=1: 신호 다음 바에 체결하는 실제 매매 시점과 맞춘다. 월평균 가격처럼 인접 수익률에
    인위적 자기상관이 있는 데이터에서 '거래 불가능한' 예측력을 걸러내는 효과도 있다.
    """
    p = prices.astype(float)
    h = bars.bars(horizon_months)
    fwd = np.log(p.shift(-h - gap) / p.shift(-gap))
    if cash is not None:
        c = np.log(cash.shift(-h - gap) / cash.shift(-gap))
        fwd = fwd.sub(c, axis=0)
    lr = np.log(p / p.shift(1))
    vol3 = _vol(lr, bars, 3).clip(lower=1e-4)
    return fwd / (vol3 * np.sqrt(horizon_months / 12))


def forward_returns(prices: pd.DataFrame, bars: BarSpec, horizon_months: float = 1, gap: int = 1) -> pd.DataFrame:
    h = bars.bars(horizon_months)
    return prices.shift(-h - gap) / prices.shift(-gap) - 1


def to_tensor(features: dict[str, pd.DataFrame], target: pd.DataFrame | None, assets: list[str],
              seq_len: int, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(X[N, A, L, F], y[N, A], mask[N, A]) — mask=True 는 결측 자산(패딩)."""
    F = np.stack([features[k][assets].values for k in FEATURES], axis=-1)  # [T, A, F]
    pos = {d: i for i, d in enumerate(features[FEATURES[0]].index)}
    X, Y, M = [], [], []
    yv = target[assets].values if target is not None else None
    for d in dates:
        i = pos[d]
        if i + 1 < seq_len:
            continue
        win = F[i + 1 - seq_len: i + 1]  # [L, A, F]
        miss = np.isnan(win).any(axis=(0, 2))
        X.append(np.nan_to_num(np.transpose(win, (1, 0, 2)), nan=0.0))
        M.append(miss)
        if yv is not None:
            Y.append(yv[i])
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=bool)
    Y = np.asarray(Y, dtype=np.float32) if yv is not None else None
    return X, Y, M
