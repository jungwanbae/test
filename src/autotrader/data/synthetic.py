"""레짐 전환형 합성 멀티에셋 일봉 생성기(테스트·ML 파이프라인 점검용).

강세/약세/인플레 레짐을 마르코프 체인으로 전환하며, 레짐별로 자산 기대수익·변동성·상관이 달라진다.
실제 시장이 아니므로 '성과'의 근거로 쓰지 않는다(엔진·전략 로직의 정상동작 확인 용도).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ["EQ_KR", "EQ_US", "EQ_TECH", "GOLD", "BOND_KR", "BOND_US", "CASH"]

# 연율 (기대수익, 변동성) — 레짐별
_REGIMES = {
    "bull": {"EQ_KR": (0.14, 0.16), "EQ_US": (0.15, 0.14), "EQ_TECH": (0.20, 0.20), "GOLD": (0.03, 0.14),
             "BOND_KR": (0.03, 0.05), "BOND_US": (0.03, 0.06), "CASH": (0.03, 0.003)},
    "bear": {"EQ_KR": (-0.25, 0.30), "EQ_US": (-0.22, 0.28), "EQ_TECH": (-0.32, 0.36), "GOLD": (0.08, 0.18),
             "BOND_KR": (0.07, 0.07), "BOND_US": (0.09, 0.08), "CASH": (0.02, 0.003)},
    "inflation": {"EQ_KR": (-0.05, 0.20), "EQ_US": (-0.04, 0.18), "EQ_TECH": (-0.10, 0.26), "GOLD": (0.18, 0.18),
                  "BOND_KR": (-0.08, 0.08), "BOND_US": (-0.10, 0.09), "CASH": (0.04, 0.003)},
}
_CORR = {
    "bull": 0.6, "bear": 0.8, "inflation": 0.5,  # 주식 간 상관
}
_TRANS = np.array([[0.996, 0.002, 0.002],
                   [0.010, 0.988, 0.002],
                   [0.006, 0.002, 0.992]])


def generate(n_days: int = 252 * 20, seed: int = 7, start: str = "2004-01-02") -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    names = list(_REGIMES)
    idx = pd.bdate_range(start, periods=n_days)
    state = 0
    regimes = []
    rets = np.zeros((n_days, len(ASSETS)))
    eq = [0, 1, 2]
    for t in range(n_days):
        state = rng.choice(3, p=_TRANS[state])
        reg = names[state]
        regimes.append(reg)
        mu = np.array([_REGIMES[reg][a][0] for a in ASSETS]) / 252
        sd = np.array([_REGIMES[reg][a][1] for a in ASSETS]) / np.sqrt(252)
        C = np.eye(len(ASSETS))
        for i in eq:
            for j in eq:
                if i != j:
                    C[i, j] = _CORR[reg]
        C[3, 0:3] = C[0:3, 3] = 0.0
        bond_eq = -0.3 if reg != "inflation" else 0.4
        for b in (4, 5):
            for i in eq:
                C[b, i] = C[i, b] = bond_eq
        C[4, 5] = C[5, 4] = 0.7
        L = np.linalg.cholesky(C + 1e-9 * np.eye(len(ASSETS)))
        z = L @ rng.standard_normal(len(ASSETS))
        rets[t] = mu - 0.5 * sd ** 2 + sd * z
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=ASSETS)
    return prices, pd.Series(regimes, index=idx, name="regime")
