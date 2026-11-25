"""비교용 벤치마크 전략."""
from __future__ import annotations

import pandas as pd

from autotrader.strategy.base import Strategy, StrategyContext


class FixedWeights(Strategy):
    """고정비중(정기 리밸런싱). 예) 60/40, 영구포트폴리오, 단일자산 보유."""

    def __init__(self, weights: dict[str, float], name: str = "fixed"):
        tot = sum(weights.values())
        if tot <= 0 or tot > 1 + 1e-9:
            raise ValueError("비중 합은 (0, 1] 이어야 합니다")
        self.weights = dict(weights)
        self.name = name

    def symbols(self) -> list[str]:
        return list(self.weights)

    def target_weights(self, prices: pd.DataFrame, ctx: StrategyContext) -> pd.Series:
        w = pd.Series(self.weights, dtype=float)
        last = prices.iloc[-1].reindex(w.index)
        w[last.isna()] = 0.0  # 아직 상장 전인 자산은 제외
        s = w.sum()
        return w / s * sum(self.weights.values()) if s > 0 else w


def equal_weight(symbols: list[str], name: str = "equal_weight") -> FixedWeights:
    return FixedWeights({s: 1.0 / len(symbols) for s in symbols}, name=name)
