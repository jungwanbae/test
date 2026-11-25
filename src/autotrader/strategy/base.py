"""전략 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from autotrader.strategy.signals import BarSpec


@dataclass
class StrategyContext:
    bars: BarSpec
    as_of: pd.Timestamp | None = None
    equity_drawdown: float = 0.0          # 전략 자산의 현재 고점대비 낙폭(<=0)
    ml_scores: pd.Series | None = None    # 자산별 ML 예측 점수(선택)
    diagnostics: dict = field(default_factory=dict)  # 전략이 판단 근거를 기록


class Strategy(ABC):
    name = "base"

    @abstractmethod
    def symbols(self) -> list[str]:
        """전략이 사용하는 모든 종목(현금성 자산 포함)."""

    @abstractmethod
    def target_weights(self, prices: pd.DataFrame, ctx: StrategyContext) -> pd.Series:
        """prices: 신호 시점까지의 (총수익 기준) 종가. 반환: 종목별 목표비중(합 ≤ 1, 나머지는 현금)."""

    def warmup_bars(self, bars: BarSpec) -> int:
        return 0
