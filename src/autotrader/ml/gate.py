"""포트폴리오 수준 ML 게이트.

IC 가 통계적으로 유의해도 이미 모멘텀을 쓰는 전략에는 정보가 겹쳐 성과가 오히려 나빠질 수 있다
(장기 실데이터에서 실제로 확인됨: IC t=1.56 인데 AMR+ML 의 Sharpe 가 더 낮았다).
그래서 '같은 표본외 예측으로 백테스트했을 때 Sharpe 가 개선되는가'를 두 번째 관문으로 둔다.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.backtest.metrics import performance
from autotrader.strategy.base import Strategy


def portfolio_uplift(prices: pd.DataFrame, make_strategy: Callable[[float], Strategy], predictions: pd.DataFrame,
                     blend: float, bt_cfg: BacktestConfig, min_uplift: float = 0.0) -> dict:
    base = run_backtest(prices, make_strategy(0.0), bt_cfg)
    with_ml = run_backtest(prices, make_strategy(blend), bt_cfg, ml_scores=predictions)
    start = max(predictions.dropna(how="all").index.min(), base.equity.index[0])
    ppy = base.bars.periods_per_year
    sb = performance(base.equity.loc[start:], ppy)
    sm = performance(with_ml.equity.loc[start:], ppy)
    uplift = sm["sharpe"] - sb["sharpe"]
    return {"sharpe_base": round(sb["sharpe"], 3), "sharpe_ml": round(sm["sharpe"], 3),
            "cagr_base": round(sb["cagr"], 4), "cagr_ml": round(sm["cagr"], 4),
            "sharpe_uplift": round(uplift, 3), "blend": blend, "passed": bool(uplift > min_uplift)}
