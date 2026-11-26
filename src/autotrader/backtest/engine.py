"""이벤트 기반 백테스터.

미래참조 방지 원칙
  * 신호: 리밸런싱 주기의 마지막 바(예: 월말) 종가까지의 데이터만 전략에 전달
  * 체결: execution_lag 바 뒤의 종가(일봉 기본 1 = 다음 거래일, 월봉 기본 0 = 같은 월말 종가)
  * 비용: 증권사 수수료 + 유관기관 제비용 + 슬리피지 + (주식이면) 매도세
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from autotrader.backtest.metrics import drawdown, performance
from autotrader.fees import FeeSchedule, get_fee_schedule
from autotrader.strategy.base import Strategy, StrategyContext
from autotrader.strategy.signals import BarSpec


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000_000
    fees: FeeSchedule = field(default_factory=lambda: get_fee_schedule("nh_plug"))
    slippage_bps: float = 5.0
    rebalance: str = "M"                 # M(월) | Q(분기) | W(주)
    execution_lag: int | None = None     # None → 일봉 1, 월봉 0
    no_trade_band: float = 0.01          # |목표-현재| < 밴드×자산 이면 매매 생략
    drift_threshold: float | None = None  # 주기 사이 비중이탈이 이 값을 넘으면 추가 리밸런싱
    fractional: bool = True              # False 면 1주 단위
    cash_rate_annual: float = 0.0        # 미투자 현금 이자(연)
    max_leverage: float = 1.0            # 총노출 상한(>1 이면 차입 허용)
    borrow_spread: float = 0.015         # 차입금리 = 기준(현금자산 수익률 또는 cash_rate) + 스프레드(연)
    borrow_rate_symbol: str | None = None  # 차입 기준금리로 쓸 현금성 자산(예: CASH_KRW)
    stock_symbols: frozenset[str] = frozenset()  # 매도세 대상(개별주식). 나머지는 ETF 로 간주
    start: str | None = None             # 이 날짜 이후 첫 신호부터 운용


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    stats: dict
    bars: BarSpec
    diagnostics: list[dict] = field(default_factory=list)


def _signal_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    key = {"M": index.to_period("M"), "Q": index.to_period("Q"), "W": index.to_period("W")}[freq]
    s = pd.Series(index, index=index)
    return set(s.groupby(key).max().values)


def run_backtest(prices: pd.DataFrame, strategy: Strategy, cfg: BacktestConfig | None = None,
                 ml_scores: pd.DataFrame | None = None, bars: BarSpec | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    syms = [s for s in strategy.symbols() if s in prices.columns]
    px = prices[syms].sort_index().ffill()
    bars = bars or BarSpec.infer(px.index)
    lag = cfg.execution_lag if cfg.execution_lag is not None else (0 if bars.periods_per_year <= 12 else 1)
    signals = _signal_dates(px.index, cfg.rebalance)
    start_ts = pd.Timestamp(cfg.start) if cfg.start else None
    warmup = strategy.warmup_bars(bars)
    slip = cfg.slippage_bps / 10_000
    cash_rate_bar = (1 + cfg.cash_rate_annual) ** (1 / bars.periods_per_year) - 1

    shares = pd.Series(0.0, index=syms)
    cash = float(cfg.initial_capital)
    started = False
    pending: tuple[int, pd.Series] | None = None
    last_target: pd.Series | None = None
    eq_hist: list[float] = []
    eq_index: list[pd.Timestamp] = []
    w_rows, trades, turn, costs, diags = [], [], {}, {}, []

    def execute(i: int, target: pd.Series) -> None:
        nonlocal cash
        prow = px.iloc[i]
        valid = prow.notna() & (prow > 0)
        tgt = target.reindex(syms).fillna(0.0).where(valid, 0.0)
        equity = cash + float((shares * prow.where(valid, 0.0)).sum())
        if equity <= 0:
            return
        cur_val = shares * prow.where(valid, 0.0)
        tgt_val = tgt * equity
        delta = tgt_val - cur_val
        traded, cost_sum = 0.0, 0.0
        sells, buys = {}, {}
        for s in syms:
            if not valid[s]:
                continue
            d = delta[s]
            full_exit = tgt[s] == 0 and shares[s] > 0
            if abs(d) < cfg.no_trade_band * equity and not full_exit:
                continue
            q = d / prow[s]
            if not cfg.fractional:
                q = math.floor(tgt_val[s] / prow[s]) - shares[s]
            if q < 0:
                sells[s] = min(-q, shares[s])
            elif q > 0:
                buys[s] = q
        for s, q in sells.items():
            if q <= 0:
                continue
            notional = q * prow[s] * (1 - slip)
            c = cfg.fees.cost(notional, is_sell=True, is_etf=s not in cfg.stock_symbols)
            cash += notional - c
            shares[s] -= q
            traded += notional
            cost_sum += c + q * prow[s] * slip
            trades.append((px.index[i], s, -q, prow[s], c))
        need = sum(q * prow[s] * (1 + slip) * (1 + cfg.fees.total_rate) for s, q in buys.items())
        budget = cash + max(0.0, cfg.max_leverage - 1.0) * equity  # 레버리지 한도까지 차입 가능
        ratio = min(1.0, budget / need) if need > 0 and budget > 0 else (1.0 if need <= 0 else 0.0)
        for s, q in buys.items():
            q = q * ratio
            if not cfg.fractional:
                q = math.floor(q)
            if q <= 0:
                continue
            notional = q * prow[s] * (1 + slip)
            c = cfg.fees.cost(notional, is_sell=False)
            cash -= notional + c
            shares[s] += q
            traded += notional
            cost_sum += c + q * prow[s] * slip
            trades.append((px.index[i], s, q, prow[s], c))
        turn[px.index[i]] = traded / equity
        costs[px.index[i]] = cost_sum

    brw = prices[cfg.borrow_rate_symbol].sort_index().ffill() if cfg.borrow_rate_symbol else None
    spread_bar = (1 + cfg.borrow_spread) ** (1 / bars.periods_per_year) - 1
    for i, t in enumerate(px.index):
        prow = px.iloc[i]
        if i > 0 and cash > 0:
            cash *= 1 + cash_rate_bar
        elif i > 0 and cash < 0:  # 차입 이자
            base = (brw.loc[t] / brw.loc[px.index[i - 1]] - 1) if brw is not None else cash_rate_bar
            cash *= 1 + base + spread_bar
        if pending and pending[0] == i:
            execute(i, pending[1])
            pending = None
        equity = cash + float((shares * prow.fillna(0.0)).sum())
        # 주기 사이 비중 이탈 리밸런싱
        if cfg.drift_threshold and started and last_target is not None and pending is None and t not in signals:
            cur_w = (shares * prow.fillna(0.0)) / equity if equity > 0 else shares * 0
            if (cur_w - last_target.reindex(syms).fillna(0.0)).abs().max() > cfg.drift_threshold:
                execute(i, last_target)
                equity = cash + float((shares * prow.fillna(0.0)).sum())
        if t in signals and i + 1 >= warmup and (start_ts is None or t >= start_ts):
            dd = 0.0
            if eq_hist:
                peak = max(eq_hist)
                dd = equity / peak - 1 if peak > 0 else 0.0
            ctx = StrategyContext(bars=bars, as_of=t, equity_drawdown=dd,
                                  ml_scores=(ml_scores.loc[:t].iloc[-1] if ml_scores is not None
                                             and len(ml_scores.loc[:t]) else None))
            target = strategy.target_weights(px.iloc[: i + 1], ctx)
            diags.append({"date": t, **ctx.diagnostics})
            last_target = target
            started = True
            if lag == 0:
                execute(i, target)
                equity = cash + float((shares * prow.fillna(0.0)).sum())
            elif i + lag < len(px):
                pending = (i + lag, target)
        if started:
            eq_index.append(t)
            eq_hist.append(equity)
            w_rows.append(((shares * prow.fillna(0.0)) / equity if equity > 0 else shares * 0).rename(t))

    equity_s = pd.Series(eq_hist, index=pd.DatetimeIndex(eq_index), name=strategy.name)
    weights = pd.DataFrame(w_rows)
    trades_df = pd.DataFrame(trades, columns=["date", "symbol", "qty", "price", "fee"])
    turnover = pd.Series(turn, dtype=float)
    cost_s = pd.Series(costs, dtype=float)
    stats = performance(equity_s, bars.periods_per_year, turnover=turnover) if len(equity_s) > 1 else {}
    if len(equity_s) > 1:
        cash_sym = getattr(getattr(strategy, "p", None), "cash_asset", None)
        risk_cols = [c for c in weights.columns if c != cash_sym]
        stats["avg_risk_exposure"] = float(weights[risk_cols].sum(axis=1).mean())
        stats["total_cost_pct"] = float(cost_s.sum() / cfg.initial_capital)
    return BacktestResult(equity_s, weights, trades_df, turnover, cost_s, stats, bars, diags)


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        s = r.stats
        rows[name] = {"CAGR": s.get("cagr"), "Vol": s.get("vol"), "Sharpe": s.get("sharpe"),
                      "Sortino": s.get("sortino"), "MDD": s.get("max_drawdown"), "Calmar": s.get("calmar"),
                      "WorstYear": s.get("worst_year"), "Underwater(y)": s.get("longest_underwater_years"),
                      "Turnover/y": s.get("turnover_per_year"), "x": s.get("final_multiple")}
    return pd.DataFrame(rows).T


def align_common(results: dict[str, BacktestResult]) -> dict[str, pd.Series]:
    start = max(r.equity.index[0] for r in results.values())
    out = {}
    for k, r in results.items():
        e = r.equity.loc[start:]
        out[k] = e / e.iloc[0]
    return out


__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "compare", "align_common", "drawdown"]
