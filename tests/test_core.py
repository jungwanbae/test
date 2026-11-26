from datetime import date

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.calendar import KRXCalendar
from autotrader.config import AppConfig, load_config
from autotrader.fees import get_fee_schedule
from autotrader.models import AccountSnapshot, Position, Side
from autotrader.portfolio.rebalancer import plan_rebalance
from autotrader.strategy.amr import AMRParams, AMRStrategy
from autotrader.strategy.base import StrategyContext
from autotrader.strategy.benchmarks import FixedWeights
from autotrader.strategy.signals import (DAILY, MONTHLY, BarSpec, cov_matrix, erc_weights, inverse_vol_weights,
                                         momentum_score)
from autotrader.ticks import round_to_tick, shift_ticks, tick_size


# ------------------------------------------------------------------ ticks
@pytest.mark.parametrize("price,kind,tick", [(1999, "etf", 1), (2000, "etf", 5), (45_000, "etf", 5),
                                             (1_500, "stock", 1), (4_990, "stock", 5), (19_990, "stock", 10),
                                             (49_950, "stock", 50), (150_000, "stock", 100),
                                             (499_000, "stock", 500), (700_000, "stock", 1000)])
def test_tick_size(price, kind, tick):
    assert tick_size(price, kind) == tick


def test_round_and_shift():
    assert round_to_tick(12_347, "etf", "up") == 12_350
    assert round_to_tick(12_347, "etf", "down") == 12_345
    assert round_to_tick(1_999.4, "etf", "up") == 2_000
    assert shift_ticks(2_000, -1, "etf") == 1_999
    assert shift_ticks(1_999, 1, "etf") == 2_000
    assert shift_ticks(12_345, 2, "etf") == 12_355


# ------------------------------------------------------------------ signals
def test_barspec_infer():
    assert BarSpec.infer(pd.bdate_range("2020-01-01", periods=50)) == DAILY
    assert BarSpec.infer(pd.date_range("2000-01-31", periods=50, freq="ME")) == MONTHLY


def test_momentum_excess_over_cash(trend_prices):
    m = momentum_score(trend_prices[["UP1", "DOWN"]], [63, 126, 252], cash=trend_prices["CASH"])
    assert m["UP1"] > 0 > m["DOWN"]


def test_inverse_vol_and_erc():
    vol = pd.Series({"A": 0.1, "B": 0.2})
    w = inverse_vol_weights(vol)
    assert w["A"] == pytest.approx(2 / 3)
    cov = pd.DataFrame([[0.04, 0.006, 0.0], [0.006, 0.01, 0.002], [0.0, 0.002, 0.0225]],
                       index=list("XYZ"), columns=list("XYZ"))
    e = erc_weights(cov)
    rc = e * (cov.values @ e.values)
    assert e.sum() == pytest.approx(1)
    assert rc.std() / rc.mean() < 1e-6  # 위험기여 동일


def test_cov_shrink_shape(trend_prices):
    c = cov_matrix(trend_prices, 63, 252)
    assert c.shape == (5, 5) and np.allclose(c.values, c.values.T)


# ------------------------------------------------------------------ AMR
def _amr(**kw):
    return AMRStrategy(AMRParams(risk_assets=["UP1", "UP2", "FLAT", "DOWN"], cash_asset="CASH", **kw))


def test_amr_selects_trending_and_fills_cash(trend_prices):
    ctx = StrategyContext(bars=DAILY)
    w = _amr(top_k=2, target_vol=None, max_weight=1.0).target_weights(trend_prices, ctx)
    assert set(ctx.diagnostics["candidates"]) == {"UP1", "UP2"}
    assert w["DOWN"] == 0 and w["FLAT"] == 0
    assert w["UP1"] > w["UP2"]  # 변동성 낮은 쪽이 더 큼(역변동성)
    assert w.sum() == pytest.approx(1.0)


def test_amr_all_negative_goes_cash(trend_prices):
    falling = trend_prices.copy()
    for c in ["UP1", "UP2", "FLAT", "DOWN"]:
        falling[c] = trend_prices["DOWN"] * (1 + 0.01 * ["UP1", "UP2", "FLAT", "DOWN"].index(c))
    w = _amr().target_weights(falling, StrategyContext(bars=DAILY))
    assert w["CASH"] == pytest.approx(1.0)


def test_amr_vol_target_scales_down(trend_prices):
    loose = _amr(top_k=2, target_vol=None, max_weight=1.0).target_weights(trend_prices, StrategyContext(bars=DAILY))
    tight = _amr(top_k=2, target_vol=0.02, max_weight=1.0).target_weights(trend_prices, StrategyContext(bars=DAILY))
    assert tight.drop("CASH").sum() < loose.drop("CASH").sum()
    assert tight.sum() == pytest.approx(1.0)


def test_amr_max_weight_cap(trend_prices):
    w = _amr(top_k=1, target_vol=None, max_weight=0.4).target_weights(trend_prices, StrategyContext(bars=DAILY))
    assert w.drop("CASH").max() <= 0.4 + 1e-12 and w["CASH"] == pytest.approx(0.6)


def test_amr_insufficient_history_is_cash(trend_prices):
    w = _amr().target_weights(trend_prices.iloc[:100], StrategyContext(bars=DAILY))
    assert w["CASH"] == 1.0


def test_amr_ml_only_reranks_not_eligibility(trend_prices):
    ctx = StrategyContext(bars=DAILY, ml_scores=pd.Series({"UP1": -5, "UP2": -5, "FLAT": 0, "DOWN": 10}))
    w = _amr(top_k=1, ml_blend=1.0, target_vol=None, max_weight=1.0).target_weights(trend_prices, ctx)
    # ML 이 DOWN 을 1위로 올려도 실제 모멘텀이 음수면 보유하지 않는다
    assert w["DOWN"] == 0 and w["CASH"] == pytest.approx(1.0)


# ------------------------------------------------------------------ backtest
def test_backtest_no_lookahead(trend_prices):
    strat = _amr()
    cfg = BacktestConfig(fees=get_fee_schedule("zero"), slippage_bps=0)
    base = run_backtest(trend_prices, strat, cfg)
    cut = trend_prices.index[400]
    shocked = trend_prices.copy()
    shocked.loc[shocked.index > cut, "DOWN"] *= 5  # 미래를 바꿔도
    alt = run_backtest(shocked, strat, cfg)
    pd.testing.assert_series_equal(base.equity.loc[:cut], alt.equity.loc[:cut])  # 과거 결과는 동일


def test_backtest_buy_and_hold_matches_asset(trend_prices):
    cfg = BacktestConfig(fees=get_fee_schedule("zero"), slippage_bps=0, execution_lag=0, no_trade_band=1.0)
    r = run_backtest(trend_prices, FixedWeights({"UP1": 1.0}), cfg)
    asset = trend_prices["UP1"].loc[r.equity.index]
    assert r.equity.iloc[-1] / r.equity.iloc[0] == pytest.approx(asset.iloc[-1] / asset.iloc[0], rel=1e-9)


def test_backtest_costs_reduce_return(trend_prices):
    free = run_backtest(trend_prices, _amr(), BacktestConfig(fees=get_fee_schedule("zero"), slippage_bps=0))
    paid = run_backtest(trend_prices, _amr(), BacktestConfig(fees=get_fee_schedule("kis_standard"), slippage_bps=20))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.stats["total_cost_pct"] > 0


def test_backtest_integer_shares(trend_prices):
    r = run_backtest(trend_prices, _amr(), BacktestConfig(fractional=False, initial_capital=5_000_000))
    assert (r.trades["qty"] == r.trades["qty"].round()).all()


# ------------------------------------------------------------------ rebalancer
def _acct(cash, **pos):
    return AccountSnapshot(cash=cash, positions={s: Position(s, q, q, 1.0, 1.0) for s, q in pos.items()})


def test_plan_initial_buy():
    t = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    plan = plan_rebalance(t, _acct(10_000_000), {"A": 10_000, "B": 20_000, "C": 5_000}, cash_buffer_pct=0.0,
                          fee_rate=0.0)
    q = {o.symbol: o.qty for o in plan.orders}
    assert q == {"A": 500, "B": 150, "C": 400}
    assert all(o.side == Side.BUY for o in plan.orders)


def test_plan_sells_first_and_full_exit():
    t = pd.Series({"A": 0.0, "B": 1.0})
    plan = plan_rebalance(t, _acct(0, A=100), {"A": 10_000, "B": 10_000}, cash_buffer_pct=0.0, fee_rate=0.0)
    assert plan.orders[0].side == Side.SELL and plan.orders[0].qty == 100
    assert plan.orders[1].side == Side.BUY and plan.orders[1].qty == 100


def test_plan_band_and_capital_limit():
    t = pd.Series({"A": 0.5, "B": 0.5})
    plan = plan_rebalance(t, _acct(100_000_000), {"A": 10_000, "B": 10_000}, capital_limit=1_000_000,
                          cash_buffer_pct=0.0, fee_rate=0.0, min_order_value=1)
    assert sum(o.value for o in plan.orders) <= 1_000_000
    small = plan_rebalance(t, _acct(10_000, A=50, B=49), {"A": 10_000, "B": 10_000}, cash_buffer_pct=0.0,
                           fee_rate=0.0, min_order_value=50_000)
    assert small.orders == []


def test_plan_leaves_unmanaged_holdings():
    t = pd.Series({"A": 1.0})
    acct = _acct(1_000_000)
    acct.positions["XYZ"] = Position("XYZ", 10, 10, 1.0, 1_000_000)  # 다른 보유종목(1천만원)
    plan = plan_rebalance(t, acct, {"A": 10_000}, cash_buffer_pct=0.0, fee_rate=0.0)
    assert {o.symbol for o in plan.orders} == {"A"} and plan.base_capital == 1_000_000


# ------------------------------------------------------------------ calendar / config
def test_krx_calendar():
    cal = KRXCalendar()
    assert cal.is_trading_day(date(2026, 11, 25))
    assert not cal.is_trading_day(date(2026, 11, 28))  # 토
    assert not cal.is_trading_day(date(2026, 2, 17))   # 설날
    assert not cal.is_trading_day(date(2026, 9, 25))   # 추석
    assert not cal.is_trading_day(date(2026, 12, 31))  # 연말 휴장
    assert not cal.is_trading_day(date(2026, 5, 1))    # 근로자의날
    assert cal.first_trading_day_of_month(2026, 3) == date(2026, 3, 3)  # 3/1 일요일 → 3/2 대체공휴일
    assert not KRXCalendar(broker_check=lambda d: False).is_trading_day(date(2026, 11, 25))


def test_default_config_loads():
    cfg = load_config("config/default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.broker.name == "nh_plug" and cfg.broker.env == "mock"
    assert len(cfg.universe.risk_assets) == 4 and cfg.universe.cash_asset.symbol == "459580"
    assert cfg.strategy.name == "msap" and cfg.msap.stock_weight == 0.70


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("strategy:\n  topk: 3\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_ml_portfolio_gate_rejects_noise(trend_prices):
    from autotrader.ml.gate import portfolio_uplift
    rng = np.random.default_rng(1)
    assets = ["UP1", "UP2", "FLAT", "DOWN"]
    noise = pd.DataFrame(rng.normal(size=(len(trend_prices), 4)), index=trend_prices.index, columns=assets)
    make = lambda b: _amr(top_k=2, ml_blend=b)  # noqa: E731
    res = portfolio_uplift(trend_prices, make, noise, 1.0, BacktestConfig())
    assert set(res) >= {"sharpe_base", "sharpe_ml", "sharpe_uplift", "passed"}
    oracle = trend_prices[assets].pct_change(21).shift(-22)  # 미래를 아는 예측 → 잡음보다 나빠선 안 됨
    assert portfolio_uplift(trend_prices, make, oracle, 1.0, BacktestConfig())["sharpe_ml"] >= res["sharpe_ml"] - 1e-9


def test_price_deviation_is_instrument_aware():
    from autotrader.config import RiskConfig
    from autotrader.models import Quote
    from autotrader.portfolio.risk import RiskManager, RiskViolation
    rm = RiskManager(RiskConfig(), is_live=False)
    jump = Quote("X", 11_500, prev_close=10_000)  # +15%
    rm.check_quote(jump, None, "stock")            # 주식은 가격제한폭 이내 → 통과
    with pytest.raises(RiskViolation):
        rm.check_quote(jump, None, "etf")          # ETF 가 +15% 면 이상치
