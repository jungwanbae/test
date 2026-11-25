"""실행엔진 E2E — PaperBroker(즉시 체결) + 합성 일봉으로 실제 주문 흐름을 검증."""
from datetime import datetime

import pandas as pd
import pytest

from autotrader.brokers.paper import PaperBroker
from autotrader.calendar import KRXCalendar
from autotrader.config import AppConfig, Asset, UniverseConfig
from autotrader.data.synthetic import generate
from autotrader.execution.engine import KST, TradingEngine
from autotrader.execution.state import StateStore
from autotrader.models import OrderRequest, OrderStatus, OrderType, Quote, Side


@pytest.fixture
def market():
    px, _ = generate(n_days=252 * 4, seed=3)
    px.index = pd.bdate_range(end="2026-10-30", periods=len(px))
    px = px * 100  # 원화 ETF 가격대(1만원 안팎)
    return px


def _make(tmp_path, market, monkeypatch, *, when=datetime(2026, 11, 2, 9, 31, tzinfo=KST), live=False, **risk):
    monkeypatch.chdir(tmp_path)
    risky = [c for c in market.columns if c != "CASH"]
    cfg = AppConfig()
    cfg.universe = UniverseConfig([Asset(s, s) for s in risky], Asset("CASH", "CASH"))
    cfg.storage.db_path = str(tmp_path / "db.sqlite")
    cfg.storage.cache_dir = str(tmp_path / "cache")
    cfg.execution.poll_interval_sec = 0
    cfg.execution.fill_timeout_sec = 0
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    last = market.iloc[-1]

    def quote(sym):
        p = float(last[sym])
        return Quote(sym, p, bid=p, ask=p)

    def hist(sym, start, end=None):
        s = market[sym].loc[pd.Timestamp(start):pd.Timestamp(end) if end else None]
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1.0}).rename_axis("date")

    broker = PaperBroker(quote, initial_cash=50_000_000, state_file=str(tmp_path / "paper.json"), history_fn=hist)
    broker.is_live = live
    eng = TradingEngine(cfg, broker, store=StateStore(cfg.storage.db_path), calendar=KRXCalendar(),
                        clock=lambda: when, sleep=lambda s: None)
    return eng, broker, cfg


def test_monthly_rebalance_executes_then_skips(tmp_path, market, monkeypatch):
    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    rep = eng.run_once()
    assert rep.status == "done", rep.message
    acct = broker.get_account()
    held = {s: p.market_value for s, p in acct.positions.items()}
    equity = acct.cash + sum(held.values())
    for s, w in rep.targets.items():  # 목표비중에 근접(1주 단위·버퍼 오차 이내)
        assert abs(held.get(s, 0) / equity - w) < 0.02, (s, w)
    assert eng.store.rebalanced("2026-11")
    again = eng.run_once()
    assert again.status == "skipped"  # 같은 달 재실행 없음


def test_dry_run_places_no_orders(tmp_path, market, monkeypatch):
    eng, broker, _ = _make(tmp_path, market, monkeypatch)
    rep = eng.run_once(dry_run=True)
    assert rep.status == "dry_run" and "목표" in rep.plan_text
    assert broker.get_account().positions == {} and not eng.store.rebalanced("2026-11")


def test_signal_uses_only_past_bars(tmp_path, market, monkeypatch):
    eng, *_ = _make(tmp_path, market, monkeypatch)
    _, diag = eng.compute_targets(datetime(2026, 11, 2).date())
    assert diag["signal_date"] == "2026-10-30"


def test_holiday_and_hours_skip(tmp_path, market, monkeypatch):
    eng, *_ = _make(tmp_path, market, monkeypatch, when=datetime(2026, 9, 25, 9, 31, tzinfo=KST))
    assert eng.run_once().status == "skipped"  # 추석
    eng2, *_ = _make(tmp_path, market, monkeypatch, when=datetime(2026, 11, 2, 15, 10, tzinfo=KST))
    assert eng2.run_once().status == "skipped"  # 마감 이후


def test_kill_switch_and_live_confirmation(tmp_path, market, monkeypatch):
    eng, *_ = _make(tmp_path, market, monkeypatch)
    (tmp_path / "STOP").write_text("")
    assert eng.run_once().status == "error"
    (tmp_path / "STOP").unlink()
    monkeypatch.delenv("AUTOTRADER_LIVE_CONFIRM", raising=False)
    live, *_ = _make(tmp_path, market, monkeypatch, live=True)
    rep = live.run_once()
    assert rep.status == "error" and "AUTOTRADER_LIVE_CONFIRM" in rep.message


def test_capital_limit_respected(tmp_path, market, monkeypatch):
    eng, broker, _ = _make(tmp_path, market, monkeypatch, capital_limit=10_000_000)
    assert eng.run_once().status == "done"
    acct = broker.get_account()
    assert sum(p.market_value for p in acct.positions.values()) <= 10_000_000
    assert acct.cash >= 40_000_000 - 1


def test_stale_open_order_is_cancelled_before_trading(tmp_path, market, monkeypatch):
    eng, broker, _ = _make(tmp_path, market, monkeypatch)
    sym = market.columns[0]
    r = broker.place_order(OrderRequest(sym, Side.BUY, 1, OrderType.LIMIT, float(market[sym].iloc[-1]) * 0.5))
    assert [o.status for o in broker.get_orders()] == [OrderStatus.OPEN]
    assert eng.run_once().status == "done"
    assert {o.order_id: o.status for o in broker.get_orders()}[r.order_id] == OrderStatus.CANCELLED


def test_risk_turnover_limit_blocks(tmp_path, market, monkeypatch):
    eng, broker, _ = _make(tmp_path, market, monkeypatch, max_turnover_pct=0.1)
    rep = eng.run_once()
    assert rep.status == "error" and "회전율" in rep.message
    assert broker.get_account().positions == {}


def test_daemon_runs_once_per_day(tmp_path, market, monkeypatch):
    eng, broker, _ = _make(tmp_path, market, monkeypatch)
    calls = []
    orig = eng.run_once
    eng.run_once = lambda **kw: calls.append(kw) or orig(**kw)
    eng.run_daemon(stop_after=5)
    assert len(calls) == 1 and eng.store.rebalanced("2026-11")
