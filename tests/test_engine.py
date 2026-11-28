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
    cfg.strategy.name = "amr"
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


def test_msap_combines_stock_and_etf_sleeves(tmp_path, market, monkeypatch):
    """멀티 슬리브: 주식 점수 상위 N + ETF 리스크패리티를 합쳐 실제 주문까지(데이터 계층은 스텁)."""
    import autotrader.equity.live as live
    import autotrader.equity.panel as panel_mod

    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    cfg.strategy.name = "msap"
    cfg.msap.stock.top_n = 3
    cfg.msap.stock.sync = False
    stocks = ["005930", "000660", "035420", "051910", "068270"]
    prices = {s: 50_000.0 + 1000 * i for i, s in enumerate(stocks)}
    base_quote = broker._quote_fn
    broker._quote_fn = lambda s: (Quote(s, prices[s], bid=prices[s], ask=prices[s]) if s in prices else base_quote(s))

    class FakePanel:
        names = pd.Series({s: f"종목{s}" for s in stocks})

    monkeypatch.setattr(panel_mod, "build_panel", lambda *a, **k: FakePanel())
    rows = pd.DataFrame({"code": stocks, "date": pd.Timestamp("2026-10-30")})
    monkeypatch.setattr(live, "latest_rows", lambda *a, **k: rows)
    monkeypatch.setattr(live, "score_rows", lambda r, p: pd.Series([5, 4, 3, 2, 1.0], index=stocks))
    rep = eng.run_once()
    assert rep.status == "done", rep.message
    held = broker.get_account().positions
    assert {"005930", "000660", "035420"} <= set(held) and "068270" not in held
    stock_w = sum(rep.targets[s] for s in stocks[:3])
    assert abs(stock_w - cfg.msap.stock_weight) < 1e-3  # 주식 슬리브 70% (리포트는 소수 4자리 반올림)
    assert abs(sum(rep.targets.values()) - 1.0) < 1e-3


def test_quad_adds_high_rate_kospi_part(tmp_path, market, monkeypatch):
    """QUAD: 국내 레그 = MSAP-M × (1−w) + 고금리 대비형 코스피 부분 × w (KOSPI 1배·2배·현금 ETF)."""
    import autotrader.equity.live as live
    import autotrader.equity.panel as panel_mod
    from autotrader.config import Asset

    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    cfg.strategy.name = "msap"
    cfg.msap.stock.top_n = 3
    cfg.msap.stock.sync = False
    hk = cfg.msap.hr_kospi
    hk.weight, hk.kospi, hk.leveraged, hk.cash = 1 / 6, "GOLD", "EQ_TECH", "CASH"   # 상승 추세 자산을 '코스피'로
    hk.us_rate = 0.02                                                                 # 저금리 → 최대 2배 허용
    cfg.universe.risk_assets = [a for a in cfg.universe.risk_assets if a.symbol != "EQ_TECH"]
    cfg.universe.extra_assets = [Asset("EQ_TECH", "레버리지")]
    eng.names = {s: s for s in cfg.universe.all_symbols()}
    stocks = ["005930", "000660", "035420"]
    prices = {s: 50_000.0 for s in stocks}
    base_quote = broker._quote_fn
    broker._quote_fn = lambda s: (Quote(s, prices[s], bid=prices[s], ask=prices[s]) if s in prices else base_quote(s))

    class FakePanel:
        names = pd.Series({s: s for s in stocks})

    monkeypatch.setattr(panel_mod, "build_panel", lambda *a, **k: FakePanel())
    monkeypatch.setattr(live, "latest_rows", lambda *a, **k: pd.DataFrame({"code": stocks, "date": pd.Timestamp("2026-10-30")}))
    monkeypatch.setattr(live, "score_rows", lambda r, p: pd.Series([3, 2, 1.0], index=stocks))
    w, diag = eng.compute_targets(pd.Timestamp("2026-11-02").date())
    assert abs(sum(w[s] for s in stocks) - 0.7 * 5 / 6) < 1e-6                     # 주식 슬리브 70% × (1 − 1/6)
    assert abs(w.sum() - 1.0) < 1e-6
    h = diag["hr_kospi"]
    assert h["exposure"] == 2.0 and abs(w["EQ_TECH"] - 1 / 6) < 1e-6              # 강한 상승 추세 → 2배 ETF 100%
    hk.us_rate = 0.045                                                                # 고금리 → 최대 1배
    eng._cached = None
    w2, d2 = eng.compute_targets(pd.Timestamp("2026-11-02").date())
    assert d2["hr_kospi"]["exposure"] == 1.0 and w2.get("EQ_TECH", 0) == 0


# ---------------------------------------------------------------- 주문 실패·부분체결 (외부 피드백 4번)
def _patch_place(broker, fn):
    orig = broker.place_order
    broker.place_order = lambda req: fn(req, orig)
    return broker


def test_partial_fills_converge_without_overbuying(tmp_path, market, monkeypatch):
    """증권사가 매번 절반만 체결해도 라운드마다 실제 잔고로 재계산 → 목표를 넘겨 사지 않는다."""
    from autotrader.models import OrderRequest as OR
    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    cfg.execution.max_rounds = 6

    def half(req, orig):
        q = max(1, req.qty // 2)
        return orig(OR(req.symbol, req.side, q, req.order_type, req.price, req.tag))
    _patch_place(broker, half)
    rep = eng.run_once()
    assert rep.status in ("done", "partial")
    acct = broker.get_account()
    equity = acct.cash + sum(p.market_value for p in acct.positions.values())
    for s, p in acct.positions.items():
        assert p.market_value / equity <= rep.targets.get(s, 0) + 0.02, s   # 과매수 없음
    assert len(rep.orders) > len(rep.targets)                               # 여러 라운드에 걸쳐 나눠 체결


def test_rejected_symbol_reports_partial_and_others_fill(tmp_path, market, monkeypatch):
    from autotrader.models import OrderResult
    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    target, _ = eng.compute_targets(datetime(2026, 11, 2).date(), broker.get_account())
    bad = target[target > 0.05].index[0]

    def reject(req, orig):
        if req.symbol == bad:
            return OrderResult(req, False, "", "VI 발동으로 주문 거부")
        return orig(req)
    _patch_place(broker, reject)
    rep = eng.run_once()
    assert rep.status == "partial" and "잔여" in rep.message
    assert bad not in broker.get_account().positions
    assert any(o["symbol"] != bad and o["accepted"] for o in rep.orders)
    assert not eng.store.rebalanced("2026-11")   # 미완료 → 다음 실행 때 재시도


def test_network_error_aborts_then_next_run_reconciles(tmp_path, market, monkeypatch):
    """주문 도중 네트워크 오류 → 즉시 중단(재전송 없음). 다음 실행은 실제 잔고 기준으로 남은 것만 주문."""
    from autotrader.brokers.base import BrokerError
    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    calls = {"n": 0}

    def flaky(req, orig):
        calls["n"] += 1
        if calls["n"] == 2:
            raise BrokerError("연결 끊김", category="network")
        return orig(req)
    _patch_place(broker, flaky)
    rep = eng.run_once()
    assert rep.status == "error" and calls["n"] == 2
    first = {s: p.qty for s, p in broker.get_account().positions.items()}
    assert len(first) == 1                       # 오류 전 1건만 체결, 그 뒤로는 보내지 않음
    rep2 = eng.run_once()
    assert rep2.status == "done"
    second = broker.get_account()
    equity = second.cash + sum(p.market_value for p in second.positions.values())
    for s, p in second.positions.items():
        assert p.market_value / equity <= rep2.targets.get(s, 0) + 0.02   # 이중 매수 없음


def test_msap_minervini_stop_sells_and_blocks_rebuy(tmp_path, market, monkeypatch):
    import autotrader.equity.live as live
    import autotrader.equity.panel as panel_mod

    day = {"d": datetime(2026, 11, 2, 9, 31, tzinfo=KST)}
    eng, broker, cfg = _make(tmp_path, market, monkeypatch)
    eng.clock = lambda: day["d"]
    cfg.strategy.name = "msap"
    cfg.msap.stock.top_n = 3
    cfg.msap.stock.sync = False
    cfg.msap.stock.stop_loss = 0.15
    stocks = ["005930", "000660", "035420", "051910"]
    px = {s: 50_000.0 for s in stocks}
    prev = dict(px)
    base_quote = broker._quote_fn
    broker._quote_fn = lambda s: (Quote(s, px[s], bid=px[s], ask=px[s], prev_close=prev[s]) if s in px else base_quote(s))

    class FakePanel:
        names = pd.Series({s: s for s in stocks})

    monkeypatch.setattr(panel_mod, "build_panel", lambda *a, **k: FakePanel())
    monkeypatch.setattr(live, "latest_rows", lambda *a, **k: pd.DataFrame({"code": stocks, "date": pd.Timestamp("2026-10-30")}))
    monkeypatch.setattr(live, "score_rows", lambda r, p: pd.Series([4, 3, 2, 1.0], index=stocks))
    assert eng.run_once().status == "done"
    assert "005930" in broker.get_account().positions
    # 다음날: 005930 전일종가가 매입가 대비 −20% → 손절
    day["d"] = datetime(2026, 11, 3, 9, 31, tzinfo=KST)
    px["005930"] = prev["005930"] = 40_000.0
    rep = eng.run_once()
    assert rep.status == "stopped" and "손절" in rep.message
    assert "005930" not in broker.get_account().positions
    assert eng.store.stopped("2026-11") == {"005930"}
    # 같은 달 강제 리밸런싱에서도 재매수하지 않음(슬롯은 현금성 ETF)
    rep2 = eng.run_once(force=True)
    assert rep2.targets.get("005930", 0) == 0 and "005930" not in broker.get_account().positions
