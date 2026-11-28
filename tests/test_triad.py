"""TRIAD 미국 레그 — NH 해외주식 어댑터(명세 형식 재현)·NYSE 달력·센트 호가·신호·실행엔진 E2E."""
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrader.brokers.nh_plug_overseas import ET, NHPlugOverseasBroker
from autotrader.brokers.paper import PaperBroker
from autotrader.calendar import USCalendar
from autotrader.config import load_config
from autotrader.execution.state import StateStore
from autotrader.execution.us_engine import USTradingEngine
from autotrader.models import OrderRequest, OrderStatus, OrderType, Quote, Side
from autotrader.strategy.triad import us_leg_targets
from autotrader.ticks import round_to_tick, shift_ticks
from conftest import FakeResp, FakeSession

TOKEN = {"access_token": "tok-1", "expires_in": 86400}


def _body(kw):
    return json.loads(kw["data"].decode() if isinstance(kw["data"], bytes) else kw["data"])["Input_0"]


def _us(handler, tmp_path, **kw):
    sess = FakeSession(handler)
    b = NHPlugOverseasBroker("key", "secret", session=sess, token_cache_dir=str(tmp_path), sleep=lambda s: None,
                             account="12345678901", **kw)
    return b, sess


# ====================================================================== 어댑터
def test_overseas_quote_history_on_live_host(tmp_path):
    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        assert url.startswith("https://api.nhplug.com:8443/gbstock/quote/")  # 시세는 운영 전용
        b = _body(kw)
        if url.endswith("/current"):
            assert b == {"iem_cd": "QQQ"}
            return FakeResp(200, {"Output_0": {"trdprc": 651.03, "best_bid1": 651.0, "best_ask1": 651.05,
                                               "hst_trdprc": 659.51, "iem_nm": "INVESCO QQQ", "currency_prc": 1468.2}})
        assert b["gubun"] == "3" and b["xtick"] == "0001" and b["maxavg"] == "0"  # 해외 일봉 = 3 (국내와 다름)
        end = b["end_dt"]
        rows = [{"bsop_date": d.strftime("%Y%m%d"), "open_prc": 10.0, "high": 11.0, "low": 9.0, "close_prc": 10.5,
                 "movolume": 100} for d in pd.bdate_range(end=end, periods=3)[::-1]]
        return FakeResp(200, {"Output_1": rows})

    b, _ = _us(h, tmp_path, env="mock")
    q = b.get_quote("QQQ")
    assert (q.price, q.bid, q.ask, q.prev_close) == (651.03, 651.0, 651.05, 659.51)
    assert b.fx_rate("QQQ") == pytest.approx(1468.2)
    df = b.get_daily_history("QQQ", date(2026, 11, 18), date(2026, 11, 20))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df["close"].iloc[-1] == 10.5


def test_overseas_account_orders_cancel_on_mock_host(tmp_path):
    seen = []

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        assert url.startswith("https://moapi.nhplug.com:8443/gbstock/")  # 모의투자 주문·잔고 = 모의 도메인
        b = _body(kw)
        seen.append((url.rsplit("/", 1)[-1], b))
        if url.endswith("/balance"):
            assert b["fc_sec_trd_nat_cd"] == "200" and b["cur_cd"] == "USD"
            return FakeResp(200, {"Output_0": {"fc_dca": 1200.5},
                                  "Output_1": [{"iem_cd": "QQQ", "cns_bse_bnc_qty": 10, "sll_pbl_qty1": 10,
                                                "fc_avg_phs_pr": 600.0, "fc_sec_end_pr": 650.0, "iem_nm": "QQQ"}]})
        if url.endswith("/buyableAmount"):
            return FakeResp(200, {"Output_0": {"orr_pbl_amt": 1100.0, "sll_pbl_qty": 10}})
        if url.endswith("/buy"):
            assert b["ahi_nmn_pr_tp_cd"] == "00" and b["fc_orr_uit_pr"] == 651.07 and b["wtm_cur_knd_cd"] == "1"
            return FakeResp(200, {"Output_0": {"orr_no": 777}, "rsp_msg": "주문완료"})
        if url.endswith("/sell"):
            assert b["ahi_nmn_pr_tp_cd"] == "03" and "fc_orr_uit_pr" not in b
            return FakeResp(200, {"Output_0": {"orr_no": 778}})
        if url.endswith("/cancel"):
            assert b["org_orr_no"] == 777 and b["all_pat_dit_cd"] == "1"
            return FakeResp(200, {"Output_0": {"orr_no": 779}})
        if url.endswith("/unexecuted"):
            return FakeResp(200, {"Output_0": [
                {"orr_no": 777, "oss_sby_dit_cd": "2", "iem_cd": "QQQ", "orr_qty": 3, "cns_qty": 1, "cns_pr": 651.0,
                 "ny_cns_orr_qty": 2, "cor_can_dit_nm": "정상"},
                {"orr_no": 778, "oss_sby_dit_cd": "1", "iem_cd": "TQQQ", "orr_qty": 5, "cns_qty": 0,
                 "ny_cns_orr_qty": 0, "oss_ato_orr_sts_cd": "2", "rjt_rsn_cts": "레버리지 ETP 교육 미이수"},
                {"orr_no": 779, "org_orr_no": 777, "iem_cd": "QQQ", "orr_qty": 2, "cor_can_dit_nm": "취소"}]})
        raise AssertionError(url)

    b, _ = _us(h, tmp_path, env="mock")
    a = b.get_account()
    assert a.cash == 1100.0 and a.positions["QQQ"].qty == 10 and a.total_equity == pytest.approx(1100 + 6500)
    r = b.place_order(OrderRequest("QQQ", Side.BUY, 3, OrderType.LIMIT, 651.0700001))
    assert r.accepted and r.order_id == "777"
    assert b.place_order(OrderRequest("TQQQ", Side.SELL, 5, OrderType.MARKET)).order_id == "778"
    assert b.cancel_order("777", "QQQ").accepted
    st = {s.order_id: s for s in b.get_orders(date(2026, 11, 24))}
    assert st["777"].status == OrderStatus.PARTIAL and st["777"].is_open
    assert st["778"].status == OrderStatus.REJECTED and not st["778"].is_open
    assert "779" not in st                                          # 취소주문 자체 행은 제외
    assert b.sellable_quantity("QQQ") == 10
    days = {x[1]["orr_dt"] for x in seen if x[0] == "unexecuted"}
    assert days == {"20261124", "20261125"}                          # 미국·한국 날짜 모두 조회


# ====================================================================== 달력·호가
def test_nyse_calendar_and_cent_ticks():
    c = USCalendar()
    assert not c.is_trading_day(date(2026, 11, 26))                  # 추수감사절
    assert not c.is_trading_day(date(2026, 4, 3))                    # 성금요일(한국은 영업일)
    assert c.is_month_end(date(2026, 11, 30)) and not c.is_month_end(date(2026, 11, 27))
    assert c.is_month_end(date(2027, 12, 31))
    assert round_to_tick(101.231, "us", "up") == 101.24 and round_to_tick(101.239, "us", "down") == 101.23
    assert shift_ticks(101.23, 2, "us") == 101.25 and shift_ticks(101.23, -1, "us") == 101.22


# ====================================================================== 신호
def _us_market(n=252 * 7, crash_at=None, seed=1, end="2026-10-30"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    r = pd.DataFrame(rng.normal(0.0004, 0.011, (n, 4)), index=idx, columns=["QQQ", "TLT", "GLD", "DBC"])
    if crash_at is not None:   # 120일간 하락(−45%) 후 회복
        r.iloc[crash_at:crash_at + 120, 0] = -0.005
        r.iloc[crash_at + 120:crash_at + 400, 0] = 0.003
    return 100 * (1 + r).cumprod()


def test_us_leg_targets_shape_and_crash_boost():
    cfg = load_config(CFG)
    cal = USCalendar()
    px = _us_market()
    d: dict = {}
    w = us_leg_targets(px, cfg.triad_us, cal.is_month_end, d)
    assert set(w.index) == set(cfg.triad_us.tickers.values()) and w.sum() <= 1 + 1e-9 and (w >= 0).all()
    assert d["triad_us"]["vt_row"] <= d["triad_us"]["signal_date"]
    # QUAD: 고금리 대비형 미국 부분이 들어가고(합 = hr_weight), 금리 4% ≥ 3.5% 라 나스닥 2배·금 1배 상한
    assert d["triad_us"]["hr_weights"] and d["triad_us"]["high_rate_cap"]
    assert "GLDx2" not in d["triad_us"]["hr_weights"]                                # 고금리 상한: 금 최대 1배
    from dataclasses import replace
    w0 = us_leg_targets(px, replace(cfg.triad_us, hr_weight=0.0), cal.is_month_end)   # TRIAD 표준
    assert (w0 - w).abs().max() > 0.01
    # 폭락 → 골든크로스 회복 구간이면 VT 슬리브 노출이 3배로
    px2 = _us_market(crash_at=252 * 7 - 330)
    d2: dict = {}
    us_leg_targets(px2, cfg.triad_us, cal.is_month_end, d2)
    t = d2["triad_us"]
    assert t["qqq_dd_2y"] <= -0.30 and t["golden_cross"] and t["boost_on"] and t["vt_exposure"] == pytest.approx(3.0)
    with pytest.raises(RuntimeError):
        us_leg_targets(px.iloc[-300:], cfg.triad_us, cal.is_month_end)  # 이력 부족


# ====================================================================== 실행엔진 E2E
CFG = Path(__file__).parents[1] / "config/triad_us.yaml"


def _engine(tmp_path, monkeypatch, when):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    cfg = load_config(CFG)
    cfg.storage.db_path = str(tmp_path / "us.sqlite")
    cfg.storage.cache_dir = str(tmp_path / "cache")
    cfg.execution.poll_interval_sec = 0
    cfg.execution.fill_timeout_sec = 0
    base = _us_market(end="2026-11-27")
    lev = {"TQQQ": ("QQQ", 3), "TMF": ("TLT", 3), "UGL": ("GLD", 2)}
    px = {t: base[t] for t in base}
    for t, (b, L) in lev.items():
        px[t] = 30 * (1 + L * base[b].pct_change().fillna(0)).cumprod()
    px["SGOV"] = pd.Series(100.0, index=base.index)
    px = pd.DataFrame(px)

    def quote(sym):
        p = float(px[sym].iloc[-1])
        return Quote(sym, p, bid=round(p - 0.01, 2), ask=round(p + 0.01, 2), prev_close=p)

    def hist(sym, start, end=None):
        s = px[sym].loc[pd.Timestamp(start):pd.Timestamp(end) if end else None]
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1.0}).rename_axis("date")

    broker = PaperBroker(quote, initial_cash=100_000, state_file=str(tmp_path / "paper.json"), history_fn=hist)
    eng = USTradingEngine(cfg, broker, store=StateStore(cfg.storage.db_path), clock=lambda: when,
                          sleep=lambda s: None)
    return eng, broker


def test_us_engine_rebalances_in_et_session(tmp_path, monkeypatch):
    when = datetime(2026, 12, 1, 10, 31, tzinfo=ET)
    eng, broker = _engine(tmp_path, monkeypatch, when)
    rep = eng.run_once()
    assert rep.status == "done", rep.message
    acct = broker.get_account()
    eq = acct.cash + sum(p.market_value for p in acct.positions.values())
    for s, w in rep.targets.items():
        held = acct.positions[s].market_value if s in acct.positions else 0.0
        assert abs(held / eq - w) < 0.03, (s, w)
    assert all(o["type"] == "LIMIT" and round(o["price"], 2) == o["price"] for o in rep.orders)  # 센트 지정가
    assert eng.run_once().status == "skipped"                         # 같은 달·신호 불변 → 재실행 없음


def test_us_engine_skips_outside_session_and_holiday(tmp_path, monkeypatch):
    eng, _ = _engine(tmp_path, monkeypatch, datetime(2026, 12, 1, 9, 0, tzinfo=ET))
    assert "주문 가능 시간" in eng.run_once().message                  # 개장 전
    eng2, _ = _engine(tmp_path / "b", monkeypatch, datetime(2026, 11, 26, 11, 0, tzinfo=ET))
    assert eng2.run_once().status == "skipped"                        # 추수감사절
