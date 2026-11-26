"""종목 알파 슬리브(패널·팩터·시뮬레이터·선정) 테스트 — marcap 형식의 합성 데이터 사용."""
import numpy as np
import pandas as pd
import pytest

from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.evaluate import deflated_sharpe
from autotrader.equity.factors import build_dataset, sample_dates
from autotrader.equity.live import StockSleeveConfig, select
from autotrader.equity.panel import build_panel, is_common_stock
from autotrader.strategy.base import StrategyContext
from autotrader.strategy.multi_sleeve import combine
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity
from autotrader.strategy.signals import MONTHLY


def _write_marcap(tmp_path, n_days=420, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    for k in range(n_stocks):
        code = f"{k + 1:05d}0"
        px = 10_000.0
        for i, d in enumerate(dates):
            r = rng.normal(0.0004 * (k % 5 - 2), 0.02)
            new = max(round(px * (1 + r)), 1000)
            if k == 0 and i == 200:  # 1:2 액면분할: 종가 절반, 기준가도 절반 → 수익률은 정상
                base = px / 2
                new = round(base * 1.01)
            else:
                base = px
            rows.append({"Date": d, "Code": code.lstrip("0") if k == 1 else code, "Name": f"S{k}", "Market": "KOSPI",
                         "Dept": None, "Open": base, "High": max(base, new) * 1.01, "Low": min(base, new) * 0.99,
                         "Close": new, "Changes": new - base,
                         "Amount": 5e8 * (1 + k), "Marcap": new * 1e7 * (1 + k), "Volume": 1000.0})
            px = new
    # 우선주·스팩(제외 대상)
    rows.append({**rows[0], "Code": "000015", "Name": "S0우"})
    rows.append({**rows[0], "Code": "000990", "Name": "하나스팩1호"})
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "marcap-2020.parquet")
    return dates


def test_common_stock_filter():
    assert is_common_stock("005930", "삼성전자") and is_common_stock("0126Z0", "삼성에피스홀딩스")
    assert not is_common_stock("005935", "삼성전자우") and not is_common_stock("00680K", "미래에셋증권2우B")
    assert not is_common_stock("000990", "하나스팩1호") and not is_common_stock("330590", "롯데리츠")


def test_panel_split_adjusted_and_zero_padded(tmp_path):
    _write_marcap(tmp_path)
    p = build_panel(tmp_path)
    assert "000020" in p.ret.columns          # '20' → 000020 으로 복원
    assert "000015" not in p.ret.columns and "000990" not in p.ret.columns
    split_day = p.dates[200]
    assert p.ret.loc[split_day, "000010"] == pytest.approx(0.01, abs=1e-3)  # 분할일에도 -50% 가 아님(정수 호가 반올림 오차)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("mc")
    _write_marcap(d)
    p = build_panel(d)
    dates = sample_dates(p.dates, "W")
    dates = dates[dates >= p.dates[260]]
    ds = build_dataset(p, dates, universe_size=30, min_amount=1e8, min_age=252, chunk=7)
    return p, ds


def test_dataset_no_lookahead(small, tmp_path):
    p, ds = small
    assert ds.groupby("date")["code"].count().max() <= 30
    # 미래 가격을 바꿔도 과거 피처는 동일해야 한다
    d0 = ds["date"].min()
    p2 = type(p)(**{f: getattr(p, f).copy() for f in ("ret", "gap", "close", "amount", "marcap", "flag")},
                 names=p.names, market=p.market)
    p2.ret.loc[p2.ret.index > d0 + pd.Timedelta(days=40)] = 0.05
    ds2 = build_dataset(p2, pd.DatetimeIndex([d0]), universe_size=30, min_amount=1e8, min_age=252, chunk=7)
    a = ds[ds.date == d0].set_index("code").sort_index()
    b = ds2.set_index("code").sort_index()
    pd.testing.assert_frame_equal(a[["r_1m", "vol_60", "hi_52", "size"]], b[["r_1m", "vol_60", "hi_52", "size"]])


def test_simulate_costs_and_delisting(small):
    p, ds = small
    ds = ds.copy()
    ds["score"] = ds["r_1m"]
    tg = make_targets(ds, "score", PortfolioRules(top_n=5))
    free = simulate(p, tg, StockCosts(commission=0, slippage_model="fixed", slippage=0, sell_tax_override=0))
    paid = simulate(p, tg, StockCosts())
    assert paid.equity.iloc[-1] < free.equity.iloc[-1] and paid.costs_paid > 0
    # 보유 종목이 상장폐지되면 haircut 만큼 손실
    code = next(iter(tg[min(tg)].index))
    p2 = type(p)(**{f: getattr(p, f).copy() for f in ("ret", "gap", "close", "amount", "marcap", "flag")},
                 names=p.names, market=p.market)
    cut = p2.dates.get_loc(min(tg)) + 5
    p2.ret.iloc[cut:, p2.ret.columns.get_loc(code)] = np.nan
    first = {min(tg): tg[min(tg)]}
    a = simulate(p, first, StockCosts(delist_haircut=0.0), end=str(p.dates[cut + 2].date()))
    b = simulate(p2, first, StockCosts(delist_haircut=1.0), end=str(p.dates[cut + 2].date()))
    assert b.equity.iloc[-1] < a.equity.iloc[-1]


def test_select_hysteresis():
    cfg = StockSleeveConfig(top_n=3, buffer=2.0)
    scores = pd.Series({c: s for c, s in zip("ABCDEFGH", [8, 7, 6, 5, 4, 3, 2, 1])})
    w = select(scores, held=["E"], cfg=cfg)          # E 는 5위 < 3×2=6 → 유지
    assert set(w.index) == {"A", "B", "E"} and w.sum() == pytest.approx(1)
    w2 = select(scores, held=["H"], cfg=cfg)         # H 는 8위 → 교체
    assert set(w2.index) == {"A", "B", "C"}


def test_combine_sleeves_sells_dropped_stocks():
    out = combine(pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"ETF": 0.6, "CASH": 0.4}), 0.7, ["A", "Z"])
    assert out["A"] == pytest.approx(0.35) and out["ETF"] == pytest.approx(0.18) and out["Z"] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_risk_parity_equal_risk_and_vol_target():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"LO": 100 * np.cumprod(1 + rng.normal(0.005, 0.01, 60)),
                       "HI": 100 * np.cumprod(1 + rng.normal(0.005, 0.05, 60)),
                       "CASH": 100 * np.cumprod(np.full(60, 1.002))}, index=idx)
    s = TrendRiskParity(RPParams(["LO", "HI"], "CASH", target_vol=None, trend_months=None))
    w = s.target_weights(px, StrategyContext(bars=MONTHLY))
    assert w["LO"] > 3 * w["HI"] and w.sum() == pytest.approx(1)
    s2 = TrendRiskParity(RPParams(["LO", "HI"], "CASH", target_vol=0.02, trend_months=None))
    w2 = s2.target_weights(px, StrategyContext(bars=MONTHLY))
    assert w2[["LO", "HI"]].sum() < w[["LO", "HI"]].sum() and w2["CASH"] > 0


def test_deflated_sharpe_penalises_many_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 300))
    few = deflated_sharpe(r, n_trials=2, trial_sharpes_std=0.3)
    many = deflated_sharpe(r, n_trials=500, trial_sharpes_std=0.3)
    assert 0 <= many["dsr"] < few["dsr"] <= 1


def test_stop_loss_limits_single_position_loss(small):
    p, ds = small
    code = p.ret.columns[3]
    d0 = p.dates[300]
    tg = {d0: pd.Series({code: 1.0})}
    p2 = type(p)(**{f: getattr(p, f).copy() for f in ("ret", "gap", "close", "amount", "marcap", "flag", "hi", "lo")},
                 names=p.names, market=p.market)
    crash = p2.dates.get_loc(d0) + 5
    p2.ret.iloc[crash, p2.ret.columns.get_loc(code)] = -0.25   # 하루 −25% 급락(장중 저가도 그 근처)
    p2.gap.iloc[crash, p2.ret.columns.get_loc(code)] = -0.02
    free = StockCosts(commission=0, slippage_model="fixed", slippage=0, sell_tax_override=0)
    no_stop = simulate(p2, tg, free, start=str(d0.date()), end=str(p2.dates[crash + 3].date()))
    with_stop = simulate(p2, tg, free, start=str(d0.date()), end=str(p2.dates[crash + 3].date()), stop_loss=0.08)
    assert with_stop.holdings["_stops"] == 1
    assert with_stop.equity.iloc[-1] > no_stop.equity.iloc[-1]
    assert with_stop.equity.iloc[-1] >= 0.90   # 손실은 대략 손절선(−8%) 수준에서 멈춤
