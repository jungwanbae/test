"""SEPA-K(미너비니 정량화) 테스트 — 트렌드 템플릿·손절·포지션 크기."""
import numpy as np
import pandas as pd

from autotrader.equity.backtest import StockCosts
from autotrader.equity.panel import Panel
from autotrader.equity.sepa import SEPAParams, compute_indicators, simulate_sepa


def _panel(n_days=420, n=30, seed=0, crash_day=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    codes = [f"{i:05d}0" for i in range(n)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (n_days, n)), index=idx, columns=codes)
    ret.iloc[:, 0] = 0.003 + rng.normal(0, 0.004, n_days)      # 꾸준한 상승(2단계)
    ret.iloc[:, 1] = -0.003 + rng.normal(0, 0.004, n_days)     # 꾸준한 하락(4단계)
    if crash_day is not None:
        ret.iloc[crash_day, 0] = -0.20
    close = 10_000 * (1 + ret).cumprod()
    one = pd.DataFrame(1.0, index=idx, columns=codes)
    lo = one * 0.99
    if crash_day is not None:
        lo.iloc[crash_day, 0] = 0.98
    return Panel(ret=ret, gap=ret * 0.3, close=close, amount=one * 5e9, marcap=one * 1e12, flag=one * 0,
                 names=pd.Series(codes, index=codes), market=pd.Series("KOSPI", index=codes), hi=one * 1.01, lo=lo)


def test_trend_template_separates_stage2_from_stage4():
    p = _panel()
    ind = compute_indicators(p, SEPAParams(min_adv=1e8, min_marcap=1e9))
    t = len(ind.dates) - 1
    assert ind.tt[t, 0] and ind.stage2[t, 0]          # 상승추세 종목 통과
    assert not ind.tt[t, 1] and not ind.stage2[t, 1]  # 하락추세 종목 탈락
    assert ind.rs[t, 0] > 95 and ind.rs[t, 1] < 5


def test_stop_limits_loss_and_sizing_respects_risk_budget():
    p = _panel(crash_day=360)
    prm = SEPAParams(min_adv=1e8, min_marcap=1e9, market_filter="none", max_rs=100, min_rs=0)
    ind = compute_indicators(p, prm)
    # 강제 셋업: 300일에 0번 종목 매수 신호, 피벗 = 당일 종가
    ind.setup[:] = False
    ind.setup[300, 0] = True
    ind.pivot[300, 0] = ind.P[300, 0]
    res = simulate_sepa(ind, prm, StockCosts(commission=0, slip_base=0, sell_tax_override=0), capital=1e8,
                        start=str(ind.dates[290].date()))
    tr = res.trades
    assert len(tr) == 1
    t0 = tr.iloc[0]
    assert t0.value0 / 1e8 <= prm.max_position + 1e-9                    # 종목당 최대 비중
    assert t0.value0 / 1e8 * t0.risk_pct <= prm.risk_per_trade * 1.05   # 1회 위험 ≈ 자산의 1.25%
    if t0.reason in ("손절", "이익보호 손절"):
        assert t0.ret >= -0.25                                           # 갭하락이어도 손실이 제한됨
    assert res.equity.min() > 0.9                                        # 계좌 전체 손실은 소폭
