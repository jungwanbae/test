"""질문: 고금리 대비형을 더 반영하면? 공격형+안정형을 섞으면? — 같은 엔진·같은 계산법으로 비교.

  python scripts/hr_mix.py
산출: reports/hr_mix.csv, reports/hr_mix_yearly.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from tournament_analysis import blend  # noqa: E402
from triad_variants import msap_curve, to_us_calendar  # noqa: E402

from autotrader.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from autotrader.equity.combine import daily_etf_returns, etf_daily_from_weights  # noqa: E402
from autotrader.equity.evaluate import monthly_signal_dates  # noqa: E402
from autotrader.equity.panel import load_panel  # noqa: E402
from autotrader.research.data import build_returns  # noqa: E402
from autotrader.research.tournament import (Ctx, after_tax, implement, lev_mix, recovery_boost, ret_n,  # noqa: E402
                                            simulate, stats)
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity  # noqa: E402

warnings.filterwarnings("ignore")
RATE_CAP = 0.035   # 고금리 대비형 규칙: 미국 T-bill ≥ 3.5% 이면 레버리지 상한 축소


def vt_ratecap(c: Ctx, L=3.0):
    """확인형 역발상 VT + 고금리 상한: 금리 ≥ 3.5% 인 신호일에는 나스닥 노출 최대 2배."""
    rows = recovery_boost("QQQ", 0.25, -0.30, L, 252)(c).fillna(0.0)
    out = {}
    for d, r in rows.iterrows():
        E = r.get("QQQ", 0) + 3 * r.get("QQQx3", 0)
        if c.rf_us.loc[d] >= RATE_CAP:
            E = min(E, 2.0)
        out[d] = lev_mix("QQQ", E, "CASH_USD", 3)
    return pd.DataFrame(out).T


def lrp_ratecap(c: Ctx, target=0.18):
    """레버리지 RP + 고금리 상한: 금리 ≥ 3.5% 이면 상한 나스닥 2·장기채 2·금 1·원자재 1배."""
    assets = ["QQQ", "TLT", "GLD", "DBC"]
    Ls = {"QQQ": 3, "TLT": 3, "GLD": 2, "DBC": 1}
    r = c.P[assets].pct_change()
    rows = {}
    for d in c.me:
        win = r.loc[:d].iloc[-126:]
        if len(win) < 126 or win.isna().any().any():
            continue
        vol = win.iloc[-63:].std() * np.sqrt(252)
        cov = np.outer(vol, vol) * win.corr().values
        w = (1 / vol) / (1 / vol).sum()
        w = w.where(ret_n(c.P[assets], d, 252) > c.rf_us.loc[d], 0.0)
        pv = np.sqrt(w.values @ cov @ w.values)
        if pv <= 0:
            rows[d] = {"CASH_USD": 1.0}
            continue
        caps = {"QQQ": 2, "TLT": 2, "GLD": 1, "DBC": 1} if c.rf_us.loc[d] >= RATE_CAP else Ls
        E = (w * target / pv).clip(upper=pd.Series(caps))
        rows[d] = implement(E.to_dict(), Ls)
    return pd.DataFrame(rows).T


def main():
    p = load_panel("data/krx")
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    sig = monthly_signal_dates(ds)
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    rp = run_backtest(px, TrendRiskParity(RPParams(["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"], "CASH_KRW",
                                                   target_vol=0.10, trend_months=None)),
                      BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10))
    etf = etf_daily_from_weights(daily_etf_returns(p, px), rp.weights)
    R, aux = build_returns()
    c = Ctx(R=R, P=aux["P_sig"], vix=aux["vix"], rf_us=aux["rf_us"])
    tour = pd.read_parquet("data/krx/tournament_equity.parquet")
    sim = lambda tg: simulate(R, tg, cost=0.001, start="2001-01-02")[0]  # noqa: E731
    C = pd.DataFrame({
        "K": to_us_calendar(msap_curve(p, ds, sig, etf, "lgbm_ens3"), R.index),
        "V3": tour["역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%"],
        "V2": sim(recovery_boost("QQQ", 0.25, -0.30, 2.0, 252)(c)),
        "V25": sim(recovery_boost("QQQ", 0.25, -0.30, 2.5, 252)(c)),
        "Vcap": sim(vt_ratecap(c)),
        "R": tour["레버리지 리스크패리티 18%+추세"],
        "Rcap": sim(lrp_ratecap(c)),
        "HR": tour["[기존] 고금리 대비형(근사)"],
    })
    t = 1 / 3
    specs = {   # 이름: (가중치, 해외비중)
        "TRIAD 표준": ({"K": t, "V3": t, "R": t}, 2 / 3),
        "[기존] 고금리 대비형 단독(근사)": ({"HR": 1}, 0.8),
        "A. TRIAD + 고금리 대비형 4등분": ({"K": .25, "V3": .25, "R": .25, "HR": .25}, .7),
        "B. 역발상 자리를 고금리 대비형으로 교체": ({"K": t, "HR": t, "R": t}, 0.6),
        "C. 역발상·고금리 대비형 반반(나스닥 칸 공유)": ({"K": t, "V3": t / 2, "HR": t / 2, "R": t}, 0.63),
        "D. TRIAD 50 + 고금리 대비형 50": ({"K": t / 2, "V3": t / 2, "R": t / 2, "HR": .5}, 0.73),
        "E. TRIAD + 고금리 규칙(금리≥3.5%면 레버리지 상한 축소)": ({"K": t, "Vcap": t, "Rcap": t}, 2 / 3),
        "공격형 (한국¼·역발상½·RP¼, 부스트 3배)": ({"K": .25, "V3": .5, "R": .25}, .75),
        "안정형 (⅓씩, 부스트 2배)": ({"K": t, "V2": t, "R": t}, 2 / 3),
        "F. 공격형+안정형 50:50 (정확한 혼합)": ({"K": (.25 + t) / 2, "V3": .25, "V2": t / 2, "R": (.25 + t) / 2}, .71),
        "G. 혼합 근사 단일 설정 (한국 0.3·역발상 0.4·RP 0.3, 부스트 2.5배)": ({"K": .3, "V25": .4, "R": .3}, .7),
    }
    rows, curves = {}, {}
    for lab, (w, osh) in specs.items():
        eq = blend(C, w)
        curves[lab] = eq
        f, a, o, r5 = stats(eq, "2001-02-01"), stats(eq, "2005-01-03"), stats(eq, "2016-01-01"), stats(eq, "2021-11-26")
        hi = (aux["rf_us"] >= RATE_CAP).reindex(eq.index).fillna(False)
        rr = eq.pct_change()
        rows[lab] = {"2001~ CAGR": f["CAGR"], "2005~ CAGR": a["CAGR"],
                     "세후(근사, 2005~)": stats(after_tax(eq.loc["2005-01-03":], osh))["CAGR"],
                     "Sharpe(2005~)": a["Sharpe"], "MDD(2005~)": a["MDD"], "최악의 해": a["WorstY"],
                     "2016~": o["CAGR"], "최근 5년": r5["CAGR"],
                     "고금리기 연환산": (1 + rr[hi & (rr.index >= "2005-01-03")]).prod() ** (252 / max(hi.sum(), 1)) - 1,
                     "변동성": a["Vol"]}
    T = pd.DataFrame(rows).T
    T.to_csv("reports/hr_mix.csv")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    print("고금리(T-bill≥3.5%) 거래일 비중(2005~):", float((aux["rf_us"] >= RATE_CAP).loc["2005-01-03":].mean()))
    corr = C.loc["2005-01-03":].resample("W-FRI").last().pct_change().dropna().corr().round(2)
    print(corr.to_string())
    yr = pd.DataFrame(curves).loc["2005-12-30":].resample("YE").last().pct_change().dropna(how="all")
    yr.index = yr.index.year
    yr.to_csv("reports/hr_mix_yearly.csv")


if __name__ == "__main__":
    main()
