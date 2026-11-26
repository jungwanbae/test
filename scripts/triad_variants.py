"""추천 순위용 변형 비교 — 모두 같은 엔진·같은 계산법(시드 3개 앙상블 종목 모델, 미국 거래일 기준 정렬).

  python scripts/triad_variants.py
산출: reports/triad_variants.csv, reports/triad_variants_yearly.csv
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

sys.path.insert(0, "scripts")
from tournament_analysis import blend  # noqa: E402

from autotrader.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets  # noqa: E402
from autotrader.equity.backtest import simulate as ssim  # noqa: E402
from autotrader.equity.combine import daily_etf_returns, daily_mix, etf_daily_from_weights  # noqa: E402
from autotrader.equity.evaluate import monthly_signal_dates  # noqa: E402
from autotrader.equity.panel import load_panel  # noqa: E402
from autotrader.research.data import build_returns  # noqa: E402
from autotrader.research.tournament import Ctx, after_tax, recovery_boost, simulate, stats  # noqa: E402
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity  # noqa: E402

warnings.filterwarnings("ignore")


def msap_curve(p, ds, sig, etf, tag: str) -> pd.Series:
    d = ds.copy()
    d["s"] = pd.read_parquet(f"data/krx/scores_{tag}.parquet")["score"].values
    tg = make_targets(d[d.date.isin(sig[sig >= "2000-12-01"])], "s", PortfolioRules(top_n=15, buffer=2.5))
    st = ssim(p, tg, StockCosts(), start="2001-01-02", stop_loss=0.15, stop_mode="close").equity
    return daily_mix(st, etf, 0.7)


def to_us_calendar(eq: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """토너먼트와 같은 방식: 한국 곡선을 미국 거래일로 옮긴다(미국 휴장일 수익은 다음 미국 거래일에 합산)."""
    return eq.reindex(idx.union(eq.index)).ffill().reindex(idx).dropna()


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
    K7 = to_us_calendar(msap_curve(p, ds, sig, etf, "lgbm"), R.index)       # 기존(시드 7 단일 모델)
    K = to_us_calendar(msap_curve(p, ds, sig, etf, "lgbm_ens3"), R.index)    # 새 실전 방식(시드 3개 앙상블)
    V = tour["역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%"]
    V2, _ = simulate(R, recovery_boost("QQQ", 0.25, -0.30, 2.0, 252)(c), cost=0.001, start="2001-01-02")
    LRP = tour["레버리지 리스크패리티 18%+추세"]
    HR, A3, AQ = tour["[기존] 고금리 대비형(근사)"], tour["[기존] 3차 공격형(근사)"], tour["[기존] AQUILA-X(근사)"]
    C = pd.DataFrame({"K7": K7, "K": K, "V": V, "V2": V2, "R": LRP, "HR": HR, "A3": A3, "AQ": AQ})
    third = 1 / 3
    specs = {  # (가중치, 해외 비중)
        "TRIAD 표준 — 기존 계산(종목 모델 시드 1개)": ({"K7": third, "V": third, "R": third}, 2 / 3),
        "① TRIAD 표준 (한국⅓·확인형 역발상⅓·레버리지RP⅓)": ({"K": third, "V": third, "R": third}, 2 / 3),
        "② TRIAD 안정형 (부스트 2배)": ({"K": third, "V2": third, "R": third}, 2 / 3),
        "③ TRIAD 방어형 (한국½·역발상¼·RP¼)": ({"K": .5, "V": .25, "R": .25}, .5),
        "④ TRIAD 공격형 (한국¼·역발상½·RP¼)": ({"K": .25, "V": .5, "R": .25}, .75),
        "⑤ 국내 전용 MSAP-M": ({"K": 1.0}, 0.0),
        "참고: 미국 전용 (역발상½·RP½)": ({"V": .5, "R": .5}, 1.0),
        "참고: 기존 3전략 ⅓씩": ({"HR": third, "A3": third, "AQ": third}, 0.93),
    }
    rows, curves = {}, {}
    for lab, (w, osh) in specs.items():
        eq = blend(C, w)
        curves[lab] = eq
        f, a, i, o, r5 = (stats(eq, "2001-02-01"), stats(eq, "2005-01-03"), stats(eq, "2005-01-03", "2015-12-31"),
                          stats(eq, "2016-01-01"), stats(eq, "2021-11-25"))
        rows[lab] = {"2001~ CAGR": f["CAGR"], "2005~ CAGR": a["CAGR"], "세후(근사, 2005~)": stats(after_tax(eq.loc["2005-01-03":], osh))["CAGR"],
                     "Sharpe(2001~)": f["Sharpe"], "일별 MDD(2001~)": f["MDD"], "최악의 해": f["WorstY"],
                     "2005~15": i["CAGR"], "2016~": o["CAGR"], "최근 5년": r5["CAGR"], "5년 롤링 최저": f["Roll5Min"],
                     "변동성": f["Vol"]}
    T = pd.DataFrame(rows).T
    T.to_csv("reports/triad_variants.csv")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    yr = pd.DataFrame(curves).resample("YE").last().pct_change().dropna(how="all")
    yr.index = yr.index.year
    yr.to_csv("reports/triad_variants_yearly.csv")
    pd.DataFrame(curves).to_parquet("data/krx/triad_variants_equity.parquet")


if __name__ == "__main__":
    main()
