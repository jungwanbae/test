"""한국 종목 엔진에 '시대 적응' 학습(최근 N년·최근 가중·유사 국면 가중)을 적용하면? — 같은 백테스트로 비교.

  python scripts/stock_adaptive_eval.py     (먼저 stock_scores.py lgbm lgbm@5 lgbm@8 lgbm~decay5 lgbm~decay3 lgbm~similar)
산출: reports/stock_adaptive.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets
from autotrader.equity.backtest import simulate as ssim
from autotrader.equity.combine import daily_etf_returns, daily_mix, etf_daily_from_weights
from autotrader.equity.evaluate import monthly_signal_dates
from autotrader.equity.panel import load_panel
from autotrader.research.tournament import stats
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity

warnings.filterwarnings("ignore")
# 모두 같은 조건(전체 학습 날짜). 시드 = 같은 설정을 다른 난수로 다시 학습한 것(모델 잡음 측정)
VARIANTS = {"lgbm": "전체 과거 · 시드7(현행 백테스트)", "lgbm_seed8": "전체 과거 · 시드8", "lgbm_seed9": "전체 과거 · 시드9",
            "lgbm_ens3": "전체 과거 · 시드 3개 앙상블(새 실전 방식)", "lgbm_w8": "최근 8년 · 시드7", "lgbm_w5": "최근 5년 · 시드7",
            "lgbm_decay5_seed7": "최근 가중(반감 5년) · 시드7", "lgbm_decay5_seed8": "최근 가중(반감 5년) · 시드8",
            "lgbm_similar_seed7": "유사 국면 가중 · 시드7", "lgbm_similar_seed8": "유사 국면 가중 · 시드8"}


def main():
    variants, out = VARIANTS, "reports/stock_adaptive.csv"
    if len(sys.argv) > 1:   # 예: python scripts/stock_adaptive_eval.py seeds lgbm_s4 lgbm_s4_seed8 …
        out = f"reports/stock_{sys.argv[1]}.csv"
        variants = {t: t for t in sys.argv[2:]}
    p = load_panel("data/krx")
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    sig = monthly_signal_dates(ds)
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    rp = run_backtest(px, TrendRiskParity(RPParams(["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"], "CASH_KRW",
                                                   target_vol=0.10, trend_months=None)),
                      BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10))
    etf = etf_daily_from_weights(daily_etf_returns(p, px), rp.weights)
    import sys as _s
    _s.path.insert(0, "scripts")
    from tournament_analysis import blend
    tour = pd.read_parquet("data/krx/tournament_equity.parquet")
    V, L = tour["역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%"], tour["레버리지 리스크패리티 18%+추세"]
    rows = {}
    for tag, lab in variants.items():
        f = Path(f"data/krx/scores_{tag}.parquet")
        if not f.exists():
            print("없음:", f)
            continue
        d = ds.copy()
        d["s"] = pd.read_parquet(f)["score"].values
        tg = make_targets(d[d.date.isin(sig[sig >= "2000-12-01"])], "s", PortfolioRules(top_n=15, buffer=2.5))
        st = ssim(p, tg, StockCosts(), start="2001-01-02", stop_loss=0.15, stop_mode="close").equity
        ms = daily_mix(st, etf, 0.7)
        tri = blend(pd.DataFrame({"K": ms, "V": V, "R": L}), {"K": 1 / 3, "V": 1 / 3, "R": 1 / 3})
        for nm, eq in ((f"종목 알파 · {lab}", st), (f"MSAP-M · {lab}", ms), (f"TRIAD · {lab}", tri)):
            a, i, o, r5 = stats(eq), stats(eq, None, "2015-12-31"), stats(eq, "2016-01-01"), stats(eq, "2021-11-25")
            rows[nm] = {"CAGR": a["CAGR"], "Sharpe": a["Sharpe"], "MDD": a["MDD"], "최악의 해": a["WorstY"],
                        "2001~15": i["CAGR"], "2016~": o["CAGR"], "2016~ Sharpe": o["Sharpe"], "최근 5년": r5["CAGR"]}
        print(lab, {k: round(v, 3) for k, v in rows[f"종목 알파 · {lab}"].items()}, flush=True)
    R = pd.DataFrame(rows).T
    R.to_csv(out)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
