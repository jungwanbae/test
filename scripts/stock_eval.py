"""종목 전략 비교 백테스트(실비용, 다음날 시가 체결) — IS 2001-2012 / OOS 2013-2026.

  python scripts/stock_eval.py --models composite linear lgbm mlp [--grid]
"""
import argparse
import itertools
import logging

import pandas as pd

from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.evaluate import composite_score, monthly_signal_dates, period_table
from autotrader.equity.panel import load_panel

logging.basicConfig(level=logging.WARNING)
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["composite", "linear", "lgbm", "mlp"])
ap.add_argument("--grid", action="store_true")
ap.add_argument("--out", default="reports/stock_eval.csv")
args = ap.parse_args()

p = load_panel("data/krx")
ds = pd.read_parquet("data/krx/dataset_cap.parquet")
sig = monthly_signal_dates(ds)
sig = sig[sig >= "2000-12-01"]
for m in args.models:
    if m == "composite":
        ds["composite"] = composite_score(ds)
    else:
        sc = pd.read_parquet(f"data/krx/scores_{m}.parquet")
        ds[m] = sc["score"].values
curves = {}
mk = p.market_return("KOSPI")
curves["KOSPI(시총가중)"] = (1 + mk.loc["2001":]).cumprod()
grid = (list(itertools.product([20, 30], ["equal", "inv_vol"], [None, "ma200", "dual"]))
        if args.grid else [(30, "equal", None), (30, "equal", "dual")])
for m in args.models:
    for n, wt, rg in grid:
        rules = PortfolioRules(top_n=n, weighting=wt, regime=rg)
        tg = make_targets(ds[ds["date"].isin(sig)], m, rules)
        res = simulate(p, tg, StockCosts(), start="2001-01-02")
        curves[f"{m}|n{n}|{wt}|{rg}"] = res.equity
        print(m, n, wt, rg, "done", flush=True)
tab = period_table(curves)
pd.set_option("display.width", 250)
print(tab.astype(float).round(3).to_string())
tab.to_csv(args.out)
