"""MSAP 개선 후보 평가 — 피드백 대응(최근 성과 둔화·일별 낙폭) + 미너비니 손절 규율.

후보: 종목 모델(확장 윈도 / 최근 8년 / 최근 5년 롤링 윈도) × 손절(없음 / 미너비니식 −15%·−12%, 종가 기준·장중)
선택 규칙(사전에 고정): 설계 구간(2001~2012) MSAP Sharpe 최대. 검증·최근 10년·일별 MDD 는 참고로 함께 보고.
  python scripts/msap_upgrade.py
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.combine import daily_etf_returns, daily_mix, etf_daily_from_weights, monthly_mix
from autotrader.equity.evaluate import monthly_signal_dates, stats
from autotrader.equity.panel import load_panel
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity

warnings.filterwarnings("ignore")
END = "2026-09-30"
W = 0.70


def main() -> None:
    p = load_panel("data/krx")
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    models = [m for m in ("lgbm", "lgbm_w8", "lgbm_w5") if Path(f"data/krx/scores_{m}.parquet").exists()]
    for m in models:
        ds[m] = pd.read_parquet(f"data/krx/scores_{m}.parquet")["score"].values
    sig = monthly_signal_dates(ds)
    sub = ds[ds.date.isin(sig[sig >= "2000-12-01"])]
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    risk = ["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"]
    rp = run_backtest(px, TrendRiskParity(RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)),
                      BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10))
    etf_d = etf_daily_from_weights(daily_etf_returns(p, px), rp.weights)
    rows = {}
    for m, (stop, mode) in itertools.product(models, [(None, "-"), (0.15, "close"), (0.12, "close"), (0.15, "intraday")]):
        tg = make_targets(sub, m, PortfolioRules(top_n=15, buffer=2.5))
        st = simulate(p, tg, StockCosts(), start="2001-01-02", stop_loss=stop,
                      stop_mode=mode if stop else "intraday").equity
        mm = monthly_mix(st, rp.equity, W, END)
        md = daily_mix(st, etf_d, W, end=END)
        a, b, c, r10 = stats(mm, "2001", "2012"), stats(mm, "2013", END), stats(mm, "2001", END), stats(mm, "2016-10", END)
        name = f"{m} | 손절 {'없음' if stop is None else f'-{int(stop * 100)}% ({mode})'}"
        rows[name] = {"설계 CAGR": a["CAGR"], "설계 Sharpe": a["Sharpe"], "검증 CAGR": b["CAGR"], "검증 Sharpe": b["Sharpe"],
                      "최근10년 CAGR": r10["CAGR"], "최근10년 Sharpe": r10["Sharpe"], "전체 CAGR": c["CAGR"],
                      "전체 Sharpe": c["Sharpe"], "일별 MDD": stats(md)["MDD"], "검증 일별 MDD": stats(md, "2013")["MDD"]}
        print(name, flush=True)
    t = pd.DataFrame(rows).T
    t.to_csv("reports/msap_upgrade.csv")
    best = t["설계 Sharpe"].astype(float).idxmax()
    fmt = {c: (lambda v: f"{v * 100:.1f}%") if ("CAGR" in c or "MDD" in c) else (lambda v: f"{v:.2f}") for c in t.columns}
    lines = ["# MSAP 개선 후보 평가 (주식 70% + ETF 30%, 실비용, 2001-02 ~ " + END + ")", "",
             "선택 규칙(사전 고정): 설계 구간(2001~2012) Sharpe 최대 → **" + best + "**", "",
             "| 후보 | " + " | ".join(t.columns) + " |", "|---|" + "---:|" * len(t.columns)]
    for k, r in t.iterrows():
        lines.append(f"| {'**' + k + '**' if k == best else k} | " + " | ".join(fmt[c](float(r[c])) for c in t.columns) + " |")
    Path("reports/msap_upgrade.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
