"""AMR 파라미터 민감도(과최적화 점검).

기본값 하나만 좋은 게 아니라 '주변 파라미터 전반에서' 우위가 유지되는지 확인한다.
  python scripts/sensitivity.py --out reports/sensitivity.md
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.data.research import build_us_monthly
from autotrader.strategy.amr import AMRParams, AMRStrategy
from autotrader.strategy.benchmarks import FixedWeights

GRID = {
    "lookbacks": [(12,), (6, 12), (3, 6, 12), (1, 3, 6, 12)],
    "top_k": [1, 2, 3],
    "target_vol": [0.07, 0.10, 0.14, None],
    "weighting": ["inverse_vol", "erc", "equal"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/sensitivity.md")
    args = ap.parse_args()
    px, _ = build_us_monthly("data/research")
    px = px.loc["1972-01":]
    cfg = BacktestConfig(start="1973-01-01", execution_lag=1)
    risk = ["SPX", "GOLD", "UST10"]
    bench = {k: run_backtest(px, FixedWeights(w, k), cfg).stats for k, w in
             {"60/40": {"SPX": 0.6, "UST10": 0.4}, "S&P500": {"SPX": 1.0},
              "1/3균등": {"SPX": 1 / 3, "GOLD": 1 / 3, "UST10": 1 / 3}}.items()}
    rows = []
    for lb, k, tv, wt in itertools.product(*GRID.values()):
        p = AMRParams(risk_assets=risk, lookback_months=list(lb), top_k=k, target_vol=tv, weighting=wt,
                      max_weight=max(0.4, 1.0 / k))
        st = run_backtest(px, AMRStrategy(p), cfg).stats
        rows.append({"lookbacks": "/".join(map(str, lb)), "top_k": k, "target_vol": tv or "off", "weighting": wt,
                     "cagr": st["cagr"], "vol": st["vol"], "sharpe": st["sharpe"], "mdd": st["max_drawdown"],
                     "calmar": st["calmar"]})
    df = pd.DataFrame(rows)
    b6040 = bench["60/40"]
    lines = ["# AMR 파라미터 민감도 (미국 장기 실데이터 1973~, 월간, 1개월 체결지연, 현금수익 0%)", "",
             f"- 조합 수: {len(df)} (lookback {len(GRID['lookbacks'])} × top_k {len(GRID['top_k'])} × "
             f"target_vol {len(GRID['target_vol'])} × weighting {len(GRID['weighting'])})",
             f"- 60/40 대비 Sharpe 우위 비율: **{(df.sharpe > b6040['sharpe']).mean():.0%}**, "
             f"MDD 우위(더 얕음) 비율: **{(df.mdd > b6040['max_drawdown']).mean():.0%}**", "",
             "| 지표 | 최소 | 25% | 중앙값 | 75% | 최대 | 60/40 | S&P500 | 1/3균등 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for col, key, pct in (("cagr", "cagr", True), ("vol", "vol", True), ("sharpe", "sharpe", False),
                          ("mdd", "max_drawdown", True), ("calmar", "calmar", False)):
        q = df[col].quantile([0, .25, .5, .75, 1]).values
        f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.2f}")
        lines.append(f"| {col} | " + " | ".join(f(v) for v in q) + " | " +
                     " | ".join(f(bench[b][key]) for b in ("60/40", "S&P500", "1/3균등")) + " |")
    lines += ["", "## 요인별 중앙값 Sharpe", ""]
    for g in ("lookbacks", "top_k", "target_vol", "weighting"):
        m = df.groupby(g)[["sharpe", "cagr", "mdd"]].median()
        lines.append(f"**{g}**: " + ", ".join(f"{i} → Sharpe {r.sharpe:.2f} / CAGR {r.cagr * 100:.1f}% / "
                                             f"MDD {r.mdd * 100:.1f}%" for i, r in m.iterrows()))
        lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
