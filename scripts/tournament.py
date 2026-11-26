"""전략 토너먼트 실행 → reports/tournament.csv, data/krx/tournament_equity.parquet

  python scripts/tournament.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

from autotrader.equity.panel import load_panel
from autotrader.research.data import build_returns
from autotrader.research.tournament import Ctx, Spec, after_tax, fixed, registry, safe_contrarian_specs, simulate, stats

warnings.filterwarnings("ignore")
COMMON = ("2005-01-03", None)
IS_, OOS_ = ("2005-01-03", "2015-12-31"), ("2016-01-01", None)


def kr_series(p) -> dict[str, pd.Series]:
    """한국 종목 알파 슬리브(미너비니 −15% 손절)와 MSAP-M(주식 70 + ETF RP 30) 일별 자산곡선."""
    cache = Path("data/krx/kr_series.parquet")
    if cache.exists():
        df = pd.read_parquet(cache)
        return {c: df[c].dropna() for c in df}
    from autotrader.backtest.engine import BacktestConfig, run_backtest
    from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets
    from autotrader.equity.backtest import simulate as ssim
    from autotrader.equity.combine import daily_etf_returns, daily_mix, etf_daily_from_weights
    from autotrader.equity.evaluate import monthly_signal_dates
    from autotrader.strategy.risk_parity import RPParams, TrendRiskParity
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    ds["lgbm"] = pd.read_parquet("data/krx/scores_lgbm.parquet")["score"].values
    sig = monthly_signal_dates(ds)
    tg = make_targets(ds[ds.date.isin(sig[sig >= "2000-12-01"])], "lgbm", PortfolioRules(top_n=15, buffer=2.5))
    st = ssim(p, tg, StockCosts(), start="2001-01-02", stop_loss=0.15, stop_mode="close").equity
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    risk = ["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"]
    rp = run_backtest(px, TrendRiskParity(RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)),
                      BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10))
    msap = daily_mix(st, etf_daily_from_weights(daily_etf_returns(p, px), rp.weights), 0.7)
    out = {"KR_ALPHA": st, "MSAPM": msap}
    pd.DataFrame(out).to_parquet(cache)
    return out


def main() -> None:
    t0 = time.time()
    p = load_panel("data/krx")
    kr = kr_series(p)
    R, aux = build_returns(p, stock_sleeve=kr["KR_ALPHA"])
    m = kr["MSAPM"]
    R["MSAPM"] = m.reindex(R.index.union(m.index)).ffill().reindex(R.index).pct_change().fillna(0.0)
    c = Ctx(R=R, P=aux["P_sig"], vix=aux["vix"], rf_us=aux["rf_us"])
    specs = registry() + safe_contrarian_specs() + [
        Spec("한국 종목 알파(ML·미너비니 손절)", "E 종목선택(한국)", fixed({"KR_ALPHA": 1}), overseas_share=0, cost=0.0,
             start="2001-02-01"),
        Spec("MSAP-M(종목 70 + ETF RP 30)", "E 종목선택(한국)", fixed({"MSAPM": 1}), overseas_share=0, cost=0.0,
             start="2001-02-01")]
    rows, curves = {}, {}
    for sp in specs:
        t = time.time()
        tg = sp.build(c)
        eq, turn = simulate(R, tg, cost=sp.cost, start=sp.start)
        curves[sp.name] = eq
        full, com, a, b = stats(eq), stats(eq, *COMMON), stats(eq, *IS_), stats(eq, *OOS_)
        at = stats(after_tax(eq.loc[COMMON[0]:], sp.overseas_share))
        rows[sp.name] = {"분류": sp.category, "시작": str(eq.index[0].date()),
                         "CAGR(2005~)": com.get("CAGR"), "세후 CAGR(근사)": at.get("CAGR"), "Sharpe": com.get("Sharpe"),
                         "일별 MDD": com.get("MDD"), "Calmar": com.get("Calmar"), "최악의 해": com.get("WorstY"),
                         "5년 롤링 최저": com.get("Roll5Min"), "2005~15": a.get("CAGR"), "2016~": b.get("CAGR"),
                         "2016~ MDD": b.get("MDD"), "전체기간 CAGR": full.get("CAGR"), "전체기간 MDD": full.get("MDD"),
                         "연회전율": turn.sum() / max(full.get("Years", 1), 1)}
        print(f"{sp.name:<48} {com.get('CAGR', float('nan')) * 100:6.1f}%  MDD {com.get('MDD', float('nan')) * 100:6.1f}%  "
              f"({time.time() - t:.0f}s)", flush=True)
    df = pd.DataFrame(rows).T
    df.to_csv("reports/tournament.csv")
    pd.DataFrame(curves).to_parquet("data/krx/tournament_equity.parquet")
    R.to_parquet("data/krx/research_returns.parquet")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
