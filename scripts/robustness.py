"""MSAP 강건성 점검(외부 피드백 대응) → reports/robustness.md

1) 구간별·롤링 성과(3·5년 롤링, 5년 블록, 최근 5·10년)
2) 일별 기준 MDD(월말 MDD 의 함정 점검): 주식 슬리브 일별 + ETF 슬리브 일별(가용 자산은 실제 일별,
   금·미국채는 월수익률의 일별 기하분할 — 해당 자산의 월중 변동은 과소추정됨을 명시)
3) 환율·금리 국면 의존도: 원화 강세/약세 월, 원화 강세 추세(12개월), 주식·채권 동반하락 월, 2022년,
   슬리브 간 36개월 롤링 상관
4) 대안 점검: 채권·금에만 추세필터를 둔 ETF 슬리브(동반하락기 방어 개선 여부)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.data.krw_assets import kospi200_daily, usdkrw_daily
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.evaluate import monthly_signal_dates, stats
from autotrader.equity.panel import load_panel
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity

warnings.filterwarnings("ignore")
STOCK_W = 0.70
END = "2026-09-30"


def pct(x):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def daily_etf_returns(p, px_m: pd.DataFrame) -> pd.DataFrame:
    """ETF 슬리브 자산의 일별 원화 수익률. 실제 일별: KOSPI200, S&P500(2022년까지)×환율. 나머지는 월수익 기하분할."""
    days = p.dates[(p.dates >= "1996-01-01")]
    out = pd.DataFrame(index=days)
    out["KOSPI200"] = kospi200_daily(p).reindex(days).fillna(0.0)
    fx = usdkrw_daily().reindex(days).ffill()
    try:
        from skfolio.datasets import load_sp500_index
        spd = load_sp500_index()["SP500"].astype(float).reindex(days).ffill()
        spx_d = (spd * fx).pct_change()
    except Exception:
        spx_d = pd.Series(np.nan, index=days)
    mret = px_m.pct_change()
    for col in ("SPX_KRW", "GOLD_KRW", "UST10_KRW", "CASH_KRW"):
        m = mret[col]
        month = days.to_period("M")
        n = pd.Series(1, index=days).groupby(month).transform("count")
        mr = pd.Series(m.values, index=m.index.to_period("M")).reindex(month).values
        geo = (1 + pd.Series(mr, index=days)) ** (1 / n) - 1
        out[col] = geo
        if col == "SPX_KRW":
            real = spx_d.dropna()
            out.loc[real.index, col] = real   # 일별 실데이터가 있는 구간은 대체
    return out.fillna(0.0)


def main() -> None:
    p = load_panel("data/krx")
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    ds["lgbm"] = pd.read_parquet("data/krx/scores_lgbm.parquet")["score"].values
    sig = monthly_signal_dates(ds)
    sig = sig[sig >= "2000-12-01"]
    stock_d = simulate(p, make_targets(ds[ds.date.isin(sig)], "lgbm", PortfolioRules(top_n=15, buffer=2.5)),
                       StockCosts(), start="2001-01-02").equity
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    risk = ["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"]
    cfg = BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10)
    rp_res = run_backtest(px, TrendRiskParity(RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)), cfg)
    rp_m = rp_res.equity
    # ---- 월간 MSAP(기준)
    ra = stock_d.resample("ME").last().pct_change()
    r = pd.concat([ra, rp_m.pct_change()], axis=1, sort=True).loc["2001-02":END].dropna()
    msap_m = (1 + STOCK_W * r.iloc[:, 0] + (1 - STOCK_W) * r.iloc[:, 1]).cumprod()
    # ---- 일별 ETF 슬리브(월말 비중 고정) + 일별 MSAP(월초 70/30 리밸런싱, 월중 드리프트 반영)
    dr = daily_etf_returns(p, px)
    w = rp_res.weights.reindex(columns=dr.columns).fillna(0.0)
    w.index = w.index.to_period("M")
    wd = w.reindex(dr.index.to_period("M") - 1)   # 전월말 비중으로 이번 달 운용
    wd.index = dr.index
    etf_d = (wd.fillna(0.0) * dr).sum(axis=1)
    sd = stock_d.pct_change().reindex(dr.index).fillna(0.0)
    days = dr.index[(dr.index >= "2001-02-01") & (dr.index <= END)]
    vals, eq = [STOCK_W, 1 - STOCK_W], []
    cur_m = None
    for d in days:
        if d.to_period("M") != cur_m:
            tot = sum(vals)
            vals = [tot * STOCK_W, tot * (1 - STOCK_W)]
            cur_m = d.to_period("M")
        vals = [vals[0] * (1 + sd.loc[d]), vals[1] * (1 + etf_d.loc[d])]
        eq.append(sum(vals))
    msap_d = pd.Series(eq, index=days)

    L = ["# MSAP 강건성 점검 (외부 피드백 대응)", "",
         f"기준: MSAP = LightGBM 종목 슬리브 {int(STOCK_W * 100)}% + 원화 리스크패리티 ETF {int(100 - STOCK_W * 100)}%, "
         f"실비용·다음날 시가 체결, 2001-02 ~ {END}", ""]

    # 1) 구간별·롤링
    L += ["## 1. 구간별 성과 (\"2013년 이후 둔화\" 점검)", "",
          "| 구간 | MSAP CAGR | Sharpe | MDD(일별) | 종목 슬리브 | ETF 슬리브 | KOSPI |", "|---|---:|---:|---:|---:|---:|---:|"]
    kospi = (1 + p.market_return("KOSPI")).cumprod()
    blocks = [("2001-2005", "2001", "2005"), ("2006-2010", "2006", "2010"), ("2011-2015", "2011", "2015"),
              ("2016-2020", "2016", "2020"), ("2021-2026.09", "2021", END), ("최근 10년(2016.10~)", "2016-10", END),
              ("최근 5년(2021.10~)", "2021-10", END), ("최근 3년(2023.10~)", "2023-10", END)]
    for name, a, b in blocks:
        sm, sdd = stats(msap_m, a, b), stats(msap_d, a, b)
        L.append(f"| {name} | {pct(sm.get('CAGR'))} | {sm.get('Sharpe', float('nan')):.2f} | {pct(sdd.get('MDD'))} | "
                 f"{pct(stats(stock_d, a, b).get('CAGR'))} | {pct(stats(rp_m, a, b).get('CAGR'))} | "
                 f"{pct(stats(kospi, a, b).get('CAGR'))} |")
    mr = msap_m.pct_change().dropna()
    roll = {}
    for yrs in (3, 5):
        n = 12 * yrs
        cg = (1 + mr).rolling(n).apply(np.prod, raw=True) ** (1 / yrs) - 1
        sh = mr.rolling(n).mean() / mr.rolling(n).std() * np.sqrt(12)
        roll[yrs] = (cg.dropna(), sh.dropna())
    L += ["", "| 롤링 창 | CAGR 최저 | 하위 10% | 중앙값 | 상위 10% | Sharpe 최저 | 중앙값 | 손실 구간 비율 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for yrs, (cg, sh) in roll.items():
        L.append(f"| {yrs}년 | {pct(cg.min())} | {pct(cg.quantile(.1))} | {pct(cg.median())} | {pct(cg.quantile(.9))} | "
                 f"{sh.min():.2f} | {sh.median():.2f} | {(cg < 0).mean():.0%} |")

    # 2) 일별 MDD
    def dd_info(e):
        dd = e / e.cummax() - 1
        t = dd.idxmin()
        peak = e.loc[:t].idxmax()
        rec = e.loc[t:][e.loc[t:] >= e.loc[peak]].index.min()
        return dd.min(), peak, t, rec
    m_mdd = stats(msap_m)["MDD"]
    d_mdd, pk, tr, rc = dd_info(msap_d)
    s_mdd, spk, strr, src = dd_info(stock_d.loc[:END])
    weekly = msap_d.resample("W").last()
    L += ["", "## 2. 일별 기준 최대낙폭 (\"월말 MDD 의 함정\" 점검)", "",
          "| 측정 | MSAP | 종목 슬리브 단독 |", "|---|---:|---:|",
          f"| 월말 기준 MDD | {pct(m_mdd)} | {pct(stats(stock_d.resample('ME').last(), None, END)['MDD'])} |",
          f"| 주말 기준 MDD | {pct(stats(weekly)['MDD'])} | {pct(stats(stock_d.resample('W').last(), None, END)['MDD'])} |",
          f"| **일별 기준 MDD** | **{pct(d_mdd)}** | **{pct(s_mdd)}** |",
          f"| 일별 MDD 구간 | {pk.date()} → {tr.date()} (회복 {rc.date() if pd.notna(rc) else '미회복'}) | "
          f"{spk.date()} → {strr.date()} |", "",
          "- ETF 슬리브의 금·미국채는 일별 데이터가 없어 월수익을 일별로 나눴다(월중 변동 과소추정). "
          "실제 일별 MDD 는 위 값보다 1~2%p 더 깊을 수 있다.",
          f"- 일별 기준 연간 변동성 {pct(msap_d.pct_change().std() * np.sqrt(252))}, "
          f"하루 최대 하락 {pct(msap_d.pct_change().min())}"]

    # 3) 환율·금리 국면
    fx_m = usdkrw_daily().resample("ME").last()
    fx_chg = fx_m.pct_change().reindex(r.index)
    fx_12 = fx_m.pct_change(12).reindex(r.index)
    pm = px.pct_change().reindex(r.index)
    msap_r = STOCK_W * r.iloc[:, 0] + (1 - STOCK_W) * r.iloc[:, 1]
    regimes = {
        "전체": pd.Series(True, index=r.index),
        "원화 강세 월(원/달러 −1% 이하)": fx_chg <= -0.01,
        "원화 약세 월(원/달러 +1% 이상)": fx_chg >= 0.01,
        "원화 강세 추세(12개월 −5% 이하)": fx_12 <= -0.05,
        "주식·채권 동반하락 월(S&P·미국채 원화 모두 −)": (pm["SPX_KRW"] < 0) & (pm["UST10_KRW"] < 0),
        "KOSPI 급락 월(−5% 이하)": pm["KOSPI200"] <= -0.05,
        "2022년": pd.Series(r.index.year == 2022, index=r.index),
    }
    L += ["", "## 3. 환율·금리 국면별 성과 (\"30% 자산배분의 국면 의존도\" 점검)", "",
          "| 국면 | 개월 | MSAP 월평균 | ETF 슬리브 | 종목 슬리브 | KOSPI200 | ETF·종목 상관 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, mask in regimes.items():
        mask = mask.fillna(False)
        if mask.sum() < 3:
            continue
        c = r[mask].corr().iloc[0, 1] if mask.sum() > 5 else np.nan
        L.append(f"| {k} | {int(mask.sum())} | {pct(msap_r[mask].mean())} | {pct(r.iloc[:, 1][mask].mean())} | "
                 f"{pct(r.iloc[:, 0][mask].mean())} | {pct(pm['KOSPI200'][mask].mean())} | "
                 f"{'-' if pd.isna(c) else f'{c:.2f}'} |")
    rc36 = r.iloc[:, 0].rolling(36).corr(r.iloc[:, 1]).dropna()
    L += ["", f"- 종목·ETF 슬리브 36개월 롤링 상관: 최저 {rc36.min():.2f}, 중앙값 {rc36.median():.2f}, 최고 {rc36.max():.2f} "
          f"(최근 {rc36.iloc[-1]:.2f})"]

    # 4) 대안: 채권·금 추세필터
    alt = {}
    for nm, prm in (("기본 ERC", RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)),
                    ("ERC + 전 자산 10개월 추세필터(이탈 시 절반)", RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=10,
                                                              trend_floor=0.5))):
        alt[nm] = run_backtest(px, TrendRiskParity(prm), cfg).equity
    L += ["", "## 4. 대안 점검: ETF 슬리브 추세필터", "",
          "| ETF 슬리브 | CAGR | Sharpe | MDD | 2022년 | 동반하락 월 평균 |", "|---|---:|---:|---:|---:|---:|"]
    co = regimes["주식·채권 동반하락 월(S&P·미국채 원화 모두 −)"].fillna(False)
    for nm, e in alt.items():
        st = stats(e, "2001", END)
        y22 = e.loc["2022-12-31"] / e.loc["2021-12-31"] - 1 if "2022-12-31" in e.index else np.nan
        L.append(f"| {nm} | {pct(st['CAGR'])} | {st['Sharpe']:.2f} | {pct(st['MDD'])} | {pct(y22)} | "
                 f"{pct(e.pct_change().reindex(r.index)[co].mean())} |")
    out = Path("reports/robustness.md")
    out.write_text("\n".join(L), encoding="utf-8")
    msap_d.to_frame("msap_daily").to_csv("reports/msap_daily_equity.csv")
    print("\n".join(L))


if __name__ == "__main__":
    main()
