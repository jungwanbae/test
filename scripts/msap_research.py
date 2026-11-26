"""MSAP 최종 검증 리포트 생성 → reports/strategy_report.md (+ 차트 PNG)

  선행: python scripts/stock_scores.py linear lgbm mlp xsformer  (워크포워드 표본외 점수)
  실행: python scripts/msap_research.py

모든 수치는 '실비용(NH 수수료+유관기관+연도별 거래세+유동성 기반 슬리피지)·다음날 시가 체결·생존편향 없음' 기준.
IS(설계·선택 구간) 2001-2012 / OOS(검증 구간) 2013-2026.11 을 분리해 표기한다.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.evaluate import composite_score, deflated_sharpe, monthly_signal_dates, stats
from autotrader.equity.factors import STOCK_FEATURES
from autotrader.equity.panel import load_panel
from autotrader.strategy.benchmarks import FixedWeights
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
OUT = Path("reports")
RULES = PortfolioRules(top_n=15, buffer=2.5)
STOCK_W = 0.70
MODELS = [m for m in ("linear", "lgbm", "mlp", "xsformer") if Path(f"data/krx/scores_{m}.parquet").exists()]
LABEL = {"composite": "규칙기반 합성팩터", "linear": "선형(릿지)", "lgbm": "LightGBM", "mlp": "딥러닝 MLP",
         "xsformer": "딥러닝 Transformer", "ens_dl": "딥러닝 앙상블(MLP+Transformer)",
         "ens_all": "전체 앙상블(LGBM+DL)"}


def pct(x):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def num(x):
    return "-" if x is None or pd.isna(x) else f"{x:.2f}"


def row(name, e, start="2001"):
    a, b, c = stats(e, start, "2012"), stats(e, "2013"), stats(e, start)
    return (f"| {name} | {pct(c.get('CAGR'))} | {num(c.get('Sharpe'))} | {pct(c.get('MDD'))} | {pct(c.get('WorstY'))} | "
            f"{pct(a.get('CAGR'))} | {num(a.get('Sharpe'))} | {pct(b.get('CAGR'))} | {num(b.get('Sharpe'))} | {pct(b.get('MDD'))} |")


HEAD = ("| 전략 | CAGR | Sharpe | MDD | 최악의 해 | IS CAGR | IS Sharpe | OOS CAGR | OOS Sharpe | OOS MDD |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main() -> None:
    p = load_panel("data/krx")
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    for m in MODELS:
        ds[m] = pd.read_parquet(f"data/krx/scores_{m}.parquet")["score"].values
    ds["composite"] = composite_score(ds)
    rank = lambda c: ds.groupby("date")[c].rank(pct=True)  # noqa: E731
    dl = [m for m in ("mlp", "xsformer") if m in MODELS]
    if dl:
        ds["ens_dl"] = sum(rank(m) for m in dl) / len(dl)
    if "lgbm" in MODELS and dl:
        ds["ens_all"] = (rank("lgbm") + sum(rank(m) for m in dl)) / (1 + len(dl))
    sig = monthly_signal_dates(ds)
    sig = sig[sig >= "2000-12-01"]
    sub = ds[ds["date"].isin(sig)]

    # ---------------- 1. 팩터 IC
    m = ds.dropna(subset=["target"])
    me = m.groupby(m["date"].dt.to_period("M"))["date"].transform("max")
    m = m[m["date"] == me]
    ic_rows = []
    for f in STOCK_FEATURES:
        if f == "kospi":
            continue
        s1 = m[m.date <= "2012-12-31"].groupby("date").apply(lambda g, f=f: g[f].corr(g["target"], method="spearman"))
        s2 = m[m.date >= "2013-01-01"].groupby("date").apply(lambda g, f=f: g[f].corr(g["target"], method="spearman"))
        ic_rows.append((f, s1.mean(), s1.mean() / s1.std() * np.sqrt(len(s1)), s2.mean(), s2.mean() / s2.std() * np.sqrt(len(s2))))
    ic_rows.sort(key=lambda r: r[2])

    # ---------------- 2. 종목 슬리브: 모델별
    curves, turns = {}, {}
    scores = ["composite"] + MODELS + [c for c in ("ens_dl", "ens_all") if c in ds]
    for sc in scores:
        r = simulate(p, make_targets(sub, sc, RULES), StockCosts(), start="2001-01-02")
        curves[sc], turns[sc] = r.equity, r.turnover.sum() / 25.9
    kospi = (1 + p.market_return("KOSPI").loc["2001":]).cumprod()
    # 모델 선택: IS Sharpe 최고
    best = max(scores, key=lambda s: stats(curves[s], "2001", "2012")["Sharpe"])

    # ---------------- 3. ETF 슬리브(원화 리스크패리티)
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    risk = ["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"]
    bcfg = BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10)
    rp = run_backtest(px, TrendRiskParity(RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)), bcfg).equity
    spx = run_backtest(px, FixedWeights({"SPX_KRW": 1.0}), bcfg).equity
    ew4 = run_backtest(px, FixedWeights({a: 0.25 for a in risk}), bcfg).equity

    def mix(stock_eq, w, b=rp):
        ra = stock_eq.dropna().resample("ME").last().pct_change()
        r = pd.concat([ra, b.pct_change()], axis=1, sort=True).loc["2001-02":"2026-09"].dropna()
        return (1 + w * r.iloc[:, 0] + (1 - w) * r.iloc[:, 1]).cumprod()

    # ---------------- 4. 비중 프런티어 + 최종
    frontier = {w: mix(curves[best], w) for w in (0.5, 0.6, 0.7, 0.8, 1.0)}
    final = frontier[STOCK_W]
    final_dl = mix(curves["ens_dl"], STOCK_W) if "ens_dl" in curves else None

    # ---------------- 5. 비용 스트레스
    stress = {}
    for nm, c in (("기본 비용", StockCosts()), ("슬리피지 2배", StockCosts(slip_base=0.001, slip_k=0.006)),
                  ("거래세 0.30% 고정", StockCosts(sell_tax_override=0.003)),
                  ("상장폐지 손실 100%", StockCosts(delist_haircut=1.0))):
        e = simulate(p, make_targets(sub, best, RULES), c, start="2001-01-02").equity
        stress[nm] = mix(e, STOCK_W)

    # ---------------- 6. DSR (시험한 설정 수 보정)
    trial_sh = []
    for f in ("reports/grid_lgbm.csv", "reports/combo2.csv", "reports/sleeveB_grid.csv"):
        if Path(f).exists():
            t = pd.read_csv(f, index_col=0)
            col = "ALL_Sharpe" if "ALL_Sharpe" in t else "ALL_Sh"
            trial_sh += list(t[col].dropna().values)
    n_trials = max(len(trial_sh) + len(scores) * 2, 10)
    dsr = deflated_sharpe(final.pct_change().dropna(), n_trials, float(np.std(trial_sh)) if trial_sh else 0.3)

    # ---------------- 7. 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"):
        if fam in have:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    base = final.index[0]
    lines = {"MSAP(최종, 주식70+ETF30)": final, "종목 알파 단독": curves[best].resample("ME").last().loc[base:],
             "ETF 리스크패리티 단독": rp.loc[base:], "KOSPI": kospi.resample("ME").last().loc[base:],
             "S&P500(원화)": spx.loc[base:]}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for i, (k, e) in enumerate(lines.items()):
        e = e / e.iloc[0]
        lw = 2.4 if i == 0 else 1.2
        a1.plot(e.index, e.values, label=k, lw=lw)
        a2.plot(e.index, (e / e.cummax() - 1).values * 100, lw=lw)
    a1.set_yscale("log"); a1.legend(frameon=False); a1.grid(alpha=.25); a1.set_ylabel("누적(로그)")
    a2.set_ylabel("낙폭 %"); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "msap_equity.png", dpi=130); plt.close(fig)

    # ---------------- 8. 리포트
    L = ["# MSAP 전략 검증 리포트", "",
         "**MSAP(Multi-Sleeve Adaptive Portfolio)** = 종목 알파 슬리브(머신러닝 랭킹, 70%) + 원화 리스크패리티 ETF 슬리브(30%)", "",
         "- 데이터: KRX 전종목 일별(1995.05~2026.11, 상장폐지 포함 4,041종목, FinanceData/marcap), 원/달러(FRED H.10), "
         "S&P500·금·미국채(공개 데이터셋)",
         "- 비용: NH PLUG 수수료 0.01% + 유관기관 0.0036% + 연도별 증권거래세(0.30%→0.15%→2026년 0.20%) + "
         "유동성 기반 슬리피지(√ 충격모형, 대형주 ≈0.08%·소형주 ≈0.26%)",
         "- 체결: 월말 종가 신호 → 다음 거래일 시가. 거래정지 종목은 매매 불가, 상장폐지 시 30% 손실 가정",
         "- 모델은 매년 그 시점 이전 데이터로만 재학습(워크포워드), 학습·테스트 사이 35일 퍼지",
         "- **IS = 2001~2012(설계·선택), OOS = 2013~2026.11(검증)**. 모델·비중·파라미터는 IS 성과로만 골랐다", "",
         "## 1. 결론", "", HEAD, row("**MSAP 최종**", final)]
    if final_dl is not None:
        L.append(row("MSAP(딥러닝 앙상블 버전)", final_dl))
    L += [row("종목 알파 단독(" + LABEL[best] + ")", curves[best]), row("ETF 리스크패리티 단독", rp),
          row("KOSPI(시총가중, 배당 제외)", kospi), row("S&P500 보유(원화)", spx), row("4자산 균등(원화)", ew4), "",
          f"- Deflated Sharpe Ratio: **{dsr['dsr']:.3f}** (시험한 설정 {dsr['n_trials']}개 보정, 기준 Sharpe "
          f"{dsr['sr0_annual']:.2f}) → 1에 가까울수록 '운 좋은 최댓값'이 아닐 확률이 높다", "",
          "## 2. 종목 알파 모델 비교 (상위 15종목·동일가중·월 리밸런싱, 표본외 점수)", "", HEAD]
    for sc in scores:
        L.append(row(f"{LABEL.get(sc, sc)} (회전율 {turns[sc]:.1f}배/년)", curves[sc]))
    L += ["", f"→ IS Sharpe 기준 선택 모델: **{LABEL.get(best, best)}**", "",
          "## 3. 주식 비중 프런티어 (나머지 = ETF 리스크패리티)", "", HEAD]
    for w, e in frontier.items():
        L.append(row(f"주식 {int(w * 100)}% / ETF {int(100 - w * 100)}%", e))
    L += ["", "## 4. 비용 스트레스 테스트 (MSAP 최종)", "", HEAD]
    for k, e in stress.items():
        L.append(row(k, e))
    L += ["", "## 5. 연도별 수익률", "", "| 연도 | MSAP | 종목 알파 | ETF RP | KOSPI |", "|---|---:|---:|---:|---:|"]
    ys = {k: v.resample("YE").last().pct_change() for k, v in
          {"a": final, "b": curves[best], "c": rp, "d": kospi}.items()}
    for y in range(2002, 2027):
        cells = []
        for k in "abcd":
            v = ys[k][ys[k].index.year == y]
            cells.append(pct(v.iloc[0]) if len(v) else "-")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    L += ["", "## 6. 팩터 IC (월간 순위상관, 높을수록 다음 달 수익 예측력)", "",
          "| 팩터 | IS IC | IS t | OOS IC | OOS t |", "|---|---:|---:|---:|---:|"]
    for f, a, at, b, bt in ic_rows:
        L.append(f"| {f} | {a:+.3f} | {at:+.1f} | {b:+.3f} | {bt:+.1f} |")
    L += ["", "![MSAP](msap_equity.png)", "",
          "## 7. 한계", "- 과거 성과는 미래를 보장하지 않는다. 특히 OOS 수익률이 IS 보다 낮다(알파 감소).",
          "- 종목 데이터는 배당 제외 가격수익률(보수적). ETF 슬리브의 월평균 시계열은 1개월 체결지연으로 평활편향 제거.",
          "- 원화 현금수익은 한국은행 기준금리 연평균 근사치.",
          "- 15종목 집중 포트폴리오라 개별 종목 사건 위험이 있다. 운용자금이 크면(수십억+) 슬리피지가 커진다."]
    (OUT / "strategy_report.md").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame({"final": final}).to_csv(OUT / "msap_monthly_equity.csv")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
