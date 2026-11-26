"""토너먼트 후속 분석 — 상관·위기 구간·장기(1970~) 꼬리위험·역발상 민감도·슬리브 조합 최적화.

  python scripts/tournament_analysis.py     (먼저 scripts/tournament.py 실행)
산출: reports/tournament_{corr,crisis,long,boost_grid,blends}.csv, reports/tournament_final.png
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from autotrader.equity.evaluate import deflated_sharpe
from autotrader.research.data import build_long_usd
from autotrader.research.tournament import (Ctx, after_tax, dd_ladder, fixed, golden_cross, ladder_with_stop,
                                            recovery_boost, simulate, stats, vol_target)

warnings.filterwarnings("ignore")
IS_, OOS_, COMMON = ("2005-01-03", "2015-12-31"), ("2016-01-01", None), ("2005-01-03", None)
SLEEVES = {  # 조합 후보(사전 지정: 토너먼트 상위 + 성격이 다른 것) → 해외 비중(세금)
    "MSAP-M(종목 70 + ETF RP 30)": ("KR", 0.0),
    "변동성 타기팅 QQQ 25%(최대 3배)": ("VT", 1.0),
    "역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%": ("VT+", 1.0),
    "[기존] 고금리 대비형(근사)": ("HR", 0.8),
    "[기존] 3차 공격형(근사)": ("A3", 1.0),
    "[기존] AQUILA-X(근사)": ("AQ", 1.0),
    "레버리지 리스크패리티 18%+추세": ("LRP", 1.0),
}


def pct(x):
    return f"{x * 100:.1f}%" if pd.notna(x) else "—"


def blend(curves: pd.DataFrame, w: dict, cost=0.001, band: float | None = None) -> pd.Series:
    """슬리브 간 리밸런싱(슬리브 내부는 각자 운용). band=None 이면 매월말, band=x 면 어느 슬리브든
    목표 대비 ±x 를 벗어난 월말에만(실전: 레그 간 환전을 가끔만). 비용 = 이동 금액 × cost."""
    cols = [k for k, v in w.items() if v > 0]
    eq = curves[cols].dropna()
    r = eq.pct_change().fillna(0.0)
    tw = np.array([w[c] for c in cols])
    cur, val, out = tw.copy(), 1.0, []
    me = set(eq.groupby(eq.index.to_period("M")).tail(1).index)
    for d, row in zip(eq.index, r.values):
        g = 1 + row @ cur
        cur = cur * (1 + row) / g
        val *= g
        if d in me and (band is None or np.abs(cur - tw).max() > band):
            val *= 1 - np.abs(cur - tw).sum() / 2 * cost * (2 if band is None else 1)
            cur = tw.copy()
        out.append(val)
    return pd.Series(out, index=eq.index)


TRIAD = {"MSAP-M(종목 70 + ETF RP 30)": 1 / 3, "역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%": 1 / 3,
         "레버리지 리스크패리티 18%+추세": 1 / 3}


def variants(curves: pd.DataFrame) -> dict[str, dict]:
    """최적화 없는(사전에 정할 수 있는) 단순 배분 규칙들 — 결론이 배분 방식에 민감한지 확인."""
    n = list(SLEEVES)
    inv = {v[0]: k for k, v in SLEEVES.items()}
    us6 = [k for k in n if SLEEVES[k][0] != "KR"]
    return {
        "TRIAD: 한국 MSAP-M + 확인형 역발상 VT + 레버리지 RP (각 1/3)": TRIAD,
        "7슬리브 동일가중": {k: 1 / 7 for k in n},
        "6슬리브 동일가중(VT 제외)": {k: 1 / 6 for k in n if SLEEVES[k][0] != "VT"},
        "6슬리브 동일가중(확인형 역발상 제외)": {k: 1 / 6 for k in n if SLEEVES[k][0] != "VT+"},
        "클러스터 1/3(한국 / 나스닥 추세군 / 멀티에셋 RP군)": {inv["KR"]: 1 / 3, inv["VT+"]: 1 / 9, inv["HR"]: 1 / 9,
                                                   inv["AQ"]: 1 / 9, inv["LRP"]: 1 / 6, inv["A3"]: 1 / 6},
        "한국 30 + 미국 6슬리브 동일": {**{inv["KR"]: 0.3}, **{k: 0.7 / 6 for k in us6}},
        "한국 50 + 미국 6슬리브 동일": {**{inv["KR"]: 0.5}, **{k: 0.5 / 6 for k in us6}},
        "[기존] 3전략 1/3(고금리·3차·AQUILA)": {inv["HR"]: 1 / 3, inv["A3"]: 1 / 3, inv["AQ"]: 1 / 3},
        "한국 + [기존] 3전략 1/4": {inv["KR"]: .25, inv["HR"]: .25, inv["A3"]: .25, inv["AQ"]: .25},
    }


def long_history():
    """1970~ 미국 S&P500(달러) — 2000·2008 같은 장기 약세장이 여러 번 들어간 꼬리위험 검증."""
    R, P, rf = build_long_usd()
    c = Ctx(R=R, P=P, vix=pd.Series(dtype=float), rf_us=rf)
    specs = {"S&P500 보유": fixed({"SPY": 1}),
             "S&P500 3배 보유": fixed({"SPYx3": 1}),
             "역발상 낙폭 사다리(−20/−30/−40% → 1.5/2/3배)": dd_ladder("SPY"),
             "역발상 사다리 + 추가하락 −25% 손절": ladder_with_stop("SPY"),
             "골든크로스 3배": golden_cross("SPY", 3),
             "변동성 타기팅 25%(최대 3배)": vol_target("SPY", 0.25, 3),
             "확인형 역발상(폭락 후 골든크로스 1년 3배)+VT25%": recovery_boost("SPY")}
    rows, curves = {}, {}
    for n, b in specs.items():
        eq, _ = simulate(R, b(c), cost=0.001, start="1971-01-04")
        curves[n] = eq
        s = stats(eq)
        dec = {f"{y}s": stats(eq, f"{y}-01-01", f"{y + 9}-12-31").get("CAGR") for y in (1970, 1980, 1990, 2000, 2010)}
        rows[n] = {"CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MDD": s["MDD"], "최악의 해": s["WorstY"],
                   "5년 롤링 최저": s["Roll5Min"], **dec}
    return pd.DataFrame(rows).T, pd.DataFrame(curves)


def boost_grid(c: Ctx, R: pd.DataFrame):
    """확인형 역발상 파라미터 민감도 — 특정 값만 좋은 '과최적화'인지 확인."""
    rows = []
    for crash, hold, L in itertools.product((-0.20, -0.25, -0.30, -0.35, -0.40), (126, 252, 378), (2.0, 2.5, 3.0)):
        eq, _ = simulate(R, recovery_boost("QQQ", 0.25, crash, L, hold)(c), cost=0.001, start="2001-01-02")
        a, i, o = stats(eq, *COMMON), stats(eq, *IS_), stats(eq, *OOS_)
        rows.append({"crash": crash, "hold": hold, "L": L, "CAGR": a["CAGR"], "Sharpe": a["Sharpe"], "MDD": a["MDD"],
                     "IS CAGR": i["CAGR"], "OOS CAGR": o["CAGR"], "전체 CAGR": stats(eq)["CAGR"]})
    return pd.DataFrame(rows)


def main():
    curves = pd.read_parquet("data/krx/tournament_equity.parquet")
    R = pd.read_parquet("data/krx/research_returns.parquet")
    names = list(SLEEVES)
    short = {k: v[0] for k, v in SLEEVES.items()}
    # 1) 상관(일별, 2005~) — 미국 휴장일 정렬 차이를 줄이려고 주간 수익률도 함께
    sub = curves[names].loc["2005-01-03":]
    wk = sub.resample("W-FRI").last().pct_change().dropna().rename(columns=short)
    corr = wk.corr().round(2)
    corr.to_csv("reports/tournament_corr.csv")
    print("주간 수익률 상관(2005~)\n", corr.to_string())
    # 2) 위기 구간
    crises = {"2008 금융위기(07.10~09.03)": ("2007-10-09", "2009-03-09"), "2011 유럽위기(11.04~11.10)": ("2011-04-29", "2011-10-03"),
              "2018 Q4": ("2018-09-28", "2018-12-24"), "2020 코로나(02.19~03.23)": ("2020-02-19", "2020-03-23"),
              "2022 금리급등(01~10)": ("2022-01-03", "2022-10-14"), "2024.08 엔캐리": ("2024-07-10", "2024-08-07")}
    allc = ["나스닥100 보유", "KOSPI200 보유", "60/40(주식/중기채)", "역발상 낙폭 사다리 QQQ(−20/−30/−40% → 1.5/2/3배)"] + names
    cr = pd.DataFrame({k: {short.get(n, n[:14]): curves[n].loc[a:b].iloc[-1] / curves[n].loc[a:b].iloc[0] - 1
                           for n in allc} for k, (a, b) in crises.items()})
    cr.to_csv("reports/tournament_crisis.csv")
    print("\n위기 구간 수익률\n", (cr * 100).round(1).to_string())
    # 3) 장기 꼬리위험
    lh, lcurves = long_history()
    lh.to_csv("reports/tournament_long.csv")
    print("\n1971~ 미국(달러, S&P500 기반)\n", lh.map(lambda v: round(v, 3)).to_string())
    # 4) 확인형 역발상 민감도
    from autotrader.research.data import us_prices
    px = us_prices()["QQQ"].reindex(R.index).ffill()
    c = Ctx(R=R, P=pd.DataFrame({"QQQ": px}), vix=pd.Series(dtype=float), rf_us=pd.Series(0.0, index=R.index))
    g = boost_grid(c, R)
    g.to_csv("reports/tournament_boost_grid.csv", index=False)
    vt = stats(curves["변동성 타기팅 QQQ 25%(최대 3배)"], *COMMON)
    print(f"\n확인형 역발상 45개 설정: CAGR 중앙값 {g.CAGR.median():.3f} (범위 {g.CAGR.min():.3f}~{g.CAGR.max():.3f}), "
          f"VT25% 대비 우위 {(g.CAGR > vt['CAGR']).mean():.0%}, MDD 중앙값 {g.MDD.median():.3f}, "
          f"OOS CAGR 중앙값 {g['OOS CAGR'].median():.3f}")
    # 5) 슬리브 조합 — 사전 선언 규칙: IS(2005~2015) Calmar 최대, 단 IS CAGR ≥ 20% · 각 슬리브 ≤ 50% · 합 100%(10% 단위)
    grid = []
    steps = np.arange(0, 0.51, 0.1)
    for combo in itertools.product(steps, repeat=len(names)):
        if abs(sum(combo) - 1) > 1e-9:
            continue
        grid.append(dict(zip(names, combo)))
    print(f"\n조합 후보 {len(grid)}개 평가 중…", flush=True)
    rows, ok_curves = [], {}
    rets = curves[names].loc["2004-12-01":].pct_change()
    for w in grid:
        # 빠른 근사(일별 리밸런싱)로 1차 선별 → 상위만 월간 리밸런싱으로 정밀 평가
        wv = np.array([w[n] for n in names])
        r = (rets.fillna(0.0).values @ wv)
        eq = pd.Series(np.cumprod(1 + r), index=rets.index)
        i = stats(eq, *IS_)
        rows.append({**{short[n]: w[n] for n in names}, "IS CAGR": i["CAGR"], "IS Sharpe": i["Sharpe"], "IS MDD": i["MDD"],
                     "IS Calmar": i["Calmar"]})
    G = pd.DataFrame(rows)
    cand = G[G["IS CAGR"] >= 0.20].sort_values("IS Calmar", ascending=False).head(40)
    fine = []
    for _, row in cand.iterrows():
        w = {n: row[short[n]] for n in names}
        eq = blend(curves, w)
        a, i, o = stats(eq, *COMMON), stats(eq, *IS_), stats(eq, *OOS_)
        osh = sum(w[n] * SLEEVES[n][1] for n in names)
        at = stats(after_tax(eq.loc[COMMON[0]:], osh))
        fine.append({**{short[n]: w[n] for n in names}, "IS CAGR": i["CAGR"], "IS Calmar": i["Calmar"], "IS Sharpe": i["Sharpe"],
                     "OOS CAGR": o["CAGR"], "OOS Sharpe": o["Sharpe"], "OOS MDD": o["MDD"], "CAGR": a["CAGR"], "세후": at["CAGR"],
                     "Sharpe": a["Sharpe"], "MDD": a["MDD"], "Calmar": a["Calmar"], "최악의 해": a["WorstY"],
                     "5년 롤링 최저": a["Roll5Min"]})
        ok_curves[tuple(w.values())] = eq
    F = pd.DataFrame(fine).sort_values("IS Calmar", ascending=False)
    F.to_csv("reports/tournament_blends.csv", index=False)
    print(F.head(15).round(3).to_string())
    best = F.iloc[0]
    wbest = {n: best[short[n]] for n in names}
    eqb = ok_curves[tuple(wbest.values())]
    # 6) 최적화 없는 단순 배분 규칙들(사전 지정 가능) — 최적화 조합과 비교
    q = R["QQQ"]
    summary, vcurves = {}, {}
    allv = {"IS 규칙 최적 조합": wbest, **variants(curves)}
    for lab, w in allv.items():
        w = {k: v for k, v in w.items() if v > 0}
        eq = ok_curves[tuple(wbest.values())] if lab == "IS 규칙 최적 조합" else blend(curves, w)
        vcurves[lab] = eq
        a, i, o = stats(eq, *COMMON), stats(eq, *IS_), stats(eq, *OOS_)
        osh = sum(w[n] * SLEEVES[n][1] for n in w)
        rr = eq.pct_change()
        beta = rr.rolling(63).cov(q.reindex(eq.index)) / q.reindex(eq.index).rolling(63).var()
        summary[lab] = {"CAGR": a["CAGR"], "세후(근사)": stats(after_tax(eq.loc[COMMON[0]:], osh))["CAGR"],
                        "Sharpe": a["Sharpe"], "변동성": a["Vol"], "일별 MDD": a["MDD"], "Calmar": a["Calmar"],
                        "2005~15": i["CAGR"], "2016~": o["CAGR"], "2016~ Sharpe": o["Sharpe"], "2016~ MDD": o["MDD"],
                        "최악의 해": a["WorstY"], "5년 롤링 최저": a["Roll5Min"],
                        "나스닥 β 중앙값": beta.loc["2005":].median(), "나스닥 β 최대": beta.loc["2005":].max(),
                        "최악의 하루": rr.loc["2005":].min()}
    S = pd.DataFrame(summary).T
    S.to_csv("reports/tournament_final.csv")
    print("\n배분 규칙 비교(2005~)\n", S.round(3).to_string())
    print("IS 규칙 가중치:", {short[n]: wbest[n] for n in names})
    eqb = ok_curves[tuple(wbest.values())]
    # 7) TRIAD 상세: 2001~ 전체(닷컴 붕괴 포함), 레그 간 밴드 리밸런싱(실전 방식), DSR
    tri = blend(curves, TRIAD)
    tri_band = blend(curves, TRIAD, cost=0.002, band=0.10)    # ±10%p 벗어날 때만 환전(환전비용 0.2%)
    rows = {}
    for lab, eq in (("TRIAD 매월 리밸런싱", tri), ("TRIAD ±10%p 밴드 리밸런싱", tri_band)):
        for per, (a_, b_) in {"2001~ 전체": (None, None), "2005~": COMMON, "2001~2004": ("2001-01-01", "2004-12-31"),
                              "2005~2015": IS_, "2016~": OOS_, "최근 5년": ("2021-11-25", None)}.items():
            st = stats(eq, a_, b_)
            rows[(lab, per)] = {"CAGR": st.get("CAGR"), "Sharpe": st.get("Sharpe"), "MDD": st.get("MDD"),
                                "최악의 해": st.get("WorstY"), "5년 롤링 최저": st.get("Roll5Min")}
    T = pd.DataFrame(rows).T
    T.to_csv("reports/tournament_triad.csv")
    print("\nTRIAD 상세\n", T.round(3).to_string())
    n_trials = len(curves.columns) + len(G) + len(allv) + 45 + 7   # 토너먼트 + 조합격자 + 배분규칙 + 민감도 + 장기
    sh = pd.concat([G["IS Sharpe"], pd.Series([stats(curves[c], *IS_).get("Sharpe", np.nan) for c in curves])]).std()
    for lab, eq in (("TRIAD", tri), ("IS 규칙 최적 조합", eqb)):
        d = deflated_sharpe(eq.loc[OOS_[0]:].pct_change().dropna(), n_trials=n_trials, trial_sharpes_std=sh,
                            periods_per_year=252)
        print(f"{lab} 2016~ DSR:", {k: round(float(v), 3) for k, v in d.items()})
        pd.Series({k: float(v) for k, v in d.items()}).to_csv(f"reports/tournament_dsr_{'triad' if lab == 'TRIAD' else 'isopt'}.csv")
    eqe = tri
    # 연도별
    yr = pd.DataFrame({"TRIAD": tri, "MSAP-M": curves[names[0]], "확인형 역발상 VT": curves[names[2]],
                       "레버리지 RP": curves[names[6]], "나스닥100": curves["나스닥100 보유"],
                       "KOSPI200": curves["KOSPI200 보유"]}).loc["2000-12-29":].resample("YE").last().pct_change().dropna()
    yr.index = yr.index.year
    yr.to_csv("reports/tournament_yearly.csv")
    print("\n연도별\n", (yr * 100).round(1).to_string())
    pd.DataFrame({"triad": tri, "triad_band": tri_band, "isopt": eqb}).to_parquet("data/krx/tournament_blend_equity.parquet")
    lcurves.to_parquet("data/krx/tournament_long_equity.parquet")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        show = {"TRIAD": eqe, "MSAP-M (KR)": curves[names[0]], "Confirmed-contrarian VT (US)": curves[names[2]],
                "Levered RP (US)": curves[names[6]], "Nasdaq100": curves["나스닥100 보유"],
                "Contrarian DD ladder": curves["역발상 낙폭 사다리 QQQ(−20/−30/−40% → 1.5/2/3배)"]}
        for k, v in show.items():
            v = v.loc["2001-02-01":]
            ax[0].plot(v / v.iloc[0], label=k, lw=2.0 if k == "TRIAD" else 1)
            ax[1].plot(v / v.cummax() - 1, lw=2.0 if k == "TRIAD" else 0.8)
        ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
        ax[0].set_title("TRIAD vs components (KRW, daily, 2001~)"); ax[1].set_ylabel("drawdown")
        fig.tight_layout(); fig.savefig("reports/tournament_final.png", dpi=110)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
