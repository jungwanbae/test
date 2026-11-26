"""질문 1: 기존 3전략 + TRIAD 슬리브를 합쳐 '최적화'하면 나아지나? — 워크포워드(과거만 보고 매년 재최적화)로 검증.

슬리브 7개(MSAP-M, VT, 확인형 역발상 VT, 고금리 대비형, 3차 공격형, AQUILA-X, 레버리지 RP).
매년 말, 그때까지의 데이터만으로 10% 격자(각 ≤50%, 6,538개) 중 목적함수 최대 조합을 고르고 다음 1년 보유.
산출: reports/combo_wf.csv, reports/combo_wf_weights.csv
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from tournament_analysis import SLEEVES, TRIAD, blend  # noqa: E402

from autotrader.research.tournament import stats  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    cur = pd.read_parquet("data/krx/tournament_equity.parquet")
    names = list(SLEEVES)
    short = [SLEEVES[n][0] for n in names]
    eq = cur[names].loc["2005-10-03":].dropna()
    r = eq.pct_change().fillna(0.0)
    grid = np.array([c for c in itertools.product(np.arange(0, 0.51, 0.1), repeat=len(names))
                     if abs(sum(c) - 1) < 1e-9])
    P = r.values @ grid.T                                    # 일별 리밸런싱 근사(선택용)
    idx = r.index
    years = sorted(set(idx.year))
    objectives = {
        "전체과거 Sharpe 최대": ("sharpe", None), "최근 5년 Sharpe 최대": ("sharpe", 5), "최근 3년 Sharpe 최대": ("sharpe", 3),
        "전체과거 Calmar 최대": ("calmar", None), "최근 5년 CAGR 최대(MDD≤30%)": ("cagr_cap", 5),
        "최근 3년 CAGR 최대(모멘텀 추종)": ("cagr", 3), "최소분산(전체과거)": ("minvar", None),
    }
    out_w = {k: {} for k in objectives}
    for y in years[3:]:                                       # 최소 3년 학습 후 2009년부터
        hi = idx < pd.Timestamp(f"{y}-01-01")
        for lab, (obj, win) in objectives.items():
            m = hi & (idx >= pd.Timestamp(f"{y - win}-01-01")) if win else hi
            x = P[m]
            mu, sd = x.mean(0), x.std(0)
            if obj == "sharpe":
                score = mu / sd
            elif obj == "minvar":
                score = -sd
            else:
                curve = np.cumprod(1 + x, axis=0)
                mdd = (curve / np.maximum.accumulate(curve, axis=0) - 1).min(0)
                cagr = curve[-1] ** (252 / len(x)) - 1
                score = cagr / np.abs(mdd) if obj == "calmar" else np.where(mdd >= -0.30, cagr, -9) \
                    if obj == "cagr_cap" else cagr
            out_w[lab][y] = grid[int(np.argmax(score))]
    rows, wtab = {}, []
    for lab, wy in out_w.items():
        parts = []
        for y, w in wy.items():
            seg = r.loc[str(y)]
            parts.append(pd.Series(seg.values @ w, index=seg.index))
            wtab.append({"규칙": lab, "연도": y, **dict(zip(short, w))})
        ret = pd.concat(parts)
        e = (1 + ret).cumprod()
        rows[lab] = e
    fixed = {"TRIAD(고정 1/3)": TRIAD, "7슬리브 동일가중": {n: 1 / 7 for n in names},
             "TRIAD 50% + 기존3전략 50%": {**{k: v / 2 for k, v in TRIAD.items()},
                                         **{n: 1 / 6 for n in names if SLEEVES[n][0] in ("HR", "A3", "AQ")}},
             "기존3전략 1/3": {n: 1 / 3 for n in names if SLEEVES[n][0] in ("HR", "A3", "AQ")}}
    for lab, w in fixed.items():
        rows[lab] = blend(cur, w)
    res = {}
    for lab, e in rows.items():
        e = e.loc["2009-01-01":]
        a, i, o = stats(e), stats(e, "2009-01-01", "2015-12-31"), stats(e, "2016-01-01")
        res[lab] = {"CAGR(2009~)": a["CAGR"], "Sharpe": a["Sharpe"], "MDD": a["MDD"], "최악의 해": a["WorstY"],
                    "2009~15": i["CAGR"], "2016~": o["CAGR"], "2016~ Sharpe": o["Sharpe"], "2016~ MDD": o["MDD"]}
    R = pd.DataFrame(res).T
    R.to_csv("reports/combo_wf.csv")
    pd.DataFrame(wtab).to_csv("reports/combo_wf_weights.csv", index=False)
    print(R.round(3).to_string())
    W = pd.DataFrame(wtab)
    print(W[W["규칙"] == "최근 3년 Sharpe 최대"].set_index("연도").drop(columns="규칙").to_string())


if __name__ == "__main__":
    main()
