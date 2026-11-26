"""질문 2: '확인형 역발상 VT' 는 평소 현금에 있나? 노출 분포와, 남는 현금을 다른 자산에 두는 대안 검증.

산출: reports/vt_cash_study.csv, reports/vt_exposure.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from tournament_analysis import blend  # noqa: E402

from autotrader.research.data import build_returns  # noqa: E402
from autotrader.research.tournament import Ctx, lev_risk_parity, recovery_boost, simulate, stats  # noqa: E402

warnings.filterwarnings("ignore")


def exposure_path(R, tg):
    """목표 행(신호일) → 일별 보유비중(다음날부터, 행 사이에는 고정 목표로 근사)."""
    t = tg.reindex(columns=R.columns).fillna(0.0)
    t.index = [R.index[R.index.get_loc(d) + 1] if R.index.get_loc(d) + 1 < len(R.index) else d for d in t.index]
    return t.reindex(R.index).ffill().dropna(how="all")


def main():
    R, aux = build_returns()
    c = Ctx(R=R, P=aux["P_sig"], vix=aux["vix"], rf_us=aux["rf_us"])
    cur = pd.read_parquet("data/krx/tournament_equity.parquet")
    msap = cur["MSAP-M(종목 70 + ETF RP 30)"]
    tv = recovery_boost("QQQ")(c)
    tr = lev_risk_parity(0.18, 3, True)(c)
    ev = exposure_path(R, tv).loc["2005":]
    er = exposure_path(R, tr).loc["2005":]
    E = ev["QQQ"] + 3 * ev["QQQx3"]
    cash_v, cash_r = ev["CASH_USD"], er["CASH_USD"]
    ex = pd.DataFrame({"VT 노출(배)": E, "VT 현금비중": cash_v, "RP 현금비중": cash_r})
    ex.resample("ME").last().to_csv("reports/vt_exposure.csv")
    print("확인형 역발상 VT 나스닥 노출(배) 분포(2005~, 일별):")
    print(E.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    print(f"노출 ≥1배(현금 0) 비율 {(E >= 0.999).mean():.0%} · 평균 현금비중 {cash_v.mean():.1%} · 현금 ≥50% 비율 {(cash_v >= .5).mean():.0%}")
    print(f"레버리지 RP 평균 현금비중 {cash_r.mean():.1%} · 현금 ≥50% 비율 {(cash_r >= .5).mean():.0%}")
    rows = {}
    alts = {"미국 단기채(T-bill, 현행)": None, "중기채 IEF": {"IEF": 1}, "장기채 TLT": {"TLT": 1}, "금 GLD": {"GLD": 1},
            "중기채50+금50": {"IEF": .5, "GLD": .5}, "S&P500 1배": {"SPY": 1}, "나스닥 1배(=항상 ≥1배)": {"QQQ": 1},
            "레버리지 RP 슬리브로": "LRP"}
    for lab, alt in alts.items():
        if alt == "LRP":
            # VT 현금 몫을 레버리지 RP 비중으로 대체(각 시점 RP 목표비중 × VT 현금비중)
            u = tv.index.union(tr.index)        # 행 안의 빈 칸(NaN=0%)을 먼저 0으로 → 그다음 날짜 정렬
            rp_rows = tr.fillna(0.0).reindex(u).ffill()
            vt_rows = tv.fillna(0.0).reindex(u).ffill()
            cw = vt_rows["CASH_USD"].fillna(0.0)
            new = vt_rows.drop(columns="CASH_USD").fillna(0.0).add(rp_rows.fillna(0.0).mul(cw, axis=0), fill_value=0.0)
        elif alt is None:
            new = tv
        else:
            new = tv.copy().fillna(0.0)
            cw = new.pop("CASH_USD")
            for k, v in alt.items():
                new[k] = new.get(k, 0.0) + cw * v
        eq, turn = simulate(R, new, cost=0.001, start="2001-01-02")
        tri = blend(pd.DataFrame({"K": msap, "V": eq, "R": cur["레버리지 리스크패리티 18%+추세"]}),
                    {"K": 1 / 3, "V": 1 / 3, "R": 1 / 3})
        a, o, f = stats(eq, "2005-01-03"), stats(eq, "2016-01-01"), stats(eq)
        ta, to = stats(tri, "2005-01-03"), stats(tri, "2016-01-01")
        rows[lab] = {"VT CAGR": a["CAGR"], "VT Sharpe": a["Sharpe"], "VT MDD": a["MDD"], "VT 2016~": o["CAGR"],
                     "VT 2001~": f["CAGR"], "VT 2001~ MDD": f["MDD"], "TRIAD CAGR": ta["CAGR"], "TRIAD Sharpe": ta["Sharpe"],
                     "TRIAD MDD": ta["MDD"], "TRIAD 2016~": to["CAGR"]}
    S = pd.DataFrame(rows).T
    S.to_csv("reports/vt_cash_study.csv")
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
