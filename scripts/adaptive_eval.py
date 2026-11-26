"""시대 적응형 ML/딥러닝 평가 — 예측력·나스닥 레그 적용·전 자산 일관성·유사 국면 가중의 실체.

  python scripts/adaptive_eval.py      (먼저 scripts/adaptive_ml.py)
산출: reports/adaptive_{auc,qqq,assets,similar}.csv, reports/adaptive_report.md
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from autotrader.research import adaptive as A  # noqa: E402
from autotrader.research.data import tbill_daily, us_prices  # noqa: E402
from autotrader.research.tournament import Ctx, lev_mix, month_ends, recovery_boost, simulate, stats  # noqa: E402

warnings.filterwarnings("ignore")
PRED = Path("data/krx/adaptive_preds")
PERIODS = {"1985~1999": ("1985", "1999"), "2000~2009": ("2000", "2009"), "2010~2019": ("2010", "2019"),
           "2020~2026": ("2020", "2026")}
NAMES = {"lgbm": "LightGBM", "mlp": "MLP(딥러닝)", "cnn": "CNN(차트 딥러닝)", "expanding": "전체 과거",
         "rolling10": "최근 10년", "decay5": "최근 가중(반감 5년)", "decay2": "최근 가중(반감 2년)",
         "similar": "유사 국면 가중", "similar_q": "유사 국면 가중(분기 재학습)", "bag": "시기 섞기 배깅"}


def label(tag: str) -> str:
    if tag.startswith("ENS"):
        return {"ENS_lgbm": "앙상블: LightGBM 7방식", "ENS_dl": "앙상블: 딥러닝(MLP+CNN)",
                "ENS_all": "앙상블: 전체"}[tag]
    m, s = tag.split("_", 1)
    return f"{NAMES[m]} · {NAMES.get(s, s)}"


def load_preds() -> dict[str, pd.DataFrame]:
    out = {f.stem: pd.read_parquet(f) for f in sorted(PRED.glob("*.parquet"))}
    common = None
    for v in out.values():
        common = v.index if common is None else common.intersection(v.index)
    out = {k: v.loc[common] for k, v in out.items()}
    groups = {"ENS_lgbm": [k for k in out if k.startswith("lgbm")],
              "ENS_dl": [k for k in out if k.startswith(("mlp", "cnn"))], "ENS_all": list(out)}
    for g, ks in groups.items():
        if ks:
            out[g] = sum(out[k] for k in ks) / len(ks)
    return out


def score_pct(s: pd.Series, win=756, minp=126) -> pd.Series:
    """자산별 과거 OOS 점수 분포 안에서 현재 점수의 백분위(인과적)."""
    return s.rolling(win, min_periods=minp).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan,
                                                  raw=True)


def auc_table(P, preds) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([P.date, P.asset])
    lab = pd.DataFrame({"y": P.y, "z": P.z}, index=key)
    rows = {}
    for tag, pr in preds.items():
        d = pr.join(lab, how="inner")
        d = d[d.y >= 0]
        s = d.p_up - d.p_down
        r = {}
        for per, (a, b) in PERIODS.items():
            m = (d.index.get_level_values(0) >= a) & (d.index.get_level_values(0) <= f"{b}-12-31")
            x = d[m]
            if len(x) < 1000:
                continue
            r[f"상승AUC {per}"] = roc_auc_score(x.y == 2, x.p_up)
            r[f"하락AUC {per}"] = roc_auc_score(x.y == 0, x.p_down)
        r["상승AUC 전체"] = roc_auc_score(d.y == 2, d.p_up)
        r["하락AUC 전체"] = roc_auc_score(d.y == 0, d.p_down)
        q = s.groupby(level=1).transform(lambda v: v.rank(pct=True))
        r["상위20% 평균z"] = d.z[q >= 0.8].mean()
        r["하위20% 평균z"] = d.z[q <= 0.2].mean()
        rows[label(tag)] = r
    return pd.DataFrame(rows).T


def qqq_strategies(preds, R, c, start="2001-01-02"):
    """나스닥 레그 적용: VT(변동성 25%) / VT+(확인형 역발상) 노출에 ML 게이트·틸트를 곱한다(월말 신호)."""
    q = c.P["QQQ"]
    vol = q.pct_change().rolling(21).std() * np.sqrt(252)
    me = [d for d in c.me if d >= pd.Timestamp("2000-06-30")]
    vtp_rows = recovery_boost("QQQ")(c)
    vtp_E = (vtp_rows.get("QQQ", 0).fillna(0) + 3 * vtp_rows.get("QQQx3", 0).fillna(0))
    base = {"VT": pd.Series({d: min(3.0, 0.25 / vol.loc[d]) for d in me}),
            "VT+": vtp_E[vtp_E.index >= me[0]]}   # 월말 + 부스트 시작/종료 같은 사건일(백테스트와 같은 신호일)
    res, curves = {}, {}

    def run(name, E):
        tg = pd.DataFrame({d: lev_mix("QQQ", float(e), "CASH_USD", 3) for d, e in E.dropna().items()}).T
        eq, turn = simulate(R, tg, cost=0.001, start=start)
        curves[name] = eq
        a, i, o = stats(eq, start), stats(eq, start, "2012-12-31"), stats(eq, "2013-01-01")
        res[name] = {"CAGR": a["CAGR"], "Sharpe": a["Sharpe"], "MDD": a["MDD"], "최악의 해": a["WorstY"],
                     "2001~12": i["CAGR"], "2013~": o["CAGR"], "2013~ Sharpe": o["Sharpe"], "평균 노출": E.mean()}

    for b, E in base.items():
        run(f"{b} (규칙만)", E)
    for tag, pr in preds.items():
        s = (pr.p_up - pr.p_down).xs("QQQ", level=1)
        pct = score_pct(s).reindex(c.days).ffill()
        for b, E in base.items():
            p = pct.reindex(E.index)
            run(f"{b} × ML게이트 [{label(tag)}]", E.where(p.isna() | (p >= 0.2), 0.0))
            run(f"{b} × ML틸트 [{label(tag)}]", (E * (0.5 + p.fillna(0.5))).clip(upper=3.0))
    return pd.DataFrame(res).T, curves


def asset_robustness(P, preds, assets) -> pd.DataFrame:
    """전 자산(1배, 달러): 보유 vs ML 게이트(하위 20% 점수면 현금) — Sharpe 개선 비율·구간별."""
    rows = {}
    for tag, pr in preds.items():
        s_all = pr.p_up - pr.p_down
        out = []
        for a in sorted(set(P.asset)):
            if a not in s_all.index.get_level_values(1):
                continue
            px = assets[a]["close"]
            r = px.pct_change()
            rf = tbill_daily(px.index) / 252
            pct = score_pct(s_all.xs(a, level=1)).reindex(px.index).ffill()
            mes = month_ends(px.index)
            pos = pd.Series(np.nan, index=px.index)
            pos.loc[mes] = (pct.loc[mes] >= 0.2).astype(float).where(pct.loc[mes].notna(), 1.0)
            pos = pos.ffill().shift(1)
            ok = pos.notna()
            for per, (lo, hi) in {"1985~2004": ("1985", "2004"), "2005~2026": ("2005", "2026")}.items():
                m = ok & (px.index >= lo) & (px.index <= f"{hi}-12-31")
                if m.sum() < 500:
                    continue
                hold = r[m]
                gate = pos[m] * r[m] + (1 - pos[m]) * rf[m] - pos[m].diff().abs().fillna(0) * 0.001
                sh = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
                ddh = ((1 + hold).cumprod() / (1 + hold).cumprod().cummax() - 1).min()
                ddg = ((1 + gate).cumprod() / (1 + gate).cumprod().cummax() - 1).min()
                out.append({"asset": a, "per": per, "dSharpe": sh(gate) - sh(hold), "dMDD": ddg - ddh})
        o = pd.DataFrame(out)
        if o.empty:
            continue
        r = {}
        for per in ("1985~2004", "2005~2026"):
            x = o[o.per == per]
            if len(x):
                r[f"Sharpe 개선 자산 비율 {per}"] = (x.dSharpe > 0).mean()
                r[f"Sharpe 변화 중앙값 {per}"] = x.dSharpe.median()
                r[f"MDD 개선 중앙값 {per}"] = x.dMDD.median()
                r[f"자산 수 {per}"] = len(x)
        rows[label(tag)] = r
    return pd.DataFrame(rows).T


def similar_diag(P) -> pd.DataFrame:
    """'유사 국면 가중'이 실제로 어느 시대 표본에 무게를 줬나(재학습 시점별 가중치 비중, %)."""
    st = P.X[A.STATE].to_numpy(np.float64)
    rows = {}
    for when in ("2000-01-01", "2008-01-01", "2009-01-01", "2020-01-01", "2022-01-01", "2023-01-01", "2026-01-01"):
        t = pd.Timestamp(when)
        tr = np.where((P.date < t - pd.Timedelta(days=A.PURGE_DAYS)) & (P.y >= 0))[0]
        now = P.X[A.STATE].iloc[np.searchsorted(P.date, t)].to_numpy(np.float64)
        w = A.scheme_weights("similar", P.date[tr], t, st[tr], now)
        dec = (P.date[tr].year // 10) * 10
        share = pd.Series(w).groupby(dec).sum() / w.sum()
        rows[when[:4] + " 재학습"] = {f"{k}년대": v for k, v in share.items()}
    return (pd.DataFrame(rows).T.fillna(0) * 100).round(0)


def main():
    P = pickle.load(open("data/krx/adaptive_panel.pkl", "rb"))
    preds = load_preds()
    print("예측 세트:", list(preds))
    auc = auc_table(P, preds)
    auc.to_csv("reports/adaptive_auc.csv")
    print(auc.round(3).to_string())
    R = pd.read_parquet("data/krx/research_returns.parquet")
    q = us_prices()["QQQ"].reindex(R.index).ffill()
    c = Ctx(R=R, P=pd.DataFrame({"QQQ": q}), vix=pd.Series(dtype=float), rf_us=pd.Series(0.0, index=R.index))
    qt, curves = qqq_strategies(preds, R, c)
    qt.to_csv("reports/adaptive_qqq.csv")
    print(qt.sort_values("Sharpe", ascending=False).round(3).to_string())
    assets = A.load_assets((1 + R["K200"]).cumprod())
    rb = asset_robustness(P, preds, assets)
    rb.to_csv("reports/adaptive_assets.csv")
    print(rb.round(3).to_string())
    sd = similar_diag(P)
    sd.to_csv("reports/adaptive_similar.csv")
    print(sd.to_string())
    pd.DataFrame(curves).to_parquet("data/krx/adaptive_qqq_curves.parquet")
    # TRIAD 에 넣으면? (사전 지정 후보: 전체 앙상블 × 틸트/게이트) + 전 후보 DSR
    import sys
    sys.path.insert(0, "scripts")
    from tournament_analysis import blend
    from autotrader.equity.evaluate import deflated_sharpe
    tour = pd.read_parquet("data/krx/tournament_equity.parquet")
    K, L = tour["MSAP-M(종목 70 + ETF RP 30)"], tour["레버리지 리스크패리티 18%+추세"]
    tri = {}
    for name in ("VT+ (규칙만)", "VT+ × ML틸트 [앙상블: 전체]", "VT+ × ML게이트 [앙상블: 전체]",
                 "VT+ × ML틸트 [앙상블: LightGBM 7방식]", "VT+ × ML틸트 [앙상블: 딥러닝(MLP+CNN)]"):
        if name not in curves:
            continue
        eq = blend(pd.DataFrame({"K": K, "V": curves[name], "R": L}), {"K": 1 / 3, "V": 1 / 3, "R": 1 / 3})
        a, i, o = stats(eq, "2001-02-01"), stats(eq, "2001-02-01", "2012-12-31"), stats(eq, "2013-01-01")
        tri["TRIAD · " + name] = {"CAGR": a["CAGR"], "Sharpe": a["Sharpe"], "MDD": a["MDD"], "최악의 해": a["WorstY"],
                                  "2001~12": i["CAGR"], "2013~": o["CAGR"], "2013~ Sharpe": o["Sharpe"]}
    T = pd.DataFrame(tri).T
    T.to_csv("reports/adaptive_triad.csv")
    print(T.round(3).to_string())
    # 기준 규칙 대비 초과수익(정보비율)과 다중검정 보정(DSR) — 'Sharpe>0' 이 아니라 '규칙보다 나은가'를 검정
    rows = []
    for b in ("VT", "VT+"):
        base_r = curves[f"{b} (규칙만)"].pct_change()
        for name in [k for k in curves if k.startswith(f"{b} ×")]:
            a_ = (curves[name].pct_change() - base_r).loc["2001-01-02":].dropna()
            ir = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
            rows.append({"base": b, "variant": name, "IR": ir(a_), "IR 2001~12": ir(a_.loc[:"2012"]),
                         "IR 2013~": ir(a_.loc["2013":])})
    Act = pd.DataFrame(rows)
    Act.to_csv("reports/adaptive_active_all.csv", index=False)
    out = {}
    for b, g in Act.groupby("base"):
        best = g.loc[g.IR.idxmax()]
        a_ = (curves[best.variant].pct_change() - curves[f"{b} (규칙만)"].pct_change()).loc["2001-01-02":].dropna()
        d = deflated_sharpe(a_, n_trials=len(g), trial_sharpes_std=g.IR.std(), periods_per_year=252)
        out[b] = {"변형 수": len(g), "초과수익 IR>0 비율": (g.IR > 0).mean(),
                  "두 구간 모두 IR>0 비율": ((g["IR 2001~12"] > 0) & (g["IR 2013~"] > 0)).mean(), "최고 IR": best.IR,
                  "최고 변형": best.variant, "최고 IR 2001~12": best["IR 2001~12"], "최고 IR 2013~": best["IR 2013~"],
                  "DSR(최고 초과수익)": d["dsr"]}
    pd.DataFrame(out).T.to_csv("reports/adaptive_active.csv")
    print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    main()
