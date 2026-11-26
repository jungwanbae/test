"""전설적 투자자 전략 토너먼트 — 같은 데이터·같은 비용·같은 체결 규칙으로 비교.

규칙: 신호는 t 일 종가까지의 정보로 계산 → t+1 일 종가에 매매(1거래일 지연). 원화 기준 일별 수익.
비용: 매매대금 대비 편도 0.10%(해외 ETF, NH 0.09%+스프레드), 국내 0.03%.
레버리지 상품은 일일 리셋 합성(금리·보수 차감) — data.build_returns 참조.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- 시뮬레이터
def simulate(R: pd.DataFrame, targets: pd.DataFrame, cost: float = 0.001, start: str | None = None,
             end: str | None = None) -> tuple[pd.Series, pd.Series]:
    """targets: 신호일 × 자산 비중(합 ≤ 1, 나머지는 원화 현금 아님 — 명시적으로 CASH 열을 넣어야 함).
    반환: (자산곡선, 회전율)."""
    days = R.index
    tg = targets.reindex(columns=R.columns).fillna(0.0)
    tg = tg[tg.index.isin(days)]
    pos = {d: i for i, d in enumerate(days)}
    execs = {}
    for d, row in tg.iterrows():
        i = pos[d] + 1
        if i < len(days):
            execs[i] = row.values
    s = int(np.searchsorted(days, pd.Timestamp(start))) if start else min(execs) if execs else 0
    e = int(np.searchsorted(days, pd.Timestamp(end), side="right")) - 1 if end else len(days) - 1
    Rv = R.values
    w = np.zeros(R.shape[1])
    val = 1.0
    eq, turn = np.empty(e - s + 1), {}
    started = False
    for k, i in enumerate(range(s, e + 1)):
        if started:
            g = 1 + Rv[i] @ w
            w = w * (1 + Rv[i]) / g if g != 0 else w
            val *= g
        if i in execs:
            tgt = execs[i]
            tv = np.abs(tgt - w).sum()
            val *= 1 - tv * cost
            w = tgt.copy()
            turn[days[i]] = tv
            started = True
        eq[k] = val
    idx = days[s: e + 1]
    out = pd.Series(eq, index=idx)
    first = next((d for d in turn), idx[0])
    return out.loc[first:], pd.Series(turn)


def stats(eq: pd.Series, start=None, end=None) -> dict:
    e = eq.loc[start:end].dropna()
    if len(e) < 60:
        return {}
    e = e / e.iloc[0]
    r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    mdd = (e / e.cummax() - 1).min()
    yr = e.resample("YE").last().pct_change().dropna()
    m = e.resample("ME").last()
    r5 = (m / m.shift(60)) ** (1 / 5) - 1
    return {"CAGR": cagr, "Vol": r.std() * np.sqrt(252), "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "MDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "WorstY": yr.min() if len(yr) else np.nan,
            "Roll5Min": r5.min(), "Years": yrs}


def after_tax(eq: pd.Series, overseas_share: float = 1.0, rate: float = 0.22) -> pd.Series:
    """해외 계좌 양도세 근사: 매년 플러스 수익의 22%(손실 이월 없음, 기본공제 250만원 무시 → 보수적)."""
    ye = eq.resample("YE").last()
    out = eq.copy()
    adj = 1.0
    prev = eq.iloc[0]
    for d, v in ye.items():
        gain = v / prev - 1
        tax = rate * overseas_share * max(gain, 0.0)
        mask = out.index > d
        adj *= 1 - tax / (1 + gain) if gain > -1 else 1
        out[mask] = eq[mask] * adj
        prev = v
    return out


# ---------------------------------------------------------------------- 신호 도우미
def month_ends(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby(days.to_period("M")).max().values)


def mom13612(P: pd.DataFrame, d) -> pd.Series:
    p = P.loc[:d]
    if len(p) < 253:
        return pd.Series(np.nan, index=P.columns)
    last = p.iloc[-1]
    r = lambda n: last / p.iloc[-1 - n] - 1  # noqa: E731
    return (12 * r(21) + 4 * r(63) + 2 * r(126) + r(252)) / 19


def ret_n(P: pd.DataFrame, d, n: int) -> pd.Series:
    p = P.loc[:d]
    return p.iloc[-1] / p.iloc[-1 - n] - 1 if len(p) > n else pd.Series(np.nan, index=P.columns)


def lev_mix(base: str, E: float, cash: str, maxL: int = 3) -> dict:
    """노출 E(0~maxL)를 1배·maxL배 상품과 현금 조합으로 구현(자본 합 = 1)."""
    E = float(np.clip(E, 0, maxL))
    if E <= 1:
        return {base: E, cash: 1 - E}
    w3 = (E - 1) / (maxL - 1)
    return {f"{base}x{maxL}": w3, base: 1 - w3}


def implement(E: dict, Ls: dict, cash: str = "CASH_USD") -> dict:
    """자산별 노출 E 를 자본 합 1 이내로 구현: 1배 ETF 우선, 총노출이 100% 를 넘는 부분만 L배 상품으로."""
    assets = list(E)
    e = np.array([max(E[a], 0.0) for a in assets])
    L = np.array([Ls.get(a, 1) for a in assets], float)
    tot = e.sum()
    lw = e / L
    f = 0.0
    if tot > 1:
        denom = tot - lw.sum()
        f = min(1.0, (tot - 1) / denom) if denom > 0 else 1.0
    u, l_ = e * (1 - f), lw * f
    ssum = u.sum() + l_.sum()
    if ssum > 1:
        u, l_ = u / ssum, l_ / ssum
    w: dict = {}
    for j, a in enumerate(assets):
        if L[j] == 1:
            w[a] = w.get(a, 0) + u[j] + l_[j]
        else:
            w[a] = w.get(a, 0) + u[j]
            w[f"{a}x{int(L[j])}"] = w.get(f"{a}x{int(L[j])}", 0) + l_[j]
    w[cash] = max(0.0, 1 - sum(w.values()))
    return w


@dataclass
class Spec:
    name: str
    category: str
    build: Callable
    overseas_share: float = 1.0
    note: str = ""
    start: str = "2001-01-02"
    cost: float = 0.001


@dataclass
class Ctx:
    R: pd.DataFrame
    P: pd.DataFrame          # 신호용 가격(USD/원화)
    vix: pd.Series
    rf_us: pd.Series
    days: pd.DatetimeIndex = field(init=False)
    me: pd.DatetimeIndex = field(init=False)

    def __post_init__(self):
        self.days = self.R.index
        self.me = month_ends(self.days)


def fixed(weights: dict) -> Callable:
    def b(c: Ctx):
        return pd.DataFrame([weights] * len(c.me), index=c.me)
    return b


# ---------------------------------------------------------------------- B. 추세·모멘텀
def faber_gtaa(c: Ctx):
    assets = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    rows = {}
    for d in c.me:
        p = c.P[assets].loc[:d]
        if len(p) < 210 or p.iloc[-1].isna().any():
            continue
        sma = p.iloc[-210:].mean()
        on = p.iloc[-1] > sma
        w = {a: 0.2 for a in assets if on[a]}
        w["CASH_USD"] = 1 - sum(w.values())
        rows[d] = w
    return pd.DataFrame(rows).T


def antonacci_gem(c: Ctx):
    rows = {}
    for d in c.me:
        r12 = ret_n(c.P[["SPY", "EFA", "IEF"]], d, 252)
        cash12 = (1 + c.rf_us.loc[:d].iloc[-252:].mean()) - 1
        if r12.isna().any():
            continue
        if r12["SPY"] > cash12:
            rows[d] = {"SPY" if r12["SPY"] >= r12["EFA"] else "EFA": 1.0}
        else:
            rows[d] = {"IEF": 1.0}
    return pd.DataFrame(rows).T


def keller_haa(c: Ctx):
    off = ["SPY", "IWM", "EFA", "EEM", "VNQ", "DBC", "IEF", "TLT"]
    rows = {}
    for d in c.me:
        m = mom13612(c.P[off + ["TIP"]], d)
        if m[["TIP"] + off].isna().any():
            continue
        cash_m = mom13612(c.P[["SHY"]], d)["SHY"]
        defensive = "IEF" if m["IEF"] > cash_m else "CASH_USD"
        if m["TIP"] < 0:
            rows[d] = {defensive: 1.0}
            continue
        top = m[off].sort_values(ascending=False).index[:4]
        w = {}
        for a in top:
            k = a if m[a] > 0 else defensive
            w[k] = w.get(k, 0) + 0.25
        rows[d] = w
    return pd.DataFrame(rows).T


def golden_cross(base="QQQ", L=3):
    def b(c: Ctx):
        p = c.P[base]
        on = p.rolling(50).mean() > p.rolling(200).mean()
        chg = on.ne(on.shift()) | c.days.isin(c.me)
        rows = {}
        for d in c.days[chg.values & p.rolling(200).mean().notna().values]:
            rows[d] = {f"{base}x{L}" if L > 1 else base: 1.0} if on.loc[d] else {"CASH_USD": 1.0}
        return pd.DataFrame(rows).T
    return b


def turtle(c: Ctx):
    assets = ["SPY", "QQQ", "GLD", "TLT"]
    rows, state = {}, {a: False for a in assets}
    hi55 = c.P[assets].rolling(55).max().shift(1)
    lo20 = c.P[assets].rolling(20).min().shift(1)
    for d in c.days[260:]:
        p = c.P[assets].loc[d]
        changed = False
        for a in assets:
            if not state[a] and p[a] > hi55.loc[d, a]:
                state[a], changed = True, True
            elif state[a] and p[a] < lo20.loc[d, a]:
                state[a], changed = False, True
        if changed or d in c.me:
            w = {a: 0.25 for a in assets if state[a]}
            w["CASH_USD"] = 1 - sum(w.values())
            rows[d] = w
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------- C. 변동성·리스크패리티
def vol_target(base="QQQ", target=0.25, maxL=3):
    def b(c: Ctx):
        r = c.P[base].pct_change()
        vol = r.rolling(21).std() * np.sqrt(252)
        rows = {}
        for d in c.me:
            v = vol.loc[d]
            if not np.isfinite(v):
                continue
            rows[d] = lev_mix(base, min(maxL, target / v), "CASH_USD", maxL)
        return pd.DataFrame(rows).T
    return b


def lev_risk_parity(target=0.18, maxL=3, trend=True):
    """레버리지 리스크패리티(브리지워터·AQR 계열): 역변동성 비중 → 목표변동성까지 확대, 추세 이탈 자산 제외."""
    assets = ["QQQ", "TLT", "GLD", "DBC"]
    Ls = {"QQQ": 3, "TLT": 3, "GLD": 2, "DBC": 1}

    def b(c: Ctx):
        r = c.P[assets].pct_change()
        rows = {}
        for d in c.me:
            win = r.loc[:d].iloc[-126:]
            if len(win) < 126 or win.isna().any().any():
                continue
            vol = win.iloc[-63:].std() * np.sqrt(252)
            cov = np.outer(vol, vol) * win.corr().values
            w = (1 / vol) / (1 / vol).sum()
            if trend:
                r12 = ret_n(c.P[assets], d, 252)
                w = w.where(r12 > c.rf_us.loc[d], 0.0)
            pv = np.sqrt(w.values @ cov @ w.values)
            if pv <= 0:
                rows[d] = {"CASH_USD": 1.0}
                continue
            E = (w * target / pv).clip(upper=pd.Series({a: min(maxL, Ls[a]) for a in assets}))
            rows[d] = implement(E.to_dict(), Ls)
        return pd.DataFrame(rows).T
    return b


# ---------------------------------------------------------------------- D. 역발상(낙폭 매수·레버리지)
def dd_ladder(base="QQQ", ladder=((-0.2, 1.5), (-0.3, 2.0), (-0.4, 3.0)), reset=-0.10, maxL=3, cash="CASH_USD",
              confirm_ma: int | None = None):
    """고점 대비 낙폭이 깊어질수록 노출을 단계적으로 늘리고(물타기·피라미딩), 낙폭이 reset 이내로 회복하면 1배로.
    confirm_ma: 지정하면 '낙폭 구간이면서 종가가 N일선 위(하락이 멈춘 확인)'일 때만 레버리지를 늘린다."""
    def b(c: Ctx):
        p = c.P[base]
        dd = p / p.cummax() - 1
        ma = p.rolling(confirm_ma).mean() if confirm_ma else None
        rows, cur = {}, None
        for d in c.days[252:]:
            x = dd.loc[d]
            if not np.isfinite(x):
                continue
            lvl = 1.0
            for th, L in ladder:
                if x <= th:
                    lvl = L
            cur0 = 1.0 if cur is None else cur
            if x >= reset:
                target = 1.0
            elif confirm_ma and not (p.loc[d] > ma.loc[d]):
                target = cur0                      # 확인 전에는 늘리지 않음(줄이지도 않음)
            else:
                target = max(cur0, lvl)            # 떨어질수록 더 산다, 회복 전까지 유지
            if cur is None or target != cur or d in c.me:
                rows[d] = lev_mix(base, target, cash, maxL)
                cur = target
        return pd.DataFrame(rows).T
    return b


def vix_spike(base="SPY", on=35, off=20, L=2):
    """VIX 가 on 이상(극단적 공포)이면 L배로 확대, off 이하로 진정되면 1배로."""
    def b(c: Ctx):
        rows, active, last = {}, False, None
        for d in c.days[252:]:
            v = c.vix.loc[d]
            if np.isfinite(v):
                if not active and v >= on:
                    active = True
                elif active and v <= off:
                    active = False
            if active != last or d in c.me:
                rows[d] = {f"{base}x{L}": 1.0} if active else {base: 1.0}
                last = active
        return pd.DataFrame(rows).T
    return b


# ---------------------------------------------------------------------- F. 사용자 기존 전략(근사 재현)
def hr_replica(c: Ctx):
    """'고금리 대비형' 근사: 나스닥 50%(GC 30 + VT 20), 금 30%(200일선), 코스피 20%(추세 점수 단계),
    T-bill ≥ 3.5% 이면 나스닥 최대 2배·금·코스피 최대 1배. (7팩터 틸트는 단순화)"""
    q, g, k = c.P["QQQ"], c.P["GLD"], c.P["K200"]
    q50, q200, g200 = q.rolling(50).mean(), q.rolling(200).mean(), g.rolling(200).mean()
    qvol = q.pct_change().rolling(21).std() * np.sqrt(252)
    k200, k50 = k.rolling(200).mean(), k.rolling(50).mean()
    rows = {}
    for d in c.me:
        if not np.isfinite(q200.loc[d]) or not np.isfinite(k200.loc[d]) or not np.isfinite(g200.loc[d]):
            continue
        cap = c.rf_us.loc[d] >= 0.035
        capq, capo = (2, 1) if cap else (3, 2)
        w = {}

        def add(dct, s):
            for kk, vv in dct.items():
                w[kk] = w.get(kk, 0) + vv * s
        Lgc = min(3 if q50.loc[d] > q200.loc[d] else 0, capq)
        add(lev_mix("QQQ", Lgc, "CASH_USD", 3), 0.30)
        Lvt = min(3 * min(1.0, 0.5 / (3 * qvol.loc[d])), capq)
        add(lev_mix("QQQ", Lvt, "CASH_USD", 3), 0.20)
        Lg = min(2 if g.loc[d] > g200.loc[d] else 0, capo)
        add(lev_mix("GLD", Lg, "CASH_USD", 2), 0.30)
        score = (0.5 * np.sign(k.loc[d] / k200.loc[d] - 1) + 0.3 * np.sign(k50.loc[d] / k200.loc[d] - 1)
                 + 0.2 * np.sign(ret_n(c.P[["K200"]], d, 252)["K200"]))
        Lk = 2.0 if score >= 0.5 else 1.5 if score >= 0.2 else 1.0 if score >= -0.2 else 0.5 if score >= -0.5 else 0.0
        Lk = min(Lk, capo)
        add({"K200x2": Lk - 1, "K200": 2 - Lk} if Lk >= 1 else {"K200": Lk, "CASH_KRW": 1 - Lk}, 0.20)
        rows[d] = w
    return pd.DataFrame(rows).T


def aggressive3_replica(c: Ctx):
    """'3차 공격형' 근사: QQQ·TLT·GLD·DBC, 12개월 추세(현금 대비), 역변동성(QQQ 2배 가중), 목표변동성 40%,
    노출 상한 QQQ3·TLT3·GLD2·DBC1, 4개 구성원 평균. (A/B 반쪽 리밸런싱은 월말 하나로 단순화)"""
    caps = {"QQQ": 3.0, "TLT": 3.0, "GLD": 2.0, "DBC": 1.0}
    members = [(["QQQ", "TLT", "GLD"], False), (["QQQ", "TLT", "GLD"], True),
               (["QQQ", "TLT", "GLD", "DBC"], False), (["QQQ", "TLT", "GLD", "DBC"], True)]
    r = c.P[list(caps)].pct_change()
    rows = {}
    for d in c.me:
        win = r.loc[:d]
        if len(win) < 260 or win.iloc[-126:].isna().any().any():
            continue
        Es = []
        for assets, renorm in members:
            r12 = ret_n(c.P[assets], d, 252)
            rc = c.rf_us.loc[d]
            s_ = (r12 > rc).astype(float)
            sig = win[assets].iloc[-63:].std() * np.sqrt(252)
            cov = np.outer(sig, sig) * win[assets].iloc[-126:].corr().values
            raw = (1 / sig) / (1 / sig).sum()
            raw = raw * np.array([2.0 if a == "QQQ" else 1.0 for a in assets])
            raw = raw / raw.sum()
            if renorm:
                ww = raw * (s_ > 0)
                if ww.sum() > 0:
                    ww = ww / ww.sum()
                    e = ww * s_ * 0.40 / np.sqrt(ww.values @ cov @ ww.values)
                else:
                    e = ww * 0
            else:
                e = raw * s_ * 0.40 / np.sqrt(raw.values @ cov @ raw.values)
            Es.append(e.clip(upper=pd.Series(caps)[assets]))
        E = pd.concat(Es, axis=1).fillna(0).mean(axis=1).reindex(list(caps)).fillna(0)
        rows[d] = implement(E.to_dict(), {"QQQ": 3, "TLT": 3, "GLD": 2, "DBC": 1})
    return pd.DataFrame(rows).T


def aquila_replica(c: Ctx):
    """'AQUILA-X' 근사: HAA 50% + GC 25%(QQQ 50>200 → TQQQ, 아니면 현금) + VT 25%(TQQQ 비중 = 0.1/QQQ 20일 변동성)."""
    haa = keller_haa(c)
    q = c.P["QQQ"]
    gc = q.rolling(50).mean() > q.rolling(200).mean()
    vol = q.pct_change().rolling(20).std() * np.sqrt(252)
    rows = {}
    for d, hrow in haa.iterrows():
        w = {k: v * 0.5 for k, v in hrow.dropna().items() if v > 0}
        w["QQQx3" if gc.loc[d] else "CASH_USD"] = w.get("QQQx3" if gc.loc[d] else "CASH_USD", 0) + 0.25
        vt = float(np.clip(round(0.1 / vol.loc[d], 1), 0, 1)) if np.isfinite(vol.loc[d]) else 0
        w["QQQx3"] = w.get("QQQx3", 0) + 0.25 * vt
        w["CASH_USD"] = w.get("CASH_USD", 0) + 0.25 * (1 - vt)
        rows[d] = w
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------- 등록부
def registry() -> list[Spec]:
    S = []
    # A. 수동형(전설적 자산배분)
    S += [Spec("보글: S&P500 보유", "A 수동형", fixed({"SPY": 1})),
          Spec("나스닥100 보유", "A 수동형", fixed({"QQQ": 1})),
          Spec("KOSPI200 보유", "A 수동형", fixed({"K200": 1}), overseas_share=0, cost=0.0003),
          Spec("60/40(주식/중기채)", "A 수동형", fixed({"SPY": .6, "IEF": .4})),
          Spec("버핏 90/10", "A 수동형", fixed({"SPY": .9, "SHY": .1}), note="버핏 유언장 권고"),
          Spec("달리오 올웨더", "A 수동형", fixed({"SPY": .3, "TLT": .4, "IEF": .15, "GLD": .075, "DBC": .075})),
          Spec("브라운 영구포트폴리오", "A 수동형", fixed({"SPY": .25, "TLT": .25, "GLD": .25, "SHY": .25})),
          Spec("골든 버터플라이", "A 수동형", fixed({"SPY": .2, "IWM": .2, "TLT": .2, "SHY": .2, "GLD": .2}))]
    # B. 추세·모멘텀
    S += [Spec("페이버 GTAA5(10개월선)", "B 추세·모멘텀", faber_gtaa),
          Spec("안토나치 듀얼모멘텀 GEM", "B 추세·모멘텀", antonacci_gem),
          Spec("켈러 HAA", "B 추세·모멘텀", keller_haa, start="2005-01-03"),
          Spec("골든크로스 QQQ 1배", "B 추세·모멘텀", golden_cross("QQQ", 1)),
          Spec("골든크로스 TQQQ(3배)", "B 추세·모멘텀", golden_cross("QQQ", 3)),
          Spec("터틀 돌파(55/20일)", "B 추세·모멘텀", turtle)]
    # C. 변동성·리스크패리티
    S += [Spec("변동성 타기팅 QQQ 25%(최대 3배)", "C 변동성·RP", vol_target("QQQ", 0.25, 3)),
          Spec("레버리지 리스크패리티 18%+추세", "C 변동성·RP", lev_risk_parity(0.18, 3, True))]
    # D. 역발상
    S += [Spec("역발상 낙폭 사다리 QQQ(−20/−30/−40% → 1.5/2/3배)", "D 역발상", dd_ladder("QQQ")),
          Spec("역발상 낙폭 사다리 S&P500", "D 역발상", dd_ladder("SPY")),
          Spec("역발상 낙폭 사다리 KOSPI200(최대 2배)", "D 역발상",
               dd_ladder("K200", ((-0.2, 1.5), (-0.3, 2.0)), maxL=2, cash="CASH_KRW"), overseas_share=0, cost=0.0003),
          Spec("역발상+반등확인 QQQ(낙폭 & 50일선 위)", "D 역발상", dd_ladder("QQQ", confirm_ma=50)),
          Spec("VIX 35 공포 매수 S&P500 2배", "D 역발상", vix_spike("SPY", 35, 20, 2))]
    # F. 사용자 기존 전략
    S += [Spec("[기존] 고금리 대비형(근사)", "F 기존 전략", hr_replica, overseas_share=0.8),
          Spec("[기존] 3차 공격형(근사)", "F 기존 전략", aggressive3_replica),
          Spec("[기존] AQUILA-X(근사)", "F 기존 전략", aquila_replica, start="2005-01-03")]
    return S


# ---------------------------------------------------------------------- D'. 안전한 역발상 변형
def dip_in_uptrend(base="QQQ", dip=-0.08, L_dip=2.0, maxL=3, cash="CASH_USD"):
    """강세장 눌림목: 200일선 위(상승추세)에서 52주 고점 대비 dip 이하로 눌리면 L_dip 배, 평소 1배,
    200일선 아래(하락추세)면 현금. — '두려울 때 사되, 추세가 살아 있을 때만'."""
    def b(c: Ctx):
        p = c.P[base]
        ma200 = p.rolling(200).mean()
        dd52 = p / p.rolling(252, min_periods=200).max() - 1
        rows, last = {}, None
        for d in c.days[260:]:
            if not np.isfinite(ma200.loc[d]):
                continue
            if p.loc[d] < ma200.loc[d]:
                E = 0.0
            elif dd52.loc[d] <= dip:
                E = L_dip
            else:
                E = 1.0
            if E != last or d in c.me:
                rows[d] = lev_mix(base, E, cash, maxL)
                last = E
        return pd.DataFrame(rows).T
    return b


def ladder_with_stop(base="QQQ", ladder=((-0.2, 1.5), (-0.3, 2.0), (-0.4, 3.0)), reset=-0.10, stop=-0.25, maxL=3,
                     cash="CASH_USD"):
    """역발상 사다리 + 손절: 레버리지 진입 이후 기초자산이 추가로 stop 만큼 더 빠지면 1배로 복귀하고,
    새 고점을 회복할 때까지 다시 레버리지를 쓰지 않는다(파산 위험 차단)."""
    def b(c: Ctx):
        p = c.P[base]
        dd = p / p.cummax() - 1
        rows, cur, entry, banned = {}, 1.0, None, False
        for d in c.days[252:]:
            x = dd.loc[d]
            if not np.isfinite(x):
                continue
            if x >= -0.001:
                banned = False
            lvl = 1.0
            for th, L in ladder:
                if x <= th:
                    lvl = L
            target = cur
            if x >= reset:
                target, entry = 1.0, None
            elif not banned:
                if lvl > cur:
                    target = lvl
                    entry = entry or p.loc[d]
                if entry and p.loc[d] <= entry * (1 + stop):
                    target, entry, banned = 1.0, None, True
            if target != cur or d in c.me:
                rows[d] = lev_mix(base, target, cash, maxL)
                cur = target
        return pd.DataFrame(rows).T
    return b


def recovery_boost(base="QQQ", target=0.25, crash=-0.30, boost_L=3.0, hold=252, maxL=3, cash="CASH_USD"):
    """변동성 타기팅 + '확인된 역발상': 최근 2년(504거래일) 안에 사상 최고가 대비 crash 이하까지 빠진 적이 있고
    (깊은 낙폭 구간), 골든크로스(50일선이 200일선을 새로 상향 돌파)가 나오면 hold 거래일 동안 노출을 boost_L 로
    올린다(골든크로스가 깨지면 해제). 폭락 '도중'이 아니라 '추세 회복 확인 후' 레버리지를 키우는 방식.
    QQQ 기준 발동: 2003(2회)·2004·2005·2006·2008.06·2009·2010·2011·2013·2023 — 2000년 고점 대비 −30% 아래에
    오래 머문 2003~2013년에 주로 발동했다."""
    def b(c: Ctx):
        p = c.P[base]
        r = p.pct_change()
        vol = r.rolling(21).std() * np.sqrt(252)
        dd = p / p.cummax() - 1
        crashed = dd.rolling(504, min_periods=1).min() <= crash
        gc = p.rolling(50).mean() > p.rolling(200).mean()
        start_gc = gc & ~gc.shift(1, fill_value=False) & crashed
        rows, boost_until = {}, None
        for i, d in enumerate(c.days):
            if i < 260 or not np.isfinite(vol.loc[d]) or vol.loc[d] <= 0:
                continue
            if start_gc.loc[d]:
                boost_until = i + hold
            E = min(target / vol.loc[d], maxL)
            if boost_until is not None and i <= boost_until and gc.loc[d]:
                E = max(E, boost_L)
            if d in c.me or start_gc.loc[d] or (boost_until is not None and i == boost_until + 1):
                rows[d] = lev_mix(base, E, cash, maxL)
        return pd.DataFrame(rows).T
    return b


def safe_contrarian_specs() -> list[Spec]:
    return [
        Spec("역발상·안전형: 강세장 눌림목 QQQ(200일선 위 & −8% → 2배)", "D' 안전한 역발상", dip_in_uptrend("QQQ")),
        Spec("역발상·안전형: 강세장 눌림목 S&P500", "D' 안전한 역발상", dip_in_uptrend("SPY")),
        Spec("역발상·안전형: 고정비중 리밸런싱 TQQQ 1/3 + 중기채 2/3", "D' 안전한 역발상",
             fixed({"QQQx3": 1 / 3, "IEF": 2 / 3})),
        Spec("역발상·안전형: 낙폭 사다리 + 추가하락 −25% 손절 QQQ", "D' 안전한 역발상", ladder_with_stop("QQQ")),
        Spec("역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%", "D' 안전한 역발상", recovery_boost("QQQ")),
    ]
