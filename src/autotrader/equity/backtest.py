"""종목 포트폴리오 일별 시뮬레이터.

체결 규칙(실거래와 동일): t 일 종가로 신호 → t+1 일 '시가'에 매매
  * 전일 보유분은 시가갭(전일종가→시가)을 먼저 반영한 뒤 시가로 리밸런싱, 이후 시가→종가 수익
  * 거래정지(당일 수익률 결측) 종목은 매매 불가 → 기존 보유 유지
  * 상장폐지(이후 데이터 없음): 마지막 가격에서 haircut 만큼 손실 처리 후 현금화
비용: 증권사 수수료(NH PLUG 0.01%+유관기관 0.0036%) + 슬리피지(편도) + 매도 시 증권거래세(연도별 세율)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from autotrader.equity.panel import Panel

# 증권거래세(+농특세) 합계, 코스피·코스닥 공통 근사. (시행일, 세율)
SELL_TAX = ((date(1900, 1, 1), 0.0030), (date(2019, 6, 3), 0.0025), (date(2021, 1, 1), 0.0023),
            (date(2023, 1, 1), 0.0020), (date(2024, 1, 1), 0.0018), (date(2025, 1, 1), 0.0015),
            (date(2026, 1, 1), 0.0020))


@dataclass
class StockCosts:
    """slippage_model: fixed → 편도 slippage 고정 / sqrt → base + k/√(60일 평균 거래대금/1억)
    (예: 거래대금 100억 0.08%, 10억 0.145%, 2억 0.26%)."""
    commission: float = 0.000136
    slippage: float = 0.002
    delist_haircut: float = 0.3
    sell_tax_override: float | None = None
    slippage_model: str = "sqrt"
    slip_base: float = 0.0005
    slip_k: float = 0.003
    # 운용금액 기반 시장충격(capital 지정 시): 충격 = impact_k·σ일·√(주문금액/20일 평균 거래대금)
    impact_k: float = 0.8
    adv_cap: float = 0.10         # 하루 주문 ≤ 거래대금의 10% (초과분은 미체결 → 다음 리밸런싱/부분체결)

    def sell_tax(self, d: date) -> float:
        if self.sell_tax_override is not None:
            return self.sell_tax_override
        rate = SELL_TAX[0][1]
        for start, r in SELL_TAX:
            if d >= start:
                rate = r
        return rate


@dataclass
class PortfolioRules:
    top_n: int = 30
    buffer: float = 1.5           # 보유 종목은 순위가 top_n×buffer 이내면 유지(회전율 절감)
    weighting: str = "equal"      # equal | inv_vol
    max_weight: float = 0.08
    regime: str | None = None     # None | ma200 | breadth | dual | voltarget | voltarget_ma
    risk_off_exposure: float = 0.0
    target_mkt_vol: float = 0.20  # voltarget: 노출 = clip(목표 / 시장 20일 변동성, 하한, 1)
    min_trade: float = 0.002      # 자산 대비 0.2% 미만 차이는 매매 생략
    cash_for_missing: bool = False  # True: 후보가 top_n 보다 적으면 빈 슬롯은 현금(1/top_n 고정 비중)


@dataclass
class SimResult:
    equity: pd.Series
    exposure: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)
    costs_paid: float = 0.0


def regime_exposure(row: pd.Series, rules: PortfolioRules) -> float:
    if rules.regime is None:
        return 1.0
    on_ma = row.get("m_ma200", 0.0) > 0
    br = row.get("breadth", 0.5)
    if rules.regime == "ma200":
        return 1.0 if on_ma else rules.risk_off_exposure
    if rules.regime == "breadth":
        return 1.0 if br > 0.5 else (0.5 if br > 0.35 else rules.risk_off_exposure)
    if rules.regime in ("voltarget", "voltarget_ma"):
        mv = float(row.get("m_vol_20", rules.target_mkt_vol) or rules.target_mkt_vol)
        e = float(np.clip(rules.target_mkt_vol / max(mv, 1e-6), rules.risk_off_exposure, 1.0))
        if rules.regime == "voltarget_ma" and not on_ma:
            e *= 0.6
        return e
    if rules.regime == "dual":
        if on_ma and br > 0.45:
            return 1.0
        if on_ma or br > 0.45:
            return 0.5 + rules.risk_off_exposure / 2
        return rules.risk_off_exposure
    raise ValueError(rules.regime)


def make_targets(ds: pd.DataFrame, score: str, rules: PortfolioRules,
                 signal_dates: pd.DatetimeIndex | None = None) -> dict[pd.Timestamp, pd.Series]:
    """신호일별 목표비중(합 = 국면 노출도)."""
    out: dict[pd.Timestamp, pd.Series] = {}
    held: list[str] = []
    for d, full in ds.groupby("date"):
        if signal_dates is not None and d not in signal_dates:
            continue
        df = full.dropna(subset=[score])
        if df.empty:                       # 통과 종목이 없으면 전량 현금(이전 보유도 매도)
            out[d] = pd.Series(dtype=float)
            held = []
            continue
        df = df.sort_values(score, ascending=False)
        order = list(df["code"])
        pos = {c: i for i, c in enumerate(order)}
        keep = [c for c in held if c in pos and pos[c] < rules.top_n * rules.buffer]
        keep = sorted(keep, key=lambda c: pos[c])[: rules.top_n]
        sel = keep + [c for c in order if c not in set(keep)][: rules.top_n - len(keep)]
        sub = df.set_index("code").loc[sel]
        if rules.weighting == "inv_vol":
            v = sub["raw_vol_60"].clip(lower=0.005)
            w = (1 / v) / (1 / v).sum()
        else:
            w = pd.Series(1.0 / len(sel), index=sel)
        w = w.clip(upper=rules.max_weight)
        w = w / w.sum()
        if rules.cash_for_missing and len(sel) < rules.top_n:
            w = w * len(sel) / rules.top_n
        exp = regime_exposure(df.iloc[0], rules)
        out[d] = w * exp
        held = sel
    return out


def simulate(p: Panel, targets: dict[pd.Timestamp, pd.Series], costs: StockCosts | None = None,
             start: str | None = None, end: str | None = None, cash_rate: float = 0.0,
             min_trade: float = 0.002, stop_loss: float | None = None,
             breakeven_after: float | None = None, capital: float | None = None,
             stop_mode: str = "intraday") -> SimResult:
    """stop_loss: 평균 매입가 대비 하락률(예 0.08) — 장중 저가가 닿으면 손절가(갭하락이면 시가)로 청산,
    다음 리밸런싱까지 현금 보유(미너비니식 손실 제한). breakeven_after: 이 수익률 도달 후 손절선을 본전으로.
    stop_mode: intraday(장중 저가 터치 시 손절가 체결 — 증권사 스톱지정가 주문) |
               close(종가가 손절선 아래면 다음날 시가 매도 — 하루 1회 실행하는 데몬과 동일)."""
    costs = costs or StockCosts()
    dates = p.dates
    R = p.ret.values
    Gp = p.gap.values
    LO = p.lo.values if (stop_loss and p.lo is not None) else None
    if costs.slippage_model == "sqrt":
        adv = p.amount.rolling(60, min_periods=10).mean().values
        SL = costs.slip_base + costs.slip_k / np.sqrt(np.clip(np.nan_to_num(adv, nan=1e7) / 1e8, 0.05, None))
    else:
        SL = None
    ADV = SIG = None
    if capital:
        ADV = p.amount.rolling(20, min_periods=10).mean().values
        SIG = p.ret.rolling(20, min_periods=15).std().values

    def slip_of(t: int, j: int, value: float, equity_now: float = 1.0) -> float:
        """value: 시뮬레이터 단위 주문금액. 운용금액을 capital 로 '고정'한 원화 주문 크기로 환산
        (복리로 자산이 불어나도 같은 운용 규모에서의 체결비용을 본다)."""
        base = SL[t - 1, j] if SL is not None else costs.slippage
        if ADV is None:
            return base
        order = value / max(equity_now, 1e-12) * capital  # 원화 주문금액
        adv, sg = ADV[t - 1, j], SIG[t - 1, j]
        adv = adv if np.isfinite(adv) and adv > 0 else 1e8
        sg = sg if np.isfinite(sg) and sg > 0 else 0.03
        return max(base, costs.slip_base + costs.impact_k * sg * np.sqrt(order / adv))

    def cap_of(t: int, j: int, equity_now: float = 1.0) -> float:
        """하루 체결 가능한 최대 주문금액(시뮬레이터 단위) = 거래대금 × adv_cap 을 운용금액 비율로 환산."""
        if ADV is None:
            return np.inf
        adv = ADV[t - 1, j]
        adv = adv if np.isfinite(adv) and adv > 0 else 0.0
        return costs.adv_cap * adv / capital * equity_now

    col = {c: i for i, c in enumerate(p.ret.columns)}
    last_valid = (~np.isnan(R)).cumsum(axis=0).argmax(axis=0)  # 종목별 마지막 관측 위치
    pos_of = {d: i for i, d in enumerate(dates)}
    execs: dict[int, pd.Series] = {}
    for d, w in targets.items():
        i = pos_of.get(pd.Timestamp(d))
        if i is not None and i + 1 < len(dates):
            execs[i + 1] = w
    s = pos_of[dates[dates >= pd.Timestamp(start)][0]] if start else (min(execs) if execs else 0)
    e = pos_of[dates[dates <= pd.Timestamp(end)][-1]] if end else len(dates) - 1
    cr = (1 + cash_rate) ** (1 / 252) - 1
    positions: dict[int, float] = {}
    basis: dict[int, float] = {}   # 평균 매입가 기준 원금(가치 / 매입가 대비 배율)
    peak: dict[int, float] = {}
    cash = 1.0
    eq, ex, tv = [], [], {}
    paid = 0.0
    holdings = {}
    stops = 0
    pending_exit: set[int] = set()
    for t in range(s, e + 1):
        d = dates[t].date()
        tax = costs.sell_tax(d)
        # 상장폐지 처리
        for j in [j for j in positions if np.isnan(R[t, j]) and t > last_valid[j]]:
            cash += positions.pop(j) * (1 - costs.delist_haircut)
            basis.pop(j, None)
            peak.pop(j, None)
        # 시가 반영(전일종가 → 시가)
        for j in positions:
            g = Gp[t, j]
            if not np.isnan(g):
                positions[j] *= 1 + g
        for j in [j for j in pending_exit if j in positions and not np.isnan(R[t, j])]:   # close 모드 손절 체결
            value = positions.pop(j)
            slip = slip_of(t, j, value, cash + sum(positions.values()) + value) * 1.5
            fee = value * (costs.commission + slip + tax)
            cash += value - fee
            paid += fee
            basis.pop(j, None)
            peak.pop(j, None)
            stops += 1
        pending_exit = {j for j in pending_exit if j in positions}
        if t in execs:
            equity = cash + sum(positions.values())
            tgt = {col[c]: float(w) * equity for c, w in execs[t].items() if c in col}
            traded = 0.0
            for j in list(set(positions) | set(tgt)):     # 매도 먼저
                cur = positions.get(j, 0.0)
                delta = tgt.get(j, 0.0) - cur
                if np.isnan(R[t, j]) or abs(delta) < 1e-12:
                    continue  # 거래정지 → 매매 불가
                if delta < 0 and (-delta >= costs_min(equity, min_trade) or tgt.get(j, 0.0) == 0.0):
                    sell = min(-delta, cap_of(t, j, equity))   # 유동성 한도 초과분은 미체결
                    slip = slip_of(t, j, sell, equity)
                    fee = sell * (costs.commission + slip + tax)
                    if j in basis and cur > 0:
                        basis[j] *= (cur - sell) / cur
                    positions[j] = cur - sell
                    cash += sell - fee
                    paid += fee
                    traded += sell
                    if positions[j] <= 1e-12:
                        positions.pop(j)
                        basis.pop(j, None)
                        peak.pop(j, None)
            buys = {j: min(tgt[j] - positions.get(j, 0.0), cap_of(t, j, equity)) for j in tgt
                    if not np.isnan(R[t, j]) and tgt[j] - positions.get(j, 0.0) >= costs_min(equity, min_trade)}
            buys = {j: b for j, b in buys.items() if b > 0}
            slips = {j: slip_of(t, j, b, equity) for j, b in buys.items()}
            need = sum(b * (1 + costs.commission + slips[j]) for j, b in buys.items())
            ratio = min(1.0, cash / need) if need > 0 else 1.0
            for j, b in buys.items():
                b *= ratio
                fee = b * (costs.commission + slips[j])
                positions[j] = positions.get(j, 0.0) + b
                basis[j] = basis.get(j, 0.0) + b
                cash -= b + fee
                paid += fee
                traded += b
            tv[dates[t]] = traded / equity if equity > 0 else 0.0
            holdings[dates[t]] = {p.ret.columns[j]: v / equity for j, v in positions.items()}
        # 장중(시가 → 저가 → 종가): 손절 점검 후 종가 반영
        for j in list(positions):
            r, g = R[t, j], Gp[t, j]
            if np.isnan(r):
                continue
            intraday = (1 + r) / (1 + (0.0 if np.isnan(g) else g))
            if LO is not None and j in basis and basis[j] > 0:
                v_open = positions[j]
                ratio_open = v_open / basis[j]
                level = 1 - stop_loss
                if breakeven_after and peak.get(j, 1.0) >= 1 + breakeven_after:
                    level = max(level, 1.005)
                ratio_low = ratio_open * intraday * (LO[t, j] if np.isfinite(LO[t, j]) else 1.0)
                if stop_mode == "close":
                    if ratio_open * intraday <= level:
                        pending_exit.add(j)
                    ratio_low = np.inf
                if ratio_low <= level:
                    exit_ratio = min(ratio_open, level)
                    value = basis[j] * exit_ratio
                    slip = slip_of(t, j, value, cash + sum(positions.values())) * 1.5
                    fee = value * (costs.commission + slip + tax)
                    cash += value - fee
                    paid += fee
                    positions.pop(j)
                    basis.pop(j)
                    peak.pop(j, None)
                    stops += 1
                    continue
            positions[j] *= intraday
            if j in basis and basis[j] > 0:
                peak[j] = max(peak.get(j, 1.0), positions[j] / basis[j])
        cash *= 1 + cr
        total = cash + sum(positions.values())
        eq.append(total)
        ex.append(sum(positions.values()) / total if total > 0 else 0.0)
    idx = dates[s: e + 1]
    res = SimResult(pd.Series(eq, index=idx), pd.Series(ex, index=idx), pd.Series(tv, dtype=float), holdings, paid)
    res.holdings["_stops"] = stops
    return res


def costs_min(equity: float, frac: float = 0.002) -> float:
    return equity * frac
