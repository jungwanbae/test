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
    g = ds.dropna(subset=[score]).groupby("date")
    for d, df in g:
        if signal_dates is not None and d not in signal_dates:
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
        exp = regime_exposure(df.iloc[0], rules)
        out[d] = w * exp
        held = sel
    return out


def simulate(p: Panel, targets: dict[pd.Timestamp, pd.Series], costs: StockCosts | None = None,
             start: str | None = None, end: str | None = None, cash_rate: float = 0.0,
             min_trade: float = 0.002) -> SimResult:
    costs = costs or StockCosts()
    dates = p.dates
    R = p.ret.values
    Gp = p.gap.values
    if costs.slippage_model == "sqrt":
        adv = p.amount.rolling(60, min_periods=10).mean().values
        SL = costs.slip_base + costs.slip_k / np.sqrt(np.clip(np.nan_to_num(adv, nan=1e7) / 1e8, 0.05, None))
    else:
        SL = None
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
    cash = 1.0
    eq, ex, tv = [], [], {}
    paid = 0.0
    holdings = {}
    for t in range(s, e + 1):
        d = dates[t].date()
        # 상장폐지 처리
        for j in [j for j in positions if np.isnan(R[t, j]) and t > last_valid[j]]:
            cash += positions.pop(j) * (1 - costs.delist_haircut)
        if t in execs:
            for j in positions:
                g = Gp[t, j]
                if not np.isnan(g):
                    positions[j] *= 1 + g
            equity = cash + sum(positions.values())
            tgt = {col[c]: float(w) * equity for c, w in execs[t].items() if c in col}
            traded = 0.0
            tax = costs.sell_tax(d)
            # 매도 먼저
            for j in list(set(positions) | set(tgt)):
                cur = positions.get(j, 0.0)
                delta = tgt.get(j, 0.0) - cur
                if np.isnan(R[t, j]) or abs(delta) < 1e-12:
                    continue  # 거래정지 → 매매 불가
                if delta < 0 and (-delta >= costs_min(equity, min_trade) or tgt.get(j, 0.0) == 0.0):
                    sell = -delta
                    slip = SL[t - 1, j] if SL is not None else costs.slippage
                    fee = sell * (costs.commission + slip + tax)
                    positions[j] = cur - sell
                    cash += sell - fee
                    paid += fee
                    traded += sell
                    if positions[j] <= 1e-12:
                        positions.pop(j)
            buys = {j: tgt[j] - positions.get(j, 0.0) for j in tgt
                    if not np.isnan(R[t, j]) and tgt[j] - positions.get(j, 0.0) >= costs_min(equity, min_trade)}
            slips = {j: (SL[t - 1, j] if SL is not None else costs.slippage) for j in buys}
            need = sum(b * (1 + costs.commission + slips[j]) for j, b in buys.items())
            ratio = min(1.0, cash / need) if need > 0 else 1.0
            for j, b in buys.items():
                b *= ratio
                fee = b * (costs.commission + slips[j])
                positions[j] = positions.get(j, 0.0) + b
                cash -= b + fee
                paid += fee
                traded += b
            tv[dates[t]] = traded / equity if equity > 0 else 0.0
            holdings[dates[t]] = {p.ret.columns[j]: v / equity for j, v in positions.items()}
            for j in positions:  # 시가 → 종가
                r, g = R[t, j], Gp[t, j]
                if not np.isnan(r):
                    positions[j] *= (1 + r) / (1 + (0.0 if np.isnan(g) else g))
        else:
            for j in positions:
                r = R[t, j]
                if not np.isnan(r):
                    positions[j] *= 1 + r
        cash *= 1 + cr
        total = cash + sum(positions.values())
        eq.append(total)
        ex.append(sum(positions.values()) / total if total > 0 else 0.0)
    idx = dates[s: e + 1]
    return SimResult(pd.Series(eq, index=idx), pd.Series(ex, index=idx), pd.Series(tv, dtype=float),
                     holdings, paid)


def costs_min(equity: float, frac: float = 0.002) -> float:
    return equity * frac
