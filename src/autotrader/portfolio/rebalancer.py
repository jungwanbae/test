"""목표비중 → 주문 계획(1주 단위, 매도 우선, 현금 한도 내 매수)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from autotrader.models import AccountSnapshot, Side


@dataclass
class PlannedOrder:
    symbol: str
    side: Side
    qty: int
    price: float
    reason: str = ""

    @property
    def value(self) -> float:
        return self.qty * self.price


@dataclass
class RebalancePlan:
    base_capital: float
    target_weights: pd.Series
    current_weights: pd.Series
    orders: list[PlannedOrder] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def turnover(self) -> float:
        return sum(o.value for o in self.orders) / self.base_capital if self.base_capital > 0 else 0.0

    def sells(self) -> list[PlannedOrder]:
        return [o for o in self.orders if o.side == Side.SELL]

    def buys(self) -> list[PlannedOrder]:
        return [o for o in self.orders if o.side == Side.BUY]

    def to_text(self, names: dict[str, str] | None = None) -> str:
        names = names or {}
        lines = [f"운용기준금액 {self.base_capital:,.0f}원 · 회전율 {self.turnover * 100:.1f}%"]
        for s in self.target_weights.index:
            lines.append(f"  {names.get(s, s):<24} 현재 {self.current_weights.get(s, 0) * 100:5.1f}% → 목표 "
                         f"{self.target_weights[s] * 100:5.1f}%")
        for o in self.orders:
            lines.append(f"  · {o.side.value:<4} {names.get(o.symbol, o.symbol)} {o.qty:,}주 × {o.price:,.0f} "
                         f"= {o.value:,.0f}원 ({o.reason})")
        for s, why in self.skipped.items():
            lines.append(f"  · 생략 {names.get(s, s)}: {why}")
        return "\n".join(lines)


def plan_rebalance(target: pd.Series, account: AccountSnapshot, prices: dict[str, float], *,
                   capital_limit: float | None = None, cash_buffer_pct: float = 0.005,
                   no_trade_band: float = 0.01, min_order_value: float = 50_000,
                   fee_rate: float = 0.0002) -> RebalancePlan:
    """target 에 있는 종목만 관리한다(계좌의 다른 보유종목은 건드리지 않음)."""
    syms = list(target.index)
    missing = [s for s in syms if not prices.get(s) or prices[s] <= 0]
    if missing:
        raise ValueError(f"가격이 없는 종목: {missing}")
    held_val = {s: account.qty(s) * prices[s] for s in syms}
    held_total = sum(held_val.values())
    base = held_total + max(account.cash, 0.0)
    if capital_limit:
        base = min(base, float(capital_limit))
    strategy_cash = max(base - held_total, 0.0)
    investable = base * (1 - cash_buffer_pct)
    cur_w = pd.Series({s: held_val[s] / base if base > 0 else 0.0 for s in syms})
    plan = RebalancePlan(base, target.astype(float), cur_w)
    if base <= 0:
        return plan

    threshold = max(min_order_value, no_trade_band * base)
    sells, buys = [], []
    for s in syms:
        px = prices[s]
        w = float(target.get(s, 0.0))
        cur = account.qty(s)
        tgt_qty = int(math.floor(w * investable / px)) if w > 0 else 0
        delta = tgt_qty - cur
        if delta == 0:
            continue
        val = abs(delta) * px
        full_exit = w == 0 and cur > 0
        if val < threshold and not full_exit:
            plan.skipped[s] = f"차이 {val:,.0f}원 < 기준 {threshold:,.0f}원"
            continue
        if delta < 0:
            q = min(-delta, account.sellable(s))
            if q > 0:
                sells.append(PlannedOrder(s, Side.SELL, q, px, "전량매도" if full_exit else "비중축소"))
            else:
                plan.skipped[s] = "매도가능수량 없음(미체결 주문 확인)"
        else:
            buys.append(PlannedOrder(s, Side.BUY, delta, px, "비중확대"))

    proceeds = sum(o.value for o in sells) * (1 - fee_rate)
    available = min(strategy_cash + proceeds, max(account.cash, 0.0) + proceeds) * (1 - cash_buffer_pct)
    need = sum(o.value for o in buys) * (1 + fee_rate)
    if need > available > 0:
        ratio = available / need
        scaled = []
        for o in buys:
            q = int(math.floor(o.qty * ratio))
            if q > 0 and q * o.price >= min(min_order_value, threshold):
                scaled.append(PlannedOrder(o.symbol, o.side, q, o.price, o.reason + "(현금 한도로 축소)"))
            else:
                plan.skipped[o.symbol] = "현금 부족"
        buys = scaled
    elif available <= 0:
        for o in buys:
            plan.skipped[o.symbol] = "현금 없음"
        buys = []
    plan.orders = sells + buys
    return plan
