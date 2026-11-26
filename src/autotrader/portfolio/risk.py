"""주문 전 안전장치."""
from __future__ import annotations

from pathlib import Path

from autotrader.config import RiskConfig, live_confirmed
from autotrader.models import OrderRequest, OrderType, Quote
from autotrader.portfolio.rebalancer import RebalancePlan


class RiskViolation(RuntimeError):
    pass


class RiskManager:
    def __init__(self, cfg: RiskConfig, is_live: bool):
        self.cfg = cfg
        self.is_live = is_live

    def pre_run(self) -> None:
        if Path(self.cfg.kill_switch_file).exists():
            raise RiskViolation(f"킬스위치 파일 '{self.cfg.kill_switch_file}' 이 있어 매매를 중단합니다")
        if self.is_live and self.cfg.require_live_confirm and not live_confirmed():
            raise RiskViolation("실계좌(live) 주문은 AUTOTRADER_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK 설정이 필요합니다")

    def check_plan(self, plan: RebalancePlan) -> None:
        if plan.turnover > self.cfg.max_turnover_pct:
            raise RiskViolation(f"회전율 {plan.turnover:.0%} 가 한도 {self.cfg.max_turnover_pct:.0%} 를 넘습니다")
        for o in plan.orders:
            if o.value > self.cfg.max_order_value:
                raise RiskViolation(f"{o.symbol} 주문금액 {o.value:,.0f}원이 한도 {self.cfg.max_order_value:,.0f}원 초과")

    def check_quote(self, q: Quote, ref_close: float | None, kind: str = "etf") -> None:
        ref = q.prev_close or ref_close
        limit = self.cfg.max_price_deviation_stock if kind == "stock" else self.cfg.max_price_deviation
        if ref and ref > 0 and abs(q.price / ref - 1) > limit:
            raise RiskViolation(f"{q.symbol} 시세 이상: 현재가 {q.price:,.0f} vs 기준 {ref:,.0f}")

    def check_order(self, req: OrderRequest, q: Quote) -> None:
        px = req.price if req.order_type == OrderType.LIMIT else q.price
        if req.qty * px > self.cfg.max_order_value * 1.02:
            raise RiskViolation(f"{req.symbol} 주문금액 한도 초과")
        if req.order_type == OrderType.LIMIT and abs(req.price / q.price - 1) > 0.03:
            raise RiskViolation(f"{req.symbol} 지정가 {req.price:,.0f} 가 현재가 {q.price:,.0f} 와 3% 이상 차이")
