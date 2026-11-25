"""브로커 중립 도메인 모델.

모든 증권사 어댑터(NH PLUG, KIS, Paper)는 이 타입으로만 데이터를 주고받는다.
전략·리밸런서·실행엔진은 증권사별 필드명을 전혀 알지 못한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    OPEN = "OPEN"            # 접수, 미체결 잔량 있음
    PARTIAL = "PARTIAL"      # 일부 체결
    FILLED = "FILLED"        # 전량 체결
    CANCELLED = "CANCELLED"  # 취소(잔량 취소 포함)
    REJECTED = "REJECTED"    # 거부
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    prev_close: float | None = None
    name: str = ""
    market_status: str | None = None
    timestamp: datetime | None = None


@dataclass
class Position:
    symbol: str
    qty: int
    sellable_qty: int
    avg_price: float
    last_price: float
    name: str = ""

    @property
    def market_value(self) -> float:
        return self.qty * self.last_price


@dataclass
class AccountSnapshot:
    """계좌 스냅샷.

    cash: 지금 바로 매수에 쓸 수 있는 현금(원). 증권사별로 '미수 없는 주문가능금액' 기준.
    total_equity: 증권사가 계산한 총자산(참고용). 리밸런서는 자체 계산값을 사용한다.
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_equity: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def qty(self, symbol: str) -> int:
        p = self.positions.get(symbol)
        return p.qty if p else 0

    def sellable(self, symbol: str) -> int:
        p = self.positions.get(symbol)
        return p.sellable_qty if p else 0


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    order_type: OrderType = OrderType.LIMIT
    price: float | None = None  # LIMIT 이면 필수
    tag: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"주문수량은 1 이상이어야 합니다: {self.qty}")
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            raise ValueError("지정가 주문에는 양수 가격이 필요합니다")


@dataclass
class OrderResult:
    request: OrderRequest
    accepted: bool
    order_id: str = ""
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class OrderState:
    order_id: str
    symbol: str
    side: Side | None
    qty: int
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float | None
    status: OrderStatus
    price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL) and self.remaining_qty > 0
