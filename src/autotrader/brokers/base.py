"""증권사 어댑터 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from autotrader.models import AccountSnapshot, OrderRequest, OrderResult, OrderState, Quote


class BrokerError(RuntimeError):
    """증권사 API 오류.

    category: config | auth | rate_limit | network | http | business
    retryable: 같은 요청을 (간격을 두고) 다시 보내도 되는지
    """

    def __init__(self, message: str, *, category: str = "http", code: str | None = None,
                 status: int | None = None, retryable: bool = False, raw=None):
        super().__init__(message)
        self.category = category
        self.code = code
        self.status = status
        self.retryable = retryable
        self.raw = raw

    def __str__(self) -> str:  # pragma: no cover - 표시용
        base = super().__str__()
        bits = [b for b in (self.category, self.code, str(self.status) if self.status else None) if b]
        return f"{base} [{' / '.join(bits)}]" if bits else base


class Broker(ABC):
    """모든 어댑터가 구현해야 하는 최소 기능.

    is_live: 실제 돈이 오가는 환경이면 True. 실행엔진은 이 값으로 안전장치를 건다.
    """

    name: str = "base"
    is_live: bool = False

    # ---- 시세 -------------------------------------------------------------
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_daily_history(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        """일봉. index=DatetimeIndex(날짜 오름차순), columns=[open, high, low, close, volume]."""

    # ---- 계좌 -------------------------------------------------------------
    @abstractmethod
    def get_account(self) -> AccountSnapshot: ...

    # ---- 주문 -------------------------------------------------------------
    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> OrderResult: ...

    @abstractmethod
    def get_orders(self, day: date | None = None) -> list[OrderState]:
        """당일(또는 day) 주문·체결 내역."""

    # ---- 캘린더(선택) -------------------------------------------------------
    def is_trading_day(self, day: date) -> bool | None:
        """증권사가 휴장일 API 를 제공하면 True/False, 모르면 None."""
        return None
