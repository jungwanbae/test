"""KRX 거래일 캘린더.

우선순위: 증권사 휴장일 API(KIS 실전) > 공휴일 라이브러리(holidays, 대체공휴일·음력 포함) + KRX 고유 휴장일.
KRX 고유 휴장: 연말(12/31), 근로자의날(5/1). 선거일 등 임시공휴일은 holidays 에 반영되거나
config 의 extra_holidays 로 추가한다.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Iterable

import holidays as _holidays


class KRXCalendar:
    def __init__(self, extra_holidays: Iterable[str | date] = (), broker_check: Callable[[date], bool | None] | None = None):
        self._kr = _holidays.country_holidays("KR")
        self._extra = {d if isinstance(d, date) else date.fromisoformat(str(d)) for d in extra_holidays}
        self._broker_check = broker_check

    def is_trading_day(self, d: date) -> bool:
        if self._broker_check is not None:
            v = self._broker_check(d)
            if v is not None:
                return v
        if d.weekday() >= 5 or d in self._extra or d in self._kr:
            return False
        if (d.month, d.day) in ((12, 31), (5, 1)):
            return False
        return True

    def next_trading_day(self, d: date) -> date:
        n = d + timedelta(days=1)
        while not self.is_trading_day(n):
            n += timedelta(days=1)
        return n

    def prev_trading_day(self, d: date) -> date:
        p = d - timedelta(days=1)
        while not self.is_trading_day(p):
            p -= timedelta(days=1)
        return p

    def first_trading_day_of_month(self, year: int, month: int) -> date:
        d = date(year, month, 1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d
