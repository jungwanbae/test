"""미국 레그 실행엔진(TRIAD 엔진 V·R) — NH PLUG 해외주식, 달러 계좌, 미국 동부시간.

국내 엔진(TradingEngine)의 안전장치·재조정 루프·주문 라운드·기록을 그대로 쓰고 다음만 바꾼다.
  * 달력: NYSE 휴장일(USCalendar), 시계: America/New_York, 주문 시간: 09:35 ~ deadline(기본 15:30 ET)
  * 신호: 전일 미국 종가까지의 일봉 → strategy.triad.us_leg_targets (백테스트와 같은 함수)
  * 리밸런싱 시점: ① 월말 신호 다음 거래일(정기) ② 폭락 후 골든크로스 같은 '사건 신호'로 목표가 바뀐 날
                   ③ 비중 이탈 > drift_threshold
  * 호가: 센트 단위(ticks kind="us")
백테스트 규칙(신호 t 종가 → t+1 매매)과 같다. 체결은 t+1 장중(기본 10:30 ET 이후)이라 종가 체결과 약간 다르다.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from autotrader.calendar import USCalendar
from autotrader.execution.engine import TradingEngine
from autotrader.strategy.triad import SIGNAL_ASSETS, us_leg_targets

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class USTradingEngine(TradingEngine):
    market_open = "09:35"

    def __init__(self, cfg, broker, **kw):
        kw.setdefault("calendar", USCalendar(cfg.extra_holidays))
        kw.setdefault("clock", lambda: datetime.now(ET))
        super().__init__(cfg, broker, **kw)
        self._cached: tuple[date, pd.Series, dict] | None = None

    def compute_targets(self, today: date, account=None) -> tuple[pd.Series, dict]:
        if self._cached and self._cached[0] == today:
            return self._cached[1].copy(), dict(self._cached[2])
        s = self.cfg.triad_us
        syms = [s.tickers[a] for a in SIGNAL_ASSETS]
        start = today - timedelta(days=int(365.25 * self.cfg.rebalance.history_years) + 30)
        closes = self.history.closes(syms, start, today)
        closes = closes[closes.index < pd.Timestamp(today)]      # 오늘(장중) 바 제외 → 전일 종가까지만
        if closes.empty:
            raise RuntimeError("미국 일봉 데이터가 없습니다")
        diag: dict = {}
        w = us_leg_targets(closes, s, self.calendar.is_month_end, diag)
        managed = set(self.cfg.universe.all_symbols())
        extra = sorted(set(w.index) - managed)
        if extra:
            raise RuntimeError(f"유니버스에 없는 티커가 목표에 있습니다: {extra} (config universe 확인)")
        for sym in managed - set(w.index):
            w[sym] = 0.0
        diag["signal_date"] = str(closes.index[-1].date())
        self._cached = (today, w, diag)
        return w.copy(), dict(diag)

    def rebalance_due(self, today: date, force: bool = False) -> tuple[bool, str]:
        due, why = super().rebalance_due(today, force)
        if due:
            return due, why
        last = self.store.last_targets()
        if not last:
            return True, "첫 실행"
        new, diag = self.compute_targets(today)
        chg = max(abs(float(new.get(k, 0.0)) - float(last.get(k, 0.0))) for k in set(new.index) | set(last))
        if chg > 0.05:
            note = " · 부스트(폭락 후 골든크로스)" if diag.get("triad_us", {}).get("boost_on") else ""
            return True, f"신호 변경 {chg:.1%}{note}"
        return False, why
