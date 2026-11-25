"""모의 체결 브로커(Paper trading).

* 시세는 주입한 quote_fn(예: 실제 증권사 시세, 백테스트 가격표)에서 가져오고
* 주문은 로컬에서 즉시 체결(시장가/체결 가능한 지정가)시키며
* 계좌 상태를 JSON 파일에 저장해 여러 날에 걸친 페이퍼 트레이딩을 지원한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from autotrader.brokers.base import Broker, BrokerError
from autotrader.fees import FeeSchedule, get_fee_schedule
from autotrader.models import (AccountSnapshot, OrderRequest, OrderResult, OrderState, OrderStatus,
                               OrderType, Position, Quote, Side)


@dataclass
class _PaperState:
    cash: float
    positions: dict[str, dict] = field(default_factory=dict)  # sym -> {qty, avg_price}
    orders: list[dict] = field(default_factory=list)
    seq: int = 0


class PaperBroker(Broker):
    name = "paper"
    is_live = False

    def __init__(self, quote_fn: Callable[[str], Quote], *, initial_cash: float = 10_000_000,
                 state_file: str | None = None, fees: FeeSchedule | None = None, slippage_bps: float = 2.0,
                 history_fn: Callable[[str, date, date | None], pd.DataFrame] | None = None):
        self._quote_fn = quote_fn
        self._history_fn = history_fn
        self._fees = fees or get_fee_schedule("nh_plug")
        self._slip = slippage_bps / 10_000
        self._path = Path(state_file) if state_file else None
        if self._path and self._path.exists():
            raw = json.loads(self._path.read_text())
            self._st = _PaperState(**raw)
        else:
            self._st = _PaperState(cash=float(initial_cash))
            self._save()

    # ---------------------------------------------------------------- 상태 저장
    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self._st), ensure_ascii=False, indent=1))
        os.replace(tmp, self._path)

    # ---------------------------------------------------------------- 시세
    def get_quote(self, symbol: str) -> Quote:
        return self._quote_fn(symbol)

    def get_daily_history(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        if self._history_fn is None:
            raise BrokerError("PaperBroker 에 history_fn 이 설정되지 않았습니다", category="config")
        return self._history_fn(symbol, start, end)

    # ---------------------------------------------------------------- 계좌
    def get_account(self) -> AccountSnapshot:
        positions = {}
        for sym, p in self._st.positions.items():
            if p["qty"] <= 0:
                continue
            try:
                last = self._quote_fn(sym).price
            except Exception:
                last = p["avg_price"]
            positions[sym] = Position(sym, int(p["qty"]), int(p["qty"]), float(p["avg_price"]), float(last))
        eq = self._st.cash + sum(p.market_value for p in positions.values())
        return AccountSnapshot(cash=self._st.cash, positions=positions, total_equity=eq)

    # ---------------------------------------------------------------- 주문
    def place_order(self, req: OrderRequest) -> OrderResult:
        q = self._quote_fn(req.symbol)
        ask = q.ask or q.price
        bid = q.bid or q.price
        if req.side == Side.BUY:
            exec_px = ask * (1 + self._slip)
            if req.order_type == OrderType.LIMIT:
                if req.price < ask:  # 체결 불가 지정가 → 미체결로 접수
                    return self._record(req, OrderStatus.OPEN, 0, None, "미체결(지정가<매도호가)")
                exec_px = min(exec_px, req.price)
            notional = exec_px * req.qty
            cost = self._fees.cost(notional, is_sell=False)
            if notional + cost > self._st.cash + 1e-6:
                return OrderResult(req, False, "", "주문가능금액 부족")
            self._st.cash -= notional + cost
            pos = self._st.positions.setdefault(req.symbol, {"qty": 0, "avg_price": 0.0})
            new_qty = pos["qty"] + req.qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + notional) / new_qty
            pos["qty"] = new_qty
        else:
            pos = self._st.positions.get(req.symbol)
            if not pos or pos["qty"] < req.qty:
                return OrderResult(req, False, "", "매도가능수량 부족")
            exec_px = bid * (1 - self._slip)
            if req.order_type == OrderType.LIMIT:
                if req.price > bid:
                    return self._record(req, OrderStatus.OPEN, 0, None, "미체결(지정가>매수호가)")
                exec_px = max(exec_px, req.price)
            notional = exec_px * req.qty
            self._st.cash += notional - self._fees.cost(notional, is_sell=True)
            pos["qty"] -= req.qty
            if pos["qty"] == 0:
                del self._st.positions[req.symbol]
        return self._record(req, OrderStatus.FILLED, req.qty, exec_px, "체결")

    def _record(self, req: OrderRequest, status: OrderStatus, filled: int, px: float | None, msg: str) -> OrderResult:
        self._st.seq += 1
        oid = f"P{self._st.seq:06d}"
        self._st.orders.append({"order_id": oid, "date": date.today().isoformat(), "ts": datetime.now().isoformat(),
                                "symbol": req.symbol, "side": req.side.value, "qty": req.qty, "filled": filled,
                                "price": req.price, "avg": px, "status": status.value})
        self._st.orders = self._st.orders[-2000:]
        self._save()
        return OrderResult(req, True, oid, msg)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        for o in self._st.orders:
            if o["order_id"] == order_id and o["status"] == OrderStatus.OPEN.value:
                o["status"] = OrderStatus.CANCELLED.value
                self._save()
                return OrderResult(OrderRequest(symbol, Side.SELL, 1, OrderType.MARKET), True, order_id, "취소")
        return OrderResult(OrderRequest(symbol, Side.SELL, 1, OrderType.MARKET), False, order_id, "취소대상 없음")

    def get_orders(self, day: date | None = None) -> list[OrderState]:
        d = (day or date.today()).isoformat()
        out = []
        for o in self._st.orders:
            if o["date"] != d:
                continue
            st = OrderStatus(o["status"])
            out.append(OrderState(o["order_id"], o["symbol"], Side(o["side"]), o["qty"], o["filled"],
                                  0 if st != OrderStatus.OPEN else o["qty"] - o["filled"], o["avg"], st, o["price"]))
        return out
