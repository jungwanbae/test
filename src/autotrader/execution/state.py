"""SQLite 상태 저장소: 실행 이력, 주문, 리밸런싱 완료 여부, 자산 스냅샷."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, status TEXT,
                                 message TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, ts TEXT, symbol TEXT,
                                   side TEXT, qty INTEGER, price REAL, order_type TEXT, order_id TEXT,
                                   accepted INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS rebalances (period TEXT PRIMARY KEY, ts TEXT, run_id INTEGER, targets TEXT);
CREATE TABLE IF NOT EXISTS equity (day TEXT PRIMARY KEY, ts TEXT, equity REAL, cash REAL, positions TEXT);
CREATE TABLE IF NOT EXISTS stops (period TEXT, symbol TEXT, ts TEXT, detail TEXT, PRIMARY KEY (period, symbol));
"""


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def start_run(self, mode: str) -> int:
        cur = self._db.execute("INSERT INTO runs(ts, mode, status) VALUES (?,?,?)",
                               (datetime.now().isoformat(timespec="seconds"), mode, "running"))
        self._db.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, message: str = "", detail: dict | None = None) -> None:
        self._db.execute("UPDATE runs SET status=?, message=?, detail=? WHERE id=?",
                         (status, message, json.dumps(detail or {}, ensure_ascii=False, default=str), run_id))
        self._db.commit()

    def record_order(self, run_id: int, symbol: str, side: str, qty: int, price: float | None, order_type: str,
                     order_id: str, accepted: bool, message: str) -> None:
        self._db.execute("INSERT INTO orders(run_id, ts, symbol, side, qty, price, order_type, order_id, accepted, "
                         "message) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (run_id, datetime.now().isoformat(timespec="seconds"), symbol, side, qty, price, order_type,
                          order_id, int(accepted), message))
        self._db.commit()

    def mark_rebalanced(self, period: str, run_id: int, targets: dict) -> None:
        self._db.execute("INSERT OR REPLACE INTO rebalances(period, ts, run_id, targets) VALUES (?,?,?,?)",
                         (period, datetime.now().isoformat(timespec="seconds"), run_id,
                          json.dumps(targets, ensure_ascii=False)))
        self._db.commit()

    def rebalanced(self, period: str) -> bool:
        return self._db.execute("SELECT 1 FROM rebalances WHERE period=?", (period,)).fetchone() is not None

    def last_targets(self) -> dict | None:
        row = self._db.execute("SELECT targets FROM rebalances ORDER BY ts DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def mark_stopped(self, period: str, symbol: str, detail: str = "") -> None:
        self._db.execute("INSERT OR REPLACE INTO stops(period, symbol, ts, detail) VALUES (?,?,?,?)",
                         (period, symbol, datetime.now().isoformat(timespec="seconds"), detail))
        self._db.commit()

    def stopped(self, period: str) -> set[str]:
        return {r[0] for r in self._db.execute("SELECT symbol FROM stops WHERE period=?", (period,))}

    def snapshot_equity(self, day: str, equity: float, cash: float, positions: dict) -> None:
        self._db.execute("INSERT OR REPLACE INTO equity(day, ts, equity, cash, positions) VALUES (?,?,?,?,?)",
                         (day, datetime.now().isoformat(timespec="seconds"), equity, cash,
                          json.dumps(positions, ensure_ascii=False)))
        self._db.commit()

    def equity_curve(self) -> list[tuple[str, float]]:
        return list(self._db.execute("SELECT day, equity FROM equity ORDER BY day"))

    def recent_runs(self, n: int = 10) -> list[tuple]:
        return list(self._db.execute("SELECT id, ts, mode, status, message FROM runs ORDER BY id DESC LIMIT ?", (n,)))
