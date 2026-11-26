"""KRX 호가단위(tick size) 처리.

* ETF·ETN: 2,000원 미만 1원 / 2,000원 이상 5원
* 주식(2023.01 개편, 코스피·코스닥 공통):
    2천 미만 1 · 5천 미만 5 · 2만 미만 10 · 5만 미만 50 · 20만 미만 100 · 50만 미만 500 · 이상 1,000
* 미국(kind="us"): $1 이상 $0.01 (Reg NMS). $1 미만 종목은 다루지 않는다.
"""
from __future__ import annotations

import math

_STOCK_TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))


def tick_size(price: float, kind: str = "etf") -> float:
    if kind == "us":
        return 0.01
    if kind in ("etf", "etn"):
        return 1 if price < 2_000 else 5
    for bound, tick in _STOCK_TICKS:
        if price < bound:
            return tick
    return 1_000


def round_to_tick(price: float, kind: str = "etf", direction: str = "nearest") -> float:
    """direction: up(매수 공격적) / down(매도 공격적) / nearest. KRX 는 정수(원), 미국은 센트 단위 실수."""
    if price <= 0:
        raise ValueError("가격은 양수여야 합니다")
    if kind == "us":
        c = price * 100
        n = math.ceil(c - 1e-6) if direction == "up" else math.floor(c + 1e-6) if direction == "down" else round(c)
        return max(n, 1) / 100
    t = tick_size(price, kind)
    q = price / t
    if direction == "up":
        n = math.ceil(q - 1e-9)
    elif direction == "down":
        n = math.floor(q + 1e-9)
    else:
        n = round(q)
    out = int(n * t)
    # 경계(예: 1,999→2,000)에서 단위가 바뀌는 경우 한 번 더 정규화
    t2 = tick_size(out, kind)
    if out % t2:
        out = int((math.ceil if direction == "up" else math.floor)(out / t2) * t2)
    return max(out, t2)


def shift_ticks(price: float, n: int, kind: str = "etf") -> float:
    """호가를 n 틱 이동(양수=위, 음수=아래)."""
    if kind == "us":
        return max(round(round_to_tick(price, kind) + n * 0.01, 2), 0.01)
    p = round_to_tick(price, kind)
    step = 1 if n > 0 else -1
    for _ in range(abs(n)):
        if step > 0:
            p = p + tick_size(p, kind)
        else:
            p = max(p - tick_size(p - 1, kind), 1)
    return p
