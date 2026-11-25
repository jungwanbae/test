"""어댑터 공용 유틸: 유량 제한, 토큰 파일 캐시."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path


class RateLimiter:
    """슬라이딩 1초 창 유량 제한기(스레드 안전).

    증권사 한도(예: NH 초당 5건, KIS 실전 초당 20건/모의 초당 1~2건)보다 약간 낮게 잡는다.
    """

    def __init__(self, per_second: float, *, sleep=time.sleep, clock=time.monotonic):
        self.per_second = float(per_second)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()
        self._sleep = sleep
        self._clock = clock

    def acquire(self) -> None:
        if self.per_second <= 0:
            return
        # 초당 1건 미만(예: 0.5)이면 최소 간격 방식으로 처리
        if self.per_second < 1:
            interval = 1.0 / self.per_second
            with self._lock:
                now = self._clock()
                if self._calls:
                    wait = interval - (now - self._calls[-1])
                    if wait > 0:
                        self._sleep(wait)
                        now = self._clock()
                self._calls.clear()
                self._calls.append(now)
            return
        limit = int(self.per_second)
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= 1.0:
                    self._calls.popleft()
                if len(self._calls) < limit:
                    self._calls.append(now)
                    return
                wait = 1.0 - (now - self._calls[0])
            self._sleep(max(wait, 0.001))


class TokenCache:
    """접근토큰 파일 캐시.

    두 증권사 모두 토큰 재발급을 제한(KIS 1분 1회)하거나 보안 알림(NH)을 보내므로
    프로세스가 재시작돼도 토큰을 재사용해야 한다. 파일 권한은 0600 으로 만든다.
    """

    def __init__(self, cache_dir: str | os.PathLike, namespace: str, key_material: str):
        digest = hashlib.sha256(key_material.encode()).hexdigest()[:16]
        self.path = Path(cache_dir).expanduser() / f"{namespace}-{digest}.json"

    def load(self, *, margin_sec: float = 600) -> str | None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        if data.get("exp", 0) - margin_sec > time.time() and data.get("token"):
            return data["token"]
        return None

    def save(self, token: str, exp_epoch: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"token": token, "exp": exp_epoch}, f)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
