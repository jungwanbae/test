import json

import numpy as np
import pandas as pd
import pytest


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}

    @property
    def ok(self):
        return self.status_code == 200

    def json(self):
        if isinstance(self._body, (dict, list)):
            return self._body
        raise ValueError("not json")

    @property
    def text(self):
        return json.dumps(self._body, ensure_ascii=False) if isinstance(self._body, (dict, list)) else str(self._body)


class FakeSession:
    """handler(method, url, kwargs) -> FakeResp | Exception"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def _do(self, method, url, **kw):
        self.calls.append((method, url, kw))
        r = self.handler(method, url, kw)
        if isinstance(r, Exception):
            raise r
        return r

    def post(self, url, **kw):
        return self._do("POST", url, **kw)

    def get(self, url, **kw):
        return self._do("GET", url, **kw)


@pytest.fixture
def trend_prices():
    """위험자산 4개(2개 상승, 1개 횡보, 1개 하락) + 현금, 일봉 2년."""
    idx = pd.bdate_range("2022-01-03", periods=520)
    t = np.arange(len(idx))
    zig = np.where(t % 2 == 0, 1.0, -1.0)  # 결정적 ±σ 진동: 추세·변동성을 정확히 통제

    def path(drift, vol):
        return 100 * np.exp(np.cumsum(drift + vol * zig))

    df = pd.DataFrame({
        "UP1": path(0.0008, 0.004),
        "UP2": path(0.0006, 0.010),
        "FLAT": path(0.0, 0.003),
        "DOWN": path(-0.0008, 0.006),
        "CASH": path(0.0001, 0.0),
    }, index=idx)
    return df
