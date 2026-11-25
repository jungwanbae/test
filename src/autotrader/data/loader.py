"""가격 데이터 적재: CSV 디렉터리 / 증권사 일봉(+로컬 캐시)."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from autotrader.brokers.base import Broker

log = logging.getLogger(__name__)


def load_csv_dir(path: str | Path, symbols: list[str] | None = None, column: str = "close") -> pd.DataFrame:
    """SYMBOL.csv(date, close …) 파일들을 읽어 종가 DataFrame 으로."""
    p = Path(path)
    files = sorted(p.glob("*.csv"))
    series = {}
    for f in files:
        sym = f.stem
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(f, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        col = column if column in df.columns else df.columns[0]
        series[sym] = df[col].astype(float)
    if not series:
        raise FileNotFoundError(f"{p} 에 CSV 가 없습니다")
    return pd.DataFrame(series).sort_index()


class BrokerHistoryCache:
    """증권사 일봉을 로컬 CSV 로 캐시하고 부족한 최근 구간만 추가로 받는다."""

    def __init__(self, broker: Broker, cache_dir: str | Path = "data/cache"):
        self.broker = broker
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.csv"

    def get(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        end = end or date.today()
        path = self._path(symbol)
        cached = pd.DataFrame()
        if path.exists():
            cached = pd.read_csv(path, parse_dates=["date"], index_col="date")
        fetch_from = start
        if not cached.empty:
            first, last = cached.index.min().date(), cached.index.max().date()
            if first <= start + timedelta(days=7):
                fetch_from = last - timedelta(days=7)  # 최근 며칠은 다시 받아 수정분 반영
        if cached.empty or fetch_from < end:
            fresh = self.broker.get_daily_history(symbol, fetch_from, end)
            merged = pd.concat([cached, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.index.name = "date"
            merged.to_csv(path)
            cached = merged
        return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def closes(self, symbols: list[str], start: date, end: date | None = None) -> pd.DataFrame:
        out = {}
        for s in symbols:
            df = self.get(s, start, end)
            if df.empty:
                log.warning("%s 일봉이 비어 있습니다", s)
                continue
            out[s] = df["close"].astype(float)
        return pd.DataFrame(out).sort_index()
