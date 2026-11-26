"""KRX 전종목 일별 패널(1995~현재) — FinanceData/marcap 데이터셋 기반.

* 생존편향 없음: 매일 그날 상장돼 있던 전 종목(이후 상장폐지 종목 포함)
* 수익률: 거래소 기준가(= 종가 − 전일대비)로 계산 → 액면분할·무상증자·유상증자 권리락이 반영된
  '수정' 일간수익률. 현금배당은 기준가에 반영되지 않으므로 가격수익률(배당 제외, 보수적)
* 보통주만: 코드 끝자리 '0'(신형 영숫자 코드 포함). 우선주(5·7·9·K), 스팩, 리츠, 코넥스 제외
* 위험표시: 관리종목·투자주의환기·정리매매 부서 플래그(코스닥은 부서 정보가 있는 기간만)

wide 행렬(날짜 × 종목, float32)로 만들어 parquet 로 캐시한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MARCAP_REPO = "https://github.com/FinanceData/marcap.git"
FIELDS = ("ret", "gap", "close", "amount", "marcap", "flag")  # gap = 시가/기준가 − 1 (전일종가→시가)
_RISK_DEPT = ("관리종목", "투자주의환기", "정리매매", "SPAC")


@dataclass
class Panel:
    ret: pd.DataFrame      # 일간 수정수익률(기준가 대비 종가)
    gap: pd.DataFrame      # 전일(수정)종가 → 당일 시가 수익률
    close: pd.DataFrame    # 원 종가(호가·가격필터용)
    amount: pd.DataFrame   # 거래대금(원)
    marcap: pd.DataFrame   # 시가총액(원)
    flag: pd.DataFrame     # 1 = 위험표시 종목
    names: pd.Series       # 종목명(마지막 관측)
    market: pd.Series      # KOSPI / KOSDAQ

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.ret.index

    def price_index(self) -> pd.DataFrame:
        """수정 누적가격지수(상장 첫날 = 1)."""
        return (1 + self.ret.fillna(0.0)).cumprod().where(self.ret.notna() | self.ret.shift(-1).notna())

    def market_return(self, which: str | None = None) -> pd.Series:
        """시가총액가중 시장수익률(전일 시총 가중). which: KOSPI | KOSDAQ | None(전체)."""
        cols = self.ret.columns if which is None else self.market.index[self.market == which]
        w = self.marcap[cols].shift(1)
        r = self.ret[cols]
        ok = r.notna() & w.notna()
        return ((r.fillna(0) * w.where(ok, 0)).sum(axis=1) / w.where(ok, 0).sum(axis=1)).fillna(0.0)


def is_common_stock(code: str, name: str) -> bool:
    if not isinstance(code, str) or len(code) != 6 or code[-1] != "0":
        return False
    n = str(name)
    return not any(k in n for k in ("스팩", "SPAC", "리츠", "REIT"))


def build_panel(marcap_dir: str | Path, start_year: int = 1995, end_year: int | None = None) -> Panel:
    files = sorted(Path(marcap_dir).glob("marcap-*.parquet"))
    files = [f for f in files if int(f.stem.split("-")[1]) >= start_year
             and (end_year is None or int(f.stem.split("-")[1]) <= end_year)]
    if not files:
        raise FileNotFoundError(f"{marcap_dir} 에 marcap-YYYY.parquet 이 없습니다 (git clone {MARCAP_REPO})")
    cols = ["Date", "Code", "Name", "Market", "Dept", "Open", "Close", "Changes", "Amount", "Marcap", "Volume"]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d["Code"] = d["Code"].astype(str).str.strip().str.zfill(6)  # 과거 연도는 앞자리 0 이 빠져 있음
        d = d[d["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
        d = d[[is_common_stock(c, n) for c, n in zip(d["Code"], d["Name"])]]
        parts.append(d)
        log.info("marcap %s: %d행", f.stem, len(d))
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.drop_duplicates(["Date", "Code"], keep="last")
    base = df["Close"] - df["Changes"]
    ok = (base > 0) & (df["Close"] > 0)
    df["ret"] = np.where(ok, df["Close"] / base - 1.0, np.nan)
    opn = df["Open"].where(df["Open"] > 0, df["Close"])
    df["gap"] = np.where(ok, opn / base - 1.0, np.nan)
    # 거래정지(거래량 0)일은 수익률 0, 시가 갭 0 으로(가격 변동 없음)
    halted = df["Volume"].fillna(0) <= 0
    df.loc[halted & ok, ["ret", "gap"]] = 0.0
    df["flag"] = df["Dept"].fillna("").str.contains("|".join(_RISK_DEPT)).astype(np.float32)
    # 극단치 방어(데이터 오류): 하루 ±95% 초과는 결측 처리(가격제한폭 ±30%, 과거 ±15%)
    df.loc[df["ret"].abs() > 0.95, "ret"] = np.nan
    wide = {}
    for fld, src in (("ret", "ret"), ("gap", "gap"), ("close", "Close"), ("amount", "Amount"),
                     ("marcap", "Marcap"), ("flag", "flag")):
        w = df.pivot(index="Date", columns="Code", values=src).astype(np.float32)
        wide[fld] = w.sort_index()
    last = df.sort_values("Date").groupby("Code").last()
    market = last["Market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})
    return Panel(names=last["Name"], market=market, **wide)


def save_panel(p: Panel, path: str | Path) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        getattr(p, f).to_parquet(out / f"{f}.parquet")
    pd.DataFrame({"name": p.names, "market": p.market}).to_parquet(out / "meta.parquet")


def load_panel(path: str | Path) -> Panel:
    src = Path(path)
    wide = {f: pd.read_parquet(src / f"{f}.parquet") for f in FIELDS}
    meta = pd.read_parquet(src / "meta.parquet")
    return Panel(names=meta["name"], market=meta["market"], **wide)
