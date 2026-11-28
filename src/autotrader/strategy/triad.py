"""QUAD(기본) / TRIAD — 전설적 전략 토너먼트에서 도출한 다중 엔진 포트폴리오.

QUAD(4등분, 1순위) = TRIAD 세 엔진 ¼씩 + 사용자의 '고금리 대비형' ¼
  고금리 대비형은 계좌별로 나눠 운용한다: 미국 부분 80%(나스닥 GC·VT, 금) → 해외 계좌, 코스피 부분 20% → 국내 계좌.
  자본 배분: 국내 30%(MSAP-M 25 + 고금리 대비형 코스피 5) · 미국 70%(역발상 VT 25 + 레버리지 RP 25 + 고금리 대비형 미국 20)
  검증(2005~, 원화·일별·실비용, 종목 모델 시드 3개 앙상블): CAGR 22.1%, 세후(근사) 18.7%, Sharpe 1.14, MDD −27.3%
  (2001~ 20.6%, 2016~ 21.4%) — reports/hr_mix.csv

  엔진 K(국내 레그, 1/3): MSAP-M — KRX 종목 머신러닝 랭킹(오닐·미너비니식 모멘텀/추세 + −15% 손절) 70%
                           + 원화 리스크패리티 ETF 30%  … strategy.name = msap (기존 엔진)
  엔진 V(미국 레그 ½):   확인형 역발상 변동성 타기팅 — 나스닥 목표변동성 25%(Moreira–Muir), 최대 3배.
                           최근 2년 안에 사상 최고가 대비 −30% 이하까지 빠진 적이 있고 골든크로스(50>200일선)가
                           새로 나오면 1년간 3배(템플턴·버핏의 '공포에 사라'를 추세 '확인 후'에 실행)
  엔진 R(미국 레그 ½):   레버리지 리스크패리티(브리지워터 올웨더 계열) — 나스닥·장기채·금·원자재 역변동성,
                           12개월 추세(현금 대비) 필터(페이버), 목표변동성 18%, 상품별 레버리지 상한

검증(원화, 일별, 2005~2026.11, 비용 반영 — reports/tournament_report.md):
  TRIAD(1/3씩): CAGR 21.7%, 세후(근사) 18.5%, Sharpe 1.19, 일별 MDD −26.2%, 2016~ 20.4%(MDD −24.1%), 최악의 해 −12.9%

이 모듈은 미국 레그(V+R)의 목표비중을 계산한다. research 모듈의 백테스트 함수를 그대로 호출해
실전 신호와 백테스트 신호가 같은 코드에서 나오게 했다(구현 불일치 위험 제거).
"""
from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from autotrader.config import TriadUSSettings
from autotrader.research.tournament import Ctx, hr_kr_part, hr_us_part, lev_risk_parity, recovery_boost

SIGNAL_ASSETS = ["QQQ", "TLT", "GLD", "DBC"]   # 신호 계산에 필요한 1배 기초자산(연구용 이름)
MIN_DAYS = 260 + 504                           # 200일선 + 2년 폭락 판정 창 + 12개월 추세


def total_return_proxy(closes: pd.DataFrame, yields: dict[str, float]) -> pd.DataFrame:
    """분배금 미반영 가격 × exp(연 분배율 × 경과연수) — 추세·낙폭 판정을 총수익 기준에 가깝게."""
    t = (closes.index - closes.index[0]).days.values / 365.25
    out = closes.copy()
    for c in out.columns:
        y = float(yields.get(c, 0.0))
        if y:
            out[c] = out[c] * np.exp(y * t)
    return out


def us_leg_targets(closes: pd.DataFrame, s: TriadUSSettings, is_month_end: Callable[[date], bool],
                   diag: dict | None = None) -> pd.Series:
    """closes: 신호일까지의 일봉 종가(열 = s.tickers 로 매핑된 미국 티커, 최소 QQQ·TLT·GLD·DBC 기초 ETF).

    반환: 미국 티커별 목표비중(합 1, 현금성 ETF 포함, 0 비중 티커도 포함 → 정리 매도 대상).
    """
    tick = s.tickers
    inv = {tick[a]: a for a in SIGNAL_ASSETS}
    missing = [tick[a] for a in SIGNAL_ASSETS if tick[a] not in closes.columns]
    if missing:
        raise RuntimeError(f"TRIAD 신호용 일봉 없음: {missing}")
    P = closes[[tick[a] for a in SIGNAL_ASSETS]].rename(columns=inv).dropna(how="all").ffill()
    if len(P) < MIN_DAYS:
        raise RuntimeError(f"TRIAD 신호에 일봉 {MIN_DAYS}개 이상 필요(현재 {len(P)}) → history_years 를 늘리세요")
    P = total_return_proxy(P, s.distribution_yield)
    days = P.index
    c = Ctx(R=P.pct_change().fillna(0.0), P=P, vix=pd.Series(dtype=float),
            rf_us=pd.Series(float(s.cash_rate), index=days))
    # 월말 = 실제 그달 마지막 거래일(캘린더 기준). 데이터 끝이 월중이면 그날은 월말 신호가 아니다
    c.me = pd.DatetimeIndex([d for d in days if is_month_end(d.date())])
    vt = recovery_boost("QQQ", s.vt_target, s.crash, s.boost_leverage, s.boost_days)(c)
    rp = lev_risk_parity(s.lrp_target, 3, True)(c)
    if vt.empty or rp.empty:
        raise RuntimeError("TRIAD 신호가 비어 있습니다(데이터 기간 부족)")
    v_row, r_row = vt.iloc[-1].fillna(0.0), rp.iloc[-1].fillna(0.0)
    core = (s.vt_weight * v_row).add((1 - s.vt_weight) * r_row, fill_value=0.0)
    h_row = pd.Series(dtype=float)
    if s.hr_weight > 0:       # QUAD: 고금리 대비형 미국 부분(나스닥 GC·VT + 금, 고금리면 레버리지 상한 축소)
        hr = hr_us_part(c)
        if hr.empty:
            raise RuntimeError("고금리 대비형 신호가 비어 있습니다(데이터 기간 부족)")
        h_row = hr.iloc[-1].fillna(0.0)
    w = ((1 - s.hr_weight) * core).add(s.hr_weight * h_row, fill_value=0.0)
    out: dict[str, float] = {t: 0.0 for t in tick.values()}
    for k, v in w.items():
        if k not in tick:
            raise RuntimeError(f"티커 매핑 없음: {k} (triad_us.tickers 에 추가)")
        out[tick[k]] += float(v)
    tgt = pd.Series(out).clip(lower=0.0)
    tot = tgt.sum()
    if tot > 1 + 1e-6:
        tgt = tgt / tot
    if diag is not None:
        q = P["QQQ"]
        vol21 = float(q.pct_change().iloc[-21:].std() * np.sqrt(252))
        dd2y = float((q / q.cummax() - 1).iloc[-504:].min())
        gc = bool(q.rolling(50).mean().iloc[-1] > q.rolling(200).mean().iloc[-1])
        e_vt = float(v_row.get("QQQ", 0) + 3 * v_row.get("QQQx3", 0))
        diag["triad_us"] = {
            "signal_date": str(days[-1].date()), "vt_row": str(vt.index[-1].date()), "rp_row": str(rp.index[-1].date()),
            "qqq_vol21": round(vol21, 4), "qqq_dd_2y": round(dd2y, 4), "golden_cross": gc,
            "vt_exposure": round(e_vt, 3), "boost_on": e_vt >= s.boost_leverage - 1e-6,
            "vt_weights": {k: round(float(v), 4) for k, v in v_row.items() if v}, "rp_weights":
                {k: round(float(v), 4) for k, v in r_row.items() if v},
            "hr_weights": {k: round(float(v), 4) for k, v in h_row.items() if v},
            "high_rate_cap": bool(s.cash_rate >= 0.035)}
    return tgt


def hr_kospi_targets(kospi_closes: pd.Series, symbols: dict[str, str], us_rate: float,
                     is_month_end: Callable[[date], bool], diag: dict | None = None) -> pd.Series:
    """QUAD 국내 레그의 '고금리 대비형' 코스피 부분(1로 정규화). kospi_closes = KODEX 200 등 코스피200 추종 종가.

    symbols: {"K200": 1배 ETF, "K200x2": 2배 ETF, "CASH_KRW": 현금성 ETF}. 백테스트(hr_kr_part)와 같은 함수를 쓴다.
    """
    k = kospi_closes.dropna()
    if len(k) < 260:
        raise RuntimeError(f"코스피 신호에 일봉 260개 이상 필요(현재 {len(k)})")
    P = pd.DataFrame({"K200": k})
    c = Ctx(R=P.pct_change().fillna(0.0), P=P, vix=pd.Series(dtype=float), rf_us=pd.Series(float(us_rate), index=P.index))
    c.me = pd.DatetimeIndex([d for d in P.index if is_month_end(d.date())])
    rows = hr_kr_part(c)
    if rows.empty:
        raise RuntimeError("코스피 신호가 비어 있습니다")
    row = rows.iloc[-1].fillna(0.0)
    out = pd.Series({symbols[k_]: 0.0 for k_ in ("K200", "K200x2", "CASH_KRW")})
    for k_, v in row.items():
        out[symbols[k_]] += float(v)
    if diag is not None:
        diag["hr_kospi"] = {"row": str(rows.index[-1].date()), "exposure": round(float(row.get("K200", 0) + 2 * row.get("K200x2", 0)), 2),
                            "weights": out.round(4).to_dict(), "high_rate_cap": bool(us_rate >= 0.035)}
    return out
