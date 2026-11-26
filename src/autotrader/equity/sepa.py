"""SEPA-K — 마크 미너비니 SEPA(Specific Entry Point Analysis)의 한국 시장 정량화 버전.

미너비니 방법론(『Trade Like a Stock Market Wizard』, 『Think & Trade Like a Champion』)의 핵심을
가격·거래대금 데이터로 구현한다. (재무 조건 — EPS·매출 가속, 어닝 서프라이즈 — 은 데이터가 없어 제외하고,
'대량 거래 동반 갭상승'을 촉매의 가격 대리변수로 쓸 수 있게 옵션으로 둔다.)

1) 트렌드 템플릿 8개 조건(2단계 상승추세만 매수)
   ① 종가 > 150일·200일 이동평균  ② 150일선 > 200일선  ③ 200일선이 최소 1개월 상승
   ④ 50일선 > 150일·200일선  ⑤ 종가 > 50일선  ⑥ 52주 저가 대비 +30% 이상
   ⑦ 52주 고가 대비 −25% 이내  ⑧ RS 등급 ≥ 70 (IBD식: 3개월 40% + 6·9·12개월 각 20%, 전 종목 백분위)
2) VCP(변동성 수축 패턴): 최근 10일 변동폭이 좁고(≤ tight_pct), 직전 구간보다 수축(≤ contraction),
   거래대금이 말라 있음(최근 10일 ≤ dryup × 50일 평균), 베이스 깊이 ≤ max_base_depth
3) 진입: 피벗(직전 base_len 일 최고가) 돌파 시 매수 — buy_stop(다음날 피벗 도달 시 체결, 실거래는 NH 스톱지정가)
   피벗보다 5% 이상 위에서 시작하면(과도한 추격) 매수하지 않음
4) 손절: 마지막 수축 저점 아래(최대 8%). 장중 저가가 닿으면 손절가(갭하락이면 시가)로 청산
5) 이익 보호: +2R 도달 시 손절을 본전으로, +partial_at 에서 절반 이익실현, 큰 수익 후 50일선 이탈 시 청산
   진입 후 time_stop_days 일 동안 +5% 못 가면 청산(죽은 돈)
6) 포지션 크기: 1회 위험 = 자산의 risk_per_trade(기본 1.25%) ÷ 손절폭, 종목당 최대 max_position
7) 진행형 노출(progressive exposure): 낙폭 10% 초과 시 위험 절반, 20% 초과 시 1/4
8) 시장 필터: 시장지수 200일선 위 + 트렌드템플릿 통과 종목 비율(breadth)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.equity.backtest import StockCosts
from autotrader.equity.panel import Panel

log = logging.getLogger(__name__)


@dataclass
class SEPAParams:
    # 트렌드 템플릿
    ma200_up_days: int = 21
    min_above_low: float = 0.30
    max_below_high: float = 0.25
    min_rs: float = 70.0
    max_rs: float = 100.0             # 한국 적응: 극단적 RS(과열·복권형) 제외 시 95
    max_above_ma50: float | None = None  # 50일선 대비 과도한 이격(추격매수) 제외, 예: 0.15
    # VCP / 피벗
    base_len: int = 40
    max_base_depth: float = 0.35
    tight_pct: float = 0.12
    contraction: float = 0.7
    dryup: float = 0.9
    max_extension: float = 0.05
    entry: str = "buy_stop"          # buy_stop | breakout_close
    breakout_vol: float = 1.4        # breakout_close 모드: 돌파일 거래대금 ≥ 50일 평균 × 이 값
    # 유동성 유니버스
    min_adv: float = 5e8             # 20일 평균 거래대금(원)
    min_marcap: float = 5e10         # 시가총액(원)
    min_price: float = 2000.0
    # 위험관리
    max_stop: float = 0.08
    risk_per_trade: float = 0.0125
    max_position: float = 0.20
    max_positions: int = 10
    breakeven_r: float = 2.0
    partial_at: float | None = 0.25
    trail_ma50_after: float = 0.20
    time_stop_days: int = 25
    time_stop_gain: float = 0.05
    fail_exit_days: int = 0          # >0: 진입 후 N일 내 종가가 피벗 아래로 복귀하면 다음날 시가 청산(돌파 실패)
    fail_tol: float = 0.01
    progressive: bool = True
    # 시장 필터
    market_filter: str = "ma200_breadth"  # none | ma200 | ma200_breadth | ma50_breadth
    min_breadth: float = 0.03
    # 체결 비용(ADV 대비 주문 크기 기반 시장충격)
    adv_cap: float = 0.10            # 1회 주문 ≤ 20일 평균 거래대금의 10%
    impact_k: float = 0.8            # 충격 = base + k·σ_일·√(주문/ADV)


@dataclass
class Indicators:
    dates: pd.DatetimeIndex
    codes: pd.Index
    P: np.ndarray       # 수정 종가 지수
    O: np.ndarray       # 수정 시가
    H: np.ndarray       # 수정 고가
    Lo: np.ndarray      # 수정 저가
    ma50: np.ndarray
    setup: np.ndarray   # bool: 오늘 종가 기준 셋업 성립(내일 피벗 돌파 시 매수)
    pivot: np.ndarray
    stop_ref: np.ndarray  # 마지막 수축 저점
    rs: np.ndarray
    adv20: np.ndarray
    sigma: np.ndarray   # 20일 일간 변동성
    tradable: np.ndarray  # 당일 거래 가능(정지 아님)
    market_ok: np.ndarray  # 날짜별 bool
    breadth: np.ndarray
    raw_close: np.ndarray
    tt: np.ndarray | None = None       # 트렌드 템플릿 8개 조건(RS 포함) 통과
    stage2: np.ndarray | None = None   # 기본 2단계: 종가>150·200일선, 200일선 상승


def compute_indicators(p: Panel, prm: SEPAParams, chunk: int = 500) -> Indicators:
    if p.hi is None or p.lo is None:
        raise ValueError("패널에 고가·저가(hi/lo)가 없습니다 → build_panel 로 다시 생성하세요")
    dates, codes = p.dates, p.ret.columns
    T, N = len(dates), len(codes)
    f32 = lambda: np.full((T, N), np.nan, dtype=np.float32)  # noqa: E731
    P, O, H, Lo, ma50, pivot, stop_ref, perf, adv20, sigma = (f32() for _ in range(10))
    tt = np.zeros((T, N), dtype=bool)
    st2 = np.zeros((T, N), dtype=bool)
    vcp = np.zeros((T, N), dtype=bool)
    for k in range(0, N, chunk):
        cc = codes[k: k + chunk]
        sl = slice(k, k + len(cc))
        ret = p.ret[cc].astype(np.float64)
        listed = ret.notna()
        Lg = np.log1p(ret.fillna(0.0)).cumsum().where(listed)
        Pc = np.exp(Lg)
        Hc = Pc * p.hi[cc].astype(np.float64)
        Lc = Pc * p.lo[cc].astype(np.float64)
        Oc = Pc.shift(1) * (1 + p.gap[cc].astype(np.float64))
        m50 = Pc.rolling(50, min_periods=50).mean()
        m150 = Pc.rolling(150, min_periods=150).mean()
        m200 = Pc.rolling(200, min_periods=200).mean()
        hi52 = Hc.rolling(252, min_periods=200).max()
        lo52 = Lc.rolling(252, min_periods=200).min()
        t_ok = ((Pc > m150) & (Pc > m200) & (m150 > m200) & (m200 > m200.shift(prm.ma200_up_days))
                & (m50 > m150) & (m50 > m200) & (Pc > m50)
                & (Pc >= lo52 * (1 + prm.min_above_low)) & (Pc >= hi52 * (1 - prm.max_below_high)))
        amt = p.amount[cc].astype(np.float64)
        a20 = amt.rolling(20, min_periods=15).mean()
        a50 = amt.rolling(50, min_periods=40).mean()
        a10 = amt.rolling(10, min_periods=8).mean()
        piv = Hc.rolling(prm.base_len, min_periods=prm.base_len).max()          # 오늘까지의 베이스 고점
        blow = Lc.rolling(prm.base_len, min_periods=prm.base_len).min()
        h10, l10 = Hc.rolling(10).max(), Lc.rolling(10).min()
        rng10 = (h10 - l10) / h10
        h_prior, l_prior = Hc.shift(10).rolling(30).max(), Lc.shift(10).rolling(30).min()
        rng_prior = (h_prior - l_prior) / h_prior
        v_ok = ((rng10 <= prm.tight_pct) & (rng10 <= rng_prior * prm.contraction)
                & (a10 <= a50 * prm.dryup) & ((piv - blow) / piv <= prm.max_base_depth)
                & (Pc >= piv * (1 - prm.max_extension * 2)) & (Pc <= piv))
        if prm.entry == "breakout_close":
            prev_piv = Hc.shift(1).rolling(prm.base_len, min_periods=prm.base_len).max()
            v_ok = ((Pc > prev_piv) & (amt >= a50 * prm.breakout_vol)
                    & ((prev_piv - blow) / prev_piv <= prm.max_base_depth) & (Pc <= prev_piv * (1 + prm.max_extension)))
            piv = prev_piv
        pf = (0.4 * (Pc / Pc.shift(63) - 1) + 0.2 * (Pc / Pc.shift(126) - 1) + 0.2 * (Pc / Pc.shift(189) - 1)
              + 0.2 * (Pc / Pc.shift(252) - 1))
        for arr, src in ((P, Pc), (O, Oc), (H, Hc), (Lo, Lc), (ma50, m50), (pivot, piv), (stop_ref, l10),
                         (perf, pf), (adv20, a20), (sigma, ret.where(listed).rolling(20, min_periods=15).std())):
            arr[:, sl] = src.to_numpy(np.float32)
        tt[:, sl] = t_ok.to_numpy(bool)
        st2[:, sl] = ((Pc > m150) & (Pc > m200) & (m200 > m200.shift(prm.ma200_up_days))).to_numpy(bool)
        vcp[:, sl] = v_ok.to_numpy(bool)
        log.info("SEPA 지표 %d/%d", min(k + chunk, N), N)
    rs = pd.DataFrame(perf, index=dates, columns=codes).rank(axis=1, pct=True).to_numpy(np.float32) * 99
    raw_close = p.close.to_numpy(np.float32)
    liq = (adv20 >= prm.min_adv) & (p.marcap.to_numpy(np.float32) >= prm.min_marcap) & (raw_close >= prm.min_price)
    liq &= np.nan_to_num(p.flag.to_numpy(np.float32), nan=0.0) < 0.5
    tt &= (rs >= prm.min_rs) & (rs <= prm.max_rs)
    if prm.max_above_ma50 is not None:
        tt &= P <= ma50 * (1 + prm.max_above_ma50)
    listed_any = ~np.isnan(P)
    breadth = (tt & listed_any).sum(axis=1) / np.maximum(listed_any.sum(axis=1), 1)
    mk = p.market_return().to_numpy()
    mlvl = np.cumprod(1 + mk)
    m200 = pd.Series(mlvl).rolling(200, min_periods=150).mean().to_numpy()
    m50 = pd.Series(mlvl).rolling(50, min_periods=40).mean().to_numpy()
    if prm.market_filter == "none":
        market_ok = np.ones(T, dtype=bool)
    elif prm.market_filter == "ma50_breadth":
        market_ok = (mlvl > np.nan_to_num(m50, nan=np.inf)) & (mlvl > np.nan_to_num(m200, nan=np.inf) * 0.97)
        market_ok &= breadth >= prm.min_breadth
    elif prm.market_filter == "ma200":
        market_ok = mlvl > np.nan_to_num(m200, nan=np.inf)
    else:
        market_ok = (mlvl > np.nan_to_num(m200, nan=np.inf)) | (breadth >= prm.min_breadth * 2)
        market_ok &= breadth >= prm.min_breadth
    setup = tt & vcp & liq
    tradable = ~np.isnan(p.ret.to_numpy(np.float32)) & (p.amount.to_numpy(np.float32) > 0)
    return Indicators(dates, codes, P, O, H, Lo, ma50, setup, pivot, stop_ref, rs, adv20, sigma, tradable,
                      market_ok, breadth, raw_close, tt=tt, stage2=st2)


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_px: float
    stop: float
    risk_pct: float
    value0: float
    exit_date: pd.Timestamp | None = None
    exit_px: float | None = None
    reason: str = ""
    pnl: float = 0.0
    ret: float = 0.0


@dataclass
class SEPAResult:
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    params: SEPAParams = field(repr=False, default=None)


def _impact(order_value: float, adv: float, sigma: float, prm: SEPAParams, costs: StockCosts) -> float:
    if not np.isfinite(adv) or adv <= 0:
        return 0.01
    s = sigma if np.isfinite(sigma) and sigma > 0 else 0.03
    return costs.slip_base + prm.impact_k * s * np.sqrt(max(order_value, 0.0) / adv)


def simulate_sepa(ind: Indicators, prm: SEPAParams, costs: StockCosts | None = None, capital: float = 1e8,
                  start: str = "2001-01-02", end: str | None = None) -> SEPAResult:
    """일별 거래 시뮬레이터(장중 손절·갭·가격제한·정지·ADV 기반 충격비용)."""
    costs = costs or StockCosts()
    D = ind.dates
    s = int(np.searchsorted(D, pd.Timestamp(start)))
    e = len(D) - 1 if end is None else int(np.searchsorted(D, pd.Timestamp(end), side="right")) - 1
    last_valid = (~np.isnan(ind.P)).cumsum(axis=0).argmax(axis=0)
    cash = float(capital)
    pos: dict[int, dict] = {}   # j -> {shares, entry, stop, risk, days, partial, peak_gain, trade}
    trades: list[Trade] = []
    eq_hist, ex_hist = [], []
    peak_eq = capital
    for t in range(max(s, 1), e + 1):
        d = D[t]
        tax = costs.sell_tax(d.date())
        # ---------------- 보유 종목 관리(청산 먼저)
        for j in list(pos):
            st = pos[j]
            if t > last_valid[j]:                       # 상장폐지
                px = ind.P[last_valid[j], j] * (1 - costs.delist_haircut)
                _close(st, j, px, d, "상장폐지", trades, ind, costs, tax, prm, pos, sell_all=True)
                cash += st["_proceeds"]
                continue
            if not ind.tradable[t, j]:
                continue
            o, h, lo, c = ind.O[t, j], ind.H[t, j], ind.Lo[t, j], ind.P[t, j]
            prev = ind.P[t - 1, j]
            st["days"] += 1
            exit_px, reason = None, ""
            limit_down_open = o <= prev * 0.71
            if lo <= st["stop"] and not limit_down_open:
                exit_px, reason = min(o, st["stop"]), "손절" if st["stop"] < st["entry"] else "이익보호 손절"
            elif st.get("exit_next_open"):
                exit_px, reason = o, st["exit_next_open"]
            if exit_px is not None:
                _close(st, j, exit_px, d, reason, trades, ind, costs, tax, prm, pos, sell_all=True)
                cash += st["_proceeds"]
                continue
            gain = c / st["entry"] - 1
            st["peak_gain"] = max(st["peak_gain"], gain)
            if prm.partial_at and not st["partial"] and h >= st["entry"] * (1 + prm.partial_at):
                px = max(o, st["entry"] * (1 + prm.partial_at))
                _close(st, j, px, d, "부분 이익실현", trades, ind, costs, tax, prm, pos, sell_all=False)
                cash += st["_proceeds"]
                st["partial"] = True
            if gain >= prm.breakeven_r * st["risk"]:
                st["stop"] = max(st["stop"], st["entry"] * 1.005)
            if prm.fail_exit_days and st["days"] <= prm.fail_exit_days and c < st["pivot"] * (1 - prm.fail_tol):
                st["exit_next_open"] = "돌파 실패"
            elif st["peak_gain"] >= prm.trail_ma50_after and c < ind.ma50[t, j]:
                st["exit_next_open"] = "50일선 이탈"
            elif st["days"] >= prm.time_stop_days and st["peak_gain"] < prm.time_stop_gain:
                st["exit_next_open"] = "시간 손절"
        # ---------------- 평가
        equity = cash + sum(st["shares"] * ind.P[t - 1 if np.isnan(ind.P[t, j]) else t, j] for j, st in pos.items())
        # ---------------- 신규 진입(전일 셋업 → 오늘 피벗 돌파)
        if ind.market_ok[t - 1] and len(pos) < prm.max_positions:
            cand = np.where(ind.setup[t - 1])[0]
            cand = [j for j in cand if j not in pos and ind.tradable[t, j]]
            cand.sort(key=lambda j: -ind.rs[t - 1, j])
            dd = equity / peak_eq - 1
            scale = 1.0
            if prm.progressive:
                scale = 0.25 if dd <= -0.20 else 0.5 if dd <= -0.10 else 1.0
            for j in cand:
                if len(pos) >= prm.max_positions:
                    break
                piv = ind.pivot[t - 1, j]
                o, h, prev = ind.O[t, j], ind.H[t, j], ind.P[t - 1, j]
                if not np.isfinite(piv) or o >= prev * 1.29:          # 상한가 시작 → 체결 불가
                    continue
                if prm.entry == "buy_stop":
                    if h < piv or o > piv * (1 + prm.max_extension):
                        continue
                    px = max(o, piv)
                else:
                    if o > piv * (1 + prm.max_extension):
                        continue
                    px = o
                ref = ind.stop_ref[t - 1, j]
                stop = max(px * (1 - prm.max_stop), (min(ref, px) if np.isfinite(ref) else 0.0) * 0.995)
                stop = min(stop, px * 0.98)        # 최소 2% 여유
                risk = 1 - stop / px
                target_val = min(equity * prm.risk_per_trade * scale / max(risk, 0.02), equity * prm.max_position)
                target_val = min(target_val, ind.adv20[t - 1, j] * prm.adv_cap, cash * 0.98)
                if target_val < equity * 0.01:
                    continue
                slip = _impact(target_val, ind.adv20[t - 1, j], ind.sigma[t - 1, j], prm, costs)
                fill = px * (1 + slip)
                shares = target_val / fill
                cash -= target_val * (1 + costs.commission)
                tr = Trade(str(ind.codes[j]), d, float(fill), float(stop), float(risk), float(target_val))
                pos[j] = {"shares": shares, "entry": fill, "stop": stop, "risk": risk, "days": 0, "partial": False,
                          "peak_gain": 0.0, "trade": tr, "cost": target_val * (1 + costs.commission), "pivot": piv}
                # 진입 당일 손절 체크(장중 저가, 보수적으로 돌파 후 하락으로 간주)
                if ind.Lo[t, j] <= stop:
                    st_new = pos[j]
                    _close(st_new, j, stop, d, "당일 손절", trades, ind, costs, tax, prm, pos, sell_all=True)
                    cash += st_new["_proceeds"]
                elif prm.fail_exit_days and ind.P[t, j] < piv * (1 - prm.fail_tol):
                    pos[j]["exit_next_open"] = "돌파 실패"
        equity = cash + sum(st["shares"] * ind.P[t if not np.isnan(ind.P[t, j]) else t - 1, j] for j, st in pos.items())
        peak_eq = max(peak_eq, equity)
        eq_hist.append(equity)
        ex_hist.append(1 - cash / equity if equity > 0 else 0.0)
    idx = D[max(s, 1): e + 1]
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    return SEPAResult(pd.Series(eq_hist, index=idx) / capital, pd.Series(ex_hist, index=idx), tdf, prm)


def _close(st: dict, j: int, px: float, d, reason: str, trades: list, ind: Indicators, costs: StockCosts,
           tax: float, prm: SEPAParams, pos: dict, sell_all: bool) -> None:
    """청산(전량/절반). st['_proceeds'] 에 현금 유입액을 기록."""
    t = int(np.searchsorted(ind.dates, d))
    frac = 1.0 if sell_all else 0.5
    sh = st["shares"] * frac
    gross = sh * px
    adv = ind.adv20[t - 1, j] if t > 0 else np.nan
    slip = _impact(gross, adv, ind.sigma[t - 1, j] if t > 0 else np.nan, prm, costs)
    if reason in ("손절", "이익보호 손절", "당일 손절"):
        slip *= 1.5  # 손절 체결은 불리하게
    net = gross * (1 - slip - costs.commission - tax)
    st["_proceeds"] = net
    tr: Trade = st["trade"]
    cost_part = st["cost"] * frac
    pnl = net - cost_part
    if sell_all:
        tr.exit_date, tr.exit_px, tr.reason = d, float(px), reason
        tr.pnl += pnl
        tr.ret = (tr.pnl + 0.0) / tr.value0
        trades.append(tr)
        pos.pop(j, None)
    else:
        tr.pnl += pnl
        st["shares"] -= sh
        st["cost"] -= cost_part


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    hold = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
    return {"trades": int(len(r)), "win_rate": float((r > 0).mean()), "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "payoff": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
            "expectancy": float(r.mean()), "median_hold_days": float(hold.median()),
            "reasons": trades["reason"].value_counts().to_dict()}
