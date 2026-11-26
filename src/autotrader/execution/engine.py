"""실거래 실행엔진.

한 번의 실행(run_once) 흐름
  1. 안전점검: 킬스위치 파일, 실계좌 확인 문구, 거래일·거래시간
  2. 리밸런싱 필요 여부: 이번 달 미실행(월초 첫 거래일 이후) / 비중 이탈 / 강제
  3. 신호: 증권사 일봉(로컬 캐시) → '어제 종가'까지만 사용해 목표비중 계산(미래참조 방지)
  4. 계획: 계좌 스냅샷 + 현재가 → 1주 단위 주문계획 → 위험한도 점검
  5. 실행: 매도 → 매수 순서. 라운드마다 '실제 잔고'를 다시 조회해 남은 차이만 주문(재조정 방식)
           → 부분체결·주문번호 체계 차이·네트워크 오류에도 이중주문 없이 수렴
           즉시 체결 지향 지정가(매수=매도호가+n틱) → 미체결은 취소 → 마지막 라운드는 시장가
  6. 기록·알림: SQLite(runs/orders/rebalances/equity) + 텔레그램
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from autotrader.brokers.base import Broker, BrokerError
from autotrader.calendar import KRXCalendar
from autotrader.config import AppConfig
from autotrader.data.loader import BrokerHistoryCache
from autotrader.fees import get_fee_schedule
from autotrader.models import OrderRequest, OrderStatus, OrderType, Quote, Side
from autotrader.notify import Notifier
from autotrader.portfolio.rebalancer import RebalancePlan, plan_rebalance
from autotrader.portfolio.risk import RiskManager, RiskViolation
from autotrader.execution.state import StateStore
from autotrader.strategy import build_strategy
from autotrader.strategy.base import StrategyContext
from autotrader.strategy.signals import DAILY
from autotrader.ticks import round_to_tick, shift_ticks

log = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")


@dataclass
class RunReport:
    status: str                    # skipped | dry_run | done | partial | error
    message: str = ""
    targets: dict = field(default_factory=dict)
    plan_text: str = ""
    orders: list[dict] = field(default_factory=list)


def _hm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


class TradingEngine:
    def __init__(self, cfg: AppConfig, broker: Broker, *, store: StateStore | None = None,
                 notifier: Notifier | None = None, calendar: KRXCalendar | None = None,
                 clock=lambda: datetime.now(KST), sleep=time.sleep, history: BrokerHistoryCache | None = None):
        self.cfg = cfg
        self.broker = broker
        self.store = store or StateStore(cfg.storage.db_path)
        self.notifier = notifier or Notifier(cfg.notify.telegram)
        self.calendar = calendar or KRXCalendar(cfg.extra_holidays, broker_check=broker.is_trading_day)
        self.clock = clock
        self.sleep = sleep
        self.history = history or BrokerHistoryCache(broker, cfg.storage.cache_dir)
        self.risk = RiskManager(cfg.risk, broker.is_live)
        self.fees = get_fee_schedule(cfg.costs.preset, commission_rate=cfg.costs.commission_rate)
        self.names = {s: cfg.universe.name_of(s) for s in cfg.universe.all_symbols()}

    # ------------------------------------------------------------------ 판단
    def rebalance_due(self, today: date, force: bool = False) -> tuple[bool, str]:
        if force:
            return True, "강제 실행"
        period = today.strftime("%Y-%m")
        if not self.store.rebalanced(period):
            first = self.calendar.first_trading_day_of_month(today.year, today.month)
            if today >= first:
                return True, f"{period} 정기 리밸런싱"
        thr = self.cfg.rebalance.drift_threshold
        last = self.store.last_targets()
        if last:  # 이번 달 손절한 종목은 의도된 이탈이므로 비중 이탈 계산에서 제외
            stopped = self.store.stopped(period)
            last = {s: w for s, w in last.items() if s not in stopped}
        if thr and last:
            acct = self.broker.get_account()
            prices = {s: (acct.positions[s].last_price if s in acct.positions else 0.0) for s in last}
            held = sum(acct.qty(s) * prices[s] for s in last)
            base = held + acct.cash
            if self.cfg.risk.capital_limit:
                base = min(base, self.cfg.risk.capital_limit)
            if base > 0:
                drift = max(abs(acct.qty(s) * prices[s] / base - w) for s, w in last.items())
                if drift > thr:
                    return True, f"비중 이탈 {drift:.1%} > {thr:.0%}"
        return False, "리밸런싱 불필요"

    def compute_targets(self, today: date, account=None) -> tuple[pd.Series, dict]:
        if self.cfg.strategy.name == "msap":
            return self._msap_targets(today, account if account is not None else self.broker.get_account())
        syms = self.cfg.universe.all_symbols()
        start = today - timedelta(days=365 * self.cfg.rebalance.history_years + 30)
        closes = self.history.closes(syms, start, today)
        closes = closes[closes.index < pd.Timestamp(today)]  # 오늘(장중) 바 제외 → 어제 종가까지만
        if closes.empty:
            raise RuntimeError("일봉 데이터가 없습니다")
        ml_scores, ml_note = None, {}
        blend = self.cfg.ml.blend if self.cfg.ml.enabled else 0.0
        if blend > 0:
            ml_scores, ml_note, blend = self._ml_scores(closes, blend)
        strategy = build_strategy(self.cfg, ml_blend=blend)
        ctx = StrategyContext(bars=DAILY, as_of=closes.index[-1], ml_scores=ml_scores)
        w = strategy.target_weights(closes, ctx)
        diag = {**ctx.diagnostics, "signal_date": str(closes.index[-1].date()), **ml_note}
        return w, diag

    def _msap_targets(self, today: date, account) -> tuple[pd.Series, dict]:
        """멀티 슬리브: 종목 알파(ML 랭킹) + ETF 리스크패리티."""
        from dataclasses import asdict

        from autotrader.equity.live import (StockSleeveConfig, latest_rows, liquidity_cap, score_rows, select,
                                            sync_marcap)
        from autotrader.equity.panel import build_panel
        from autotrader.strategy.multi_sleeve import MSAPConfig, combine, etf_sleeve_weights

        ms = self.cfg.msap
        st_kw = {k: v for k, v in asdict(ms.stock).items() if k in StockSleeveConfig.__dataclass_fields__}
        mcfg = MSAPConfig(stock_weight=ms.stock_weight, stock=StockSleeveConfig(**st_kw),
                          rp_target_vol=ms.rp_target_vol, rp_vol_window_months=ms.rp_vol_window_months,
                          rp_max_leverage=ms.rp_max_leverage)
        diag: dict = {}
        uni = self.cfg.universe
        risk = [a.symbol for a in uni.risk_assets]
        cash = uni.cash_asset.symbol if uni.cash_asset else None
        start = today - timedelta(days=365 * self.cfg.rebalance.history_years + 30)
        closes = self.history.closes(uni.all_symbols(), start, today)
        closes = closes[closes.index < pd.Timestamp(today)]
        etf_w = etf_sleeve_weights(closes.resample("ME").last(), risk, cash, mcfg, diag)

        data_dir = sync_marcap(mcfg.stock.marcap_dir) if ms.stock.sync else Path(mcfg.stock.marcap_dir) / "data"
        panel = build_panel(data_dir, start_year=1995)
        rows = latest_rows(panel, mcfg.stock, as_of=pd.Timestamp(today) - pd.Timedelta(days=1))
        sig_date = rows["date"].max()
        age = (pd.Timestamp(today) - sig_date).days
        if age > ms.stock.max_signal_age_days:
            raise RuntimeError(f"종목 데이터가 {age}일 지났습니다(신호일 {sig_date.date()}) → marcap 갱신 확인")
        scores = score_rows(rows, mcfg.stock.model_path)
        prev = self.store.last_targets() or {}
        etf_set = set(uni.all_symbols())
        prev_stocks = [s for s in prev if s not in etf_set]
        held = [s for s in prev_stocks if account.qty(s) > 0]
        stock_w = select(scores, held, mcfg.stock)
        stopped = self.store.stopped(today.strftime("%Y-%m"))
        if stopped & set(stock_w.index):   # 손절 종목은 다음 달 리밸런싱 전까지 재매수하지 않음(슬롯은 현금)
            diag["stopped_excluded"] = sorted(stopped & set(stock_w.index))
            stock_w = stock_w.where(~stock_w.index.isin(list(stopped)), 0.0)
        # 유동성 한도: 종목당 보유금액 ≤ 20일 평균 거래대금 × adv_cap (초과분은 현금성 ETF 로)
        equity = account.cash + sum(p.market_value for p in account.positions.values())
        if self.cfg.risk.capital_limit:
            equity = min(equity, self.cfg.risk.capital_limit)
        adv20 = self._adv20(panel, list(stock_w.index), sig_date)
        stock_w, cut = liquidity_cap(stock_w, adv20, equity * mcfg.stock_weight, ms.stock.adv_cap)
        if cut:
            diag["liquidity_cut"] = cut
            log.warning("유동성 한도로 비중 축소: %s", cut)
        self.names.update({c: str(panel.names.get(c, c)) for c in stock_w.index})
        target = combine(stock_w, etf_w, mcfg.stock_weight, prev_stocks)
        leftover = mcfg.stock_weight * (1.0 - float(stock_w.sum()))
        if leftover > 1e-9 and cash:
            target[cash] = target.get(cash, 0.0) + leftover
        diag.update({"signal_date": str(sig_date.date()), "stocks": list(stock_w.index),
                     "stock_scores": scores.loc[stock_w.index].round(4).to_dict()})
        return target, diag

    def _stop_check(self, today: date, dry_run: bool = False) -> str:
        """미너비니식 손절(종가 기준): 전일 종가가 평균 매입가 × (1 − stop_loss) 이하인 전략 보유 종목을 매도.

        백테스트의 stop_mode='close'(종가 확인 → 다음날 시가 매도)와 같은 규칙이다.
        """
        stop = float(self.cfg.msap.stock.stop_loss)
        prev = self.store.last_targets() or {}
        etf_set = set(self.cfg.universe.all_symbols())
        managed = [s for s in prev if s not in etf_set]
        if not managed:
            return ""
        acct = self.broker.get_account()
        hits = {}
        for s in managed:
            pos = acct.positions.get(s)
            if not pos or pos.qty <= 0 or pos.avg_price <= 0:
                continue
            q = self.broker.get_quote(s)
            ref = q.prev_close or q.price
            if ref <= pos.avg_price * (1 - stop):
                hits[s] = f"전일종가 {ref:,.0f} ≤ 매입가 {pos.avg_price:,.0f}×{1 - stop:.2f}"
        if not hits:
            return ""
        names = ", ".join(f"{self.names.get(s, s)}({s})" for s in hits)
        if dry_run:
            return f"[드라이런] 손절 대상 {len(hits)}종목: {names}"
        run_id = self.store.start_run("stop")
        self._execute(run_id, pd.Series({s: 0.0 for s in hits}))
        period = today.strftime("%Y-%m")
        for s, why in hits.items():
            self.store.mark_stopped(period, s, why)
        msg = f"손절 {len(hits)}종목 매도: {names}"
        self.store.finish_run(run_id, "done", msg, {"stops": hits})
        self.notifier.send(f"🛑 {msg}")
        return msg

    @staticmethod
    def _adv20(panel, codes: list[str], as_of) -> pd.Series:
        try:
            amt = panel.amount.loc[:as_of, [c for c in codes if c in panel.amount.columns]].tail(20)
            return amt.mean()
        except Exception:  # 테스트용 가짜 패널 등 거래대금이 없으면 한도 미적용
            return pd.Series({c: float("inf") for c in codes})

    def _ml_scores(self, closes: pd.DataFrame, blend: float):
        try:
            from autotrader.ml.train import load_and_predict
            scores, val = load_and_predict(self.cfg.ml.model_path, closes)
        except Exception as e:  # 모델 문제로 매매가 멈추지 않게: ML 없이 진행
            log.warning("ML 예측 실패 → ML 미반영: %s", e)
            return None, {"ml": f"실패: {e}"}, 0.0
        if self.cfg.ml.require_validation and not val.get("passed"):
            return None, {"ml": f"검증 미통과(IC={val.get('ic_mean')}) → 미반영"}, 0.0
        return scores, {"ml": "반영", "ml_ic": val.get("ic_mean")}, blend

    # ------------------------------------------------------------------ 실행
    def run_once(self, *, force: bool = False, dry_run: bool = False) -> RunReport:
        now = self.clock()
        today = now.date()
        mode = f"{self.broker.name}/{'live' if self.broker.is_live else 'mock'}" + ("/dry" if dry_run else "")
        try:
            self.risk.pre_run()
        except RiskViolation as e:
            self.notifier.send(f"⛔ 실행 중단: {e}")
            return RunReport("error", str(e))
        if not force and not self.calendar.is_trading_day(today):
            return RunReport("skipped", f"{today} 휴장일")
        if not dry_run:
            hh, mm = _hm(self.cfg.execution.deadline)
            if now.time() >= now.replace(hour=hh, minute=mm).time() or now.time() < now.replace(hour=9, minute=0).time():
                return RunReport("skipped", "주문 가능 시간(09:00~마감) 아님")
        stop_note = ""
        if self.cfg.strategy.name == "msap" and self.cfg.msap.stock.stop_loss:
            try:
                stop_note = self._stop_check(today, dry_run=dry_run)
            except (BrokerError, RiskViolation, RuntimeError, ValueError) as e:
                log.exception("손절 점검 실패")
                self.notifier.send(f"❌ 손절 점검 실패: {e}")
                return RunReport("error", f"손절 점검 실패: {e}")
        try:
            due, why = self.rebalance_due(today, force=force)
            if stop_note:
                why = f"{stop_note} · {why}"
        except BrokerError as e:
            self.notifier.send(f"❌ 잔고 조회 실패: {e}")
            return RunReport("error", f"잔고 조회 실패: {e}")
        if not due:
            return RunReport("stopped" if stop_note else "skipped", (stop_note + " · " if stop_note else "") + why)

        run_id = self.store.start_run(mode)
        try:
            account = self.broker.get_account()
            target, diag = self.compute_targets(today, account)
            targets = {s: round(float(v), 4) for s, v in target.items()}
            quotes = self._quotes(list(target.index))
            plan = self._plan(target, account, quotes)
            self.risk.check_plan(plan)
            plan_text = plan.to_text(self.names)
            log.info("리밸런싱(%s)\n%s", why, plan_text)
            if dry_run:
                self.store.finish_run(run_id, "dry_run", why, {"targets": targets, "diag": diag})
                self.notifier.send(f"🧪 [드라이런] {why}\n{plan_text}")
                return RunReport("dry_run", why, targets, plan_text)
            orders = self._execute(run_id, target)
            final_acct = self.broker.get_account()
            final_plan = self._plan(target, final_acct, self._quotes(list(target.index)))
            leftover = [o for o in final_plan.orders]
            status = "done" if not leftover else "partial"
            if status == "done":
                self.store.mark_rebalanced(today.strftime("%Y-%m"), run_id, targets)
            self._snapshot(final_acct, today)
            msg = f"{why} · 주문 {len(orders)}건 · " + ("완료" if status == "done" else f"잔여 {len(leftover)}건(다음 실행 때 재시도)")
            self.store.finish_run(run_id, status, msg, {"targets": targets, "diag": diag})
            self.notifier.send(f"{'✅' if status == 'done' else '⚠️'} {msg}\n{plan_text}")
            return RunReport(status, msg, targets, plan_text, orders)
        except (RiskViolation, BrokerError, RuntimeError, ValueError) as e:
            log.exception("실행 실패")
            self.store.finish_run(run_id, "error", str(e))
            self.notifier.send(f"❌ 자동매매 오류: {e}")
            return RunReport("error", str(e))

    def _quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out = {}
        for s in symbols:
            q = self.broker.get_quote(s)
            self.risk.check_quote(q, None, self.cfg.universe.kind_of(s))
            out[s] = q
        return out

    def _plan(self, target: pd.Series, account, quotes: dict[str, Quote]) -> RebalancePlan:
        r = self.cfg.rebalance
        return plan_rebalance(target, account, {s: q.price for s, q in quotes.items()},
                              capital_limit=self.cfg.risk.capital_limit, cash_buffer_pct=self.cfg.risk.cash_buffer_pct,
                              no_trade_band=r.no_trade_band, min_order_value=r.min_order_value,
                              fee_rate=self.fees.total_rate)

    def _order_price(self, side: Side, q: Quote, kind: str) -> int:
        n = self.cfg.execution.limit_offset_ticks
        if side == Side.BUY:
            return shift_ticks(round_to_tick(q.ask or q.price, kind, "up"), n, kind)
        return shift_ticks(round_to_tick(q.bid or q.price, kind, "down"), -n, kind)

    def _execute(self, run_id: int, target: pd.Series) -> list[dict]:
        ex = self.cfg.execution
        records: list[dict] = []
        for phase in (Side.SELL, Side.BUY):
            for rnd in range(ex.max_rounds):
                self._cancel_open_orders(set(target.index))  # 남은 미체결이 있으면 이중주문 위험 → 정리 후 진행
                account = self.broker.get_account()
                quotes = self._quotes(list(target.index))
                todo = [o for o in self._plan(target, account, quotes).orders if o.side == phase]
                if not todo:
                    break
                final = rnd == ex.max_rounds - 1 and ex.final_market_order
                placed = []
                for o in todo:
                    q = quotes[o.symbol]
                    qty = o.qty
                    if phase == Side.SELL and hasattr(self.broker, "sellable_quantity"):
                        sq = self.broker.sellable_quantity(o.symbol)
                        if sq is not None:
                            qty = min(qty, sq)
                    if qty <= 0:
                        continue
                    if final:
                        req = OrderRequest(o.symbol, phase, qty, OrderType.MARKET, tag=f"r{rnd}")
                    else:
                        px = self._order_price(phase, q, self.cfg.universe.kind_of(o.symbol))
                        req = OrderRequest(o.symbol, phase, qty, OrderType.LIMIT, float(px), tag=f"r{rnd}")
                    self.risk.check_order(req, q)
                    res = self.broker.place_order(req)
                    self.store.record_order(run_id, req.symbol, req.side.value, req.qty, req.price,
                                            req.order_type.value, res.order_id, res.accepted, res.message)
                    records.append({"symbol": req.symbol, "side": req.side.value, "qty": req.qty, "price": req.price,
                                    "type": req.order_type.value, "accepted": res.accepted, "msg": res.message})
                    if res.accepted:
                        placed.append(res)
                    else:
                        log.warning("주문 거부 %s %s: %s", req.side.value, req.symbol, res.message)
                if placed:
                    self._wait_and_cancel(placed)
        return records

    def _cancel_open_orders(self, symbols: set[str]) -> None:
        def open_orders():
            try:
                return [s for s in self.broker.get_orders() if s.is_open and s.symbol in symbols]
            except BrokerError as e:
                raise RuntimeError(f"주문내역 조회 실패로 안전을 위해 중단: {e}") from e

        opens = open_orders()
        if not opens:
            return
        for s in opens:
            try:
                self.broker.cancel_order(s.order_id, s.symbol)
            except BrokerError as e:
                log.warning("미체결 취소 실패 %s: %s", s.order_id, e)
        self.sleep(self.cfg.execution.poll_interval_sec)
        still = open_orders()
        if still:
            raise RuntimeError("미체결 주문이 취소되지 않아 중단합니다: " +
                               ", ".join(f"{s.symbol}#{s.order_id}" for s in still))

    def _wait_and_cancel(self, placed) -> None:
        ex = self.cfg.execution
        deadline = time.monotonic() + ex.fill_timeout_sec
        ids = {r.order_id for r in placed}
        while True:
            try:
                states = {s.order_id: s for s in self.broker.get_orders()}
            except BrokerError as e:
                log.warning("주문내역 조회 실패: %s", e)
                states = {}
            open_ids = [i for i in ids if i not in states or states[i].is_open]
            if not open_ids or time.monotonic() >= deadline:
                break
            self.sleep(ex.poll_interval_sec)
        for r in placed:
            st = states.get(r.order_id)
            if st is not None and st.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                continue
            try:  # 체결 완료된 주문의 취소 실패는 정상
                c = self.broker.cancel_order(r.order_id, r.request.symbol)
                log.info("미체결 취소 %s %s: %s", r.request.symbol, r.order_id, c.message)
            except BrokerError as e:
                log.warning("취소 실패 %s: %s", r.order_id, e)

    def _snapshot(self, account, today: date) -> None:
        managed = set(self.cfg.universe.all_symbols())
        pos = {s: {"qty": p.qty, "px": p.last_price} for s, p in account.positions.items() if s in managed}
        eq = account.cash + sum(v["qty"] * v["px"] for v in pos.values())
        self.store.snapshot_equity(today.isoformat(), eq, account.cash, pos)

    # ------------------------------------------------------------------ 데몬
    def run_daemon(self, *, dry_run: bool = False, stop_after: int | None = None) -> None:
        """매 거래일 run_time 에 1회 실행. stop_after: 테스트용 반복 횟수 제한."""
        hh, mm = _hm(self.cfg.execution.run_time)
        last_day = None
        retries: dict[date, int] = {}
        loops = 0
        log.info("데몬 시작: 매 거래일 %02d:%02d KST 실행 (%s)", hh, mm, self.broker.name)
        while stop_after is None or loops < stop_after:
            loops += 1
            now = self.clock()
            if now.date() != last_day and (now.hour, now.minute) >= (hh, mm):
                last_day = now.date()
                try:
                    rep = self.run_once(dry_run=dry_run)
                except Exception as e:  # 예기치 못한 오류로 데몬이 죽지 않게
                    log.exception("run_once 예외")
                    rep = RunReport("error", f"예외: {e}")
                log.info("실행 결과: %s — %s", rep.status, rep.message)
                # 부분체결·일시 오류는 같은 날 최대 3회 재시도(10분 간격). 킬스위치·실계좌 미확인은 재시도 안 함
                if rep.status in ("partial", "error") and "킬스위치" not in rep.message \
                        and "LIVE_CONFIRM" not in rep.message and retries.get(now.date(), 0) < 3:
                    retries[now.date()] = retries.get(now.date(), 0) + 1
                    last_day = None
                    self.sleep(600)
            self.sleep(30)
