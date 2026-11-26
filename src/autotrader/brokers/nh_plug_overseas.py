"""NH투자증권 나무 PLUG — 해외주식(미국) 어댑터.

근거: NH 공식 해외주식 OpenAPI 명세(gbstock.openapi.json, 261015)·공식 SDK 스니펫.
국내 어댑터(NHPlugBroker)의 인증·유량·재시도·연속조회 규약을 그대로 쓰고, 경로·필드만 해외용으로 바꾼다.

  * 시세(운영 전용):  /gbstock/quote/v1/current, /gbstock/quote/v1/period (gubun 3=일 — 국내와 코드가 다르다)
  * 계좌·주문(모의·운영): /gbstock/inquiry/v1/balance·buyableAmount·unexecuted, /gbstock/order/v1/buy·sell·cancel
  * 가격 단위: 달러(소수 2자리). 수량: 정수 주. 계좌 금액: 달러(외화예수금·외화평가금액)
  * 증거금 통화: 기본 1(해당통화=달러) → 미국 레그는 '환전해 둔 달러'만 쓴다(국내 레그 원화와 섞이지 않게)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from autotrader.brokers.base import BrokerError
from autotrader.brokers.nh_plug import NHPlugBroker, _msg_of, _num, _rows_to_ohlcv
from autotrader.models import (AccountSnapshot, OrderRequest, OrderResult, OrderState, OrderStatus, OrderType,
                               Position, Quote, Side)

log = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

NAT_US = "200"            # 외화증권거래국가코드: 200 = 미국
PX_LIMIT = "00"           # 현물호가유형: 지정가
PX_MARKET = "03"          # 시장가
GUBUN_DAY = "3"           # 해외 기간별 시세 주기: 1.틱 2.분 3.일 4.주 5.월


class NHPlugOverseasBroker(NHPlugBroker):
    name = "nh_plug_us"

    def __init__(self, *args, margin_currency: str = "1", **kwargs):
        kwargs.setdefault("market", "US")
        super().__init__(*args, **kwargs)
        if margin_currency not in ("1", "2"):
            raise BrokerError("margin_currency 는 1(달러) 또는 2(원화)", category="config")
        self.margin_currency = margin_currency

    # ------------------------------------------------------------------ 시세
    def get_quote(self, symbol: str) -> Quote:
        data, _ = self._call(self.quote_url, "/gbstock/quote/v1/current", {"iem_cd": symbol})
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        price = _num(o0.get("trdprc"))
        if price <= 0:
            raise BrokerError(f"해외 현재가 없음({symbol}): {_msg_of(data)}", category="business", raw=data)
        return Quote(symbol=symbol, price=price, bid=_num(o0.get("best_bid1")) or None,
                     ask=_num(o0.get("best_ask1")) or None, prev_close=_num(o0.get("hst_trdprc")) or None,
                     name=str(o0.get("iem_nm", "")).strip(),
                     market_status=(str(o0.get("pf_jgubun")) if o0.get("pf_jgubun") is not None else None),
                     timestamp=datetime.now(ET))

    def fx_rate(self, symbol: str = "SPY") -> float | None:
        """원/달러 환율(현재가 응답의 환율 필드). 레그 간 자본 배분 점검용."""
        data, _ = self._call(self.quote_url, "/gbstock/quote/v1/current", {"iem_cd": symbol})
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        v = _num(o0.get("currency_prc"))
        return v if v > 0 else None

    def get_daily_history(self, symbol: str, start: date, end: date | None = None,
                          page_size: int = 500) -> pd.DataFrame:
        """일봉(정규장, 배당 미반영 가격). end_dt 를 과거로 옮겨가며 start 까지 수집."""
        end = end or datetime.now(ET).date()
        rows: dict[str, dict] = {}
        cur_end = end
        for _ in range(100):
            data, _ = self._call(self.quote_url, "/gbstock/quote/v1/period", {
                "iem_cd": symbol, "end_dt": cur_end.strftime("%Y%m%d"), "count": str(page_size), "maxavg": "0",
                "gubun": GUBUN_DAY, "xtick": "0001", "today_cls": "0", "market_cls": "1",
            })
            batch = data.get("Output_1") or []
            if not batch:
                break
            oldest = None
            for r in batch:
                d = str(r.get("bsop_date") or r.get("trade_date") or "").strip()
                if len(d) != 8:
                    continue
                rows[d] = r
                oldest = d if oldest is None or d < oldest else oldest
            if oldest is None:
                break
            oldest_d = datetime.strptime(oldest, "%Y%m%d").date()
            if oldest_d <= start or len(batch) < page_size:
                break
            nxt = oldest_d - timedelta(days=1)
            if nxt >= cur_end:
                break
            cur_end = nxt
        return _rows_to_ohlcv(rows, start, end, keys=("open_prc", "high", "low", "close_prc", "movolume"))

    def is_trading_day(self, day: date) -> bool | None:
        return None  # 해외 휴장일 API 없음 → USCalendar(NYSE) 사용

    # ------------------------------------------------------------------ 계좌
    def buying_power(self, symbol: str = "SPY", price: float | None = None) -> float | None:
        """달러 주문가능금액(증거금통화=해당통화). 실패하면 None."""
        inp = {"act_no": self._acct(), "pcs_dit": "1", "fc_sec_trd_nat_cd": NAT_US, "iem_cd": symbol,
               "wtm_cur_knd_cd": self.margin_currency, "oss_orr_knd_cd": "1",
               "ahi_nmn_pr_tp_cd": PX_LIMIT if price else PX_MARKET}
        if price:
            inp["fc_orr_uit_pr"] = round(float(price), 2)
        data, _ = self._call(self.trade_url, "/gbstock/inquiry/v1/buyableAmount", inp)
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        if not o0 or o0.get("orr_pbl_amt") is None:
            return None
        return _num(o0.get("orr_pbl_amt"))

    def sellable_quantity(self, symbol: str) -> int | None:
        data, _ = self._call(self.trade_url, "/gbstock/inquiry/v1/buyableAmount", {
            "act_no": self._acct(), "pcs_dit": "3", "fc_sec_trd_nat_cd": NAT_US, "iem_cd": symbol,
            "wtm_cur_knd_cd": self.margin_currency, "oss_orr_knd_cd": "1", "ahi_nmn_pr_tp_cd": PX_MARKET})
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        for k in ("sll_pbl_qty", "sll_pbl_qty1"):
            if o0.get(k) is not None:
                return int(_num(o0.get(k)))
        return None

    def get_account(self) -> AccountSnapshot:
        """미국 주식 잔고(달러). cash = 달러 주문가능금액(없으면 외화예수금)."""
        act = self._acct()
        pages = self._paginate(self.trade_url, "/gbstock/inquiry/v1/balance", {
            "act_no": act, "qut_iqr_dit_cd": "1", "fc_sec_trd_nat_cd": NAT_US, "cur_cd": "USD", "xns_dit_cd": "0"})
        summary: dict = {}
        positions: dict[str, Position] = {}
        for p in pages:
            o0 = p.get("Output_0")
            if isinstance(o0, list):
                o0 = o0[0] if o0 else {}
            if isinstance(o0, dict) and o0:
                summary = o0
            for r in p.get("Output_1") or []:
                sym = str(r.get("iem_cd", "")).strip().upper()
                qty = int(_num(r.get("cns_bse_bnc_qty")))
                if not sym or qty <= 0:
                    continue
                sellable = r.get("sll_pbl_qty1")
                positions[sym] = Position(symbol=sym, qty=qty,
                                          sellable_qty=int(_num(sellable)) if sellable is not None else qty,
                                          avg_price=_num(r.get("fc_avg_phs_pr"), _num(r.get("fc_phs_uit_pr"))),
                                          last_price=_num(r.get("fc_sec_end_pr")),
                                          name=str(r.get("iem_nm", "")).strip())
        if not summary and not positions:
            raise BrokerError(f"해외 잔고 응답에 Output 이 없습니다: {_msg_of(pages[-1] if pages else {})}",
                              category="business", raw=pages)
        cash = None
        try:
            cash = self.buying_power()
        except BrokerError as e:
            log.warning("해외 주문가능금액 조회 실패 → 외화예수금 사용: %s", e)
        if cash is None:
            cash = _num(summary.get("fc_dca"))
        eq = cash + sum(p.qty * p.last_price for p in positions.values())
        return AccountSnapshot(cash=cash, positions=positions, total_equity=eq, raw={"summary": summary})

    # ------------------------------------------------------------------ 주문
    def place_order(self, req: OrderRequest) -> OrderResult:
        path = "/gbstock/order/v1/buy" if req.side == Side.BUY else "/gbstock/order/v1/sell"
        inp = {"act_no": self._acct(), "fc_sec_trd_nat_cd": NAT_US, "iem_cd": req.symbol, "orr_qty": int(req.qty),
               "ahi_nmn_pr_tp_cd": PX_MARKET if req.order_type == OrderType.MARKET else PX_LIMIT}
        if req.side == Side.BUY:
            inp["wtm_cur_knd_cd"] = self.margin_currency
        if req.order_type == OrderType.LIMIT:
            inp["fc_orr_uit_pr"] = round(float(req.price), 2)
        log.info("NH PLUG 해외주문 전송: %s %s %d주 %s", req.side.value, req.symbol, req.qty,
                 "시장가" if req.order_type == OrderType.MARKET else f"${float(req.price):,.2f}")
        data, _ = self._call(self.trade_url, path, inp, idempotent=False)
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        order_no = o0.get("orr_no")
        accepted = order_no is not None and _num(order_no) > 0
        return OrderResult(request=req, accepted=accepted, order_id=str(int(_num(order_no))) if accepted else "",
                           message=str(data.get("rsp_msg") or ""), raw=data)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        req = OrderRequest(symbol=symbol, side=Side.SELL, qty=1, order_type=OrderType.MARKET, tag="cancel")
        data, _ = self._call(self.trade_url, "/gbstock/order/v1/cancel", {
            "act_no": self._acct(), "org_orr_no": int(order_id), "fc_sec_trd_nat_cd": NAT_US, "iem_cd": symbol,
            "all_pat_dit_cd": "1"}, idempotent=False)
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        ok = bool(o0.get("orr_no"))
        return OrderResult(request=req, accepted=ok, order_id=str(o0.get("orr_no") or ""),
                           message=str(data.get("rsp_msg") or ""), raw=data)

    def get_orders(self, day: date | None = None) -> list[OrderState]:
        """주문·체결 내역. 미국 장중 주문은 한국 날짜가 하루 늦을 수 있어 미국·한국 날짜를 모두 조회해 합친다."""
        if day is None:
            days = sorted({datetime.now(ET).date(), datetime.now(KST).date()})
        else:
            days = sorted({day, day + timedelta(days=1)})
        seen: dict[str, OrderState] = {}
        for d in days:
            pages = self._paginate(self.trade_url, "/gbstock/inquiry/v1/unexecuted", {
                "orr_dt": d.strftime("%Y%m%d"), "act_no": self._acct(), "oss_sby_dit_cd": "0", "sot_dit": "0",
                "ost_cns_dit": "0"})
            for p in pages:
                for r in p.get("Output_0") or []:
                    st = _order_state(r)
                    if st is not None:
                        seen[st.order_id] = st
        return list(seen.values())


def _order_state(r: dict) -> OrderState | None:
    kind = str(r.get("cor_can_dit_nm") or "")
    if "취소" in kind or "정정" in kind:
        return None  # 정정·취소 주문 자체 행 — 원주문 행의 취소수량에 반영돼 있다
    sd = str(r.get("oss_sby_dit_cd") or "").strip()
    name = str(r.get("sby_dit_nm") or "")
    side = Side.SELL if sd == "1" or "매도" in name else Side.BUY if sd == "2" or "매수" in name else None
    qty = int(_num(r.get("orr_qty")))
    filled = int(_num(r.get("cns_qty")))
    remaining = int(_num(r.get("ny_cns_orr_qty")))
    cancelled = int(_num(r.get("can_qty")))
    rejected = str(r.get("oss_ato_orr_sts_cd") or "").strip() == "2" or bool(str(r.get("rjt_rsn_cts") or "").strip())
    if rejected:
        status = OrderStatus.REJECTED
        remaining = 0
    elif qty > 0 and filled >= qty:
        status = OrderStatus.FILLED
    elif remaining > 0:
        status = OrderStatus.PARTIAL if filled > 0 else OrderStatus.OPEN
    elif cancelled > 0:
        status = OrderStatus.CANCELLED
    else:
        status = OrderStatus.UNKNOWN
    oid = r.get("orr_no")
    if oid is None:
        return None
    return OrderState(order_id=str(int(_num(oid))), symbol=str(r.get("iem_cd", "")).strip().upper(), side=side,
                      qty=qty, filled_qty=filled, remaining_qty=remaining,
                      avg_fill_price=_num(r.get("cns_pr")) or None, status=status,
                      price=_num(r.get("fc_orr_uit_pr")) or None, raw=r)
