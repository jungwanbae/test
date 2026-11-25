"""NH투자증권 나무(Namuh) PLUG / N2 PLUG Open API 어댑터.

근거: NH투자증권 공식 국내주식 OpenAPI 명세(API명세서 261015, openapi.json)와 공식 SDK(nhplug-sdk).

핵심 규약(명세 요약)
  * REST: 모두 POST + JSON. 요청 {"Input_0": {...}} / 응답 Output_0(+Output_1 …) + rsp_cd·rsp_msg
  * 인증 헤더는 `authorization: Bearer <token>` 하나뿐
  * 토큰: POST {운영}/oauth2/token?appkey&appsecretkey&grant_type=client_credentials&scope=oob
          (form-urlencoded). 24시간 유효. 모의투자도 운영 도메인에서 발급. 재발급은 401 일 때만
  * 유량: REST 초당 5건 → 기본 4건으로 제한. 429(IGW42902) 시 간격을 늘려 재시도
  * 시세 API(currentPrice·period·etfCurrent)는 운영 전용 → 모의투자 모드에서도 운영 도메인에서 조회
  * 업무 성공 판정: rsp_cd 단일 코드와 비교 금지. "기대한 Output 블록에 값이 있는가 + rsp_msg" 로 판정
  * 연속조회: 응답 헤더 cts / cts_flag(Y=다음 있음) → 다음 요청 헤더에 그대로
  * 계좌: /n2/acctinfo 의 acct_type 01·02=운영, 03=모의투자. 환경과 다른 계좌를 쓰면 실패
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from autotrader.brokers.base import Broker, BrokerError
from autotrader.brokers.http import RateLimiter, TokenCache
from autotrader.models import (AccountSnapshot, OrderRequest, OrderResult, OrderState, OrderStatus,
                               OrderType, Position, Quote, Side)

log = logging.getLogger(__name__)

_HOSTS = {
    "namuh": ("api.nhplug.com", "moapi.nhplug.com"),
    "n2": ("api.n2plug.com", "moapi.n2plug.com"),
}
_PORT = 8443
_INVALID_TOKEN_RE = re.compile(r"유효하지\s*않은\s*token", re.IGNORECASE)
# cts_flag 헤더가 없을 때 '다음 페이지 있음'으로 보는 업무코드(공식 SDK 실측값)
_CONTINUE_CODES = {"00165", "00218", "XA109"}

# 명세 코드값
PRICE_LIMIT = "01"    # 보통가(지정가)
PRICE_MARKET = "05"   # 시장가
GUBUN_DAY = "1"       # 국내 period 주기: 1.일 2.주 3.월 (해외와 다르다!)


def _num(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _block(data: dict, name: str):
    v = data.get(name) if isinstance(data, dict) else None
    return v if v is not None else None


class NHPlugBroker(Broker):
    name = "nh_plug"

    def __init__(self, app_key: str, app_secret: str, *, env: str = "mock", account: str | None = None,
                 brand: str = "namuh", market: str = "KRX", rate_limit_per_sec: float = 4.0,
                 token_cache_dir: str = "~/.autotrader/tokens", session: requests.Session | None = None,
                 timeout: float = 10.0, sleep=time.sleep):
        if env not in ("mock", "live"):
            raise BrokerError("env 는 mock 또는 live 여야 합니다", category="config")
        if brand not in _HOSTS:
            raise BrokerError(f"brand 는 {list(_HOSTS)} 중 하나여야 합니다", category="config")
        if not app_key or not app_secret:
            raise BrokerError("NHPLUG_APP_KEY / NHPLUG_APP_SECRET 이 비어 있습니다(.env 확인)", category="config")
        live_host, mock_host = _HOSTS[brand]
        self.env = env
        self.is_live = env == "live"
        self.auth_url = f"https://{live_host}:{_PORT}"
        self.quote_url = f"https://{live_host}:{_PORT}"  # 시세는 운영 전용(명세 x-available-env=['live'])
        self.trade_url = f"https://{(live_host if self.is_live else mock_host)}:{_PORT}"
        self.market = market
        self._app_key = app_key
        self._app_secret = app_secret
        self._account = (account or "").replace("-", "").strip() or None
        self._session = session or requests.Session()
        self._timeout = timeout
        self._sleep = sleep
        self._limiter = RateLimiter(min(float(rate_limit_per_sec), 5.0), sleep=sleep)
        self._token_cache = TokenCache(token_cache_dir, "nhplug", f"{app_key}|{self.auth_url}")
        self._token: str | None = None

    # ------------------------------------------------------------------ 인증
    def _get_token(self, force: bool = False) -> str:
        if not force:
            if self._token:
                return self._token
            cached = self._token_cache.load()
            if cached:
                self._token = cached
                return cached
        params = {"appkey": self._app_key, "appsecretkey": self._app_secret,
                  "grant_type": "client_credentials", "scope": "oob"}
        last_err = None
        for attempt in range(3):
            try:
                res = self._session.post(f"{self.auth_url}/oauth2/token", params=params,
                                         headers={"content-type": "application/x-www-form-urlencoded"},
                                         timeout=self._timeout)
            except requests.RequestException as e:
                last_err = e
                self._sleep(1 + attempt)
                continue
            data = _safe_json(res)
            if res.status_code == 200 and isinstance(data, dict) and data.get("access_token"):
                token = data["access_token"]
                exp = time.time() + int(_num(data.get("expires_in"), 86400))
                self._token = token
                self._token_cache.save(token, exp)
                log.info("NH PLUG 접근토큰 발급(만료 %s)", datetime.fromtimestamp(exp).strftime("%m-%d %H:%M"))
                return token
            code = (data or {}).get("error_code") if isinstance(data, dict) else None
            if code == "IGW40054" and attempt < 2:  # 인증서버 일시 불가 → 재시도
                self._sleep(1 + attempt)
                continue
            raise BrokerError(f"NH PLUG 토큰 발급 실패: {_msg_of(data)}", category="auth",
                              code=code, status=res.status_code, raw=data)
        raise BrokerError(f"NH PLUG 토큰 발급 네트워크 오류: {last_err}", category="network", retryable=True)

    # ------------------------------------------------------------------ 호출
    def _call(self, base: str, path: str, input_0: dict, *, idempotent: bool = True,
              cts: str | None = None, cts_flag: str | None = None) -> tuple[dict, dict]:
        """POST 후 (본문, 소문자 헤더) 반환. HTTP 200 이 아니면 BrokerError.

        idempotent=False(주문) 이면 네트워크 오류 시 절대 재전송하지 않는다(이중주문 방지).
        """
        url = f"{base}{path}"
        body = json.dumps({"Input_0": input_0}, ensure_ascii=False)
        refreshed = False
        for attempt in range(5):
            headers = {"authorization": f"Bearer {self._get_token(force=False)}",
                       "content-type": "application/json; charset=UTF-8"}
            if cts:
                headers["cts"] = cts
            if cts_flag:
                headers["cts_flag"] = cts_flag
            self._limiter.acquire()
            try:
                res = self._session.post(url, headers=headers, data=body.encode("utf-8"), timeout=self._timeout)
            except requests.RequestException as e:
                if idempotent and attempt < 2:
                    self._sleep(1 + attempt)
                    continue
                raise BrokerError(
                    f"네트워크 오류({path}): {e}" + ("" if idempotent else " — 주문 접수 여부 불명, 주문내역을 확인하세요"),
                    category="network", retryable=idempotent) from e
            data = _safe_json(res)
            if res.status_code == 200:
                return (data if isinstance(data, dict) else {"raw": data}), {k.lower(): v for k, v in res.headers.items()}
            text = res.text or ""
            if not refreshed and (res.status_code == 401 or "IGW40043" in text or _INVALID_TOKEN_RE.search(text)):
                self._token = None
                self._token_cache.clear()
                self._get_token(force=True)
                refreshed = True
                continue
            if res.status_code == 429:
                retry_after = _num(res.headers.get("Retry-After"), 0.0)
                wait = retry_after if retry_after > 0 else 0.5 * (attempt + 1)
                log.warning("NH PLUG 유량 초과(429) %s → %.1fs 대기 후 재시도", path, wait)
                self._sleep(wait)
                continue  # 429 는 게이트웨이에서 거절된 것이므로 주문도 재전송 안전
            raise BrokerError(f"NH PLUG HTTP {res.status_code} {path}: {_msg_of(data)}", category="http",
                              code=_code_of(data), status=res.status_code, raw=data)
        raise BrokerError(f"NH PLUG 재시도 한도 초과: {path}", category="rate_limit", retryable=True)

    def _paginate(self, base: str, path: str, input_0: dict, max_pages: int = 50) -> list[dict]:
        pages, seen = [], set()
        cts = cts_flag = None
        for _ in range(max_pages):
            data, headers = self._call(base, path, input_0, cts=cts, cts_flag=cts_flag)
            pages.append(data)
            nxt = (headers.get("cts") or "").strip()
            flag = (headers.get("cts_flag") or "").strip().upper()
            if not nxt or flag == "N" or nxt in seen:
                break
            if flag != "Y" and str(data.get("rsp_cd")) not in _CONTINUE_CODES:
                break
            seen.add(nxt)
            cts, cts_flag = nxt, (flag or None)
        return pages

    # ------------------------------------------------------------------ 계좌
    def _acct(self) -> str:
        if self._account:
            return self._account
        data, _ = self._call(self.trade_url, "/n2/acctinfo", {})
        rows = data.get("Output_0") or []
        want = ("01", "02") if self.is_live else ("03",)
        for t in want:  # 01(일반) 우선
            for r in rows:
                if str(r.get("acct_type", "")).strip() == t and r.get("acct_no"):
                    self._account = str(r["acct_no"]).strip()
                    log.info("NH PLUG 계좌 자동선택: %s(acct_type=%s)", _mask(self._account), t)
                    return self._account
        raise BrokerError(f"{self.env} 환경에 맞는 계좌(acct_type {want})가 없습니다: {_msg_of(data)}",
                          category="config", raw=data)

    def list_accounts(self) -> list[dict]:
        data, _ = self._call(self.trade_url, "/n2/acctinfo", {})
        return list(data.get("Output_0") or [])

    def get_account(self) -> AccountSnapshot:
        act = self._acct()
        pages = self._paginate(self.trade_url, "/krstock/inquiry/v1/balance", {
            "act_no": act, "bnc_bse_cd": "5", "ltg_aot_dit_cd": "9", "aet_bse": "2",
            "qut_dit_cd": "UNT", "aly_qut_cd": "1",  # 1=정규장 시세 기준(필수)
        })
        summary: dict = {}
        positions: dict[str, Position] = {}
        for p in pages:
            o0 = p.get("Output_0")
            if isinstance(o0, dict) and o0:
                summary = o0  # 합계는 마지막 페이지에서 완성된다(명세)
            for r in p.get("Output_1") or []:
                sym = str(r.get("iem_cd", "")).strip().lstrip("A")
                qty = int(_num(r.get("itg_bnc_qty")))
                if not sym or qty <= 0:
                    continue
                positions[sym] = Position(symbol=sym, qty=qty, sellable_qty=qty,
                                          avg_price=_num(r.get("phs_pr")), last_price=_num(r.get("now_pr")),
                                          name=str(r.get("iem_nm", "")).strip())
        if not summary:
            raise BrokerError(f"잔고 응답에 Output_0 이 없습니다: {_msg_of(pages[-1] if pages else {})}",
                              category="business", raw=pages)
        # 매수에 쓸 현금: D+2 예수금(오늘 매도대금 재사용분 반영). 없으면 당일 예수금
        cash = _num(summary.get("nxt2_dd_dca"), _num(summary.get("dca")))
        return AccountSnapshot(cash=cash, positions=positions, total_equity=_num(summary.get("tot_aet_amt"), None),
                               raw={"summary": summary})

    def buying_power(self, symbol: str, price: float | None = None) -> float | None:
        """미수 미발생 현금 주문가능금액(csh_orr_pbl_amt)."""
        inp = {"ost_dit_cd": "1", "act_no": self._acct(), "iem_cd": symbol,
               "nmn_pr_tp_cd": PRICE_LIMIT if price else PRICE_MARKET}
        if price:
            inp["orr_pr"] = int(price)
        data, _ = self._call(self.trade_url, "/krstock/inquiry/v1/buyableQuantity", inp)
        o0 = data.get("Output_0") or {}
        if not o0 or o0.get("csh_orr_pbl_amt") is None:
            return None
        return _num(o0.get("csh_orr_pbl_amt"))

    def sellable_quantity(self, symbol: str) -> int | None:
        data, _ = self._call(self.trade_url, "/krstock/inquiry/v1/sellableQuantity",
                             {"act_no": self._acct(), "iem_cd": symbol, "cfd_lon_cd": "00"})
        o0 = data.get("Output_0") or {}
        if isinstance(o0, list):
            o0 = o0[0] if o0 else {}
        if "sll_pbl_qty" not in o0:
            return None
        return int(_num(o0.get("sll_pbl_qty")))

    # ------------------------------------------------------------------ 시세
    def get_quote(self, symbol: str) -> Quote:
        data, _ = self._call(self.quote_url, "/krstock/quote/v1/currentPrice",
                             {"market_cd": self.market, "iem_cd": symbol})
        o0 = data.get("Output_0") or {}
        price = _num(o0.get("stck_prpr"))
        if price <= 0:
            raise BrokerError(f"현재가 없음({symbol}): {_msg_of(data)}", category="business", raw=data)
        bid = _num(o0.get("bidp1"), _num(o0.get("bidp"))) or None
        ask = _num(o0.get("askp1"), _num(o0.get("askp"))) or None
        return Quote(symbol=symbol, price=price, bid=bid, ask=ask,
                     prev_close=_num(o0.get("stck_prdy_clpr")) or None, name=str(o0.get("iem_nm", "")).strip(),
                     market_status=(str(o0.get("market_status")) if o0.get("market_status") is not None else None),
                     timestamp=datetime.now())

    def get_daily_history(self, symbol: str, start: date, end: date | None = None,
                          page_size: int = 300) -> pd.DataFrame:
        """일봉(정규장 기준). edate 를 과거로 옮겨가며 start 까지 수집한다."""
        end = end or date.today()
        rows: dict[str, dict] = {}
        cur_end = end
        for _ in range(200):
            data, _ = self._call(self.quote_url, "/krstock/quote/v1/period", {
                "market_cd": "KRX",  # UNT+정규장은 2026-09-14 이후만 제공 → 장기 이력은 KRX
                "iem_cd": symbol, "gubun": GUBUN_DAY, "edate": cur_end.strftime("%Y%m%d"),
                "array_cnt": str(page_size), "view_main_yn": "Y",
            })
            batch = data.get("Output_1") or []
            if not batch:
                break
            oldest = None
            for r in batch:
                d = str(r.get("bsop_date", "")).strip()
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
            if nxt >= cur_end:  # 진행이 없으면 중단(무한루프 방지)
                break
            cur_end = nxt
        return _rows_to_ohlcv(rows, start, end,
                              keys=("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "vol"))

    # ------------------------------------------------------------------ 주문
    def place_order(self, req: OrderRequest) -> OrderResult:
        path = "/krstock/order/v1/cashBuy" if req.side == Side.BUY else "/krstock/order/v1/cashSell"
        inp = {
            "act_no": self._acct(), "iem_cd": req.symbol, "orr_qty": int(req.qty),
            "nmn_pr_tp_cd": PRICE_MARKET if req.order_type == OrderType.MARKET else PRICE_LIMIT,
            "orr_cnd_dit_cd": "00", "ssl_nmn_pr_dit_cd": "00",
            "rmt_mkt_cd": self.market, "sor_mkt_sli_yn": "N",
        }
        if req.order_type == OrderType.LIMIT:
            inp["orr_pr"] = int(round(req.price))
        log.info("NH PLUG 주문 전송: %s %s %d주 %s", req.side.value, req.symbol, req.qty,
                 "시장가" if req.order_type == OrderType.MARKET else f"{int(round(req.price)):,}원")
        data, _ = self._call(self.trade_url, path, inp, idempotent=False)
        o0 = data.get("Output_0") or {}
        order_no = o0.get("mkt_orr_no")
        accepted = bool(order_no) and _num(order_no) > 0
        return OrderResult(request=req, accepted=accepted, order_id=str(int(_num(order_no))) if accepted else "",
                           message=str(data.get("rsp_msg") or ""), raw=data)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        req = OrderRequest(symbol=symbol, side=Side.SELL, qty=1, order_type=OrderType.MARKET, tag="cancel")
        data, _ = self._call(self.trade_url, "/krstock/order/v1/cancel", {
            "act_no": self._acct(), "org_mkt_orr_no": int(order_id), "all_pat_dit_cd": "1", "iem_cd": symbol,
        }, idempotent=False)
        o0 = data.get("Output_0") or {}
        ok = bool(o0.get("mkt_orr_no"))
        return OrderResult(request=req, accepted=ok, order_id=str(o0.get("mkt_orr_no") or ""),
                           message=str(data.get("rsp_msg") or ""), raw=data)

    def get_orders(self, day: date | None = None) -> list[OrderState]:
        day = day or date.today()
        pages = self._paginate(self.trade_url, "/krstock/inquiry/v1/dailyOrderExecution", {
            "orr_dt": day.strftime("%Y%m%d"), "act_no": self._acct(), "orr_mkt_cd": "00", "ost_cns_dit": "0",
        })
        out: list[OrderState] = []
        for p in pages:
            for r in p.get("Output_0") or []:
                kind = str(r.get("sby_dit_cd_nm", ""))
                if "취소" in kind or "정정" in kind:
                    continue  # 정정/취소 주문 자체 행은 원주문 상태에 반영돼 있다
                side = Side.BUY if "매수" in kind else Side.SELL if "매도" in kind else None
                qty = int(_num(r.get("orr_qty")))
                filled = int(_num(r.get("tot_cns_qty")))
                remaining = int(_num(r.get("ny_cns_qty")))
                cancelled = int(_num(r.get("can_qty")))
                rejected = str(r.get("orr_rjt_rsn_cd_nm") or "").strip()
                if rejected:
                    status = OrderStatus.REJECTED
                elif qty > 0 and filled >= qty:
                    status = OrderStatus.FILLED
                elif remaining > 0:
                    status = OrderStatus.PARTIAL if filled > 0 else OrderStatus.OPEN
                elif cancelled > 0:
                    status = OrderStatus.CANCELLED
                else:
                    status = OrderStatus.UNKNOWN
                out.append(OrderState(
                    order_id=str(int(_num(r.get("itg_orr_no")))), symbol=str(r.get("iem_cd", "")).strip().lstrip("A"),
                    side=side, qty=qty, filled_qty=filled, remaining_qty=remaining,
                    avg_fill_price=_num(r.get("cns_avg_uit_pr")) or None, status=status,
                    price=_num(r.get("orr_pr")) or None, raw=r))
        return out


# ---------------------------------------------------------------------- helpers
def _safe_json(res):
    try:
        return res.json()
    except Exception:
        return res.text


def _msg_of(data) -> str:
    if isinstance(data, dict):
        return str(data.get("rsp_msg") or data.get("error_description") or data.get("msg1")
                   or json.dumps(data, ensure_ascii=False)[:300])
    return str(data)[:300]


def _code_of(data) -> str | None:
    if isinstance(data, dict):
        c = data.get("rsp_cd") or data.get("error_code") or data.get("msg_cd")
        return str(c) if c is not None else None
    return None


def _mask(acct: str) -> str:
    return acct[:3] + "*" * max(len(acct) - 5, 0) + acct[-2:] if len(acct) > 5 else "***"


def _rows_to_ohlcv(rows: dict[str, dict], start: date, end: date, keys: tuple[str, str, str, str, str]) -> pd.DataFrame:
    o, h, lo, c, v = keys
    recs = []
    for d, r in rows.items():
        dt = datetime.strptime(d, "%Y%m%d").date()
        if dt < start or dt > end:
            continue
        close = _num(r.get(c))
        if close <= 0:
            continue
        recs.append({"date": pd.Timestamp(dt), "open": _num(r.get(o), close), "high": _num(r.get(h), close),
                     "low": _num(r.get(lo), close), "close": close, "volume": _num(r.get(v))})
    if not recs:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(recs).set_index("date").sort_index()
