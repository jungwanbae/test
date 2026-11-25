"""한국투자증권 KIS Developers Open API 어댑터(국내주식/ETF).

근거: KIS 공식 샘플 저장소(koreainvestment/open-trading-api, examples_llm/domestic_stock).
  * 실전 https://openapi.koreainvestment.com:9443 / 모의 https://openapivts.koreainvestment.com:29443
  * 토큰 POST /oauth2/tokenP (JSON). 24시간 유효, 발급은 1분 1회 제한 → 파일 캐시 필수
  * 헤더: authorization, appkey, appsecret, tr_id, custtype=P, tr_cont(연속조회)
  * 2025년 변경된 주문 TR: 매도 TTTC0011U / 매수 TTTC0012U / 정정취소 TTTC0013U (모의는 V…)
  * 일별주문체결(3개월 이내) TTTC0081R, 잔고 TTTC8434R, 매수가능 TTTC8908R, 휴장일 CTCA0903R(실전 전용)
  * 성공 판정: rt_cd == "0"
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from autotrader.brokers.base import Broker, BrokerError
from autotrader.brokers.http import RateLimiter, TokenCache
from autotrader.brokers.nh_plug import _num, _rows_to_ohlcv
from autotrader.models import (AccountSnapshot, OrderRequest, OrderResult, OrderState, OrderStatus,
                               OrderType, Position, Quote, Side)

log = logging.getLogger(__name__)

URLS = {"live": "https://openapi.koreainvestment.com:9443",
        "mock": "https://openapivts.koreainvestment.com:29443"}

TR = {
    "price": ("FHKST01010100", "FHKST01010100"),
    "daily": ("FHKST03010100", "FHKST03010100"),
    "buy": ("TTTC0012U", "VTTC0012U"),
    "sell": ("TTTC0011U", "VTTC0011U"),
    "rvsecncl": ("TTTC0013U", "VTTC0013U"),
    "balance": ("TTTC8434R", "VTTC8434R"),
    "psbl_order": ("TTTC8908R", "VTTC8908R"),
    "daily_ccld": ("TTTC0081R", "VTTC0081R"),
    "holiday": ("CTCA0903R", None),  # 모의투자 미지원
}

_TOKEN_ERRORS = {"EGW00121", "EGW00123"}  # 유효하지 않은/만료된 토큰
_RATE_ERRORS = {"EGW00201"}               # 초당 거래건수 초과


class KISBroker(Broker):
    name = "kis"

    def __init__(self, app_key: str, app_secret: str, account: str, *, env: str = "mock",
                 rate_limit_per_sec: float | None = None, token_cache_dir: str = "~/.autotrader/tokens",
                 session: requests.Session | None = None, timeout: float = 10.0, sleep=time.sleep):
        if env not in URLS:
            raise BrokerError("env 는 mock 또는 live 여야 합니다", category="config")
        if not app_key or not app_secret:
            raise BrokerError("KIS_APP_KEY / KIS_APP_SECRET 이 비어 있습니다(.env 확인)", category="config")
        acct = (account or "").replace(" ", "")
        if "-" in acct:
            cano, prdt = acct.split("-", 1)
        else:
            cano, prdt = acct[:8], acct[8:10] or "01"
        if len(cano) != 8 or len(prdt) != 2:
            raise BrokerError("KIS_ACCOUNT 형식은 12345678-01 입니다", category="config")
        self.env = env
        self.is_live = env == "live"
        self.base = URLS[env]
        self.cano, self.prdt = cano, prdt
        self._app_key, self._app_secret = app_key, app_secret
        self._session = session or requests.Session()
        self._timeout = timeout
        self._sleep = sleep
        rate = rate_limit_per_sec if rate_limit_per_sec is not None else (15.0 if self.is_live else 1.0)
        self._limiter = RateLimiter(rate, sleep=sleep)
        self._token_cache = TokenCache(token_cache_dir, "kis", f"{app_key}|{self.base}")
        self._token: str | None = None
        self._holiday_cache: dict[str, bool] = {}

    def _tr(self, key: str) -> str:
        live, mock = TR[key]
        tr = live if self.is_live else mock
        if tr is None:
            raise BrokerError(f"{key} 는 모의투자에서 지원되지 않습니다", category="config")
        return tr

    # ------------------------------------------------------------------ 인증
    def _get_token(self, force: bool = False) -> str:
        if not force:
            if self._token:
                return self._token
            cached = self._token_cache.load()
            if cached:
                self._token = cached
                return cached
        body = {"grant_type": "client_credentials", "appkey": self._app_key, "appsecret": self._app_secret}
        try:
            res = self._session.post(f"{self.base}/oauth2/tokenP", data=json.dumps(body),
                                     headers={"content-type": "application/json"}, timeout=self._timeout)
        except requests.RequestException as e:
            raise BrokerError(f"KIS 토큰 발급 네트워크 오류: {e}", category="network", retryable=True) from e
        data = _safe_json(res)
        if res.status_code != 200 or not isinstance(data, dict) or not data.get("access_token"):
            raise BrokerError(f"KIS 토큰 발급 실패(1분 1회 제한 주의): {data}", category="auth",
                              status=res.status_code, raw=data)
        exp = time.time() + int(_num(data.get("expires_in"), 86400))
        self._token = data["access_token"]
        self._token_cache.save(self._token, exp)
        return self._token

    # ------------------------------------------------------------------ 호출
    def _request(self, method: str, path: str, tr_id: str, *, params: dict | None = None,
                 body: dict | None = None, tr_cont: str = "", idempotent: bool = True) -> tuple[dict, dict]:
        refreshed = False
        for attempt in range(5):
            headers = {"content-type": "application/json; charset=utf-8",
                       "authorization": f"Bearer {self._get_token()}", "appkey": self._app_key,
                       "appsecret": self._app_secret, "tr_id": tr_id, "custtype": "P", "tr_cont": tr_cont}
            self._limiter.acquire()
            try:
                if method == "GET":
                    res = self._session.get(f"{self.base}{path}", headers=headers, params=params, timeout=self._timeout)
                else:
                    res = self._session.post(f"{self.base}{path}", headers=headers, data=json.dumps(body or {}),
                                             timeout=self._timeout)
            except requests.RequestException as e:
                if idempotent and attempt < 2:
                    self._sleep(1 + attempt)
                    continue
                raise BrokerError(f"KIS 네트워크 오류({path}): {e}" + ("" if idempotent else " — 주문내역 확인 필요"),
                                  category="network", retryable=idempotent) from e
            data = _safe_json(res)
            code = str(data.get("msg_cd", "")) if isinstance(data, dict) else ""
            if code in _RATE_ERRORS or res.status_code == 429:
                self._sleep(0.5 * (attempt + 1))
                continue
            if (code in _TOKEN_ERRORS or res.status_code == 401) and not refreshed:
                self._token = None
                self._token_cache.clear()
                self._get_token(force=True)
                refreshed = True
                continue
            if res.status_code != 200 or not isinstance(data, dict):
                raise BrokerError(f"KIS HTTP {res.status_code} {path}: {data}", category="http",
                                  code=code or None, status=res.status_code, raw=data)
            return data, {k.lower(): v for k, v in res.headers.items()}
        raise BrokerError(f"KIS 재시도 한도 초과: {path}", category="rate_limit", retryable=True)

    @staticmethod
    def _ok(data: dict) -> bool:
        return str(data.get("rt_cd")) == "0"

    # ------------------------------------------------------------------ 시세
    def get_quote(self, symbol: str) -> Quote:
        data, _ = self._request("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", self._tr("price"),
                                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol})
        o = data.get("output") or {}
        price = _num(o.get("stck_prpr"))
        if not self._ok(data) or price <= 0:
            raise BrokerError(f"KIS 현재가 실패({symbol}): {data.get('msg1')}", category="business", raw=data)
        prev = price - _num(o.get("prdy_vrss"))
        return Quote(symbol=symbol, price=price, prev_close=prev if prev > 0 else None,
                     name=str(o.get("hts_kor_isnm", "")), timestamp=datetime.now())

    def get_daily_history(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        end = end or date.today()
        rows: dict[str, dict] = {}
        cur_end = end
        for _ in range(300):  # 1회 최대 100건
            data, _ = self._request("GET", "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                                    self._tr("daily"), params={
                                        "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
                                        "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                                        "FID_INPUT_DATE_2": cur_end.strftime("%Y%m%d"),
                                        "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0"})  # 0=수정주가
            batch = [r for r in (data.get("output2") or []) if r.get("stck_bsop_date")]
            if not batch:
                break
            for r in batch:
                rows[r["stck_bsop_date"]] = r
            oldest = datetime.strptime(min(r["stck_bsop_date"] for r in batch), "%Y%m%d").date()
            if oldest <= start or len(batch) < 100:
                break
            nxt = oldest - timedelta(days=1)
            if nxt >= cur_end:
                break
            cur_end = nxt
        return _rows_to_ohlcv(rows, start, end, keys=("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"))

    def is_trading_day(self, day: date) -> bool | None:
        if not self.is_live:
            return None
        key = day.strftime("%Y%m%d")
        if key in self._holiday_cache:
            return self._holiday_cache[key]
        try:
            data, _ = self._request("GET", "/uapi/domestic-stock/v1/quotations/chk-holiday", self._tr("holiday"),
                                    params={"BASS_DT": key, "CTX_AREA_NK": "", "CTX_AREA_FK": ""})
        except BrokerError as e:
            log.warning("KIS 휴장일 조회 실패: %s", e)
            return None
        for r in data.get("output") or []:
            self._holiday_cache[str(r.get("bass_dt"))] = str(r.get("opnd_yn")) == "Y"
        return self._holiday_cache.get(key)

    # ------------------------------------------------------------------ 계좌
    def get_account(self) -> AccountSnapshot:
        positions: dict[str, Position] = {}
        summary: dict = {}
        fk = nk = ""
        tr_cont = ""
        for _ in range(20):
            data, headers = self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance",
                                          self._tr("balance"), tr_cont=tr_cont, params={
                                              "CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "AFHR_FLPR_YN": "N",
                                              "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
                                              "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
                                              "PRCS_DVSN": "00", "CTX_AREA_FK100": fk, "CTX_AREA_NK100": nk})
            if not self._ok(data):
                raise BrokerError(f"KIS 잔고 실패: {data.get('msg1')}", category="business", raw=data)
            for r in data.get("output1") or []:
                qty = int(_num(r.get("hldg_qty")))
                if qty <= 0:
                    continue
                sym = str(r.get("pdno", "")).strip()
                positions[sym] = Position(symbol=sym, qty=qty, sellable_qty=int(_num(r.get("ord_psbl_qty"), qty)),
                                          avg_price=_num(r.get("pchs_avg_pric")), last_price=_num(r.get("prpr")),
                                          name=str(r.get("prdt_name", "")).strip())
            o2 = data.get("output2") or []
            if o2:
                summary = o2[0] if isinstance(o2, list) else o2
            if str(headers.get("tr_cont", "")).upper() in ("F", "M"):
                fk, nk, tr_cont = data.get("ctx_area_fk100", ""), data.get("ctx_area_nk100", ""), "N"
                continue
            break
        cash = _num(summary.get("prvs_rcdl_excc_amt"), _num(summary.get("dnca_tot_amt")))  # D+2 예수금
        return AccountSnapshot(cash=cash, positions=positions, total_equity=_num(summary.get("tot_evlu_amt"), None),
                               raw={"summary": summary})

    def buying_power(self, symbol: str, price: float | None = None) -> float | None:
        data, _ = self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-psbl-order", self._tr("psbl_order"),
                                params={"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": symbol,
                                        "ORD_UNPR": str(int(price)) if price else "",
                                        "ORD_DVSN": "00" if price else "01",
                                        "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N"})
        if not self._ok(data):
            return None
        return _num((data.get("output") or {}).get("nrcvb_buy_amt"), None)  # 미수없는매수금액

    # ------------------------------------------------------------------ 주문
    def place_order(self, req: OrderRequest) -> OrderResult:
        is_mkt = req.order_type == OrderType.MARKET
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": req.symbol,
                "ORD_DVSN": "01" if is_mkt else "00", "ORD_QTY": str(int(req.qty)),
                "ORD_UNPR": "0" if is_mkt else str(int(round(req.price))), "EXCG_ID_DVSN_CD": "KRX",
                "SLL_TYPE": "01" if req.side == Side.SELL else "", "CNDT_PRIC": ""}
        tr = self._tr("buy" if req.side == Side.BUY else "sell")
        data, _ = self._request("POST", "/uapi/domestic-stock/v1/trading/order-cash", tr, body=body, idempotent=False)
        o = data.get("output") or {}
        ok = self._ok(data) and bool(o.get("ODNO"))
        # 취소 시 주문조직번호가 필요하므로 'ORGNO:ODNO' 형태로 보관
        oid = f"{o.get('KRX_FWDG_ORD_ORGNO', '')}:{o.get('ODNO', '')}" if ok else ""
        return OrderResult(request=req, accepted=ok, order_id=oid, message=str(data.get("msg1", "")), raw=data)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        orgno, _, odno = order_id.partition(":")
        req = OrderRequest(symbol=symbol, side=Side.SELL, qty=1, order_type=OrderType.MARKET, tag="cancel")
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "KRX_FWDG_ORD_ORGNO": orgno, "ORGN_ODNO": odno,
                "ORD_DVSN": "00", "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": "0", "ORD_UNPR": "0",
                "QTY_ALL_ORD_YN": "Y", "EXCG_ID_DVSN_CD": "KRX"}
        data, _ = self._request("POST", "/uapi/domestic-stock/v1/trading/order-rvsecncl", self._tr("rvsecncl"),
                                body=body, idempotent=False)
        return OrderResult(request=req, accepted=self._ok(data), order_id=order_id,
                           message=str(data.get("msg1", "")), raw=data)

    def get_orders(self, day: date | None = None) -> list[OrderState]:
        day = day or date.today()
        d = day.strftime("%Y%m%d")
        out: list[OrderState] = []
        fk = nk = ""
        tr_cont = ""
        for _ in range(20):
            data, headers = self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                                          self._tr("daily_ccld"), tr_cont=tr_cont, params={
                                              "CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": d,
                                              "INQR_END_DT": d, "SLL_BUY_DVSN_CD": "00", "PDNO": "",
                                              "CCLD_DVSN": "00", "INQR_DVSN": "00", "INQR_DVSN_3": "00",
                                              "ORD_GNO_BRNO": "", "ODNO": "", "INQR_DVSN_1": "",
                                              "CTX_AREA_FK100": fk, "CTX_AREA_NK100": nk, "EXCG_ID_DVSN_CD": "KRX"})
            if not self._ok(data):
                raise BrokerError(f"KIS 체결조회 실패: {data.get('msg1')}", category="business", raw=data)
            for r in data.get("output1") or []:
                qty = int(_num(r.get("ord_qty")))
                filled = int(_num(r.get("tot_ccld_qty")))
                remaining = int(_num(r.get("rmn_qty")))
                if str(r.get("cncl_yn")) == "Y":
                    status = OrderStatus.CANCELLED
                elif qty > 0 and filled >= qty:
                    status = OrderStatus.FILLED
                elif remaining > 0:
                    status = OrderStatus.PARTIAL if filled else OrderStatus.OPEN
                else:
                    status = OrderStatus.UNKNOWN
                side = {"01": Side.SELL, "02": Side.BUY}.get(str(r.get("sll_buy_dvsn_cd")))
                out.append(OrderState(order_id=f"{r.get('ord_gno_brno', '')}:{r.get('odno', '')}",
                                      symbol=str(r.get("pdno", "")).strip(), side=side, qty=qty, filled_qty=filled,
                                      remaining_qty=remaining, avg_fill_price=_num(r.get("avg_prvs")) or None,
                                      status=status, price=_num(r.get("ord_unpr")) or None, raw=r))
            if str(headers.get("tr_cont", "")).upper() in ("F", "M"):
                fk, nk, tr_cont = data.get("ctx_area_fk100", ""), data.get("ctx_area_nk100", ""), "N"
                continue
            break
        return out


def _safe_json(res):
    try:
        return res.json()
    except Exception:
        return res.text
