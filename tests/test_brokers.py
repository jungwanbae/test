"""증권사 어댑터 테스트 — 공식 명세의 요청/응답 형식을 가짜 세션으로 재현."""
import json
from datetime import date

import pytest
import requests

from autotrader.brokers.base import BrokerError
from autotrader.brokers.kis import KISBroker
from autotrader.brokers.nh_plug import NHPlugBroker
from autotrader.models import OrderRequest, OrderStatus, OrderType, Side
from conftest import FakeResp, FakeSession

TOKEN = {"access_token": "tok-1", "expires_in": 86400}


def _nh(handler, tmp_path, **kw):
    sess = FakeSession(handler)
    b = NHPlugBroker("key", "secret", session=sess, token_cache_dir=str(tmp_path), sleep=lambda s: None, **kw)
    return b, sess


def _body(kw):
    return json.loads(kw["data"].decode() if isinstance(kw["data"], bytes) else kw["data"])


# ====================================================================== NH PLUG
def test_nh_token_and_quote_uses_live_host_in_mock(tmp_path):
    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            assert url.startswith("https://api.nhplug.com:8443")  # 토큰은 운영 도메인
            assert kw["params"]["appsecretkey"] == "secret" and kw["params"]["scope"] == "oob"
            assert kw["headers"]["content-type"] == "application/x-www-form-urlencoded"
            return FakeResp(200, TOKEN)
        assert url == "https://api.nhplug.com:8443/krstock/quote/v1/currentPrice"  # 시세는 운영 전용
        assert kw["headers"]["authorization"] == "Bearer tok-1"
        assert set(kw["headers"]) == {"authorization", "content-type"}
        assert _body(kw) == {"Input_0": {"market_cd": "KRX", "iem_cd": "069500"}}
        return FakeResp(200, {"Output_0": {"stck_prpr": 35210, "askp1": 35215, "bidp1": 35210,
                                           "stck_prdy_clpr": 35000, "iem_nm": "KODEX 200", "market_status": "1"},
                              "rsp_cd": "00000", "rsp_msg": "정상처리"})

    b, sess = _nh(h, tmp_path, env="mock")
    q = b.get_quote("069500")
    assert (q.price, q.ask, q.bid, q.prev_close) == (35210, 35215, 35210, 35000)
    b.get_quote("069500")
    assert sum(1 for c in sess.calls if c[1].endswith("/oauth2/token")) == 1  # 토큰 재사용


def test_nh_token_file_cache_survives_restart(tmp_path):
    calls = []

    def h(method, url, kw):
        calls.append(url)
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        return FakeResp(200, {"Output_0": {"stck_prpr": 100}})

    _nh(h, tmp_path)[0].get_quote("A")
    _nh(h, tmp_path)[0].get_quote("A")  # 새 인스턴스 = 프로세스 재시작
    assert sum(u.endswith("/oauth2/token") for u in calls) == 1


def test_nh_mock_account_autoselect_and_order(tmp_path):
    seen = {}

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        if url.endswith("/n2/acctinfo"):
            assert url.startswith("https://moapi.nhplug.com:8443")
            return FakeResp(200, {"Output_0": [{"acct_no": "20100000001", "acct_type": "01"},
                                               {"acct_no": "20100000003", "acct_type": "03"}]})
        if url.endswith("/krstock/order/v1/cashBuy"):
            seen["buy"] = _body(kw)["Input_0"]
            return FakeResp(200, {"Output_0": {"mkt_orr_no": 12345}, "rsp_cd": "00000", "rsp_msg": "주문이 완료되었습니다"})
        if url.endswith("/krstock/order/v1/cashSell"):
            seen["sell"] = _body(kw)["Input_0"]
            return FakeResp(200, {"rsp_cd": "40310", "rsp_msg": "매도가능수량이 부족합니다"})
        raise AssertionError(url)

    b, _ = _nh(h, tmp_path, env="mock")
    r = b.place_order(OrderRequest("069500", Side.BUY, 3, OrderType.LIMIT, 35215.0))
    assert r.accepted and r.order_id == "12345"
    assert seen["buy"] == {"act_no": "20100000003", "iem_cd": "069500", "orr_qty": 3, "nmn_pr_tp_cd": "01",
                           "orr_cnd_dit_cd": "00", "ssl_nmn_pr_dit_cd": "00", "rmt_mkt_cd": "KRX",
                           "sor_mkt_sli_yn": "N", "orr_pr": 35215}
    r2 = b.place_order(OrderRequest("069500", Side.SELL, 3, OrderType.MARKET))
    assert not r2.accepted and "부족" in r2.message  # 업무오류는 HTTP 200 + Output 없음으로 판정
    assert seen["sell"]["nmn_pr_tp_cd"] == "05" and "orr_pr" not in seen["sell"]


def test_nh_order_network_error_not_retried(tmp_path):
    n = {"order": 0}

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        n["order"] += 1
        return requests.ConnectionError("reset")

    b, _ = _nh(h, tmp_path, account="20100000003")
    with pytest.raises(BrokerError) as e:
        b.place_order(OrderRequest("069500", Side.BUY, 1, OrderType.MARKET))
    assert n["order"] == 1 and e.value.category == "network" and not e.value.retryable  # 이중주문 방지


def test_nh_401_refresh_and_429_backoff(tmp_path):
    state = {"tokens": 0, "quote": 0}

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            state["tokens"] += 1
            return FakeResp(200, {"access_token": f"tok-{state['tokens']}", "expires_in": 86400})
        state["quote"] += 1
        if state["quote"] == 1:
            return FakeResp(401, {"error_code": "IGW40043", "error_description": "유효하지 않은 token"})
        if state["quote"] == 2:
            return FakeResp(429, {"error_code": "IGW42902"}, headers={"Retry-After": "0.2"})
        assert kw["headers"]["authorization"] == "Bearer tok-2"
        return FakeResp(200, {"Output_0": {"stck_prpr": 1000}})

    b, _ = _nh(h, tmp_path)
    assert b.get_quote("A").price == 1000
    assert state == {"tokens": 2, "quote": 3}


def test_nh_balance_pagination_and_parse(tmp_path):
    pages = [
        FakeResp(200, {"Output_1": [{"iem_cd": "069500", "iem_nm": "KODEX 200", "itg_bnc_qty": 10.0,
                                     "phs_pr": 30000, "now_pr": 35000}], "rsp_cd": "00165"},
                 headers={"cts": "NEXT1", "cts_flag": "Y"}),
        FakeResp(200, {"Output_0": {"dca": 1_000_000, "nxt2_dd_dca": 1_200_000, "tot_aet_amt": 1_550_000},
                       "Output_1": [{"iem_cd": "411060", "iem_nm": "ACE KRX금현물", "itg_bnc_qty": 5,
                                     "phs_pr": 20000, "now_pr": 22000}]}, headers={"cts_flag": "N"}),
    ]
    sent_cts = []

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        inp = _body(kw)["Input_0"]
        assert inp["aly_qut_cd"] == "1" and inp["qut_dit_cd"] == "UNT" and inp["act_no"] == "20100000003"
        sent_cts.append(kw["headers"].get("cts"))
        return pages.pop(0)

    b, _ = _nh(h, tmp_path, account="201-0000-0003".replace("-", ""))
    a = b.get_account()
    assert sent_cts == [None, "NEXT1"]
    assert a.cash == 1_200_000 and a.total_equity == 1_550_000
    assert a.positions["069500"].qty == 10 and a.positions["411060"].last_price == 22000


def test_nh_daily_history_walks_back(tmp_path):
    def rows(dates):
        return [{"bsop_date": d, "stck_oprc": "100", "stck_hgpr": "110", "stck_lwpr": "90", "stck_prpr": "105",
                 "vol": "1000"} for d in dates]

    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        inp = _body(kw)["Input_0"]
        assert inp["gubun"] == "1" and inp["view_main_yn"] == "Y" and inp["market_cd"] == "KRX"
        if inp["edate"] == "20240110":
            return FakeResp(200, {"Output_1": rows(["20240110", "20240109", "20240108"])})
        assert inp["edate"] == "20240107"
        return FakeResp(200, {"Output_1": rows(["20240105", "20240104"])})

    b, _ = _nh(h, tmp_path)
    df = b.get_daily_history("069500", date(2024, 1, 4), date(2024, 1, 10), page_size=3)
    assert list(df.index.strftime("%Y%m%d")) == ["20240104", "20240105", "20240108", "20240109", "20240110"]
    assert df["close"].iloc[0] == 105.0


def test_nh_orders_status_mapping(tmp_path):
    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        return FakeResp(200, {"Output_0": [
            {"itg_orr_no": 11, "iem_cd": "069500", "sby_dit_cd_nm": "현금매수", "orr_qty": 10, "tot_cns_qty": 10,
             "ny_cns_qty": 0, "can_qty": 0, "cns_avg_uit_pr": 35000, "orr_pr": 35000},
            {"itg_orr_no": 12, "iem_cd": "411060", "sby_dit_cd_nm": "현금매도", "orr_qty": 5, "tot_cns_qty": 2,
             "ny_cns_qty": 3, "can_qty": 0},
            {"itg_orr_no": 13, "iem_cd": "411060", "sby_dit_cd_nm": "매도취소", "orr_qty": 3},
        ]})

    b, _ = _nh(h, tmp_path, account="20100000003")
    st = {s.order_id: s for s in b.get_orders(date(2026, 11, 2))}
    assert st["11"].status == OrderStatus.FILLED and st["11"].side == Side.BUY
    assert st["12"].status == OrderStatus.PARTIAL and st["12"].is_open and st["12"].side == Side.SELL
    assert "13" not in st


def test_nh_live_requires_live_account(tmp_path):
    def h(method, url, kw):
        if url.endswith("/oauth2/token"):
            return FakeResp(200, TOKEN)
        assert url.startswith("https://api.nhplug.com:8443")
        return FakeResp(200, {"Output_0": [{"acct_no": "20100000003", "acct_type": "03"}]})

    b, _ = _nh(h, tmp_path, env="live")
    with pytest.raises(BrokerError):
        b.get_account()


# ====================================================================== KIS
def _kis(handler, tmp_path, env="mock"):
    sess = FakeSession(handler)
    return KISBroker("k", "s", "12345678-01", env=env, session=sess, token_cache_dir=str(tmp_path),
                     sleep=lambda s: None, rate_limit_per_sec=0), sess


def test_kis_order_tr_ids_and_payload(tmp_path):
    seen = []

    def h(method, url, kw):
        if url.endswith("/oauth2/tokenP"):
            assert json.loads(kw["data"])["appsecret"] == "s"
            return FakeResp(200, {"access_token": "kt", "expires_in": 86400})
        seen.append((kw["headers"]["tr_id"], json.loads(kw["data"])))
        assert kw["headers"]["custtype"] == "P" and kw["headers"]["appkey"] == "k"
        return FakeResp(200, {"rt_cd": "0", "msg1": "주문 전송 완료",
                              "output": {"KRX_FWDG_ORD_ORGNO": "91252", "ODNO": "0000117057"}})

    b, _ = _kis(h, tmp_path, env="mock")
    r = b.place_order(OrderRequest("069500", Side.BUY, 2, OrderType.LIMIT, 35215))
    assert r.accepted and r.order_id == "91252:0000117057"
    b.place_order(OrderRequest("069500", Side.SELL, 2, OrderType.MARKET))
    b.cancel_order(r.order_id, "069500")
    assert [t for t, _ in seen] == ["VTTC0012U", "VTTC0011U", "VTTC0013U"]
    assert seen[0][1]["ORD_DVSN"] == "00" and seen[0][1]["ORD_UNPR"] == "35215" and seen[0][1]["CANO"] == "12345678"
    assert seen[1][1]["ORD_DVSN"] == "01" and seen[1][1]["SLL_TYPE"] == "01"
    assert seen[2][1]["ORGN_ODNO"] == "0000117057" and seen[2][1]["RVSE_CNCL_DVSN_CD"] == "02"


def test_kis_live_balance_and_token_expiry(tmp_path):
    n = {"bal": 0}

    def h(method, url, kw):
        if url.endswith("/oauth2/tokenP"):
            return FakeResp(200, {"access_token": f"kt{n['bal']}", "expires_in": 86400})
        assert url.startswith("https://openapi.koreainvestment.com:9443")
        assert kw["headers"]["tr_id"] == "TTTC8434R"
        n["bal"] += 1
        if n["bal"] == 1:
            return FakeResp(500, {"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "기간이 만료된 token 입니다."})
        return FakeResp(200, {"rt_cd": "0", "output1": [{"pdno": "069500", "prdt_name": "KODEX 200", "hldg_qty": "7",
                                                          "ord_psbl_qty": "7", "pchs_avg_pric": "30000.0",
                                                          "prpr": "35000"}],
                              "output2": [{"dnca_tot_amt": "500000", "prvs_rcdl_excc_amt": "480000",
                                           "tot_evlu_amt": "725000"}]})

    b, _ = _kis(h, tmp_path, env="live")
    a = b.get_account()
    assert a.cash == 480000 and a.positions["069500"].qty == 7 and a.total_equity == 725000


def test_kis_rejected_order(tmp_path):
    def h(method, url, kw):
        if url.endswith("/oauth2/tokenP"):
            return FakeResp(200, {"access_token": "kt", "expires_in": 86400})
        return FakeResp(200, {"rt_cd": "1", "msg1": "주문가능금액을 초과 했습니다"})

    b, _ = _kis(h, tmp_path)
    r = b.place_order(OrderRequest("069500", Side.BUY, 1, OrderType.MARKET))
    assert not r.accepted and "초과" in r.message


def test_kis_account_format_validation(tmp_path):
    with pytest.raises(BrokerError):
        KISBroker("k", "s", "1234-5", token_cache_dir=str(tmp_path))
