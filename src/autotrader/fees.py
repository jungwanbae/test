"""거래비용 모델 + 증권사 API 수수료 비교.

수수료율은 계좌 유형·이벤트에 따라 바뀌므로 config 에서 덮어쓸 수 있게 했다.
아래 프리셋은 2026-11 조사 기준이며 '확인 필요' 표시가 있는 값은 반드시 본인 계좌에서 확인할 것.
"""
from __future__ import annotations

from dataclasses import dataclass

# 유관기관 제비용(거래소·예탁원 등). 증권사 무관 공통, 매수·매도 양쪽.
EXCHANGE_FEE_RATE = 0.000036396  # 0.0036396%


@dataclass(frozen=True)
class FeeSchedule:
    name: str
    commission_rate: float            # 증권사 수수료(소수). 0.0001 = 0.01%
    exchange_fee_rate: float = EXCHANGE_FEE_RATE  # commission_rate 에 이미 포함돼 있으면 0
    stock_sell_tax_rate: float = 0.0020           # 주식 매도 시 증권거래세+농특세(2026, 코스피·코스닥 0.20%)
    etf_sell_tax_rate: float = 0.0                # 국내 상장 ETF 는 증권거래세 면제
    min_commission: float = 0.0
    note: str = ""

    @property
    def total_rate(self) -> float:
        return self.commission_rate + self.exchange_fee_rate

    def cost(self, notional: float, is_sell: bool, is_etf: bool = True) -> float:
        fee = max(abs(notional) * self.total_rate, self.min_commission if notional else 0.0)
        tax = abs(notional) * (self.etf_sell_tax_rate if is_etf else self.stock_sell_tax_rate) if is_sell else 0.0
        return fee + tax


PRESETS: dict[str, FeeSchedule] = {
    # NH투자증권 나무 PLUG API 전용 수수료(2026-08 출시 보도): 국내 0.01%, 해외 0.09%.
    # 유관기관 제비용 포함 여부는 공식 수수료표 확인 필요 → 보수적으로 별도 가산.
    "nh_plug": FeeSchedule("NH투자증권 나무 PLUG(API)", 0.0001,
                           note="국내 0.01%·해외 0.09%(API 전용). 신청 후 12개월 우대 이벤트. 제비용 별도 가정"),
    # 한국투자증권 KIS: API 수수료 = 해당 계좌의 HTS/온라인 수수료(별도 API 이용료 없음).
    "kis_bankis": FeeSchedule("한국투자증권 뱅키스(API=온라인 동일)", 0.000140527, exchange_fee_rate=0.0,
                              note="뱅키스 계좌 우대요율(제비용 포함, 확인 필요). 일반 온라인 0.147%"),
    "kis_standard": FeeSchedule("한국투자증권 일반 온라인", 0.00147, note="수수료 우대 미적용 계좌"),
    # 미래에셋증권: 2026-11 기준 개인용 Open API 미출시(준비 중 보도) → 자동매매 불가
    "zero": FeeSchedule("비용 없음(연구용)", 0.0, exchange_fee_rate=0.0, stock_sell_tax_rate=0.0),
}


def get_fee_schedule(preset: str | None = None, **overrides) -> FeeSchedule:
    base = PRESETS[preset or "nh_plug"]
    if not overrides:
        return base
    fields = {**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
    return FeeSchedule(**fields)


def compare_table(annual_turnover: float = 3.0, capital: float = 50_000_000) -> list[dict]:
    """연간 회전율(매수+매도 합계 / 자본)에 따른 연간 비용 비교."""
    rows = []
    for key, f in PRESETS.items():
        if key == "zero":
            continue
        traded = capital * annual_turnover
        rows.append({"preset": key, "name": f.name, "rate_pct": round(f.total_rate * 100, 6),
                     "annual_cost_krw": round(traded * f.total_rate), "note": f.note})
    return sorted(rows, key=lambda r: r["annual_cost_krw"])
