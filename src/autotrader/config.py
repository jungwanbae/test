"""설정 로딩: YAML(전략·운용) + .env/환경변수(비밀값)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISK"


@dataclass
class Asset:
    symbol: str
    name: str = ""
    asset_class: str = ""
    kind: str = "etf"  # etf | stock


@dataclass
class UniverseConfig:
    risk_assets: list[Asset] = field(default_factory=list)
    cash_asset: Asset | None = None

    def all_symbols(self) -> list[str]:
        s = [a.symbol for a in self.risk_assets]
        if self.cash_asset:
            s.append(self.cash_asset.symbol)
        return s

    def kind_of(self, symbol: str) -> str:
        for a in self.risk_assets + ([self.cash_asset] if self.cash_asset else []):
            if a.symbol == symbol:
                return a.kind
        return "stock"  # 유니버스(ETF) 밖 종목 = 종목 알파 슬리브의 개별주식

    def name_of(self, symbol: str) -> str:
        for a in self.risk_assets + ([self.cash_asset] if self.cash_asset else []):
            if a.symbol == symbol:
                return a.name or symbol
        return symbol


@dataclass
class StrategyConfig:
    name: str = "msap"                  # msap(멀티 슬리브, 기본) | amr(ETF 모멘텀만) | triad_us(TRIAD 미국 레그)
    lookback_months: list[float] = field(default_factory=lambda: [3, 6, 12])
    lookback_weights: list[float] | None = None
    top_k: int = 3
    vol_window_months: float = 3
    weighting: str = "inverse_vol"
    max_weight: float = 0.40
    target_vol: float | None = 0.10
    trend_sma_months: float | None = None
    drawdown_brake: float | None = None
    drawdown_brake_scale: float = 0.5


@dataclass
class StockSleeveSettings:
    marcap_dir: str = "data/marcap"          # FinanceData/marcap 로컬 복제(매일 갱신)
    model_path: str = "models/stock_lgbm.txt"
    top_n: int = 15
    buffer: float = 2.5
    universe_size: int = 600
    min_amount: float = 2e8                  # 60일 평균 거래대금 하한(원)
    max_weight: float = 0.10
    min_price: float = 1000.0
    sync: bool = True                        # 실행 시 marcap git pull
    max_signal_age_days: int = 5             # 데이터가 이보다 오래되면 주식 슬리브 중단
    adv_cap: float = 0.10                    # 종목당 보유금액 ≤ 20일 평균 거래대금의 10% (초과분은 현금성 ETF)
    stop_loss: float | None = 0.15           # 미너비니식 손절: 전일 종가가 매입가 대비 15% 이상 하락하면 매도(None=끔)


@dataclass
class MSAPSettings:
    stock_weight: float = 0.70               # 종목 알파 슬리브 비중(나머지 = ETF 리스크패리티)
    stock: StockSleeveSettings = field(default_factory=StockSleeveSettings)
    rp_target_vol: float = 0.10
    rp_vol_window_months: float = 12
    rp_max_leverage: float = 1.0


@dataclass
class TriadUSSettings:
    """TRIAD 미국 레그: 확인형 역발상 변동성 타기팅(나스닥) + 레버리지 리스크패리티(주식·장기채·금·원자재).

    research 모듈의 백테스트 함수(recovery_boost, lev_risk_parity)를 그대로 호출해 실전·백테스트 로직을 일치시킨다.
    tickers: 연구용 이름 → 미국 상장 ETF. 레버리지 상품은 일일 리셋형(TQQQ 3배·TMF 3배·UGL 2배).
    distribution_yield: 증권사 일봉은 분배금 미반영 가격이라, 추세(12개월 수익 > 현금) 판정용으로 연 분배율을
                        매끈하게 더해 총수익을 근사한다.
    """
    vt_weight: float = 0.5                 # 미국 레그 안에서 확인형 역발상 VT 비중(나머지 = 레버리지 RP)
    vt_target: float = 0.25                # 나스닥 목표변동성(연)
    crash: float = -0.30                   # '깊은 낙폭' 기준: 최근 2년 안에 사상 최고가 대비 −30% 이하였던 적
    boost_leverage: float = 3.0            # 폭락 후 골든크로스(50>200일선) 확인 시 노출
    boost_days: int = 252                  # 부스트 유지 거래일(골든크로스가 깨지면 즉시 종료)
    lrp_target: float = 0.18               # 레버리지 RP 목표변동성
    cash_rate: float = 0.040               # 미국 단기금리(연) — 추세 필터 기준. 연준 금리 변화 시 갱신
    tickers: dict[str, str] = field(default_factory=lambda: {
        "QQQ": "QQQ", "QQQx3": "TQQQ", "TLT": "TLT", "TLTx3": "TMF", "GLD": "GLD", "GLDx2": "UGL",
        "DBC": "DBC", "CASH_USD": "SGOV"})
    distribution_yield: dict[str, float] = field(default_factory=lambda: {
        "QQQ": 0.006, "TLT": 0.040, "GLD": 0.0, "DBC": 0.030})


@dataclass
class MLSettings:
    enabled: bool = False
    model_path: str = "models/ml_overlay.pt"
    blend: float = 0.3
    require_validation: bool = True  # 워크포워드 게이트 통과 모델만 사용


@dataclass
class RebalanceConfig:
    frequency: str = "monthly"            # monthly
    drift_threshold: float | None = 0.08  # 월중 비중이탈이 8%p 를 넘으면 임시 리밸런싱
    no_trade_band: float = 0.01
    min_order_value: float = 50_000
    history_years: int = 3


@dataclass
class ExecutionConfig:
    run_time: str = "09:30"               # KST. 장 시작 직후 LP 호가가 안정된 뒤
    deadline: str = "15:00"               # 이후로는 신규 주문 안 함
    limit_offset_ticks: int = 1           # 매수=매도호가+n틱, 매도=매수호가-n틱 (즉시 체결 지향)
    fill_timeout_sec: int = 60
    poll_interval_sec: int = 5
    max_rounds: int = 4
    final_market_order: bool = True       # 마지막 라운드는 시장가


@dataclass
class RiskConfig:
    capital_limit: float | None = None    # 이 전략에 배정할 최대 금액(원). None=계좌 전체(유니버스 종목+현금)
    cash_buffer_pct: float = 0.005
    max_order_value: float = 50_000_000
    max_turnover_pct: float = 2.0         # 1회 리밸런싱 매매대금/자산 상한(200%)
    max_price_deviation: float = 0.10     # ETF: 전일종가 대비 ±10% 초과 시세는 이상치로 보고 중단
    max_price_deviation_stock: float = 0.30  # 개별주식: 가격제한폭(±30%) 초과만 이상치
    kill_switch_file: str = "STOP"
    require_live_confirm: bool = True


@dataclass
class BrokerConfig:
    name: str = "nh_plug"                 # nh_plug | kis | paper
    env: str = "mock"                     # mock | live
    nh_brand: str = "namuh"               # namuh(나무) | n2
    nh_rate_limit_per_sec: float = 4.0
    kis_rate_limit_per_sec: float | None = None
    paper_initial_cash: float = 10_000_000
    paper_state_file: str = "state/paper_account.json"
    paper_quote_source: str = "none"      # none | nh_plug | kis (페이퍼 체결에 쓸 실시세)
    token_cache_dir: str = "~/.autotrader/tokens"


@dataclass
class CostsConfig:
    preset: str = "nh_plug"
    commission_rate: float | None = None
    slippage_bps: float = 5.0


@dataclass
class StorageConfig:
    db_path: str = "state/autotrader.db"
    cache_dir: str = "data/cache"
    log_dir: str = "logs"


@dataclass
class NotifyConfig:
    telegram: bool = False


@dataclass
class AppConfig:
    market: str = "KRX"                   # KRX(국내 레그, 원화) | US(미국 레그, 달러 — NH 해외주식)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    msap: MSAPSettings = field(default_factory=MSAPSettings)
    triad_us: TriadUSSettings = field(default_factory=TriadUSSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    extra_holidays: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- 로더
def _build(cls, data: Any):
    if data is None:
        return cls()
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: 알 수 없는 설정 키 {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def _coerce(tp, value):
    if value is None:
        return None
    origin = get_origin(tp)
    args = [a for a in get_args(tp) if a is not type(None)]
    if origin is list and args and is_dataclass(args[0]):
        return [_build(args[0], v) for v in value]
    if is_dataclass(tp):
        return _build(tp, value)
    if args and is_dataclass(args[0]) and isinstance(value, dict):
        return _build(args[0], value)
    return value


def load_dotenv(path: str | Path = ".env") -> None:
    """의존성 없는 .env 로더(이미 설정된 환경변수는 덮어쓰지 않음)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _build(AppConfig, raw)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def live_confirmed() -> bool:
    return env("AUTOTRADER_LIVE_CONFIRM") == LIVE_CONFIRM_PHRASE
