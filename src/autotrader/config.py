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
        return "etf"

    def name_of(self, symbol: str) -> str:
        for a in self.risk_assets + ([self.cash_asset] if self.cash_asset else []):
            if a.symbol == symbol:
                return a.name or symbol
        return symbol


@dataclass
class StrategyConfig:
    name: str = "amr"
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
    max_price_deviation: float = 0.10     # 전일종가 대비 ±10% 초과 시세는 이상치로 보고 중단
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
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
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
