"""증권사 어댑터 팩토리."""
from __future__ import annotations

from autotrader.brokers.base import Broker, BrokerError
from autotrader.config import AppConfig, env
from autotrader.fees import get_fee_schedule


def _nh(cfg: AppConfig, env_name: str):
    from autotrader.brokers.nh_plug import NHPlugBroker
    return NHPlugBroker(env("NHPLUG_APP_KEY"), env("NHPLUG_APP_SECRET"), env=env_name,
                        account=env("NHPLUG_ACCOUNT") or None, brand=cfg.broker.nh_brand,
                        rate_limit_per_sec=cfg.broker.nh_rate_limit_per_sec,
                        token_cache_dir=cfg.broker.token_cache_dir)


def _kis(cfg: AppConfig, env_name: str):
    from autotrader.brokers.kis import KISBroker
    return KISBroker(env("KIS_APP_KEY"), env("KIS_APP_SECRET"), env("KIS_ACCOUNT"), env=env_name,
                     rate_limit_per_sec=cfg.broker.kis_rate_limit_per_sec, token_cache_dir=cfg.broker.token_cache_dir)


def build_broker(cfg: AppConfig, quote_broker: Broker | None = None) -> Broker:
    name = cfg.broker.name
    if name == "nh_plug":
        return _nh(cfg, cfg.broker.env)
    if name == "kis":
        return _kis(cfg, cfg.broker.env)
    if name == "paper":
        from autotrader.brokers.paper import PaperBroker
        src = quote_broker
        if src is None and cfg.broker.paper_quote_source in ("nh_plug", "kis"):
            # 페이퍼 체결용 실시세: 시세 조회만 하므로 mock 환경 키로도 충분
            src = _nh(cfg, "mock") if cfg.broker.paper_quote_source == "nh_plug" else _kis(cfg, "mock")
        if src is None:
            raise BrokerError("paper 브로커에는 시세 소스가 필요합니다(paper_quote_source=nh_plug|kis)",
                              category="config")
        fees = get_fee_schedule(cfg.costs.preset, commission_rate=cfg.costs.commission_rate)
        return PaperBroker(src.get_quote, initial_cash=cfg.broker.paper_initial_cash,
                           state_file=cfg.broker.paper_state_file, fees=fees, slippage_bps=cfg.costs.slippage_bps,
                           history_fn=src.get_daily_history)
    raise BrokerError(f"알 수 없는 브로커: {name}", category="config")


__all__ = ["Broker", "BrokerError", "build_broker"]
