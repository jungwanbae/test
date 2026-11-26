"""종목 알파 슬리브 — 실거래용 데이터 동기화 · 모델 학습 · 종목 선정.

데이터: FinanceData/marcap(매일 장 마감 후 자동 갱신되는 KRX 전종목 일별 데이터) 로컬 복제본
흐름: sync_marcap() → build_panel() → build_dataset(최근 표본일) → 모델 점수 → 상위 N(히스테리시스)
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from autotrader.equity.factors import build_dataset, sample_dates
from autotrader.equity.models import FEATURES, _xy
from autotrader.equity.panel import MARCAP_REPO, build_panel

log = logging.getLogger(__name__)


@dataclass
class StockSleeveConfig:
    marcap_dir: str = "data/marcap"
    model_path: str = "models/stock_lgbm.txt"
    top_n: int = 15
    buffer: float = 2.5
    universe_size: int = 600
    min_amount: float = 2e8
    max_weight: float = 0.10
    min_price: float = 1000.0


def sync_marcap(marcap_dir: str | Path) -> Path:
    """marcap 저장소를 얕게 복제/갱신하고 data/ 폴더 경로를 돌려준다."""
    root = Path(marcap_dir)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", MARCAP_REPO, str(root)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "data"], check=True, capture_output=True)
    else:
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "--depth", "1"], check=False, capture_output=True)
    return root / "data"


def latest_rows(panel, cfg: StockSleeveConfig, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """신호일(가장 최근 거래일 또는 as_of 이전 마지막 거래일)의 유니버스 피처."""
    idx = panel.dates if as_of is None else panel.dates[panel.dates <= as_of]
    d = pd.DatetimeIndex([idx[-1]])
    return build_dataset(panel, d, universe_size=cfg.universe_size, min_amount=cfg.min_amount,
                         min_price=cfg.min_price, universe_by="marcap")


def train_final(ds: pd.DataFrame, path: str | Path, seed: int = 7) -> Path:
    """전체 이력으로 최종 LightGBM 학습(실거래용). 메타정보를 .json 으로 함께 저장."""
    import lightgbm as lgb
    lab = ds.dropna(subset=["target"])
    X, y = _xy(lab)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=seed, n_jobs=4, verbose=-1)
    model.fit(X, y)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out))
    out.with_suffix(".json").write_text(json.dumps({
        "features": FEATURES, "train_rows": int(len(lab)), "train_start": str(lab["date"].min().date()),
        "train_end": str(lab["date"].max().date()), "trained_on": date.today().isoformat()}, ensure_ascii=False, indent=1))
    return out


def score_rows(rows: pd.DataFrame, model_path: str | Path) -> pd.Series:
    import lightgbm as lgb
    meta = json.loads(Path(model_path).with_suffix(".json").read_text())
    if meta.get("features") != FEATURES:
        raise ValueError("모델 피처 목록이 현재 코드와 다릅니다 → autotrader stock-train 으로 재학습하세요")
    booster = lgb.Booster(model_file=str(model_path))
    return pd.Series(booster.predict(_xy(rows)[0]), index=rows["code"].values)


def select(scores: pd.Series, held: list[str], cfg: StockSleeveConfig) -> pd.Series:
    """상위 N + 히스테리시스(보유 종목은 순위 N×buffer 안이면 유지) → 동일가중."""
    order = list(scores.sort_values(ascending=False).index)
    pos = {c: i for i, c in enumerate(order)}
    keep = sorted([c for c in held if c in pos and pos[c] < cfg.top_n * cfg.buffer], key=lambda c: pos[c])[: cfg.top_n]
    sel = keep + [c for c in order if c not in set(keep)][: cfg.top_n - len(keep)]
    w = pd.Series(1.0 / len(sel), index=sel).clip(upper=cfg.max_weight)
    return w / w.sum()


def liquidity_cap(weights: pd.Series, adv20: pd.Series, sleeve_capital: float, cap: float = 0.10) -> tuple[pd.Series, dict]:
    """종목별 보유금액이 20일 평균 거래대금 × cap 을 넘지 않게 비중을 줄인다(슬리피지·부분체결 위험 통제).

    반환: (조정 비중 — 합이 1 미만일 수 있음, 축소 내역)
    """
    out, cut = weights.copy(), {}
    if sleeve_capital <= 0:
        return out, cut
    for c, w in weights.items():
        adv = float(adv20.get(c, float("nan")))
        if not adv == adv or adv <= 0:  # NaN/0 → 매수하지 않음
            out[c] = 0.0
            cut[c] = "거래대금 정보 없음"
            continue
        limit = cap * adv / sleeve_capital
        if w > limit:
            out[c] = limit
            cut[c] = f"{w:.1%} → {limit:.1%} (20일 평균 거래대금 {adv / 1e8:.1f}억원의 {cap:.0%})"
    return out, cut


def build_training_dataset(panel, cfg: StockSleeveConfig) -> pd.DataFrame:
    dates = sample_dates(panel.dates, "W")
    dates = dates[dates >= panel.dates[0] + pd.Timedelta(days=400)]
    return build_dataset(panel, dates, universe_size=cfg.universe_size, min_amount=cfg.min_amount,
                         min_price=cfg.min_price, universe_by="marcap")


def sleeve_config_dict(cfg: StockSleeveConfig) -> dict:
    return asdict(cfg)


__all__ = ["StockSleeveConfig", "sync_marcap", "latest_rows", "train_final", "score_rows", "select",
           "liquidity_cap", "build_training_dataset", "build_panel"]
