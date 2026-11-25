"""워크포워드(walk-forward) 학습·검증 + 검증 게이트.

  * 확장 윈도: 매 retrain_every_months 마다 '그 시점까지의 데이터'로만 재학습 → 다음 구간 예측
  * 퍼지(purge): 학습 마지막 타깃이 테스트 구간과 겹치지 않도록 horizon 만큼 비운다
  * 조기종료: 학습 구간의 마지막 20%(시간순)를 검증셋으로
  * 앙상블: 시드 여러 개 평균 → 분산 감소
  * 게이트: 표본외 순위 IC 평균·t값이 기준 미달이면 ML 을 전략에 반영하지 않는다(blend=0)
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.ml.features import FEATURES, forward_returns, make_features, make_target, to_tensor
from autotrader.strategy.signals import BarSpec

log = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("ML 기능에는 PyTorch 가 필요합니다: pip install 'autotrader[ml]'") from e
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch


@dataclass
class MLConfig:
    seq_len_months: float = 3
    horizon_months: float = 1
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.2
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    patience: int = 8
    seeds: int = 3
    rank_weight: float = 0.5
    retrain_every_months: int = 12
    min_train_months: int = 60
    sample_stride_days: int = 5       # 일봉에서 표본 간격(중복 완화)
    min_ic: float = 0.02
    min_ic_tstat: float = 1.5


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # 표본외 예측(날짜 × 자산)
    ic: pd.Series                      # 날짜별 순위 IC
    summary: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.summary.get("passed"))


def _seq_len(cfg: MLConfig, bars: BarSpec) -> int:
    return max(6, bars.bars(cfg.seq_len_months))


def _stride(cfg: MLConfig, bars: BarSpec) -> int:
    return 1 if bars.periods_per_year <= 52 else max(1, cfg.sample_stride_days)


def _fit_ensemble(X, Y, M, n_assets: int, cfg: MLConfig, seed0: int = 0):
    torch = _torch()
    from autotrader.ml.model import CrossAssetTemporalNet, combined_loss

    n = len(X)
    n_val = max(1, int(n * 0.2))
    tr, va = slice(0, n - n_val), slice(n - n_val, n)
    Xt, Yt, Mt = (torch.from_numpy(a) for a in (X, np.nan_to_num(Y, nan=0.0), M))
    Vt = torch.from_numpy(np.isfinite(Y) & ~M)
    models = []
    for s in range(cfg.seeds):
        torch.manual_seed(seed0 + s)
        np.random.seed(seed0 + s)
        model = CrossAssetTemporalNet(len(FEATURES), n_assets, cfg.hidden, cfg.heads, cfg.dropout)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad = math.inf, None, 0
        idx_tr = np.arange(tr.stop)
        for _ in range(cfg.epochs):
            model.train()
            np.random.shuffle(idx_tr)
            for b in range(0, len(idx_tr), cfg.batch_size):
                ib = torch.from_numpy(idx_tr[b: b + cfg.batch_size])
                loss = combined_loss(model(Xt[ib], Mt[ib]), Yt[ib], Vt[ib], cfg.rank_weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(combined_loss(model(Xt[va], Mt[va]), Yt[va], Vt[va], cfg.rank_weight))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)
    return models


def _predict(models, X, M) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        xt, mt = torch.from_numpy(X), torch.from_numpy(M)
        preds = np.mean([m(xt, mt).numpy() for m in models], axis=0)
    preds[M] = np.nan
    return preds


def rank_ic(pred: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    out = {}
    for d in pred.index:
        p = pred.loc[d]
        r = realized.loc[d] if d in realized.index else None
        if r is None:
            continue
        ok = p.notna() & r.notna()
        if ok.sum() < 3:
            continue
        out[d] = float(p[ok].rank().corr(r[ok].rank()))
    return pd.Series(out, dtype=float)


def walk_forward(prices: pd.DataFrame, bars: BarSpec | None = None, cfg: MLConfig | None = None,
                 cash_col: str | None = None, verbose: bool = True) -> WalkForwardResult:
    cfg = cfg or MLConfig()
    bars = bars or BarSpec.infer(prices.index)
    assets = [c for c in prices.columns if c != cash_col]
    cash = prices[cash_col] if cash_col else None
    feats = make_features(prices[assets], bars)
    target = make_target(prices[assets], bars, cfg.horizon_months, cash=cash)
    realized = forward_returns(prices[assets], bars, cfg.horizon_months)
    L, stride, h = _seq_len(cfg, bars), _stride(cfg, bars), bars.bars(cfg.horizon_months)
    dates = prices.index
    valid_feat = pd.concat([feats[k] for k in FEATURES], axis=1).notna().any(axis=1)
    first_ok = valid_feat[valid_feat].index.min() if valid_feat.any() else None
    if first_ok is None:
        raise ValueError("피처를 만들 데이터가 부족합니다")
    pos0 = max(dates.get_loc(first_ok) + L, bars.bars(cfg.min_train_months))
    step = bars.bars(cfg.retrain_every_months)
    preds = pd.DataFrame(np.nan, index=dates, columns=assets)
    fold = 0
    for test_start in range(pos0, len(dates), step):
        test_end = min(test_start + step, len(dates))
        train_last = test_start - h - 1  # 퍼지: 학습 타깃이 테스트 시작 이후 가격을 보지 않게
        train_dates = dates[: train_last + 1][::stride]
        train_dates = train_dates[train_dates >= first_ok]
        if len(train_dates) < 30:
            continue
        X, Y, M = to_tensor(feats, target, assets, L, train_dates)
        keep = np.isfinite(Y).any(axis=1)
        X, Y, M = X[keep], Y[keep], M[keep]
        models = _fit_ensemble(X, Y, M, len(assets), cfg, seed0=fold * 100)
        tdates = dates[test_start:test_end]
        Xs, _, Ms = to_tensor(feats, None, assets, L, tdates)
        if len(Xs):
            preds.loc[tdates[-len(Xs):]] = _predict(models, Xs, Ms)
        fold += 1
        if verbose:
            log.info("ML fold %d: train≤%s (%d표본) → test %s~%s", fold, dates[train_last].date(), len(X),
                     tdates[0].date(), tdates[-1].date())
    preds = preds.dropna(how="all")
    eval_dates = preds.index[::stride]
    ic = rank_ic(preds.loc[eval_dates], realized)
    n = len(ic)
    overlap = max(1.0, h / stride)
    tstat = float(ic.mean() / ic.std(ddof=1) * math.sqrt(n / overlap)) if n > 2 and ic.std(ddof=1) > 0 else 0.0
    summary = {"ic_mean": float(ic.mean()) if n else float("nan"), "ic_tstat": tstat, "n_obs": n,
               "hit_rate": float((ic > 0).mean()) if n else float("nan"), "folds": fold,
               "min_ic": cfg.min_ic, "min_ic_tstat": cfg.min_ic_tstat}
    summary["passed"] = bool(n > 10 and summary["ic_mean"] >= cfg.min_ic and tstat >= cfg.min_ic_tstat)
    return WalkForwardResult(preds, ic, summary)


def train_final(prices: pd.DataFrame, out_path: str | Path, bars: BarSpec | None = None,
                cfg: MLConfig | None = None, cash_col: str | None = None,
                validation: dict | None = None) -> Path:
    """전체 데이터로 최종 모델 학습 후 저장(실거래 예측용). validation 에 워크포워드 결과를 기록."""
    torch = _torch()
    cfg = cfg or MLConfig()
    bars = bars or BarSpec.infer(prices.index)
    assets = [c for c in prices.columns if c != cash_col]
    feats = make_features(prices[assets], bars)
    target = make_target(prices[assets], bars, cfg.horizon_months, cash=prices[cash_col] if cash_col else None)
    L, stride = _seq_len(cfg, bars), _stride(cfg, bars)
    h = bars.bars(cfg.horizon_months)
    dates = prices.index[: len(prices.index) - h][::stride]
    X, Y, M = to_tensor(feats, target, assets, L, dates)
    keep = np.isfinite(Y).any(axis=1)
    models = _fit_ensemble(X[keep], Y[keep], M[keep], len(assets), cfg, seed0=12345)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], "assets": assets, "cash_col": cash_col,
                "config": asdict(cfg), "bars": asdict(bars), "features": FEATURES,
                "validation": validation or {}}, out)
    out.with_suffix(".json").write_text(json.dumps({"assets": assets, "validation": validation or {}},
                                                   ensure_ascii=False, indent=1, default=str))
    return out


def load_and_predict(path: str | Path, prices: pd.DataFrame) -> tuple[pd.Series, dict]:
    """저장된 모델로 마지막 시점 점수를 예측. 반환: (자산별 점수, 검증 메타)."""
    torch = _torch()
    from autotrader.ml.model import CrossAssetTemporalNet

    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = MLConfig(**ck["config"])
    bars = BarSpec(**ck["bars"])
    assets = [a for a in ck["assets"]]
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise ValueError(f"모델 자산이 가격 데이터에 없습니다: {missing}")
    feats = make_features(prices[assets], bars)
    X, _, M = to_tensor(feats, None, assets, _seq_len(cfg, bars), prices.index[-1:])
    models = []
    for sd in ck["state_dicts"]:
        m = CrossAssetTemporalNet(len(FEATURES), len(assets), cfg.hidden, cfg.heads, cfg.dropout)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    pred = _predict(models, X, M)[0]
    return pd.Series(pred, index=assets), ck.get("validation", {})
