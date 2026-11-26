"""종목 순위 모델 — 워크포워드(확장 윈도) 학습으로 표본외 점수만 생성.

  linear  : 릿지 회귀(피처 → 횡단면 순위 타깃). 해석 가능한 기준선
  lgbm    : LightGBM 회귀(비선형·상호작용)
  mlp     : 딥러닝 MLP(잔차블록, 드롭아웃) — 손실 = MSE + ListNet(날짜별 순위)
  xsformer: 딥러닝 Cross-Sectional Transformer — 같은 날짜의 전 종목을 하나의 집합으로 보고
            종목 간 셀프어텐션으로 '시장 맥락 속 상대적 매력도'를 학습

퍼지: 학습 표본의 타깃 구간(H 거래일)이 테스트 시작일을 넘지 않도록 purge_days 만큼 비운다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.equity.factors import MARKET_FEATURES, STOCK_FEATURES

log = logging.getLogger(__name__)
FEATURES = STOCK_FEATURES + MARKET_FEATURES


@dataclass
class WFConfig:
    first_test_year: int = 2001
    retrain_years: int = 1
    purge_days: int = 35          # 21 거래일 타깃 + 여유
    min_train_years: int = 4
    seed: int = 7
    train_window_years: float | None = None   # None=확장 윈도(전체 과거), 숫자=최근 N년만(국면 변화 적응)


def _folds(dates: pd.Series, cfg: WFConfig):
    years = sorted(dates.dt.year.unique())
    start = max(cfg.first_test_year, years[0] + cfg.min_train_years)
    for y in range(start, years[-1] + 1, cfg.retrain_years):
        test_lo = pd.Timestamp(f"{y}-01-01")
        test_hi = pd.Timestamp(f"{y + cfg.retrain_years}-01-01")
        train_hi = test_lo - pd.Timedelta(days=cfg.purge_days)
        yield train_hi, test_lo, test_hi


def _xy(df: pd.DataFrame):
    X = np.array(df[FEATURES].to_numpy(np.float32), copy=True)
    X[~np.isfinite(X)] = 0.0
    return X, df["target"].to_numpy(np.float32)


# ---------------------------------------------------------------------- 모델별 학습기
def _fit_linear(tr: pd.DataFrame):
    X, y = _xy(tr)
    lam = 10.0 * len(X) / 1e4
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X.T @ y)
    return lambda te: _xy(te)[0] @ w


def _fit_lgbm(tr: pd.DataFrame, seed: int):
    import lightgbm as lgb
    X, y = _xy(tr)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=seed, n_jobs=4, verbose=-1)
    model.fit(X, y)
    return lambda te: model.predict(_xy(te)[0])


def _torch():
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch


def _date_batches(df: pd.DataFrame):
    """날짜별 (X[n,F], y[n]) 리스트."""
    out = []
    for _, g in df.groupby("date", sort=True):
        X, y = _xy(g)
        out.append((X, y))
    return out


def _listnet(torch, pred, y, temp=0.5):
    p = torch.log_softmax(pred, dim=-1)
    t = torch.softmax(y / temp, dim=-1)
    return -(t * p).sum(-1).mean()


def _fit_mlp(tr: pd.DataFrame, seed: int, epochs: int = 12):
    torch = _torch()
    from torch import nn
    torch.manual_seed(seed)
    F = len(FEATURES)

    class Res(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))

        def forward(self, x):
            return x + self.f(x)

    net = nn.Sequential(nn.Linear(F, 128), nn.GELU(), Res(128, 0.2), Res(128, 0.2), nn.LayerNorm(128),
                        nn.Dropout(0.2), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    batches = _date_batches(tr)
    n_val = max(1, len(batches) // 8)
    train_b, val_b = batches[:-n_val], batches[-n_val:]
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        for i in rng.permutation(len(train_b)):
            X, y = (torch.from_numpy(a) for a in train_b[i])
            pred = net(X).squeeze(-1)
            loss = nn.functional.mse_loss(pred, y) + 0.5 * _listnet(torch, pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(net(torch.from_numpy(X)).squeeze(-1),
                                                       torch.from_numpy(y)).item() for X, y in val_b]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    def predict(te: pd.DataFrame):
        with torch.no_grad():
            return net(torch.from_numpy(_xy(te)[0])).squeeze(-1).numpy()
    return predict


def _fit_xsformer(tr: pd.DataFrame, seed: int, epochs: int = 8, d: int = 48, max_stocks: int = 400):
    torch = _torch()
    from torch import nn
    torch.manual_seed(seed)
    F = len(FEATURES)

    class XSFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(F, d), nn.GELU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.15, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, x):          # x [1, n, F]
            h = self.inp(x)
            return self.head(h + self.enc(h)).squeeze(-1)

    net = XSFormer()
    opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=1e-3)
    batches = _date_batches(tr)
    n_val = max(1, len(batches) // 8)
    train_b, val_b = batches[:-n_val], batches[-n_val:]
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        for i in rng.permutation(len(train_b)):
            X, y = train_b[i]
            if len(X) > max_stocks:  # 학습 시 종목 부분집합 샘플링(속도·정규화)
                sel = rng.choice(len(X), max_stocks, replace=False)
                X, y = X[sel], y[sel]
            X, y = torch.from_numpy(X)[None], torch.from_numpy(y)[None]
            pred = net(X)
            loss = nn.functional.mse_loss(pred, y) + 0.5 * _listnet(torch, pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(net(torch.from_numpy(X)[None]),
                                                       torch.from_numpy(y)[None]).item() for X, y in val_b]))
        log.info("xsformer epoch %d val %.4f", ep, vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    def predict(te: pd.DataFrame):
        out = np.empty(len(te), dtype=np.float32)
        with torch.no_grad():
            for _, idx in te.groupby("date").indices.items():
                X = torch.from_numpy(_xy(te.iloc[idx])[0])[None]
                out[idx] = net(X)[0].numpy()
        return out
    return predict


FITTERS = {"linear": lambda tr, s: _fit_linear(tr), "lgbm": _fit_lgbm, "mlp": _fit_mlp, "xsformer": _fit_xsformer}


def walk_forward_scores(ds: pd.DataFrame, model: str, cfg: WFConfig | None = None,
                        train_stride: int = 1) -> pd.Series:
    """표본외 점수(Series, ds 인덱스). train_stride>1 이면 학습 날짜를 솎아 속도를 높인다."""
    cfg = cfg or WFConfig()
    fit = FITTERS[model]
    ds = ds.sort_values(["date", "code"])
    lab = ds.dropna(subset=["target"])
    all_dates = np.sort(lab["date"].unique())
    keep_dates = set(all_dates[::train_stride])
    scores = pd.Series(np.nan, index=ds.index, dtype=np.float64)
    for k, (train_hi, test_lo, test_hi) in enumerate(_folds(ds["date"], cfg)):
        tr = lab[(lab["date"] < train_hi) & lab["date"].isin(keep_dates)]
        if cfg.train_window_years:
            tr = tr[tr["date"] >= train_hi - pd.Timedelta(days=int(365.25 * cfg.train_window_years))]
        te = ds[(ds["date"] >= test_lo) & (ds["date"] < test_hi)]
        if len(te) == 0 or tr["date"].nunique() < 50:
            continue
        predict = fit(tr, cfg.seed + k)
        scores.loc[te.index] = np.asarray(predict(te), dtype=np.float64)
        log.info("%s fold %s: train %d행(~%s) → test %s", model, test_lo.year, len(tr), train_hi.date(), len(te))
    return scores
