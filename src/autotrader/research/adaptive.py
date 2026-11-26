"""시대 적응형 머신러닝·딥러닝 타이밍 연구 — '큰 상승 확률은 높고 큰 하락 확률은 낮은' 패턴 학습.

설계(사용자 제안을 그대로 시험 가능한 형태로):
  * 표본: 15개 위험자산(주식지수·섹터·신흥국·리츠·금·원자재·장기채·하이일드·KOSPI200)을 한데 모아 학습(pooled).
          가격·거래량을 변동성으로 정규화한 '차트 모양' 피처만 써서 자산이 달라도 같은 패턴을 공유하게 한다.
          1971~ 장기 시계열(S&P500·금·국채·원자재)을 넣어 1970년대 인플레·1987·2000·2008 같은 다양한 국면을 포함.
  * 타깃: 향후 21거래일 수익 / (63일 변동성·√21) = z.  z > +1 '큰 상승', z < −1 '큰 하락', 그 외 보합(3분류).
  * 학습 방식(가중치):  expanding(전체 과거 동일) · rolling10(최근 10년) · decay5/decay2(반감기 5년/2년, 최근 가중)
                       · similar(현재와 비슷한 시장 국면의 과거 표본 가중: 변동성·추세·금리·주식채권상관의 커널 거리)
                       · bag(연도 블록을 무작위로 섞어 뽑은 5개 모델 평균 — '시기를 섞어서' 과최적화 완화)
                       · ensemble(위 방식들의 확률 평균)
  * 모델: LightGBM(트리) · MLP(딥러닝) · CNN(최근 63일 수익률·거래량 '차트' 시퀀스를 직접 보는 1D 합성곱 딥러닝)
  * 검증: 워크포워드 — 매년(또는 분기) 그 시점 이전 데이터만으로 학습, 타깃 기간만큼 퍼지(purge). 미래 정보 사용 없음.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.research.data import US_DIR, tbill_daily, vix

H = 21                    # 예측 기간(거래일)
PURGE_DAYS = 35           # 학습 끝 ~ 시험 시작 간격(달력일) — 타깃이 겹치지 않게
SEQ = 63                  # CNN 이 보는 최근 일수
ETF_WITH_VOL = ["QQQ", "IWM", "EFA", "EEM", "VWO", "VNQ", "SOXX", "XLK", "HYG"]
LONG = {"SPX": "us_large_cap_sp500", "GOLD": "gold", "UST_L": "long_term_us_treasury",
        "UST_I": "intermediate_term_us_treasury", "CMDTY": "broad_commodities"}
STATE = ["st_vol", "st_trend", "st_ret252", "st_rate", "st_rate_chg", "st_sbcorr"]


# ---------------------------------------------------------------------- 데이터
def load_assets(k200: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """자산별 DataFrame(close, dvol). dvol=거래대금(없으면 NaN). 날짜는 각 자산 고유 거래일."""
    out = {}
    for k, f in LONG.items():
        d = pd.read_csv(US_DIR / f"{f}.csv", usecols=["Date", "Adj Close", "Close", "Volume"], parse_dates=["Date"])
        d = d.dropna(subset=["Adj Close"]).drop_duplicates("Date").set_index("Date").sort_index()
        dv = d["Close"] * d["Volume"] if d["Volume"].notna().mean() > 0.9 else pd.Series(np.nan, index=d.index)
        out[k] = pd.DataFrame({"close": d["Adj Close"].astype(float), "dvol": dv.astype(float)})
    px = pd.read_parquet(US_DIR / "prices.parquet", columns=["date", "ticker", "close", "volume", "adj_close"],
                         filters=[("ticker", "in", ETF_WITH_VOL)])
    for t, g in px.groupby("ticker"):
        g = g.dropna(subset=["adj_close"]).set_index(pd.to_datetime(g["date"].dropna()).dt.normalize()).sort_index()
        g = g[~g.index.duplicated()]
        out[t] = pd.DataFrame({"close": g["adj_close"].astype(float), "dvol": (g["close"] * g["volume"]).astype(float)})
    if k200 is not None:
        k = k200.dropna()
        out["K200"] = pd.DataFrame({"close": k.astype(float), "dvol": np.nan})
    return out


def market_state(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """모든 자산 공통의 시장 국면 피처(미국 기준, 1971~). 유사 국면 가중에도 쓴다."""
    spx = assets["SPX"]["close"]
    ust = assets["UST_L"]["close"].reindex(spx.index).ffill()
    r, rb = spx.pct_change(), ust.pct_change()
    rate = tbill_daily(spx.index)
    st = pd.DataFrame(index=spx.index)
    st["st_vol"] = r.rolling(63).std() * np.sqrt(252)
    st["st_trend"] = spx / spx.rolling(200).mean() - 1
    st["st_ret252"] = spx / spx.shift(252) - 1
    st["st_rate"] = rate
    st["st_rate_chg"] = rate - rate.shift(252)
    st["st_sbcorr"] = r.rolling(126).corr(rb)
    v = vix().reindex(spx.index).ffill()
    st["st_vix"] = v
    st["st_vix_rel"] = v / v.rolling(252).mean() - 1
    return st


def asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """변동성으로 정규화한 가격·거래량 피처(스케일 무관 → 여러 자산을 한 모델로)."""
    c = df["close"]
    lr = np.log(c).diff()
    s63 = lr.rolling(63).std()
    s21 = lr.rolling(21).std()
    f = pd.DataFrame(index=c.index)
    for k in (1, 5, 10, 21, 63, 126, 252):
        f[f"z{k}"] = np.log(c / c.shift(k)) / (s63 * np.sqrt(k))
    f["vol21"] = s21 * np.sqrt(252)
    f["vol63"] = s63 * np.sqrt(252)
    f["vol_ratio"] = s21 / lr.rolling(252).std()
    f["vol_pct"] = s21.rolling(756, min_periods=252).rank(pct=True)
    f["ma50"] = np.log(c / c.rolling(50).mean()) / s63
    f["ma200"] = np.log(c / c.rolling(200).mean()) / s63
    f["gc"] = np.log(c.rolling(50).mean() / c.rolling(200).mean()) / s63
    f["dd252"] = c / c.rolling(252).max() - 1
    f["dd_ath"] = c / c.cummax() - 1
    f["from_low"] = c / c.rolling(252).min() - 1
    f["days_hi"] = (252 - c.rolling(252).apply(np.argmax, raw=True)) / 252
    f["skew63"] = lr.rolling(63).skew()
    f["kurt63"] = lr.rolling(63).kurt()
    f["down_frac21"] = (lr < 0).rolling(21).mean()
    f["worst21"] = lr.rolling(21).min() / s63
    f["best21"] = lr.rolling(21).max() / s63
    dv = df["dvol"]
    if dv.notna().any():
        ldv = np.log(dv.where(dv > 0))
        f["v5_63"] = ldv.rolling(5).mean() - ldv.rolling(63).mean()
        f["v21_252"] = ldv.rolling(21).mean() - ldv.rolling(252).mean()
        up = (lr > 0).astype(float)
        f["upvol21"] = (dv * up).rolling(21).sum() / dv.rolling(21).sum()
        f["v_absret"] = ldv.rolling(63).corr(lr.abs())
    else:
        for k in ("v5_63", "v21_252", "upvol21", "v_absret"):
            f[k] = np.nan
    return f


def labels(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    lr = np.log(c).diff()
    s63 = lr.rolling(63).std()
    fwd = np.log(c.shift(-H) / c)
    z = fwd / (s63 * np.sqrt(H))
    y = pd.Series(1, index=c.index)
    y[z > 1.0] = 2
    y[z < -1.0] = 0
    y[z.isna()] = -1
    return pd.DataFrame({"y": y, "z": z, "fwd": np.exp(fwd) - 1})


@dataclass
class Panel:
    X: pd.DataFrame          # (asset, date) × features
    y: np.ndarray
    z: np.ndarray
    asset: np.ndarray
    date: pd.DatetimeIndex
    seq_src: dict            # asset → (dates, ret/σ, log volume ratio)
    feats: list[str]


def build_panel(assets: dict[str, pd.DataFrame], start="1972-01-01") -> Panel:
    st = market_state(assets)
    rows = []
    seq = {}
    for name, df in assets.items():
        f = asset_features(df)
        lab = labels(df)
        s = st.reindex(df.index.union(st.index)).ffill().reindex(df.index)
        d = pd.concat([f, s, lab], axis=1)
        d["asset"] = name
        d = d.loc[start:]
        d = d[d["z252"].notna() & d["vol63"].notna()]
        rows.append(d)
        lr = np.log(df["close"]).diff()
        s63 = lr.rolling(63).std()
        ldv = np.log(df["dvol"].where(df["dvol"] > 0))
        vr = (ldv - ldv.rolling(63).mean()) if df["dvol"].notna().any() else pd.Series(np.nan, index=df.index)
        seq[name] = (df.index, (lr / s63).clip(-6, 6).fillna(0).to_numpy(np.float32),
                     vr.clip(-3, 3).to_numpy(np.float32))
    P = pd.concat(rows).reset_index().rename(columns={"index": "date", "Date": "date"})
    P = P.sort_values(["date", "asset"]).reset_index(drop=True)
    feats = [c for c in P.columns if c not in ("date", "asset", "y", "z", "fwd")]
    return Panel(P[feats], P["y"].to_numpy(), P["z"].to_numpy(), P["asset"].to_numpy(),
                 pd.DatetimeIndex(P["date"]), seq, feats)


# ---------------------------------------------------------------------- 학습 가중치
def scheme_weights(scheme: str, dates: pd.DatetimeIndex, test_lo: pd.Timestamp, state: np.ndarray | None = None,
                   state_now: np.ndarray | None = None, ess_frac: float = 0.25) -> np.ndarray:
    age = (test_lo - dates).days.values / 365.25
    if scheme in ("expanding", "bag"):
        w = np.ones(len(dates))
    elif scheme == "rolling10":
        w = (age <= 10).astype(float)
    elif scheme.startswith("decay"):
        hl = float(scheme[5:])
        w = 0.5 ** (age / hl)
    elif scheme == "similar":
        S = state.copy()
        ok = np.isfinite(S).all(1) & np.isfinite(state_now).all()
        mu, sd = np.nanmean(S, 0), np.nanstd(S, 0) + 1e-9
        Z, z0 = (S - mu) / sd, (state_now - mu) / sd
        d2 = np.where(ok, ((Z - z0) ** 2).sum(1), np.inf)
        lo, hi = 0.05, 20.0                       # 대역폭: 유효표본 ≈ ess_frac·N 이 되도록 이분탐색
        for _ in range(30):
            bw = (lo * hi) ** 0.5
            w = np.exp(-d2 / (2 * bw * bw))
            ess = w.sum() ** 2 / max((w * w).sum(), 1e-12)
            if ess < ess_frac * ok.sum():
                lo = bw
            else:
                hi = bw
    else:
        raise ValueError(scheme)
    m = w.mean()
    return w / m if m > 0 else w


def folds(dates: pd.DatetimeIndex, first_test_year: int, step_months: int = 12):
    end = dates.max()
    t = pd.Timestamp(f"{first_test_year}-01-01")
    while t <= end:
        nxt = t + pd.DateOffset(months=step_months)
        yield t - pd.Timedelta(days=PURGE_DAYS), t, nxt
        t = nxt


# ---------------------------------------------------------------------- 모델
def _prep(X: pd.DataFrame, mu=None, sd=None):
    A = X.to_numpy(np.float32, copy=True)
    miss = ~np.isfinite(A)
    if mu is None:
        mu = np.nanmean(np.where(miss, np.nan, A), 0)
        sd = np.nanstd(np.where(miss, np.nan, A), 0) + 1e-6
        mu, sd = np.nan_to_num(mu), np.nan_to_num(sd, nan=1.0)
    A = np.where(miss, 0.0, (A - mu) / sd).clip(-6, 6)
    flag = miss[:, [X.columns.get_loc("v5_63")]].astype(np.float32)
    return np.hstack([A, flag]).astype(np.float32), mu, sd


def fit_predict_lgbm(Xtr, ytr, wtr, Xte, seed=7):
    import lightgbm as lgb
    params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=400,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                  num_threads=4, seed=seed)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=250)
    return m.predict(Xte)


def _torch_fit(net, tensors_tr, ytr, wtr, tensors_va, yva, wva, epochs=25, lr=1e-3, seed=7):
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss(reduction="none")
    Y, W = torch.tensor(ytr, dtype=torch.long), torch.tensor(wtr, dtype=torch.float32)
    Yv, Wv = torch.tensor(yva, dtype=torch.long), torch.tensor(wva, dtype=torch.float32)
    n = len(Y)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            out = net(*[t[b] for t in tensors_tr])
            loss = (lossf(out, Y[b]) * W[b]).sum() / W[b].sum()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float((lossf(net(*tensors_va), Yv) * Wv).sum() / Wv.sum())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    if best_state:
        net.load_state_dict(best_state)
    return net


def fit_predict_mlp(Xtr, ytr, wtr, Xte, dtr, seed=7):
    import torch
    nn = torch.nn
    torch.set_num_threads(4)
    va = dtr >= dtr.max() - pd.Timedelta(days=730)   # 학습구간 마지막 2년 = 검증(조기종료)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(32, 3))
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    net = _torch_fit(net, [T(Xtr[~va])], ytr[~va], wtr[~va], [T(Xtr[va])], ytr[va], wtr[va], seed=seed)
    with torch.no_grad():
        return torch.softmax(net(T(Xte)), 1).numpy()


def seq_tensor(panel: Panel, idx: np.ndarray) -> np.ndarray:
    """표본별 최근 SEQ 일의 (수익/σ, 거래량 비율, 거래량 없음 표시) 시퀀스 → (n, 3, SEQ)."""
    out = np.zeros((len(idx), 3, SEQ), np.float32)
    pos_cache = {}
    for j, i in enumerate(idx):
        a, d = panel.asset[i], panel.date[i]
        dates, r, v = panel.seq_src[a]
        if a not in pos_cache:
            pos_cache[a] = {dd: k for k, dd in enumerate(dates)}
        k = pos_cache[a].get(d)
        if k is None or k < SEQ:
            continue
        out[j, 0] = r[k - SEQ + 1:k + 1]
        vv = v[k - SEQ + 1:k + 1]
        out[j, 2] = np.isnan(vv)
        out[j, 1] = np.nan_to_num(vv)
    return out


def fit_predict_cnn(Str, Xtr, ytr, wtr, Ste, Xte, dtr, seed=7):
    import torch
    nn = torch.nn
    torch.set_num_threads(4)

    class Net(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(16, 16, 5, padding=4, dilation=2), nn.ReLU(),
                                      nn.Conv1d(16, 16, 5, padding=8, dilation=4), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(32 + k, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

        def forward(self, s, x):
            h = self.conv(s)
            return self.head(torch.cat([h.mean(2), h.amax(2), x], 1))

    va = dtr >= dtr.max() - pd.Timedelta(days=730)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    net = _torch_fit(Net(Xtr.shape[1]), [T(Str[~va]), T(Xtr[~va])], ytr[~va], wtr[~va],
                     [T(Str[va]), T(Xtr[va])], ytr[va], wtr[va], seed=seed)
    with torch.no_grad():
        return torch.softmax(net(T(Ste), T(Xte)), 1).numpy()


def walk_forward(panel: Panel, model: str, scheme: str, first_test_year: int = 1985, step_months: int = 12,
                 sub: int = 3, seed: int = 7, log=print) -> pd.DataFrame:
    """반환: 시험구간 전 표본의 OOS 확률(p_down, p_flat, p_up) — index=(date, asset)."""
    dates = panel.date
    lab_ok = panel.y >= 0
    st = panel.X[STATE].to_numpy(np.float64)
    preds = []
    for train_hi, test_lo, test_hi in folds(dates, first_test_year, step_months):
        tr = np.where((dates < train_hi) & lab_ok)[0]
        te = np.where((dates >= test_lo) & (dates < test_hi))[0]
        if len(te) == 0 or len(tr) < 5000:
            continue
        # 21일 타깃이 겹쳐 인접일은 거의 같은 정보 → 3일 간격으로 솎아 학습(속도·과적합 완화)
        day_id = (dates[tr] - dates[0]).days.values
        tr = tr[(day_id // 1) % sub == 0] if sub > 1 else tr
        now = panel.X[STATE].iloc[te[0]].to_numpy(np.float64)
        Xtr_df, Xte_df = panel.X.iloc[tr], panel.X.iloc[te]
        ytr = panel.y[tr]
        if scheme == "bag":
            yrs = np.array(sorted(set(dates[tr].year)))
            rng = np.random.default_rng(seed + test_lo.year)
            P = 0
            for b in range(5):
                keep = set(rng.choice(yrs, size=max(1, int(0.7 * len(yrs))), replace=False))
                m = np.isin(dates[tr].year, list(keep))
                P = P + _fit_one(model, panel, tr[m], te, Xtr_df[m], Xte_df, ytr[m], np.ones(m.sum()), seed + b)
            P = P / 5
        else:
            w = scheme_weights(scheme, dates[tr], test_lo, st[tr], now)
            keep = w > 1e-6
            P = _fit_one(model, panel, tr[keep], te, Xtr_df[keep], Xte_df, ytr[keep], w[keep], seed)
        preds.append(pd.DataFrame(P, columns=["p_down", "p_flat", "p_up"],
                                  index=pd.MultiIndex.from_arrays([dates[te], panel.asset[te]], names=["date", "asset"])))
        log(f"  {model}/{scheme} {test_lo.date()} 학습 {len(tr):,} 시험 {len(te):,}")
    return pd.concat(preds)


def _fit_one(model, panel, tr, te, Xtr_df, Xte_df, ytr, w, seed):
    if model == "lgbm":
        return fit_predict_lgbm(Xtr_df.to_numpy(np.float32), ytr, w, Xte_df.to_numpy(np.float32), seed)
    Xtr, mu, sd = _prep(Xtr_df)
    Xte, _, _ = _prep(Xte_df, mu, sd)
    if model == "mlp":
        return fit_predict_mlp(Xtr, ytr, w, Xte, panel.date[tr], seed)
    if model == "cnn":
        return fit_predict_cnn(seq_tensor(panel, tr), Xtr, ytr, w, seq_tensor(panel, te), Xte, panel.date[tr], seed)
    raise ValueError(model)
