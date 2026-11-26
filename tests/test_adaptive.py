"""시대 적응형 학습 가중치·워크포워드 분할의 성질(미래참조 없음)."""
import numpy as np
import pandas as pd

from autotrader.research import adaptive as A


def test_scheme_weights_properties():
    d = pd.DatetimeIndex(pd.bdate_range("1990-01-01", "2019-12-31"))
    t = pd.Timestamp("2020-01-01")
    w = A.scheme_weights("rolling10", d, t)
    assert (w[d < "2009-12-01"] == 0).all() and (w[d > "2010-02-01"] > 0).all()
    w = A.scheme_weights("decay5", d, t)
    a, b = w[np.searchsorted(d, pd.Timestamp("2014-12-31"))], w[np.searchsorted(d, pd.Timestamp("2019-12-31"))]
    assert abs(a / b - 0.5) < 0.01                                    # 5년 전 표본은 가중치 절반
    rng = np.random.default_rng(0)
    S = rng.normal(size=(len(d), 3))
    w = A.scheme_weights("similar", d, t, S, np.zeros(3), ess_frac=0.25)
    ess = w.sum() ** 2 / (w * w).sum()
    assert 0.2 * len(d) < ess < 0.3 * len(d)                          # 유효표본 ≈ 25%
    near, far = np.linalg.norm(S, axis=1) < 0.5, np.linalg.norm(S, axis=1) > 2.5
    assert w[near].mean() > 5 * w[far].mean()                         # 현재와 비슷한 국면일수록 큰 가중


def test_folds_purge_no_overlap():
    d = pd.DatetimeIndex(pd.bdate_range("1980-01-01", "2026-06-30"))
    for train_hi, lo, hi in A.folds(d, 1990, 12):
        assert (lo - train_hi).days >= A.PURGE_DAYS                  # 21거래일 타깃 > 퍼지 간격 안쪽
        assert lo < hi


def test_labels_use_only_future_window():
    c = pd.Series(np.exp(np.cumsum(np.r_[np.zeros(100), np.full(30, 0.02), np.zeros(100)])),
                  index=pd.bdate_range("2020-01-01", periods=230))
    c = c * (1 + np.random.default_rng(1).normal(0, 0.001, len(c)))
    lab = A.labels(pd.DataFrame({"close": c, "dvol": np.nan}))
    assert lab["y"].iloc[95] == 2                                     # 직후 21일 급등 → 큰 상승
    assert (lab["y"].iloc[-A.H:] == -1).all()                         # 미래가 없는 끝부분은 학습 제외


def test_stock_seed_ensemble_train_and_score(tmp_path):
    from autotrader.equity.models import FEATURES
    from autotrader.equity.live import score_rows, train_final
    rng = np.random.default_rng(3)
    n = 3000
    ds = pd.DataFrame(rng.normal(size=(n, len(FEATURES))).astype(np.float32), columns=FEATURES)
    ds["target"] = ds[FEATURES[0]] * 0.5 + rng.normal(size=n) * 0.5
    ds["date"] = pd.Timestamp("2020-01-31")
    ds["code"] = [f"{i:06d}" for i in range(n)]
    p = train_final(ds, tmp_path / "m.txt", seeds=(1, 2, 3))
    files = sorted(x.name for x in tmp_path.iterdir())
    assert files == ["m.json", "m.s2.txt", "m.s3.txt", "m.txt"]
    s = score_rows(ds.iloc[:200], p)
    assert s.between(0, 1).all() and s.corr(ds[FEATURES[0]].iloc[:200].set_axis(s.index)) > 0.5
    p1 = train_final(ds, tmp_path / "single.txt", seeds=None)           # 단일 모델(하위호환)
    assert score_rows(ds.iloc[:50], p1).shape == (50,)
