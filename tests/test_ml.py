import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from autotrader.data.synthetic import generate  # noqa: E402
from autotrader.ml.features import FEATURES, make_features, make_target, to_tensor  # noqa: E402
from autotrader.ml.model import CrossAssetTemporalNet, combined_loss  # noqa: E402
from autotrader.ml.train import MLConfig, load_and_predict, train_final, walk_forward  # noqa: E402
from autotrader.strategy.signals import MONTHLY  # noqa: E402


@pytest.fixture(scope="module")
def monthly():
    px, _ = generate(n_days=252 * 14, seed=11)
    return px.resample("ME").last()


def test_features_do_not_peek(monthly):
    f1 = make_features(monthly, MONTHLY)
    cut = monthly.index[100]
    shocked = monthly.copy()
    shocked.loc[shocked.index > cut] *= 3
    f2 = make_features(shocked, MONTHLY)
    for k in FEATURES:
        pd.testing.assert_frame_equal(f1[k].loc[:cut], f2[k].loc[:cut])


def test_target_is_forward_and_last_is_nan(monthly):
    t = make_target(monthly, MONTHLY, horizon_months=1)
    assert t.iloc[-2:].isna().all().all() and t.iloc[-3].notna().all()  # gap=1 + h=1


def test_model_shapes_and_masking():
    m = CrossAssetTemporalNet(len(FEATURES), 5, hidden=16, heads=2)
    x = torch.randn(4, 5, 6, len(FEATURES))
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[0, 1] = True
    mask[1] = True  # 전부 마스크된 행도 NaN 없이 동작
    out = m(x, mask)
    assert out.shape == (4, 5) and torch.isfinite(out).all()
    y = torch.randn(4, 5)
    loss = combined_loss(out, y, ~mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_walk_forward_and_roundtrip(monthly, tmp_path):
    cfg = MLConfig(epochs=3, seeds=1, hidden=8, heads=2, min_train_months=48, retrain_every_months=24)
    wf = walk_forward(monthly, cfg=cfg, cash_col="CASH", verbose=False)
    assert {"ic_mean", "ic_tstat", "passed", "folds"} <= set(wf.summary) and wf.summary["folds"] >= 2
    first_pred = wf.predictions.index.min()
    assert first_pred >= monthly.index[48]  # 최소 학습기간 이후부터만 표본외 예측
    path = train_final(monthly, tmp_path / "m.pt", cfg=cfg, cash_col="CASH", validation=wf.summary)
    scores, meta = load_and_predict(path, monthly)
    assert list(scores.index) == [c for c in monthly.columns if c != "CASH"]
    assert np.isfinite(scores.values).all() and meta["folds"] == wf.summary["folds"]


def test_to_tensor_masks_missing_assets(monthly):
    px = monthly.copy()
    px.iloc[:80, 0] = np.nan  # 늦게 상장된 자산
    f = make_features(px, MONTHLY)
    X, _, M = to_tensor(f, None, list(px.columns), 6, px.index[60:62])
    assert M[:, 0].all() and not M[:, 1].any() and np.isfinite(X).all()
