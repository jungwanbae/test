"""시대 적응형 ML/딥러닝 워크포워드 예측 생성 → data/krx/adaptive_preds/*.parquet

  python scripts/adaptive_ml.py [lgbm|mlp|cnn ...]
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from autotrader.research import adaptive as A

warnings.filterwarnings("ignore")
OUT = Path("data/krx/adaptive_preds")
PLAN = {
    "lgbm": [("expanding", 12), ("rolling10", 12), ("decay5", 12), ("decay2", 12), ("similar", 12), ("similar", 3),
             ("bag", 12)],
    "mlp": [("expanding", 24), ("decay5", 24), ("similar", 24)],
    "cnn": [("expanding", 24), ("decay5", 24), ("similar", 24)],
}


def panel():
    p = Path("data/krx/adaptive_panel.pkl")
    if p.exists():
        return pickle.load(open(p, "rb"))
    R = pd.read_parquet("data/krx/research_returns.parquet")
    P = A.build_panel(A.load_assets((1 + R["K200"]).cumprod()))
    pickle.dump(P, open(p, "wb"))
    return P


def main(models):
    OUT.mkdir(parents=True, exist_ok=True)
    P = panel()
    for m in models:
        for scheme, step in PLAN[m]:
            tag = f"{m}_{scheme}" + ("_q" if step == 3 else "")
            f = OUT / f"{tag}.parquet"
            if f.exists():
                print("skip", tag, flush=True)
                continue
            t = time.time()
            pr = A.walk_forward(P, m, scheme, first_test_year=1985, step_months=step, log=lambda s: None)
            pr.to_parquet(f)
            print(f"{tag}: {len(pr):,} 예측 {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["lgbm", "cnn", "mlp"])
