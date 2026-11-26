"""종목 순위 모델 워크포워드 점수 생성 → data/krx/scores_<model>.parquet

  python scripts/stock_scores.py linear lgbm mlp xsformer
"""
import logging
import os
import sys
import time

import pandas as pd

from autotrader.equity.models import WFConfig, walk_forward_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
ds = pd.read_parquet("data/krx/dataset_cap.parquet")
for arg in sys.argv[1:]:
    arg, _, scheme = arg.partition("~")  # 예: lgbm~decay5 → 반감기 5년 최근 가중, lgbm~similar → 유사 국면 가중
    model, _, win = arg.partition("@")   # 예: lgbm@8 → 최근 8년 롤링 윈도
    t = time.time()
    stride = int(os.environ.get("TRAIN_STRIDE", 2 if model == "xsformer" else 1))  # 학습 날짜 솎기(속도)
    cfg = WFConfig(retrain_years=2) if model == "xsformer" else WFConfig()  # 트랜스포머는 계산량이 커 2년마다 재학습
    if win:
        cfg.train_window_years = float(win)
    if scheme:
        cfg.weight_scheme = scheme
    if "WF_SEED" in os.environ:          # 같은 설정을 다른 난수로 다시 학습 → 모델 잡음 측정
        cfg.seed = int(os.environ["WF_SEED"])
    s = walk_forward_scores(ds, model, cfg, train_stride=stride)
    tag = model + (f"_w{win}" if win else "") + (f"_{scheme}" if scheme else "") + \
        (f"_s{stride}" if "TRAIN_STRIDE" in os.environ else "") + \
        (f"_seed{cfg.seed}" if "WF_SEED" in os.environ else "")
    pd.DataFrame({"date": ds["date"], "code": ds["code"], "score": s}).to_parquet(f"data/krx/scores_{tag}.parquet")
    print(f"{tag} done in {time.time() - t:.0f}s, scored {s.notna().sum()}", flush=True)
