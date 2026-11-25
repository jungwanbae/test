"""딥러닝 오버레이 검증: 워크포워드 IC + AMR vs AMR+ML 백테스트(장기 실데이터).

  python scripts/ml_research.py --out reports/ml_research.md

예측은 전부 표본외(그 시점 이전 데이터로만 학습한 모델)이며, 타깃은 1바 지연(t+1→t+2)으로
실제 체결 시점과 맞춘다. 따라서 월평균 데이터의 인접 자기상관 같은 '거래 불가능한 신호'는 걸러진다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.backtest.report import comparison_markdown
from autotrader.data.research import build_us_monthly
from autotrader.ml.gate import portfolio_uplift
from autotrader.ml.train import MLConfig, walk_forward
from autotrader.strategy.amr import AMRParams, AMRStrategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/ml_research.md")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    px, _ = build_us_monthly("data/research")
    px = px.loc["1972-01":]
    t0 = time.time()
    wf = walk_forward(px, cfg=MLConfig(epochs=args.epochs, seeds=args.seeds), verbose=False)
    elapsed = time.time() - t0
    cfg = BacktestConfig(start="1973-01-01", execution_lag=1)
    risk = ["SPX", "GOLD", "UST10"]
    results = {"AMR": run_backtest(px, AMRStrategy(AMRParams(risk_assets=risk, top_k=2, max_weight=0.5)), cfg)}
    for blend in (0.3, 0.5):
        p = AMRParams(risk_assets=risk, top_k=2, max_weight=0.5, ml_blend=blend)
        results[f"AMR+ML({blend})"] = run_backtest(px, AMRStrategy(p), cfg, ml_scores=wf.predictions)
    # 공정 비교: ML 예측이 존재하는 구간만
    start = wf.predictions.index.min()
    sub = {}
    for k, r in results.items():
        e = r.equity.loc[start:]
        from autotrader.backtest.metrics import performance
        r2 = type(r)(e / e.iloc[0], r.weights, r.trades, r.turnover.loc[start:], r.costs, {}, r.bars)
        r2.stats = performance(r2.equity, 12, turnover=r2.turnover)
        sub[k] = r2
    s = wf.summary
    gate2 = portfolio_uplift(px, lambda b: AMRStrategy(AMRParams(risk_assets=risk, top_k=2, max_weight=0.5,
                                                                   ml_blend=b)), wf.predictions, 0.3, cfg)
    final = bool(wf.passed and gate2["passed"])
    lines = ["# 딥러닝 오버레이 검증 (미국 장기 실데이터, 월간, 표본외)", "",
             f"- 워크포워드: {s['folds']}개 폴드(12개월마다 재학습), 표본외 관측 {s['n_obs']}개월, 학습시간 {elapsed:.0f}초",
             f"- 순위 IC 평균 **{s['ic_mean']:.3f}**, t값 **{s['ic_tstat']:.2f}**, 적중률 {s['hit_rate']:.0%} "
             f"→ 게이트①(IC≥{s['min_ic']}, t≥{s['min_ic_tstat']}) **{'통과' if s['passed'] else '미통과'}**",
             f"- 게이트②(포트폴리오 Sharpe 개선): {gate2['sharpe_base']} → {gate2['sharpe_ml']} "
             f"(Δ {gate2['sharpe_uplift']:+.3f}) **{'통과' if gate2['passed'] else '미통과'}**",
             f"- 최종 판정: **{'실거래 반영' if final else 'ML 미반영(순수 AMR 로 운용)'}**",
             f"- 타깃: 1바 지연 변동성조정 초과수익(t+1→t+2), 비교 구간 {start.date()} 이후", "",
             comparison_markdown(sub), "",
             "```json", json.dumps({**s, "portfolio_gate": gate2, "final": final}, ensure_ascii=False, indent=1), "```"]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
