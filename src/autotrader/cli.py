"""명령행 인터페이스.

  autotrader fees                         증권사 API 수수료 비교
  autotrader backtest --data research_us  장기 실데이터(1972~) 백테스트 + 리포트
  autotrader backtest --data broker       운용 ETF 실제 일봉(증권사 API)으로 백테스트
  autotrader account                      잔고 조회
  autotrader signal                       오늘 기준 목표비중(주문 없음)
  autotrader rebalance --dry-run          주문계획만 출력
  autotrader rebalance                    1회 리밸런싱 실행
  autotrader daemon                       매 거래일 정해진 시각에 자동 실행
  autotrader ml-validate --data ...       딥러닝 오버레이 워크포워드 검증(+통과 시 모델 저장)
  autotrader status                       최근 실행 이력
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from autotrader.config import AppConfig, load_config, load_dotenv

log = logging.getLogger("autotrader")


def _setup_logging(cfg: AppConfig | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg is not None:
        d = Path(cfg.storage.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(d / "autotrader.log", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _cfg(args) -> AppConfig:
    load_dotenv(args.env_file)
    path = args.config or ("config/default.yaml" if Path("config/default.yaml").exists() else None)
    return load_config(path)


# ---------------------------------------------------------------------- 명령
def cmd_fees(args) -> int:
    from autotrader.fees import compare_table
    rows = compare_table(args.turnover, args.capital)
    print(f"연 회전율 {args.turnover:.1f}배 · 운용자금 {args.capital:,.0f}원 기준 연간 매매비용")
    for r in rows:
        print(f"  {r['name']:<34} {r['rate_pct']:>9.5f}%  연 {r['annual_cost_krw']:>10,}원  {r['note']}")
    print("※ 미래에셋증권: 2026-11 기준 개인용 Open API 미출시(준비 중) → 자동매매 대상 아님")
    return 0


def _load_prices(args, cfg: AppConfig):
    """반환: (가격, 위험자산 목록, 현금자산, 메타)"""
    src = args.data
    if src == "research_us":
        from autotrader.data.research import build_us_monthly
        px, meta = build_us_monthly(args.research_dir)
        px = px.loc["1972-01":]  # 금 자유변동(1971.8) 이후
        return px, ["SPX", "GOLD", "UST10"], None, meta
    if src == "synthetic":
        from autotrader.data.synthetic import generate
        px, _ = generate(seed=args.seed)
        return px, [c for c in px.columns if c != "CASH"], "CASH", {"note": "합성 데이터(로직 점검용)"}
    if src.startswith("csv:"):
        from autotrader.data.loader import load_csv_dir
        px = load_csv_dir(src[4:])
        risk = [a.symbol for a in cfg.universe.risk_assets if a.symbol in px.columns] or list(px.columns)
        cash = cfg.universe.cash_asset.symbol if cfg.universe.cash_asset and cfg.universe.cash_asset.symbol in px.columns else None
        return px, [r for r in risk if r != cash], cash, {}
    if src == "broker":
        from autotrader.brokers import build_broker
        from autotrader.data.loader import BrokerHistoryCache
        broker = build_broker(cfg)
        cache = BrokerHistoryCache(broker, cfg.storage.cache_dir)
        start = date.fromisoformat(args.start) if args.start else date(2010, 1, 1)
        px = cache.closes(cfg.universe.all_symbols(), start)
        cash = cfg.universe.cash_asset.symbol if cfg.universe.cash_asset else None
        return px, [a.symbol for a in cfg.universe.risk_assets], cash, {}
    raise SystemExit(f"알 수 없는 데이터 소스: {src}")


def cmd_backtest(args) -> int:
    from autotrader.backtest.engine import BacktestConfig, run_backtest
    from autotrader.backtest.report import comparison_markdown, crisis_table, save_charts, yearly_markdown
    from autotrader.fees import get_fee_schedule
    from autotrader.strategy.amr import AMRParams, AMRStrategy
    from autotrader.strategy.benchmarks import FixedWeights, equal_weight

    cfg = _cfg(args)
    _setup_logging(None, args.verbose)
    px, risk, cash, meta = _load_prices(args, cfg)
    s = cfg.strategy
    top_k = args.top_k or (s.top_k if len(risk) > 3 else 2)
    params = AMRParams(risk_assets=risk, cash_asset=cash, lookback_months=s.lookback_months,
                       lookback_weights=s.lookback_weights, top_k=top_k, vol_window_months=s.vol_window_months,
                       weighting=s.weighting, max_weight=max(s.max_weight, 1.0 / top_k),
                       target_vol=args.target_vol if args.target_vol is not None else s.target_vol,
                       trend_sma_months=s.trend_sma_months, drawdown_brake=s.drawdown_brake)
    fees = get_fee_schedule(cfg.costs.preset, commission_rate=cfg.costs.commission_rate)
    monthly_avg = args.data == "research_us"
    bt = BacktestConfig(initial_capital=args.capital, fees=fees, slippage_bps=cfg.costs.slippage_bps,
                        execution_lag=1 if monthly_avg else None, start=args.start,
                        no_trade_band=cfg.rebalance.no_trade_band)
    ml_scores = None
    if args.ml:
        from autotrader.ml.train import MLConfig, walk_forward
        wf = walk_forward(px[risk + ([cash] if cash else [])], cfg=MLConfig(), cash_col=cash)
        print(f"ML 워크포워드: IC={wf.summary['ic_mean']:.4f} t={wf.summary['ic_tstat']:.2f} "
              f"→ {'통과' if wf.passed else '미통과(혼합 0)'}")
        if wf.passed:
            ml_scores = wf.predictions
            params.ml_blend = cfg.ml.blend
    results = {"AMR(전략)": run_backtest(px, AMRStrategy(params), bt, ml_scores=ml_scores)}
    if monthly_avg:
        bench = {"S&P500 보유": FixedWeights({"SPX": 1.0}, "spx"),
                 "60/40(주식/채권)": FixedWeights({"SPX": 0.6, "UST10": 0.4}, "6040"),
                 "1/3 균등(주식·금·채권)": equal_weight(risk, "ew")}
    else:
        eq = [a for a in risk if a in px.columns][:1]
        bonds = [a.symbol for a in cfg.universe.risk_assets if "채" in a.asset_class and a.symbol in px.columns][:1]
        bench = {"균등비중(위험자산)": equal_weight(risk, "ew")}
        if eq:
            bench[f"{eq[0]} 보유"] = FixedWeights({eq[0]: 1.0}, "first")
        if eq and bonds:
            bench["60/40"] = FixedWeights({eq[0]: 0.6, bonds[0]: 0.4}, "6040")
    for k, strat in bench.items():
        results[k] = run_backtest(px, strat, bt)
    # 벤치마크는 전략 첫 운용일부터 비교
    start = results["AMR(전략)"].equity.index[0]
    for k in list(results)[1:]:
        e = results[k].equity.loc[start:]
        from autotrader.backtest.metrics import performance
        results[k].equity = e / e.iloc[0] * args.capital
        results[k].stats = performance(results[k].equity, results[k].bars.periods_per_year,
                                       turnover=results[k].turnover.loc[start:])
    md = comparison_markdown(results)
    print(md)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    charts = save_charts(results, out, main="AMR(전략)", prefix=f"{args.data.split(':')[0]}_")
    crises = {"오일쇼크·스태그플레이션": ("1973-01", "1974-12"), "블랙먼데이": ("1987-08", "1987-12"),
              "닷컴버블 붕괴": ("2000-09", "2002-09"), "글로벌 금융위기": ("2007-10", "2009-02"),
              "코로나 쇼크": ("2020-01", "2020-03"), "2022 금리 급등": ("2022-01", "2022-10")}
    report = [f"# 백테스트 리포트 — {args.data}", "", f"- 메타: `{json.dumps(meta, ensure_ascii=False, default=str)}`",
              f"- 비용: {fees.name} {fees.total_rate * 100:.4f}% + 슬리피지 {cfg.costs.slippage_bps}bp, "
              f"체결지연 {bt.execution_lag if bt.execution_lag is not None else 'auto'}바", "", "## 성과 요약", "", md, ""]
    if monthly_avg:
        report += ["## 위기 구간 수익률", "", crisis_table(results, crises), ""]
    report += ["## 연도별 수익률", "", yearly_markdown(results), ""]
    report += [f"![chart]({c.name})" for c in charts]
    report_path = out / f"backtest_{args.data.split(':')[0]}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n리포트: {report_path}")
    return 0


def _engine(args, cfg: AppConfig):
    from autotrader.brokers import build_broker
    from autotrader.execution.engine import TradingEngine
    return TradingEngine(cfg, build_broker(cfg))


def cmd_account(args) -> int:
    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    from autotrader.brokers import build_broker
    b = build_broker(cfg)
    a = b.get_account()
    print(f"[{b.name}/{'live' if b.is_live else 'mock'}] 주문가능현금 {a.cash:,.0f}원 · 총자산 {a.total_equity or 0:,.0f}원")
    for p in a.positions.values():
        print(f"  {p.symbol} {p.name:<20} {p.qty:>8,}주  평단 {p.avg_price:>10,.0f}  현재 {p.last_price:>10,.0f}  "
              f"평가 {p.market_value:>14,.0f}")
    return 0


def cmd_signal(args) -> int:
    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    eng = _engine(args, cfg)
    w, diag = eng.compute_targets(date.today())
    print(f"신호 기준일: {diag.get('signal_date')}  (후보 {diag.get('candidates')}, 축소배율 {diag.get('scale')})")
    for s, v in w.items():
        print(f"  {cfg.universe.name_of(s):<26} {v * 100:6.2f}%   모멘텀 {diag.get('momentum', {}).get(s, float('nan')):+.3f}")
    return 0


def cmd_rebalance(args) -> int:
    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    rep = _engine(args, cfg).run_once(force=args.force, dry_run=args.dry_run)
    print(f"[{rep.status}] {rep.message}")
    if rep.plan_text:
        print(rep.plan_text)
    return 0 if rep.status in ("done", "dry_run", "skipped") else 1


def cmd_daemon(args) -> int:
    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    _engine(args, cfg).run_daemon(dry_run=args.dry_run)
    return 0


def cmd_status(args) -> int:
    cfg = _cfg(args)
    from autotrader.execution.state import StateStore
    st = StateStore(cfg.storage.db_path)
    for r in st.recent_runs(args.n):
        print(" | ".join(str(x) for x in r))
    return 0


def cmd_ml_validate(args) -> int:
    from autotrader.ml.train import MLConfig, train_final, walk_forward
    cfg = _cfg(args)
    _setup_logging(None, args.verbose)
    px, risk, cash, _ = _load_prices(args, cfg)
    cols = risk + ([cash] if cash else [])
    mcfg = MLConfig(epochs=args.epochs, seeds=args.seeds)
    wf = walk_forward(px[cols].dropna(how="all"), cfg=mcfg, cash_col=cash)
    print(json.dumps(wf.summary, ensure_ascii=False, indent=1))
    if wf.passed or args.save_anyway:
        p = train_final(px[cols], args.out, cfg=mcfg, cash_col=cash, validation=wf.summary)
        print(f"모델 저장: {p} (검증 {'통과' if wf.passed else '미통과 — 실거래에는 반영되지 않음'})")
    else:
        print("검증 미통과 → 모델을 저장하지 않습니다(전략은 ML 없이 동작).")
    return 0


# ---------------------------------------------------------------------- 파서
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autotrader", description="멀티에셋 알고리즘 자동매매")
    p.add_argument("--config", help="설정 YAML (기본 config/default.yaml)")
    p.add_argument("--env-file", default=".env")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fees", help="증권사 API 수수료 비교")
    f.add_argument("--turnover", type=float, default=3.0)
    f.add_argument("--capital", type=float, default=50_000_000)
    f.set_defaults(func=cmd_fees)

    def data_args(sp):
        sp.add_argument("--data", default="research_us", help="research_us | synthetic | broker | csv:디렉터리")
        sp.add_argument("--research-dir", default="data/research")
        sp.add_argument("--start", default=None)
        sp.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("backtest", help="백테스트")
    data_args(b)
    b.add_argument("--capital", type=float, default=100_000_000)
    b.add_argument("--top-k", type=int, default=None)
    b.add_argument("--target-vol", type=float, default=None)
    b.add_argument("--ml", action="store_true", help="ML 오버레이 워크포워드 결과 반영")
    b.add_argument("--out", default="reports")
    b.set_defaults(func=cmd_backtest)

    for name, fn, hlp in (("account", cmd_account, "잔고"), ("signal", cmd_signal, "목표비중")):
        sp = sub.add_parser(name, help=hlp)
        sp.set_defaults(func=fn)

    r = sub.add_parser("rebalance", help="1회 리밸런싱")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--force", action="store_true", help="일정과 무관하게 실행")
    r.set_defaults(func=cmd_rebalance)

    d = sub.add_parser("daemon", help="스케줄러")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(func=cmd_daemon)

    s = sub.add_parser("status", help="최근 실행 이력")
    s.add_argument("-n", type=int, default=10)
    s.set_defaults(func=cmd_status)

    m = sub.add_parser("ml-validate", help="딥러닝 오버레이 워크포워드 검증")
    data_args(m)
    m.add_argument("--epochs", type=int, default=60)
    m.add_argument("--seeds", type=int, default=3)
    m.add_argument("--out", default="models/ml_overlay.pt")
    m.add_argument("--save-anyway", action="store_true")
    m.set_defaults(func=cmd_ml_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
