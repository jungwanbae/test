"""미너비니 전략 검증 + 외부 피드백 대응 + 최종 MSAP-M 리포트 → reports/minervini_report.md

선행: scripts/robustness.py, scripts/msap_upgrade.py 와 연구 그리드(reports/*.csv) 실행
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.equity.backtest import PortfolioRules, StockCosts, make_targets, simulate
from autotrader.equity.combine import daily_etf_returns, daily_mix, etf_daily_from_weights, monthly_mix
from autotrader.equity.evaluate import monthly_signal_dates, stats
from autotrader.equity.panel import load_panel
from autotrader.equity.sepa import SEPAParams, compute_indicators, simulate_sepa, trade_stats
from autotrader.strategy.risk_parity import RPParams, TrendRiskParity

warnings.filterwarnings("ignore")
END = "2026-09-30"
OUT = Path("reports")


def pct(x):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def table(df: pd.DataFrame, pct_cols: set[str], bold: str | None = None) -> list[str]:
    lines = ["| 구분 | " + " | ".join(df.columns) + " |", "|---|" + "---:|" * len(df.columns)]
    for k, r in df.iterrows():
        cells = [pct(float(r[c])) if c in pct_cols else (f"{float(r[c]):.2f}" if isinstance(r[c], (float, np.floating))
                                                           else str(r[c])) for c in df.columns]
        name = f"**{k}**" if bold and k == bold else k
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = load_panel("data/krx")
    # ------------------------------------------------ SEPA-K 대표 설정(설계 구간 최선)
    prm = SEPAParams(fail_exit_days=3, max_rs=95, max_above_ma50=0.15, market_filter="ma50_breadth", partial_at=None)
    ind = compute_indicators(p, prm)
    sepa = simulate_sepa(ind, prm, capital=1e8)
    free = StockCosts(commission=0, slip_base=0, sell_tax_override=0, delist_haircut=0)
    sepa0 = simulate_sepa(ind, SEPAParams(**{**prm.__dict__, "impact_k": 0.0}), free, capital=1e8)
    ts = trade_stats(sepa.trades)
    # ------------------------------------------------ MSAP vs MSAP-M (일별)
    ds = pd.read_parquet("data/krx/dataset_cap.parquet")
    ds["lgbm"] = pd.read_parquet("data/krx/scores_lgbm.parquet")["score"].values
    sig = monthly_signal_dates(ds)
    tg = make_targets(ds[ds.date.isin(sig[sig >= "2000-12-01"])], "lgbm", PortfolioRules(top_n=15, buffer=2.5))
    st_plain = simulate(p, tg, StockCosts(), start="2001-01-02").equity
    st_m = simulate(p, tg, StockCosts(), start="2001-01-02", stop_loss=0.15, stop_mode="close").equity
    px = pd.read_csv("data/research/krw_monthly.csv", index_col=0, parse_dates=True)
    risk = ["KOSPI200", "SPX_KRW", "GOLD_KRW", "UST10_KRW"]
    rp = run_backtest(px, TrendRiskParity(RPParams(risk, "CASH_KRW", target_vol=0.10, trend_months=None)),
                      BacktestConfig(start="1997-01-01", execution_lag=1, slippage_bps=10))
    etf_d = etf_daily_from_weights(daily_etf_returns(p, px), rp.weights)
    curves_d = {"MSAP-M(권장: 미너비니 손절)": daily_mix(st_m, etf_d, 0.7, end=END),
                "MSAP(손절 없음)": daily_mix(st_plain, etf_d, 0.7, end=END),
                "SEPA-K(미너비니 정량화, 단독)": sepa.equity.loc["2001-02":END]}
    kospi = (1 + p.market_return("KOSPI")).cumprod().loc["2001-02":END]
    curves_d["KOSPI"] = kospi
    mm = {"MSAP-M": monthly_mix(st_m, rp.equity, 0.7, END), "MSAP": monthly_mix(st_plain, rp.equity, 0.7, END)}

    def rowstats(e_daily, e_month=None):
        em = e_month if e_month is not None else e_daily.resample("ME").last()
        a, b, c, r10 = stats(em, "2001", "2012"), stats(em, "2013", END), stats(em, "2001", END), stats(em, "2016-10", END)
        return {"전체 CAGR": c["CAGR"], "Sharpe": c["Sharpe"], "설계(01~12)": a["CAGR"], "검증(13~)": b["CAGR"],
                "최근10년": r10["CAGR"], "최근10년 Sharpe": r10["Sharpe"], "일별 MDD": stats(e_daily)["MDD"],
                "최악의 해": c["WorstY"]}
    final = pd.DataFrame({
        "MSAP-M(권장: 미너비니 손절)": rowstats(curves_d["MSAP-M(권장: 미너비니 손절)"], mm["MSAP-M"]),
        "MSAP(손절 없음, 수익 극대화형)": rowstats(curves_d["MSAP(손절 없음)"], mm["MSAP"]),
        "SEPA-K(미너비니 정량화, 단독)": rowstats(curves_d["SEPA-K(미너비니 정량화, 단독)"]),
        "KOSPI(배당 제외)": rowstats(kospi),
    }).T
    pc = {"전체 CAGR", "설계(01~12)", "검증(13~)", "최근10년", "일별 MDD", "최악의 해"}

    # ------------------------------------------------ 차트
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"):
        if fam in have:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for i, (k, e) in enumerate(curves_d.items()):
        e = e / e.iloc[0]
        a1.plot(e.index, e.values, label=k, lw=2.3 if i == 0 else 1.1)
        a2.plot(e.index, (e / e.cummax() - 1).values * 100, lw=1.6 if i == 0 else 0.9)
    a1.set_yscale("log"); a1.legend(frameon=False); a1.grid(alpha=.25); a1.set_ylabel("누적(로그, 일별)")
    a2.set_ylabel("일별 낙폭 %"); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "msap_m_equity.png", dpi=130); plt.close(fig)

    # ------------------------------------------------ 연도별
    yr = {k: v.resample("YE").last().pct_change() for k, v in mm.items()}
    yr["KOSPI"] = kospi.resample("YE").last().pct_change()

    v1 = pd.read_csv(OUT / "sepa_variants.csv", index_col=0)
    v2 = pd.read_csv(OUT / "sepa_variants2.csv", index_col=0)
    lay = pd.read_csv(OUT / "minervini_layers.csv", index_col=0)
    cap = pd.read_csv(OUT / "capacity.csv", index_col=0)
    L = ["# 마크 미너비니 전략 검증 + 외부 피드백 대응 → MSAP-M", "",
         "## 0. 결론", ""] + table(final, pc, bold="MSAP-M(권장: 미너비니 손절)") + [
         "", "- 기간 2001-02 ~ 2026-09, 실비용(NH 수수료·연도별 거래세·유동성 슬리피지)·다음날 시가 체결·생존편향 없음. "
         "**MDD 는 일별 기준**(월말 기준의 함정 제거).",
         "- **MSAP-M** = MSAP 에 미너비니의 손실 제한 규율(매입가 대비 −15% 종가 손절, 그달 재매수 금지)과 유동성 한도"
         "(종목당 20일 거래대금의 10%)를 더한 것. 최근 10년 수익·Sharpe 개선, 일별 MDD −41% → −32%.",
         "- 선택 기준 공개: 설계 구간(2001~2012) Sharpe 만 보면 손절 없는 MSAP 가 높다(2000년대엔 손절 후 반등이 많았음). "
         "손절 채택은 '최대 수익'이 아니라 **낙폭 축소(위험 선호)와 미너비니 규율 반영**을 위한 선택이다.",
         "", "![MSAP-M](msap_m_equity.png)", "",
         "## 1. 미너비니 방법론 요약(SEPA)", "",
         "| 요소 | 미너비니 원칙 | SEPA-K 구현 |", "|---|---|---|",
         "| 트렌드 템플릿 | 종가>150·200일선, 150>200일선, 200일선 1개월+ 상승, 50일선>150·200일선, 종가>50일선, "
         "52주 저가+30% 이상, 52주 고가−25% 이내, RS≥70 | 8개 조건 그대로, RS=IBD식(3개월 40%+6·9·12개월 20%) 전 종목 백분위 |",
         "| VCP | 2~6번 수축하며 변동폭 축소, 거래량 감소, 피벗 돌파 | 10일 변동폭≤12%·직전 대비 70% 이하, 10일 거래대금≤50일×0.9, 베이스 깊이≤35% |",
         "| 진입 | 피벗 돌파 시 매수, 피벗 +5% 이상 추격 금지 | 다음날 피벗 도달 시 체결(스톱 매수), +5% 초과 갭은 생략 |",
         "| 손절 | 최대 7~8%(평균 손실 더 작게), 마지막 수축 저점 아래 | 최대 8%·10일 저점 아래, 장중 저가 터치 시 체결(갭하락은 시가) |",
         "| 이익 보호 | 수익이 위험의 2~3배면 본전 스톱, 큰 수익은 50일선 이탈 시 매도 | +2R 본전 스톱, +20% 이후 50일선 이탈 매도, 25일 내 +5% 미달 시 시간 손절 |",
         "| 포지션 크기 | 1회 위험 1~2.5%, 소수 집중 | 위험 1.25%/손절폭, 최대 20%·10종목 |",
         "| 진행형 노출 | 잘 될 때 늘리고 낙폭 때 줄임 | 낙폭 10%↑ 위험 절반, 20%↑ 1/4 |",
         "| 시장 방향 | 강세장에서만 공격 | 시장지수 50·200일선 + 트렌드템플릿 통과 비율 |",
         "| 펀더멘털 | EPS·매출 가속, 어닝 서프라이즈 | **데이터 없음(미구현)** — 실제 SEPA 의 핵심 축 하나가 빠짐 |",
         "", "공개 실적: 1997·2021 미국투자대회(US Investing Championship) 우승(각 +155%, +334.8%). 단, 대회는 1년 단위·"
         "레버리지 허용이며, 같은 방식을 따라 한 투자자 다수의 성과는 검증된 통계가 없다(생존편향 주의).", "",
         "## 2. SEPA-K 백테스트(한국 전종목, 1억원)", "",
         f"- 대표 설정(설계 구간 최선): 전체 CAGR {pct(stats(sepa.equity, '2001', END)['CAGR'])}, "
         f"설계 {pct(stats(sepa.equity, '2001', '2012')['CAGR'])} / 검증 {pct(stats(sepa.equity, '2013', END)['CAGR'])}, "
         f"일별 MDD {pct(stats(sepa.equity)['MDD'])}, 평균 노출 {sepa.exposure.mean():.0%}",
         f"- 거래 {ts['trades']}회(연 {ts['trades'] / 25.7:.0f}회), 승률 {ts['win_rate']:.0%}, 평균 이익 {pct(ts['avg_win'])}, "
         f"평균 손실 {pct(ts['avg_loss'])}, 손익비 {ts['payoff']:.1f}:1 → **미너비니식 '작은 손실·큰 이익' 구조는 재현**",
         f"- 비용 전 수익: 설계 {pct(stats(sepa0.equity, '2001', '2012')['CAGR'])} / 검증 {pct(stats(sepa0.equity, '2013', END)['CAGR'])} "
         "→ **원천 엣지는 있으나 한국의 매도세(0.15~0.30%)·손절 슬리피지가 대부분을 상쇄**", "",
         "### 규칙 변형(설계 구간에서 진단 후 추가)", ""] + table(
        v1[["IS_CAGR", "OOS_CAGR", "ALL_CAGR", "ALL_MDD(일별)", "win", "avgW", "avgL"]].rename(columns={
            "IS_CAGR": "설계", "OOS_CAGR": "검증", "ALL_CAGR": "전체", "ALL_MDD(일별)": "일별 MDD", "win": "승률",
            "avgW": "평균이익", "avgL": "평균손실"}), {"설계", "검증", "전체", "일별 MDD", "승률", "평균이익", "평균손실"}) + [
        "", "### 더 엄격한 '고품질' 셋업(RS≥80·VCP 8%·대형 유동주·장기보유)", ""] + table(
        v2[["IS_CAGR", "OOS_CAGR", "ALL_CAGR", "ALL_MDD", "trades/y", "win"]].rename(columns={
            "IS_CAGR": "설계", "OOS_CAGR": "검증", "ALL_CAGR": "전체", "ALL_MDD": "일별 MDD", "trades/y": "연 거래", "win": "승률"}),
        {"설계", "검증", "전체", "일별 MDD", "승률"}) + [
        "", "**진단(설계 구간 거래 532건)**: 돌파 당일 종가가 진입가 +3% 이상이면 승률 48.5%·건당 +10.9%, 진입가 아래면 승률 13~23%. "
        "RS 95 이상·50일선 +20% 이상 과열 종목은 손실 — 한국의 '복권형 과열 후 되돌림' 이상현상과 일치.", "",
        "## 3. 미너비니 요소를 ML 종목 슬리브에 적용", ""] + table(
        lay.loc[[i for i in lay.index if "be=None" in i]][["IS_CAGR", "OOS_CAGR", "ALL_CAGR", "OOS_MDD", "ALL_MDD(일별)"]].rename(
            columns={"IS_CAGR": "설계", "OOS_CAGR": "검증", "ALL_CAGR": "전체", "OOS_MDD": "검증 일별 MDD",
                     "ALL_MDD(일별)": "전체 일별 MDD"}), {"설계", "검증", "전체", "검증 일별 MDD", "전체 일별 MDD"}) + [
        "", "- **트렌드 템플릿(2단계 상승추세)만 사라는 규칙은 한국에서 역효과**: 수익률이 절반 이하로 감소. "
        "한국의 초과수익은 '과열되지 않은' 종목에서 나온다(팩터 분석의 MAX·변동성·단기과열 음(−)효과와 같은 결론).",
        "- **손실 제한 규율은 2013년 이후 유효**: 검증 구간 수익 유지·일별 MDD 크게 축소 → MSAP-M 에 채택(−15%, 종가 기준).", "",
        "## 4. 외부 피드백 대응", "",
        "| 지적 | 검증 결과 | 반영 |", "|---|---|---|",
        "| ① 2013년 이후 둔화 | **사실.** 5년 블록 26%→24%→26%→**12%**(2016~20)→**14%**(2021~26). 최근 10년 MSAP 13.9% ≈ KOSPI 13.0%. "
        "단, 5년 롤링 손실 구간 0% | 리포트에 구간별·롤링 성과 상시 표기, 롤링 윈도 재학습 옵션 추가, MSAP-M 은 최근 10년 14.8% |",
        "| ② 월말 MDD 함정 | **사실.** 일별 MDD −41.1%(월말 −28.5%, 주말 −37.1%) | 모든 표를 일별 MDD 로 교체. MSAP-M 일별 −32.1% |",
        "| ③ 환율·금리 국면 의존 | **부분적 사실.** 원화 강세 월 ETF 슬리브 −0.4%/월, 주식·채권 동반하락 월 −1.4%/월. "
        "다만 그 달 주식 슬리브가 +4.3%·+2.0% 로 상쇄(36개월 상관 중앙값 0.22) | ETF 추세필터는 개선 없음 → 미채택, 국면표 공개 |",
        "| ④ 유동성·자동화 | **사실.** 운용금액별 성과 하락 확인(아래) | 종목당 20일 거래대금 10% 한도, 부분체결·거부·네트워크 오류 테스트 추가 |",
        "", "### 운용 규모별 성과(종목 슬리브 단독, 주문 크기 기반 √충격 + 하루 거래대금 10% 체결 한도)", ""] + table(
        cap[["ALL_CAGR", "OOS_CAGR", "MDD(일별)"]].rename(columns={"ALL_CAGR": "전체", "OOS_CAGR": "검증", "MDD(일별)": "일별 MDD"}),
        {"전체", "검증", "일별 MDD"}) + [
        "", "→ **개인 규모(~5억원)까지는 성과 유지, 30억원 이상은 크게 감소.** 거래대금 10억원 이상 종목만 쓰면 오히려 악화(알파가 중간 유동성 종목에 있음).",
        "", "### 자동주문 예외 처리(테스트로 검증)", "- 부분체결: 라운드마다 실제 잔고로 재계산 → 과매수 없음",
        "- 주문 거부(VI·가격제한 등): 해당 종목만 미체결, 상태 'partial', 다음 실행 때 재시도",
        "- 주문 중 네트워크 오류: 즉시 중단(재전송 금지 → 이중주문 방지), 다음 실행이 잔고 기준으로 남은 것만 주문",
        "- 미체결 잔존 시 취소 확인 전에는 다음 라운드 진행 안 함, 손절 종목은 그달 재매수 금지", "",
        "상세: [robustness.md](robustness.md) · [msap_upgrade.md](msap_upgrade.md)", "",
        "## 5. 연도별 수익률", "", "| 연도 | MSAP-M | MSAP | KOSPI |", "|---|---:|---:|---:|"]
    for y in range(2002, 2027):
        cells = []
        for k in ("MSAP-M", "MSAP", "KOSPI"):
            v = yr[k][yr[k].index.year == y]
            cells.append(pct(v.iloc[0]) if len(v) else "-")
        L.append(f"| {y}{' (9월까지)' if y == 2026 else ''} | " + " | ".join(cells) + " |")
    (OUT / "minervini_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
