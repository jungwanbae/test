"""질문 3종(조합 최적화·VT 현금·시대 적응 딥러닝) 결과 → reports/adaptive_report.md"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path("reports")


def pct(v, d=1):
    return "—" if pd.isna(v) else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "—" if pd.isna(v) else f"{v:.{d}f}"


def table(df: pd.DataFrame, fmt: dict, index_name: str = "") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join([index_name] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.get(c, pct)(row[c]) for c in cols]) + " |")
    return "\n".join(out)


def stock_note(sk: pd.DataFrame) -> str:
    a = sk[sk.index.str.startswith("종목 알파")]["CAGR"]
    t = sk[sk.index.str.startswith("TRIAD")]["CAGR"]
    base = a[a.index.str.contains("전체 과거 · 시드[789]")]
    ens = a[a.index.str.contains("앙상블")].iloc[0]
    tens = t[t.index.str.contains("앙상블")].iloc[0]
    rec = a[a.index.str.contains("최근")]
    sim = a[a.index.str.contains("유사 국면")]
    return "\n".join([
        f"* **모델 잡음이 크다**: 설정은 그대로 두고 난수(시드)만 바꿔도 종목 알파 CAGR 이 {pct(base.min())}~{pct(base.max())} 로 흔들렸다"
        "(IC 는 0.12~0.13 으로 거의 같은데 상위 15종목만 담는 구조라 작은 점수 차이가 큰 성과 차이로 번진다). "
        f"기존 보고서의 20.3% 는 이 분포의 위쪽이었다. **3개 시드 앙상블: {pct(ens)}** → TRIAD 기대치도 {pct(tens)} 로 보는 것이 정직하다.",
        f"* **최근 가중·최근 N년은 모두 전체 과거보다 낮았다**({pct(rec.min())}~{pct(rec.max())}). 2016년 이후만 보면 비슷한 범위 — "
        "최근 데이터를 더 믿는다고 최근 성과가 좋아지지도 않았다.",
        f"* **유사 국면 가중은 크게 나빴다**({pct(sim.min())}~{pct(sim.max())}). 비슷한 시장 국면의 표본만 강조하면 종목 간 패턴을 배울 "
        "표본이 줄어 잡음을 외운다.",
        "* 그래서 실전 종목 모델은 **시드 3개 앙상블(순위 평균)** 로 바꿨다(`equity/live.py train_final`, 모델 파일 3개). "
        "수익 기대치를 올리는 변경이 아니라, 한 번의 운 좋은(또는 나쁜) 학습에 기대지 않게 하는 변경이다."])


def main():
    cw = pd.read_csv(R / "combo_wf.csv", index_col=0)
    vc = pd.read_csv(R / "vt_cash_study.csv", index_col=0)
    ex = pd.read_csv(R / "vt_exposure.csv", index_col=0, parse_dates=True)
    auc = pd.read_csv(R / "adaptive_auc.csv", index_col=0)
    qq = pd.read_csv(R / "adaptive_qqq.csv", index_col=0)
    rb = pd.read_csv(R / "adaptive_assets.csv", index_col=0)
    sim = pd.read_csv(R / "adaptive_similar.csv", index_col=0)
    tri = pd.read_csv(R / "adaptive_triad.csv", index_col=0)
    sk = pd.read_csv(R / "stock_adaptive.csv", index_col=0) if (R / "stock_adaptive.csv").exists() else None

    L = ["# 후속 질문 3가지 — 조합 최적화 · 확인형 역발상의 현금 · 시대 적응형 딥러닝\n"]

    # ---------------------------------------------------------------- Q1
    L.append("## 1. 기존 3전략 + TRIAD 를 합쳐 최적화하면?\n")
    L.append("**결론: 나빠진다.** 7개 엔진(MSAP-M·VT·확인형 역발상·고금리 대비형·3차 공격형·AQUILA-X·레버리지 RP)을 "
             "매년 말 '그때까지의 데이터만으로' 다시 최적화해 다음 1년을 운용하는 워크포워드로 시험했다"
             "(10% 단위 6,538개 조합 중 목적함수 최대).\n")
    L.append(table(cw, {"Sharpe": num, "2016~ Sharpe": num}, "배분 규칙(2009~, 원화·일별)") + "\n")
    t, mix = cw.loc["TRIAD(고정 1/3)"], cw.loc["TRIAD 50% + 기존3전략 50%"]
    L.append(f"* 단순 합산(TRIAD 50 + 기존 3전략 50): 수익은 같고({pct(mix['CAGR(2009~)'])} vs {pct(t['CAGR(2009~)'])}) "
             f"Sharpe {num(t['Sharpe'])} → {num(mix['Sharpe'])}, MDD {pct(t['MDD'])} → {pct(mix['MDD'])} 로 **위험만 커진다**. "
             "기존 3전략이 TRIAD 의 미국 두 엔진과 상관 0.56~0.91 이라 분산 효과가 없고 레버리지 노출만 겹친다.")
    L.append("* 매년 재최적화: Sharpe 를 최대화하면 한국 엔진을 50%까지 싣고 수익이 낮아지고(2016~ 15.7~19.0%), "
             "최근 수익을 쫓으면 MDD 가 −32~−34% 로 커진다. 어떤 목적함수도 2016~ 에서 TRIAD(20.4%, Sharpe 1.18)를 넘지 못했다.")
    L.append("* 공정성 주의: TRIAD 구성은 전체 기간을 본 뒤 골랐으므로 TRIAD 수치에는 표본 내 요소가 있다. 그래도 "
             "'단순 고정 배분 > 매년 재최적화'는 자산배분 연구의 일관된 결과(DeMiguel 외 2009)와 같다 — 추정 오차가 이득보다 크다.\n")

    # ---------------------------------------------------------------- Q2
    E = ex["VT 노출(배)"]
    yr = E.groupby(E.index.year).mean()
    cash = ex["VT 현금비중"].groupby(ex.index.year).mean()
    L.append("## 2. '확인형 역발상' 엔진은 평소 현금에 있나? — 아니다\n")
    L.append("평소에는 **나스닥에 변동성 25% 목표로 1~3배** 투자한다(노출 = 25% ÷ 최근 21일 변동성, 최대 3배). "
             "'폭락 후 골든크로스' 부스트는 그 위에 얹는 장치일 뿐이다. 2005년 이후 월말 기준으로 보면:\n")
    L.append(f"* 나스닥 노출 중앙값 **{E.median():.2f}배**, 평균 {E.mean():.2f}배, 1배 이상(현금 0%)인 달 {pct((E >= 0.999).mean(), 0)}")
    L.append(f"* 평균 현금비중 **{pct(ex['VT 현금비중'].mean())}** — 현금이 생기는 때는 나스닥 변동성이 25%를 넘는 위기 국면뿐: "
             + ", ".join(f"{y}년 {pct(cash[y], 0)}" for y in (2008, 2009, 2020, 2022)))
    L.append("* 연도별 평균 노출(배): " + " · ".join(f"{y} {v:.1f}" for y, v in yr.items()) + "\n")
    L.append("그래도 그 현금을 다른 자산에 두면 어떨까:\n")
    L.append(table(vc[["VT CAGR", "VT Sharpe", "VT MDD", "VT 2016~", "VT 2001~ MDD", "TRIAD CAGR", "TRIAD Sharpe", "TRIAD MDD"]],
                   {"VT Sharpe": num, "TRIAD Sharpe": num}, "현금 대신(2005~, MDD 는 일별)") + "\n")
    L.append("* 차이는 VT 단독 ±0.6%p, TRIAD 기준 ±0.2%p — **잡음 수준**이다. 현금이 생기는 때가 '변동성 폭발' 국면이라 "
             "주식(S&P·나스닥)으로 대체하면 닷컴 붕괴를 포함한 2001~ MDD 가 −53% → −61~−71% 로 커진다. 채권·금은 0.3~0.6%p 낫지만 "
             "잡음 범위라, 단순함을 위해 단기채(SGOV) 유지를 권한다.\n")

    # ---------------------------------------------------------------- Q3
    L.append("## 3. 시대 적응형 딥러닝 — '큰 상승 확률↑·큰 하락 확률↓' 패턴 학습\n")
    L.append("제안을 그대로 실험으로 옮겼다(`src/autotrader/research/adaptive.py`).\n")
    L.append("* **표본**: 15개 위험자산(S&P500·나스닥100·러셀2000·선진국·신흥국 2종·리츠·반도체·기술주·하이일드·KOSPI200·금·원자재·"
             "장기/중기 국채)을 한 모델로 학습. 1971년부터(1970년대 인플레·1987·2000·2008·2020·2022 포함).")
    L.append("* **입력**: 최근 차트·거래량을 변동성으로 정규화한 27개 피처(1~252일 수익, 이동평균 괴리, 고점 대비 낙폭, "
             "변동성 국면, 왜도·첨도, 거래량 추세·상승일 거래량 비중 등) + 시장 국면 8개(변동성·추세·금리·금리변화·주식채권 상관·VIX). "
             "CNN 은 최근 63일 수익률·거래량 시퀀스(차트 자체)를 직접 본다.")
    L.append("* **타깃**: 향후 21거래일 수익이 +1σ 초과 '큰 상승'(20%), −1σ 미만 '큰 하락'(13%), 그 외. 점수 = P(큰 상승) − P(큰 하락).")
    L.append("* **학습 방식**: 전체 과거 / 최근 10년 / 최근 가중(반감기 5년·2년) / **유사 국면 가중**(현재와 비슷한 변동성·추세·금리·"
             "주식채권상관을 가진 과거 표본에 커널 가중) / 분기 재학습 유사 국면 / **시기 섞기 배깅**(연도를 무작위로 70%씩 뽑은 5개 모델 평균) "
             "/ 전체 앙상블. 모델: LightGBM · MLP · 1D-CNN.")
    L.append("* **검증**: 1985~2026 워크포워드 — 매년(딥러닝은 2년) 그 시점 이전 데이터만으로 학습, 타깃 기간만큼 퍼지. "
             "시기를 섞는 배깅도 '과거 연도들' 안에서만 섞는다(무작위 K-fold 처럼 미래를 섞으면 겹치는 타깃 때문에 성과가 부풀려진다).\n")
    act = pd.read_csv(R / "adaptive_active.csv", index_col=0)
    L.append("### 3-1. 예측력: 큰 상승은 약간, 큰 하락은 전혀\n")
    acols = [c for c in auc.columns if c.startswith(("상승AUC", "하락AUC"))]
    L.append(table(auc[acols], {c: num for c in acols}, "모델 · 학습 방식") + "\n")
    up, dn = auc["상승AUC 전체"], auc["하락AUC 전체"]
    L.append(f"AUC 0.5 = 동전 던지기. **큰 상승은 {num(up.min())}~{num(up.max())}** 로 약하게나마 맞히고(딥러닝 앙상블이 가장 높음, "
             f"2010년 이후 0.57~0.60 으로 개선), **큰 하락은 {num(dn.min())}~{num(dn.max())} — 어떤 모델·학습 방식도 예측하지 못했다.** "
             "'큰 하락 확률은 낮고 큰 상승 확률은 높은 패턴'에서 앞의 절반(폭락 회피)은 차트·거래량에 정보가 없다는 뜻이다. "
             "1985~1999 에는 상승도 0.49~0.52 로 시대에 따라 들쭉날쭉했다.\n")
    L.append("### 3-2. 나스닥 엔진에 적용: 기준 규칙을 이기지 못함\n")
    L.append("적용 규칙은 미리 정했다: ① **게이트** — 점수가 그 자산의 과거 3년 분포 하위 20%면 현금, ② **틸트** — 노출 × (0.5 + 점수 백분위). "
             "VT(변동성 25%)와 VT+(확인형 역발상, TRIAD 의 엔진) 두 기준 규칙에 곱했다(2001~, 원화, 비용 반영).\n")
    base = qq.loc[["VT+ (규칙만)", "VT (규칙만)"]]
    vplus = qq[qq.index.str.startswith("VT+ ×")].sort_values("Sharpe", ascending=False)
    show = pd.concat([base.iloc[:1], vplus.head(5), vplus.tail(2), base.iloc[1:],
                      qq[qq.index.str.startswith("VT ×")].sort_values("Sharpe", ascending=False).head(3)])
    L.append(table(show[["CAGR", "Sharpe", "MDD", "2001~12", "2013~", "평균 노출"]],
                   {"Sharpe": num, "평균 노출": num}, "전략(2001~)") + "\n")
    v, w = act.loc["VT+"], act.loc["VT"]
    L.append(f"* 핵심 지표는 **기준 규칙 대비 초과수익**이다. TRIAD 엔진(VT+)에 ML 을 곱한 {int(float(v['변형 수']))}개 중 "
             f"초과수익이 플러스인 것은 **{pct(float(v['초과수익 IR>0 비율']), 0)}** 뿐 — Sharpe 가 오른 경우도 대부분 노출을 줄여서"
             "(평균 1.9배 → 1.4~1.8배) 위험이 같이 줄어든 것이지 더 잘 맞힌 것이 아니다.")
    L.append(f"* 단순 VT 에는 {pct(float(w['초과수익 IR>0 비율']), 0)}가 플러스였고, 가장 좋은 것은 **CNN(차트 딥러닝) · 유사 국면 가중 틸트**"
             f"(초과수익 IR {num(float(w['최고 IR']))}, 2001~12 {num(float(w['최고 IR 2001~12']))} / 2013~ {num(float(w['최고 IR 2013~']))} 로 두 구간 모두 플러스). "
             f"하지만 32개를 시험한 것을 보정한 DSR 은 {num(float(w['DSR(최고 초과수익)']))} — 우연과 구별되지 않는다(0.95 이상이어야 신뢰).")
    L.append("* TRIAD 에 넣으면(사전 지정 후보: 전체 앙상블 / 모델별 앙상블):\n")
    L.append(table(tri, {"Sharpe": num, "2013~ Sharpe": num}, "TRIAD 변형(2001~)") + "\n")
    t0 = tri.loc["TRIAD · VT+ (규칙만)"]
    L.append(f"→ Sharpe 는 {num(t0['Sharpe'])} → 1.16~1.18 로 소폭 오르지만 수익은 같거나 낮고(최대 −1.1%p) MDD 는 약간 나빠진다. "
             "**수익을 늘리는 도구가 아니라 약한 위험 축소 장치**에 가깝다.\n")
    L.append("### 3-3. 다른 자산에도 통하나(15개 자산, 1배, 게이트)\n")
    rcols = [c for c in rb.columns if "비율" in c or "중앙값" in c]
    L.append(table(rb[rcols], {c: (num if "중앙값" in c and "Sharpe" in c else pct) for c in rcols}, "모델 · 학습 방식") + "\n")
    L.append("* 두 시대(1985~2004 / 2005~2026) 모두에서 과반 자산의 Sharpe 를 올린 것은 **LightGBM · 시기 섞기 배깅**(64% / 73%), "
             "전체 앙상블(64% / 67%), LightGBM 전체 과거·유사 국면 정도다. 제안하신 '시기를 섞어 학습'이 가장 일관됐다.")
    L.append("* CNN · 유사 국면 가중은 2005년 이후 93% 자산에서 개선됐지만 1985~2004 에는 45% — 시대가 바뀌면 뒤집힌다.")
    L.append("* **최근 가중(반감 2년·최근 10년)은 거의 모든 비교에서 전체 과거보다 나빴다.** 표본이 줄어 잡음을 더 외우기 때문이다. "
             "MLP 단독은 가장 불안정했다.\n")
    L.append("### 3-4. '유사 국면 가중'은 실제로 무엇을 골랐나\n")
    L.append(table(sim, {c: (lambda v: f"{v:.0f}%") for c in sim.columns}, "재학습 시점") + "\n")
    L.append("2023년(금리 급등 직후) 재학습 때 1970년대 인플레 시기 표본에 24% 가중 — '비슷한 시대'는 잘 찾는다. "
             "하지만 그 시대의 패턴이 다음 해에 반복된다는 보장은 없었고, 전체 과거 대비 개선은 일관되지 않았다.\n")
    if sk is not None:
        L.append("### 3-5. 같은 아이디어를 한국 종목 선택 엔진에\n")
        L.append("종목 엔진은 매달 600종목을 비교하므로(횡단면) 표본이 훨씬 많고, 머신러닝이 실제로 효과를 내는 곳이다. "
                 "같은 학습 가중 방식을 LightGBM 종목 모델에 적용했다(모두 전체 학습 날짜, 같은 비용·손절·유동성 백테스트. "
                 "'시드' = 같은 설정을 다른 난수로 다시 학습한 것. 이 표의 TRIAD 는 한·미 공통 거래일로 맞춰 계산해 Sharpe 가 "
                 "3-2 표와 약간 다르므로 표 안에서만 비교).\n")
        L.append(table(sk, {"Sharpe": num, "2016~ Sharpe": num}, "엔진 · 학습 방식(2001~)") + "\n")
        L.append(stock_note(sk) + "\n")
    L.append("### 3-6. 결론\n")
    L.append("* **시장 타이밍에 딥러닝을 쓰는 것은 통계적으로 의미 있는 개선을 만들지 못했다.** 큰 하락은 예측되지 않았고(AUC≈0.50), "
             "큰 상승 예측의 작은 우위(0.55~0.58)는 기존 규칙(변동성 타기팅 + 추세 확인 역발상)이 이미 담고 있는 정보와 겹친다.")
    L.append("* 학습 방식 중에서는 **시기 섞기 배깅·앙상블**이 가장 안정적이었다. **최근 가중은 시장 타이밍·종목 선택 모두에서 "
             "전체 과거보다 나빴고**, 유사 국면 가중은 시장 타이밍에서는 엇갈렸으며 종목 선택에서는 크게 나빴다. "
             "'시대가 변해도 꾸준한' 쪽은 최근에 맞춘 모델이 아니라 여러 시대를 고루 학습한 모델이었다.")
    L.append("* 반영한 변경 한 가지: 종목 모델을 **시드 3개 앙상블**로 재학습(모델 잡음 축소). 이에 따라 TRIAD 기대치를 "
             "2001~ 기준 20.9% → 약 20.5% 로 낮춰 본다.")
    L.append("* 시대 변화 대응은 지금 구조가 이미 맡고 있다: ① 변동성 타기팅(국면이 바뀌면 노출이 자동으로 바뀜), ② 추세 필터, "
             "③ 상관 낮은 세 엔진 분산, ④ 종목 모델의 정기 재학습. 모두 파라미터가 적어 과최적화 여지가 작다.")
    L.append("* 그래서 **TRIAD 는 바꾸지 않는다.** 다만 위험을 조금 더 줄이고 싶다면 '전체 앙상블 게이트'를 선택 옵션으로 넣을 수 있다"
             "(TRIAD Sharpe 1.14 → 1.18, 최악의 해 −12.9% → −10.5%, 대신 CAGR −1.1%p). 연구 코드는 "
             "`src/autotrader/research/adaptive.py`, 재현은 `scripts/adaptive_ml.py` → `scripts/adaptive_eval.py`.")
    (R / "adaptive_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/adaptive_report.md")


if __name__ == "__main__":
    main()
