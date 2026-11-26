"""토너먼트 결과 → reports/tournament_report.md (scripts/tournament.py, tournament_analysis.py 실행 후)."""
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
        cells = [str(i)] + [fmt.get(c, pct)(row[c]) if not isinstance(row[c], str) else row[c] for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    t = pd.read_csv(R / "tournament.csv", index_col=0)
    fin = pd.read_csv(R / "tournament_final.csv", index_col=0)
    tri = pd.read_csv(R / "tournament_triad.csv", index_col=[0, 1])
    yr = pd.read_csv(R / "tournament_yearly.csv", index_col=0)
    longh = pd.read_csv(R / "tournament_long.csv", index_col=0)
    corr = pd.read_csv(R / "tournament_corr.csv", index_col=0)
    grid = pd.read_csv(R / "tournament_boost_grid.csv")
    lev = pd.read_csv(R / "tournament_levcost.csv", index_col=0)
    dsr = pd.read_csv(R / "tournament_dsr_triad.csv", index_col=0).iloc[:, 0]
    dsr_o = pd.read_csv(R / "tournament_dsr_isopt.csv", index_col=0).iloc[:, 0]
    b = pd.read_parquet("data/krx/tournament_blend_equity.parquet")["triad"]
    cur_dd = b.iloc[-1] / b.cummax().iloc[-1] - 1
    T = fin.loc[[i for i in fin.index if i.startswith("TRIAD")][0]]
    OLD = fin.loc["[기존] 3전략 1/3(고금리·3차·AQUILA)"]
    full = tri.loc[("TRIAD 매월 리밸런싱", "2001~ 전체")]
    band = tri.loc[("TRIAD ±10%p 밴드 리밸런싱", "2005~")]
    vt = t.loc["변동성 타기팅 QQQ 25%(최대 3배)"]
    vtp = t.loc["역발상·확인형: 폭락 후 골든크로스 1년 3배 + VT25%"]
    lad = t.loc["역발상 낙폭 사다리 QQQ(−20/−30/−40% → 1.5/2/3배)"]

    cols = {"CAGR(2005~)": "CAGR", "세후 CAGR(근사)": "세후", "Sharpe": "Sharpe", "일별 MDD": "MDD", "Calmar": "Calmar",
            "최악의 해": "최악의 해", "2005~15": "2005~15", "2016~": "2016~", "전체기간 CAGR": "2001~ CAGR",
            "전체기간 MDD": "2001~ MDD"}
    tt = t[list(cols)].rename(columns=cols).astype(float)
    tt.insert(0, "분류", t["분류"])
    tt = tt.sort_values(["분류", "Calmar"], ascending=[True, False])
    tfmt = {"Sharpe": num, "Calmar": num}

    L = []
    L.append("# 전략 토너먼트 → 최종 전략 TRIAD\n")
    L.append("> 질문: \"전설적인 투자자들의 검증된 전략을 모두 참고해 최고의 전략을 만들어라. 낙폭이 과도할 때 역발상으로 "
             "더 사고 레버리지를 쓰는 건 어떤가? 기존에 알림 받는 두 전략도 참고하라.\"\n")
    L.append("## 1. 결론\n")
    L.append("**TRIAD = 한국 종목 알파(MSAP-M) ⅓ + 미국 확인형 역발상 변동성 타기팅 ⅓ + 미국 레버리지 리스크패리티 ⅓**\n")
    L.append("| | CAGR | 세후(근사) | Sharpe | 일별 MDD | 2016~ | 2016~ MDD | 최악의 해 | 5년 롤링 최저 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab, r in (("**TRIAD (2005~)**", T), ("[기존] 고금리·3차·AQUILA 1/3씩", OLD)):
        L.append(f"| {lab} | {pct(r['CAGR'])} | {pct(r['세후(근사)'])} | {num(r['Sharpe'])} | {pct(r['일별 MDD'])} | "
                 f"{pct(r['2016~'])} | {pct(r['2016~ MDD'])} | {pct(r['최악의 해'])} | {pct(r['5년 롤링 최저'])} |")
    for lab in ("나스닥100 보유", "보글: S&P500 보유", "KOSPI200 보유", "60/40(주식/중기채)"):
        r = t.loc[lab]
        L.append(f"| {lab} | {pct(r['CAGR(2005~)'])} | {pct(r['세후 CAGR(근사)'])} | {num(r['Sharpe'])} | {pct(r['일별 MDD'])} | "
                 f"{pct(r['2016~'])} | {pct(r['2016~ MDD'])} | {pct(r['최악의 해'])} | {pct(r['5년 롤링 최저'])} |")
    L.append("")
    L.append(f"* 닷컴 붕괴를 포함한 **2001~2026.11 전체: CAGR {pct(full['CAGR'])}, Sharpe {num(full['Sharpe'])}, "
             f"MDD {pct(full['MDD'])}** (같은 기간 나스닥100 MDD {pct(t.loc['나스닥100 보유', '전체기간 MDD'])}).")
    L.append(f"* 실전처럼 레그 간 비중이 ±10%p 벗어날 때만 환전(비용 0.2%)해도 CAGR {pct(band['CAGR'])}, MDD {pct(band['MDD'])}.")
    L.append(f"* 6,600여 개 설정을 시험한 것을 보정한 Deflated Sharpe(2016~ 표본): **{num(dsr['dsr'])}** "
             f"(IS 최적화 조합은 {num(dsr_o['dsr'])}) — 운 좋은 최댓값일 확률이 낮다는 뜻.")
    L.append(f"* 기존 세 전략(사용자 알림 전략 근사)과 수익은 비슷하고({pct(OLD['CAGR'])} vs {pct(T['CAGR'])}) **Sharpe "
             f"{num(OLD['Sharpe'])} → {num(T['Sharpe'])}, MDD {pct(OLD['일별 MDD'])} → {pct(T['일별 MDD'])}** 으로 위험이 크게 줄었다. "
             "핵심은 미국 레버리지 추세군과 상관 0.1~0.3 인 '한국 종목 알파' 엔진을 더한 것.")
    L.append(f"* ⚠️ 2026-11-24 현재 TRIAD 는 고점 대비 **{pct(cur_dd)}** 낙폭 중(나스닥·코스피 동반 조정). "
             "과거 최대낙폭(−26%)을 넘을 수 있다는 전제로 비중을 정하세요.\n")
    L.append("![TRIAD](tournament_final.png)\n")

    L.append("## 2. 같은 조건에서 공정하게 비교했다\n")
    L.append("* **원화 기준 일별** 수익(환율 반영), 일별 MDD(월말 MDD 의 함정 제거). 신호 t일 종가 → **t+1일 매매**.")
    L.append("* 비용: 해외 편도 0.10%(NH 해외 0.09%+스프레드), 국내 ETF 0.03%. 레버리지 ETF 는 일일 리셋 합성 "
             "`L·r − (L−1)·단기금리 − 연 0.95%`(변동성 손실 자동 반영). 한국 종목 슬리브는 실제 거래세·슬리피지·유동성 한도 반영.")
    L.append("* 데이터: 미국 ETF 1999~, 상장 전 금·채권·원자재는 장기 총수익지수로 보강, 한국은 생존편향 없는 KRX 전종목.")
    L.append("* 검증: 2005~2015(IS) / 2016~(OOS) 분리, 5년 롤링 최저, 세후 근사(해외 양도세 22%, 손실이월·공제 무시 → 보수적), "
             "1971~ 미국 장기 꼬리위험 검증, Deflated Sharpe.")
    L.append("* 한계: 버핏·그레이엄·린치·그린블랫·피오트로스키 같은 **재무제표 기반 전략은 무료 시점별(point-in-time) 재무 데이터가 "
             "없어 시험하지 못했다**(현재 재무로 과거를 시험하면 미래참조). 사용자 기존 전략은 알림 메시지에서 읽은 규칙으로 만든 **근사**다.\n")

    L.append("## 3. 토너먼트 결과(31개, 2005~ 공통구간)\n")
    L.append(table(tt, tfmt, "전략") + "\n")
    L.append("읽는 법: 전설적 자산배분(A)은 MDD 가 작지만 원화 기준 연 8~11%. 추세·모멘텀(B)의 고전 규칙은 2008 이후 "
             "수익이 낮다(GTAA 5.7%, 터틀 5.4%). 20%를 넘는 것은 **레버리지를 변동성·추세로 통제한 전략(C·F)** 과 "
             "**한국 종목 알파(E)** 뿐이다.\n")

    L.append("## 4. \"낙폭이 클 때 역발상으로 더 사고 레버리지\" — 데이터의 답\n")
    L.append(f"**무조건 역발상 레버리지는 파산형이다.** 고점 대비 −20/−30/−40% 에서 1.5/2/3배로 늘리는 사다리는 "
             f"2005~ CAGR {pct(lad['CAGR(2005~)'])} 로 높아 보이지만 **MDD {pct(lad['일별 MDD'])}**, 2001년에 시작했다면 "
             f"**CAGR {pct(lad['전체기간 CAGR'])}, MDD {pct(lad['전체기간 MDD'])}** 였다. 2000~2002년 나스닥은 −83%까지 "
             "빠졌고, '충분히 빠졌다'고 레버리지를 늘린 지점들 아래로 또 반토막이 났다.\n")
    lh = longh[["CAGR", "Sharpe", "MDD", "최악의 해", "1970s", "1980s", "1990s", "2000s", "2010s"]]
    L.append("1971년부터 S&P500(달러)으로 다시 시험해도 같다(1973~74, 1987, 2000~02, 2008 포함):\n")
    L.append(table(lh, {"Sharpe": num}, "1971~ 미국") + "\n")
    L.append("왜 그런가: ① 레버리지 ETF 는 하락장의 변동성 손실로 기초자산보다 훨씬 깊게 빠진다(3배 보유 MDD −98%). "
             "② 폭락은 '평균 회귀'보다 '추세 지속'이 먼저 온다 — 변동성이 높을 때 레버리지를 늘리는 것은 "
             "Moreira–Muir(2017)가 보인 것과 정반대(고변동성일 때 줄여야 위험 대비 수익이 오른다). "
             "③ 손절을 붙여도(−72%) VIX 공포 매수(−65%)도 꼬리를 못 막는다.\n")
    L.append("**하지만 아이디어의 핵심 — '공포 뒤의 회복은 크다' — 는 맞다.** 그래서 순서를 바꿨다: "
             "폭락 '중'이 아니라, **최근 2년 안에 사상 최고가 대비 −30% 이하까지 빠졌던 자산에서 골든크로스"
             "(50일선이 200일선을 새로 상향 돌파)로 추세 회복이 확인된 뒤** 1년간 3배로 싣는다(골든크로스가 깨지면 해제). "
             "템플턴의 '최대 비관의 시점'을 추세로 확인하는 방식으로, 나스닥에서는 2003~2013년(2000년 고점 아래 구간)과 "
             "2023년에 11번 발동했다.\n")
    L.append(f"| | 2005~ CAGR | MDD | 2016~ | 2001~ CAGR |\n|---|---|---|---|---|\n"
             f"| 변동성 타기팅 25% | {pct(vt['CAGR(2005~)'])} | {pct(vt['일별 MDD'])} | {pct(vt['2016~'])} | {pct(vt['전체기간 CAGR'])} |\n"
             f"| **확인형 역발상 + VT 25%** | **{pct(vtp['CAGR(2005~)'])}** | {pct(vtp['일별 MDD'])} | {pct(vtp['2016~'])} | {pct(vtp['전체기간 CAGR'])} |\n")
    g = grid.groupby("L")[["CAGR", "MDD", "OOS CAGR"]].median()
    L.append(f"* 과최적화 점검: 폭락 기준(−20~−40%)·유지기간(6~18개월)·배수(2~3배) **45개 조합 모두**가 VT 단독보다 CAGR 이 높았다"
             f"(중앙값 {pct(grid['CAGR'].median())}, 범위 {pct(grid['CAGR'].min())}~{pct(grid['CAGR'].max())}). "
             f"배수 2배면 CAGR {pct(g.loc[2.0, 'CAGR'])}·MDD {pct(g.loc[2.0, 'MDD'])}, 3배면 {pct(g.loc[3.0, 'CAGR'])}·{pct(g.loc[3.0, 'MDD'])}.")
    L.append(f"* 표본 밖 검증: 이 규칙을 만들 때 보지 않은 1971~2000년 S&P500 에서도 VT 단독 대비 우위(1971~ CAGR "
             f"{pct(longh.iloc[-1]['CAGR'])} vs {pct(longh.loc['변동성 타기팅 25%(최대 3배)', 'CAGR'])}, MDD 동일).")
    L.append("* 단, 단독으로는 MDD −46% 라 TRIAD 안에서 ⅓ 비중으로만 쓴다.\n")

    L.append("## 5. 기존 두 전략(알림 받는 전략) 평가\n")
    for k in ("[기존] 고금리 대비형(근사)", "[기존] 3차 공격형(근사)", "[기존] AQUILA-X(근사)"):
        r = t.loc[k]
        L.append(f"* **{k}**: CAGR {pct(r['CAGR(2005~)'])}, Sharpe {num(r['Sharpe'])}, MDD {pct(r['일별 MDD'])}, "
                 f"2016~ {pct(r['2016~'])} — 토너먼트 상위권. 설계가 좋다.")
    L.append("\n세 전략 모두 '나스닥 추세·변동성 레버리지' 또는 '레버리지 리스크패리티'에 속해 서로 상관이 0.7~0.9 로 높다"
             f"(아래 표). 셋을 섞어도 MDD 가 {pct(OLD['일별 MDD'], 0)} 로 남는 이유다. TRIAD 의 미국 두 엔진은 이 둘의 대표(고금리 대비형/AQUILA ≈ VT 계열, "
             "3차 공격형 ≈ 레버리지 RP, 상관 0.83~0.89)이고, 여기에 상관 0.09~0.31 인 한국 종목 엔진을 더해 위험을 낮췄다.\n")
    L.append(table(corr, {c: num for c in corr.columns}, "주간 상관") + "\n")
    L.append("(KR=MSAP-M, VT=변동성 타기팅, VT+=확인형 역발상, HR=고금리 대비형, A3=3차 공격형, AQ=AQUILA-X, LRP=레버리지 RP)\n")

    L.append("## 6. 조합은 '최적화'보다 '단순 규칙'이 강했다\n")
    fcols = ["CAGR", "세후(근사)", "Sharpe", "일별 MDD", "2005~15", "2016~", "2016~ MDD", "최악의 해", "나스닥 β 최대"]
    L.append(table(fin[fcols], {"Sharpe": num, "나스닥 β 최대": num}, "배분 규칙(2005~)") + "\n")
    L.append("* 사전에 정한 규칙(IS 2005~2015 에서 CAGR ≥ 20% 중 Calmar 최대, 10% 단위 격자 6,538개)으로 고른 조합은 "
             "IS 20.7% → OOS 18.5% 로 식었다(전형적인 과최적화 감쇠). 반면 최적화 없는 단순 배분들은 **어느 것을 골라도 "
             "CAGR 20~22%, MDD −25~−31%** — 결론이 배분 방식에 둔감하다(DeMiguel 외 2009 '1/N' 결과와 같은 현상).")
    L.append(f"* 그중 TRIAD(서로 다른 세 엔진 1/3씩)를 택한 이유: 가장 단순(엔진 3개)하고, 2016~ Sharpe({num(T['2016~ Sharpe'])})·"
             f"MDD({pct(T['2016~ MDD'])})가 모든 배분 규칙 중 가장 좋으며, 세 엔진이 각각 다른 '전설'의 검증된 원리"
             "(오닐·미너비니의 종목 모멘텀과 손절, 변동성 관리+템플턴식 역발상, 달리오의 리스크패리티)라서. "
             "고른 뒤의 선택 편향은 아래 Deflated Sharpe 에 시험 횟수로 반영했다.\n")

    L.append("## 7. TRIAD 상세\n")
    tri_t = tri.copy()
    tri_t.index = [f"{a} · {b_}" for a, b_ in tri_t.index]
    L.append(table(tri_t, {"Sharpe": num}, "구간") + "\n")
    L.append("**연도별 수익(원화, %)**\n")
    yr_t = (yr * 100).round(1)
    L.append(table(yr_t, {c: (lambda v: f"{v:.1f}") for c in yr_t.columns}, "연도") + "\n")
    L.append("**위기 구간**\n")
    L.append("| 구간 | TRIAD | 나스닥100 | KOSPI200 |\n|---|---|---|---|")
    for k, (a, e) in {"2001~02 닷컴": ("2001-02-01", "2002-10-09"), "2008 금융위기": ("2007-10-09", "2009-03-09"),
                      "2011 유럽위기": ("2011-04-29", "2011-10-03"), "2018 Q4": ("2018-09-28", "2018-12-24"),
                      "2020 코로나": ("2020-02-19", "2020-03-23"), "2022 금리급등": ("2022-01-03", "2022-10-14"),
                      "2024.08 엔캐리": ("2024-07-10", "2024-08-07"), "2026.10~11 조정": ("2026-10-08", "2026-11-24")}.items():
        cur = pd.read_parquet("data/krx/tournament_equity.parquet", columns=["나스닥100 보유", "KOSPI200 보유"])
        x, q, kk = b.loc[a:e], cur["나스닥100 보유"].loc[a:e], cur["KOSPI200 보유"].loc[a:e]
        L.append(f"| {k} | {pct(x.iloc[-1] / x.iloc[0] - 1)} | {pct(q.iloc[-1] / q.iloc[0] - 1)} | {pct(kk.iloc[-1] / kk.iloc[0] - 1)} |")
    L.append("")
    L.append("**레버리지 ETF 비용 민감도**(실제 스왑 금리는 단기금리보다 높다):\n")
    L.append(table(lev[["TRIAD CAGR", "Sharpe", "MDD", "2016~"]], {"Sharpe": num}, "추가 금융비용") + "\n")
    L.append(f"**꼬리위험**: TRIAD 의 나스닥 63일 β 는 중앙값 {num(T['나스닥 β 중앙값'])}, 최대 {num(T['나스닥 β 최대'])}. "
             "1987년 블랙먼데이 같은 하루 −20% 가 오면 이론상 하루 −17%(보통) ~ −31%(부스트 최대 노출) 손실이 가능하다. "
             "백테스트 최악의 하루는 " + pct(T["최악의 하루"]) + ". 3배 상품의 하루 갭 위험은 손절로 막을 수 없으므로 "
             f"감당할 수 없다면 `boost_leverage: 2.0` 을 쓰세요(확인형 역발상 엔진 기준 CAGR {pct(g.loc[2.0, 'CAGR'] - g.loc[3.0, 'CAGR'])}p, "
             f"MDD {pct(g.loc[3.0, 'MDD'])} → {pct(g.loc[2.0, 'MDD'])}).\n")

    L.append("## 8. 실전 운용\n")
    L.append("| 레그 | 비중 | 설정 | 계좌 | 실행 시각 |\n|---|---|---|---|---|")
    L.append("| 국내: MSAP-M | ⅓ | `config/default.yaml` | NH 국내(원화) | 매 거래일 09:30 KST |")
    L.append("| 미국: 확인형 역발상 VT + 레버리지 RP | ⅔ | `config/triad_us.yaml` | NH 해외(달러) | 매 거래일 10:30 ET |")
    L.append("")
    L.append("```bash\nautotrader --config config/triad_us.yaml signal          # 미국 레그 목표비중·부스트 상태\n"
             "autotrader --config config/triad_us.yaml rebalance --dry-run\nautotrader --config config/triad_us.yaml daemon          # 모의투자로 먼저\n"
             "autotrader triad                                          # 레그 간 배분 점검·환전 권고\n```\n")
    L.append("* 미국 레그 매매 상품: QQQ·TQQQ(3배)·TLT·TMF(3배)·GLD·UGL(2배)·DBC·SGOV(현금). 신호는 백테스트와 **같은 함수**"
             "(`research.tournament`)를 호출하며, 과거 40개월 신호가 백테스트와 정확히 일치하는 것을 확인했다.")
    L.append("* 레그 간 배분은 ±10%p 를 벗어날 때만 수동 환전(`autotrader triad` 가 금액을 알려줌). 환전 우대율을 확인하세요.")
    L.append("* 확인할 것: 해외 레버리지 ETP 매매 요건(사전교육·기본예탁금 등 금융당국 규제), NH PLUG 해외주식 API 신청, "
             "해외 양도소득세(연 250만원 공제 후 22%, 다음 해 5월 신고).")
    L.append("* 세후 수익은 계좌 구조에 따라 달라진다. 국내 레그 종목 매매차익은 대주주가 아니면 비과세다.\n")

    L.append("## 9. 정직한 한계\n")
    L.append("* '부스트' 발동은 25년간 11번이고 그중 대부분이 2000년 고점 회복 전(2003~2013) 한 국면에 몰려 있다. "
             "45개 설정·1971년 이후 검증으로 보완했지만 독립 사례는 많지 않다(2008.06 발동처럼 곧바로 실패한 사례도 있다).")
    L.append("* 레버리지 ETF 는 합성 수익률이다(실제 추적오차·스왑 비용은 위 민감도 참조).")
    L.append(f"* 한국 종목 엔진(MSAP-M)의 최근 10년 수익은 초기보다 낮다(2016~ {pct(t.loc['MSAP-M(종목 70 + ETF RP 30)', '2016~'])}) "
             "— 단독이 아니라 ⅓ 로 둔 이유 중 하나.")
    L.append("* 미래 수익을 보장하지 않는다. 모의투자 → 소액 실전 → 증액 순서를 지키세요.")
    (R / "tournament_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/tournament_report.md")


if __name__ == "__main__":
    main()
