"""백테스트 결과 리포트(마크다운 + 차트 PNG)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from autotrader.backtest.engine import BacktestResult, align_common, compare
from autotrader.backtest.metrics import drawdown, yearly_returns


def _pct(x) -> str:
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def _num(x) -> str:
    return "-" if pd.isna(x) else f"{x:.2f}"


def comparison_markdown(results: dict[str, BacktestResult]) -> str:
    df = compare(results)
    head = "| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar | 최악의 해 | 최장 회복(년) | 연회전율 | 최종배수 |\n"
    head += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for name, r in df.iterrows():
        rows.append(f"| {name} | {_pct(r.CAGR)} | {_pct(r.Vol)} | {_num(r.Sharpe)} | {_num(r.Sortino)} | "
                    f"{_pct(r.MDD)} | {_num(r.Calmar)} | {_pct(r.WorstYear)} | {_num(r['Underwater(y)'])} | "
                    f"{_num(r['Turnover/y'])} | {_num(r.x)}x |")
    return head + "\n".join(rows)


def yearly_markdown(results: dict[str, BacktestResult], last_n: int | None = None) -> str:
    yr = pd.DataFrame({k: yearly_returns(r.equity) for k, r in results.items()})
    if last_n:
        yr = yr.iloc[-last_n:]
    head = "| 연도 | " + " | ".join(yr.columns) + " |\n|---|" + "---:|" * len(yr.columns) + "\n"
    return head + "\n".join(f"| {y} | " + " | ".join(_pct(v) for v in row) + " |" for y, row in yr.iterrows())


def crisis_table(results: dict[str, BacktestResult], periods: dict[str, tuple[str, str]]) -> str:
    head = "| 위기 구간 | " + " | ".join(results) + " |\n|---|" + "---:|" * len(results) + "\n"
    rows = []
    for label, (a, b) in periods.items():
        cells = []
        for r in results.values():
            e = r.equity.loc[a:b]
            cells.append(_pct(e.iloc[-1] / e.iloc[0] - 1) if len(e) > 1 else "-")
        rows.append(f"| {label} ({a}~{b}) | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def save_charts(results: dict[str, BacktestResult], out_dir: str | Path, main: str | None = None,
                prefix: str = "") -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return []
    have = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"):
        if fam in have:  # 한글 라벨용 폰트
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curves = align_common(results)
    files = []
    palette = ["#2a6fdb", "#e07b39", "#3aa76d", "#9b59b6", "#7f8c8d", "#c0392b"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for i, (k, e) in enumerate(curves.items()):
        lw = 2.2 if k == main else 1.2
        ax1.plot(e.index, e.values, label=k, lw=lw, color=palette[i % len(palette)])
        ax2.plot(e.index, drawdown(e).values * 100, lw=lw, color=palette[i % len(palette)])
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    p = out / f"{prefix}equity_drawdown.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    if main and main in results:
        w = results[main].weights.resample("ME").last()
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.stackplot(w.index, w.T.values, labels=w.columns, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weight")
        ax.legend(loc="upper left", ncol=min(7, len(w.columns)), frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        p = out / f"{prefix}weights.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    return files
