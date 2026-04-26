#!/usr/bin/env python3
"""Multi Portfolio の資産推移グラフを生成しレポートに追加する"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import base64
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 日本語フォント設定 (macOS)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Hiragino Maru Gothic Pro", "AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
from backtesting import Backtest
from config.settings import INITIAL_EQUITY, COMMISSION

from strategy.trend_follow import prepare_data as prep_tf, TrendFollowStrategy
from strategy.supertrend import prepare_data as prep_st, SupertrendStrategy
from strategy.macd_momentum import prepare_data as prep_macd, MACDMomentumStrategy
from strategy.ichimoku import prepare_data as prep_ichi, IchimokuStrategy


SPREAD_PIPS = 0.4
AVG_PRICE = 130.0
SPREAD_RATIO = SPREAD_PIPS * 0.01 / AVG_PRICE

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "usdjpy_h1_25years.csv",
)

STRATEGIES = {
    "TF": {
        "name": "Trend Follow",
        "color": "#4263eb",
        "prep": lambda df: prep_tf(df.copy(), entry_period=15, exit_period=15),
        "strategy": TrendFollowStrategy,
        "params": dict(entry_period=15, exit_period=15, atr_sl_mult=3.0,
                       atr_trail_mult=3.0, use_ma_filter=0, max_hold_hours=72),
    },
    "ST": {
        "name": "Supertrend",
        "color": "#e03131",
        "prep": lambda df: prep_st(df.copy(), period=10, multiplier=2.0),
        "strategy": SupertrendStrategy,
        "params": dict(st_period=10, st_multiplier=2.0,
                       atr_sl_mult=2.0, atr_trail_mult=3.0, max_hold_hours=72),
    },
    "MACD": {
        "name": "MACD",
        "color": "#f59f00",
        "prep": lambda df: prep_macd(df.copy(), fast_period=12, slow_period=26, signal_period=12),
        "strategy": MACDMomentumStrategy,
        "params": dict(atr_sl_mult=2.5, atr_trail_mult=2.5, use_ma_filter=0),
    },
    "ICHI": {
        "name": "Ichimoku",
        "color": "#9333ea",
        "prep": lambda df: prep_ichi(df.copy(), tenkan_period=9, kijun_period=26, senkou_b_period=52),
        "strategy": IchimokuStrategy,
        "params": dict(atr_sl_mult=2.0, atr_trail_mult=3.0,
                       use_tenkan_kijun=1, max_hold_hours=72),
    },
}

ALLOC = 0.25


def _run_bt(df, strategy_class, cash, margin=1.0, **params):
    bt = Backtest(
        df, strategy_class,
        cash=cash, commission=COMMISSION,
        spread=SPREAD_RATIO, exclusive_orders=True, margin=margin,
    )
    return bt.run(**params)


def _eq_series(stats):
    eq = stats["_equity_curve"]["Equity"]
    return eq.resample("D").last().dropna()


def _get_combined_equity(df_raw, leverage=1.0):
    """指定レバレッジで合算エクイティを取得する。"""
    margin = 1.0 / leverage
    equities = []
    for key, spec in STRATEGIES.items():
        df = spec["prep"](df_raw)
        stats = _run_bt(df, spec["strategy"],
                        cash=INITIAL_EQUITY * ALLOC, margin=margin,
                        **spec["params"])
        eq = _eq_series(stats)
        equities.append((key, spec["name"], spec["color"], eq, stats))

    # 合算
    combined = equities[0][3].copy()
    for _, _, _, eq, _ in equities[1:]:
        idx = combined.index.intersection(eq.index)
        combined = combined.loc[idx] + eq.loc[idx]

    return combined, equities


def _fig_to_base64(fig):
    """matplotlib図をbase64エンコードされたPNGに変換する。"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _format_yen(x, _):
    if x >= 1e8:
        return f"{x/1e8:.1f}億"
    if x >= 1e4:
        return f"{x/1e4:.0f}万"
    return f"{x:.0f}"


def _make_multi_portfolio_chart(df_raw):
    """Multi Portfolio の資産推移チャート (1x, ドローダウン含む)"""
    combined, equities = _get_combined_equity(df_raw, leverage=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
                              sharex=True)

    # 1. 各戦略の資産推移 + 合算
    ax = axes[0]
    for key, name, color, eq, _ in equities:
        ax.plot(eq.index, eq.values, color=color, alpha=0.6, linewidth=1.2,
                label=f"{name} (25%)")
    ax.plot(combined.index, combined.values, color="#16a34a", linewidth=2.2,
            label="Combined Portfolio", zorder=10)
    ax.axhline(INITIAL_EQUITY, color="#888", linestyle="--", linewidth=0.8, alpha=0.5,
               label="初期資金")

    # ピーク値をマーク
    peak_idx = combined.idxmax()
    peak_val = combined.max()
    ax.annotate(f"最高値\n{peak_val:,.0f}円",
                xy=(peak_idx, peak_val),
                xytext=(peak_idx, peak_val * 1.1),
                fontsize=10, color="#16a34a", ha="center",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1))

    ax.set_ylabel("資産 (円)", fontsize=11)
    ax.set_title("Multi Portfolio 資産推移 (1x レバレッジ、25年間)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_yen))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 2. ドローダウン
    ax = axes[1]
    peak = combined.cummax()
    dd = (combined - peak) / peak * 100
    ax.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#dc2626", linewidth=1)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.axhline(dd.min(), color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"最大DD {dd.min():.1f}%")
    ax.set_ylabel("ドローダウン (%)", fontsize=11)
    ax.set_title("ドローダウン (資産ピークからの下落率)",
                 fontsize=12, pad=8)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 月次P&L
    ax = axes[2]
    monthly_pnl = combined.resample("ME").last().diff().dropna()
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in monthly_pnl.values]
    ax.bar(monthly_pnl.index, monthly_pnl.values, color=colors, width=25, alpha=0.7)
    ax.axhline(0, color="#333", linewidth=0.5)

    m_wins = (monthly_pnl > 0).sum()
    m_total = len(monthly_pnl)
    m_wr = m_wins / m_total * 100
    ax.set_title(f"月次P&L (月次勝率 {m_wr:.1f}% | {m_wins}/{m_total}ヶ月勝ち)",
                 fontsize=12, pad=8)
    ax.set_ylabel("月次損益 (円)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_yen))
    ax.grid(True, alpha=0.3)

    # X軸フォーマット
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _fig_to_base64(fig), combined


def _make_leverage_comparison_chart(df_raw):
    """レバレッジ別の資産推移比較チャート"""
    leverage_levels = [1, 2, 3, 5, 10]
    colors = ["#16a34a", "#4263eb", "#f59f00", "#e03131", "#9333ea"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [2, 1]},
                              sharex=True)

    # 1. 資産推移 (log scale)
    ax = axes[0]
    ax.set_yscale("log")

    results = []
    for lev, color in zip(leverage_levels, colors):
        combined, _ = _get_combined_equity(df_raw, leverage=lev)
        final = combined.iloc[-1]
        results.append((lev, combined, final))
        ax.plot(combined.index, combined.values, color=color, linewidth=1.8,
                label=f"{lev}x (最終 {final:,.0f}円)")

    ax.axhline(INITIAL_EQUITY, color="#888", linestyle="--", linewidth=0.8, alpha=0.5,
               label="初期資金 100万")
    ax.set_ylabel("資産 (円・対数スケール)", fontsize=11)
    ax.set_title("レバレッジ別 資産推移比較 (25年間)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_yen))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # 2. ドローダウン比較
    ax = axes[1]
    for (lev, combined, _), color in zip(results, colors):
        peak = combined.cummax()
        dd = (combined - peak) / peak * 100
        ax.plot(dd.index, dd.values, color=color, linewidth=1.3, label=f"{lev}x (MDD {dd.min():.1f}%)")
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.axhline(-50, color="#dc2626", linestyle=":", linewidth=1, alpha=0.5,
               label="-50%線 (危険ライン)")
    ax.set_ylabel("ドローダウン (%)", fontsize=11)
    ax.set_title("レバレッジ別 ドローダウン比較",
                 fontsize=12, pad=8)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _fig_to_base64(fig)


def _inject_chart_into_report(filepath, chart_html, marker_tag="<h2>組み合わせ"):
    """既存HTMLレポートの指定位置にチャートを挿入する。"""
    if not os.path.exists(filepath):
        print(f"  ファイルが存在しないためスキップ: {filepath}")
        return False

    with open(filepath, "r") as f:
        html = f.read()

    # すでに挿入されている場合はスキップ
    if 'id="equity-chart"' in html:
        # 既存を置換 (先頭のh2も含めて削除)
        import re
        html = re.sub(
            r'<h2>資産推移グラフ[^<]*</h2>\s*<div id="equity-chart".*?</div>',
            chart_html.strip(),
            html,
            count=1,
            flags=re.DOTALL,
        )
    else:
        # マーカーの前に挿入
        if marker_tag in html:
            html = html.replace(marker_tag, chart_html + "\n\n" + marker_tag, 1)
        else:
            # </body>直前に
            html = html.replace("</body>", chart_html + "\n</body>")

    with open(filepath, "w") as f:
        f.write(html)
    return True


def main():
    print("="*60)
    print("Multi Portfolio 資産推移チャート生成")
    print("="*60)

    print("\nデータ読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)

    # 1. Multi Portfolio チャート
    print("\n[1/2] Multi Portfolio 資産推移チャート生成中...")
    chart_b64, combined = _make_multi_portfolio_chart(df_raw)
    print(f"  チャート生成完了 ({len(chart_b64):,} バイト)")
    print(f"  最終資産: {combined.iloc[-1]:,.0f} 円")

    # report_multi_portfolio.html に挿入
    chart_html = f"""
<h2>資産推移グラフ</h2>
<div id="equity-chart" class="card">
  <img src="data:image/png;base64,{chart_b64}" alt="Multi Portfolio 資産推移"
       style="width:100%;height:auto;display:block"/>
  <p style="color:#888;font-size:11px;margin-top:8px">
    上段: 各戦略(薄色)と合算ポートフォリオ(緑太線)の資産推移 /
    中段: ドローダウン (資産ピークからの下落率) /
    下段: 月次損益 (緑=勝ち月、赤=負け月)
  </p>
</div>
"""
    ok = _inject_chart_into_report(
        "report_multi_portfolio.html", chart_html,
        marker_tag="<h2>組み合わせ",
    )
    print(f"  → report_multi_portfolio.html {'更新' if ok else 'スキップ'}")

    # 2. レバレッジ比較チャート
    print("\n[2/2] レバレッジ比較チャート生成中...")
    lev_chart_b64 = _make_leverage_comparison_chart(df_raw)
    print(f"  チャート生成完了 ({len(lev_chart_b64):,} バイト)")

    lev_chart_html = f"""
<h2>資産推移グラフ (レバレッジ別比較)</h2>
<div id="equity-chart" class="card">
  <img src="data:image/png;base64,{lev_chart_b64}" alt="Leverage Comparison"
       style="width:100%;height:auto;display:block"/>
  <p style="color:#888;font-size:11px;margin-top:8px">
    上段: レバレッジ別の資産推移 (対数スケール) /
    下段: レバレッジ別のドローダウン (点線 -50% は危険ライン)
  </p>
</div>
"""
    ok = _inject_chart_into_report(
        "report_multi_leverage.html", lev_chart_html,
        marker_tag='<div class="warning">',
    )
    print(f"  → report_multi_leverage.html {'更新' if ok else 'スキップ'}")

    print("\n完了")


if __name__ == "__main__":
    main()
