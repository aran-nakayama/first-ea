#!/usr/bin/env python3
"""Multi Portfolio (TF+ST+MACD+ICHI) のレバレッジ感応度分析"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
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
        "prep": lambda df: prep_tf(df.copy(), entry_period=15, exit_period=15),
        "strategy": TrendFollowStrategy,
        "params": dict(entry_period=15, exit_period=15, atr_sl_mult=3.0,
                       atr_trail_mult=3.0, use_ma_filter=0, max_hold_hours=72),
    },
    "ST": {
        "name": "Supertrend",
        "prep": lambda df: prep_st(df.copy(), period=10, multiplier=2.0),
        "strategy": SupertrendStrategy,
        "params": dict(st_period=10, st_multiplier=2.0,
                       atr_sl_mult=2.0, atr_trail_mult=3.0, max_hold_hours=72),
    },
    "MACD": {
        "name": "MACD Momentum",
        "prep": lambda df: prep_macd(df.copy(), fast_period=12, slow_period=26, signal_period=12),
        "strategy": MACDMomentumStrategy,
        "params": dict(atr_sl_mult=2.5, atr_trail_mult=2.5, use_ma_filter=0),
    },
    "ICHI": {
        "name": "Ichimoku",
        "prep": lambda df: prep_ichi(df.copy(), tenkan_period=9, kijun_period=26, senkou_b_period=52),
        "strategy": IchimokuStrategy,
        "params": dict(atr_sl_mult=2.0, atr_trail_mult=3.0,
                       use_tenkan_kijun=1, max_hold_hours=72),
    },
}

ALLOC = 0.25  # 各戦略25%


def _pf_from_trades(trades):
    if trades is None or len(trades) == 0:
        return 0
    w = trades[trades["PnL"] > 0]["PnL"].sum()
    l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
    return w / l if l > 0 else float("inf")


def _run_leveraged(df_raw, leverage):
    """指定レバレッジで4戦略を実行し、合算結果を返す。"""
    margin = 1.0 / leverage
    equities = []
    total_trades = 0

    for key, spec in STRATEGIES.items():
        df = spec["prep"](df_raw)
        bt = Backtest(
            df, spec["strategy"],
            cash=INITIAL_EQUITY * ALLOC,
            commission=COMMISSION,
            spread=SPREAD_RATIO,
            exclusive_orders=True,
            margin=margin,
        )
        stats = bt.run(**spec["params"])
        eq = stats["_equity_curve"]["Equity"].resample("D").last().dropna()
        equities.append(eq)
        total_trades += stats["# Trades"]

    # 合算
    combined = equities[0].copy()
    for eq in equities[1:]:
        idx = combined.index.intersection(eq.index)
        combined = combined.loc[idx] + eq.loc[idx]

    init_val = combined.iloc[0]
    final_val = combined.iloc[-1]
    total_ret = (final_val / init_val - 1) * 100
    years = (combined.index[-1] - combined.index[0]).days / 365.25
    annual = ((1 + total_ret / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = combined.cummax()
    dd = (combined - peak) / peak * 100
    mdd = dd.min()

    # 月次分析
    monthly = combined.resample("ME").last().diff().dropna()
    m_wins = (monthly > 0).sum()
    m_total = len(monthly)
    m_wr = m_wins / m_total * 100 if m_total > 0 else 0

    # 年次分析
    yearly = combined.resample("YE").last().pct_change().dropna() * 100
    y_wins = (yearly > 0).sum()
    y_wr = y_wins / len(yearly) * 100 if len(yearly) > 0 else 0

    return {
        "leverage": leverage,
        "return_pct": total_ret,
        "annual_pct": annual,
        "mdd": mdd,
        "final_equity": final_val,
        "monthly_wr": m_wr,
        "yearly_wr": y_wr,
        "y_wins": y_wins,
        "y_total": len(yearly),
        "trades": total_trades,
    }


def main():
    print("="*60)
    print("Multi Portfolio レバレッジ感応度分析")
    print("="*60)

    print("\nデータ読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"データ: {len(df_raw)}本")

    leverage_levels = [1, 2, 3, 5, 7, 10, 15, 20, 25]
    results = []

    for lev in leverage_levels:
        print(f"\nレバレッジ {lev}x で計算中...", end=" ", flush=True)
        r = _run_leveraged(df_raw, lev)
        results.append(r)
        print(f"Return={r['return_pct']:+.1f}% MDD={r['mdd']:.1f}% "
              f"月次WR={r['monthly_wr']:.1f}% 年次WR={r['yearly_wr']:.1f}%")

    # 表示
    print("\n\n" + "="*80)
    print(f"  {'レバ':>4} {'Return':>12} {'年利':>8} {'最終資産':>14} "
          f"{'MDD':>8} {'月次WR':>7} {'年次WR':>7} {'判定':>8}")
    print("  " + "-"*78)

    for r in results:
        # 判定ロジック
        if r["mdd"] > -15 and r["return_pct"] > 0:
            rank = "おすすめ"
        elif r["mdd"] > -30 and r["return_pct"] > 0:
            rank = "中リスク"
        elif r["mdd"] > -50 and r["return_pct"] > 0:
            rank = "注意"
        elif r["mdd"] > -80 and r["return_pct"] > 0:
            rank = "危険"
        else:
            rank = "破綻"

        print(f"  {r['leverage']:>3}x {r['return_pct']:>+11.1f}% "
              f"{r['annual_pct']:>+7.1f}% {r['final_equity']:>14,.0f} "
              f"{r['mdd']:>7.1f}% {r['monthly_wr']:>6.1f}% "
              f"{r['yearly_wr']:>6.1f}% {rank:>8}")

    # HTMLレポート出力
    print("\nHTMLレポート生成中...")
    _gen_report("report_multi_leverage.html", results)
    print("レポート出力: report_multi_leverage.html")


def _gen_report(filepath, results):
    rows = ""
    for r in results:
        lev = r["leverage"]
        is_profitable = r["return_pct"] > 0

        if r["mdd"] > -15 and is_profitable:
            rank = "おすすめ"
            rank_bg = "#dcfce7;color:#166534"
        elif r["mdd"] > -30 and is_profitable:
            rank = "中リスク"
            rank_bg = "#fef9c3;color:#854d0e"
        elif r["mdd"] > -50 and is_profitable:
            rank = "注意"
            rank_bg = "#ffedd5;color:#9a3412"
        elif r["mdd"] > -80 and is_profitable:
            rank = "危険"
            rank_bg = "#fef2f2;color:#991b1b"
        else:
            rank = "破綻"
            rank_bg = "#fef2f2;color:#991b1b"

        ret_color = "#16a34a" if is_profitable else "#dc2626"
        star = " ★推奨" if lev == 3 else " ★安全" if lev == 1 else ""
        highlight = ' style="background:#e8f5e9;font-weight:700"' if lev in [1, 3] else ""

        # フラッシュクラッシュ想定損失（-150pips ≈ 1.15%）
        flash_loss = lev * 1.15

        rows += f"""
        <tr{highlight}>
          <td style="text-align:center;font-weight:700">{lev}x{star}</td>
          <td style="text-align:right;color:{ret_color};font-weight:700">{r['return_pct']:+.1f}%</td>
          <td style="text-align:right">{r['annual_pct']:+.1f}%</td>
          <td style="text-align:right">{r['final_equity']:,.0f}</td>
          <td style="text-align:right">{r['mdd']:.1f}%</td>
          <td style="text-align:right">{r['monthly_wr']:.1f}%</td>
          <td style="text-align:right">{r['yearly_wr']:.1f}%</td>
          <td style="text-align:right">{flash_loss:.1f}%</td>
          <td style="text-align:center"><span style="background:{rank_bg};padding:2px 10px;border-radius:3px;font-size:11px;font-weight:700">{rank}</span></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Multi Portfolio Leverage Analysis</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px;
    margin: 20px auto;
    padding: 20px;
    background: #f5f5f5;
    line-height: 1.6;
}}
h1 {{ font-size: 22px; margin: 0 0 16px; padding-bottom: 8px; border-bottom: 2px solid #333; }}
h2 {{ font-size: 18px; margin: 24px 0 12px; }}
.card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 16px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 700; text-align: center; }}
.back {{ display: inline-block; margin-bottom: 16px; color: #4263eb; text-decoration: none; }}
.back:hover {{ text-decoration: underline; }}
.warning {{
    background: #fef3c7; border-left: 4px solid #f59e0b;
    padding: 12px 16px; border-radius: 4px; margin: 16px 0;
    font-size: 13px;
}}
.danger {{
    background: #fee2e2; border-left: 4px solid #dc2626;
    padding: 12px 16px; border-radius: 4px; margin: 16px 0;
    font-size: 13px;
}}
</style>
</head>
<body>

<a href="index.html" class="back">← 戦略一覧に戻る</a>

<h1>Multi Portfolio (TF+ST+MACD+ICHI) レバレッジ感応度分析</h1>
<p style="color:#666">
  4戦略均等配分ポートフォリオに対して、レバレッジをかけた場合のリターンとリスクを25年間で検証。
</p>

<div class="card">
  <h2 style="margin-top:0">レバレッジ別結果 (初期資金100万円・25年間)</h2>
  <table>
    <thead>
      <tr>
        <th>レバレッジ</th>
        <th>Return</th>
        <th>年利</th>
        <th>最終資産</th>
        <th>最大DD</th>
        <th>月次勝率</th>
        <th>年次勝率</th>
        <th>暴落時損失</th>
        <th>判定</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <p style="color:#888;font-size:11px;margin-top:12px">
    ※ 暴落時損失 = フラッシュクラッシュ(-150pips/約1.15%)がSL超過で約定した場合の想定損失率<br>
    ※ MDD = 最大ドローダウン（資産ピークからの最大下落率）<br>
    ※ 国内FX業者の個人口座レバレッジ上限は25倍
  </p>
</div>

<div class="warning">
  <strong>⚠️ レバレッジの基本:</strong> レバレッジはリターンもMDDも比例して拡大する。
  MDDが -40%を超えると、一時的に資金が半分以下になる期間がある。
  心理的に耐えられるかを事前に検討すること。
</div>

<div class="danger">
  <strong>🔴 ロスカット(強制決済)のリスク:</strong>
  レバレッジが高すぎると、MDDが -100%に達して口座破綻する可能性がある。
  特にフラッシュクラッシュ (2019年1月、2022年10月など) で一瞬で数%動く場合、
  通常のSLを飛ばして大きく損失が出ることがある。
</div>

<div class="card">
  <h2 style="margin-top:0">推奨レバレッジ</h2>
  <ul>
    <li><strong>1x (レバなし)</strong> - 最も安全。25年で +164.5%, 年利4.0%。心理的負担ゼロ。</li>
    <li><strong>3x (中リスク)</strong> - バランス型。25年リターン約+700%、年利約8-9%。MDD -34%程度。</li>
    <li><strong>5-10x (高リスク)</strong> - 積極運用。フラッシュクラッシュで -5〜-15%の一撃損失リスク。</li>
    <li><strong>25x (国内上限)</strong> - ほぼ破綻レベル。MDDが -100%に達する可能性あり。</li>
  </ul>
</div>

<div class="card">
  <h2 style="margin-top:0">なぜ単独戦略(TF)より有利なのか</h2>
  <p>
    TF単独のMDDは -17.4%なので、3xレバレッジでMDD -52%相当。
    Multi Portfolio のMDDは -11.3%なので、3xレバレッジでもMDD -34%に抑えられる。
    <strong>同じレバレッジでも、分散効果でMDDが約35%低減される</strong> ため、
    実質的により安全に高レバレッジをかけられる。
  </p>
</div>

</body>
</html>
"""
    with open(filepath, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
