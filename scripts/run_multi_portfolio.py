#!/usr/bin/env python3
"""Multi-Strategy Portfolio: 4つ以上の戦略を組み合わせて月次勝率を最大化"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import re
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtesting import Backtest
from config.settings import INITIAL_EQUITY, COMMISSION

from strategy.trend_follow import prepare_data as prep_tf, TrendFollowStrategy
from strategy.volatility_breakout import prepare_data as prep_vb, VolatilityBreakoutStrategy
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


# 各戦略の最適パラメータ（事前の最適化結果から）
STRATEGIES = {
    "TF": {
        "name": "Trend Follow",
        "prep": lambda df: prep_tf(df.copy(), entry_period=15, exit_period=15),
        "strategy": TrendFollowStrategy,
        "params": dict(entry_period=15, exit_period=15, atr_sl_mult=3.0,
                       atr_trail_mult=3.0, use_ma_filter=0, max_hold_hours=72),
    },
    "VB": {
        "name": "Volatility Breakout",
        "prep": lambda df: prep_vb(df.copy()),
        "strategy": VolatilityBreakoutStrategy,
        "params": dict(k_factor=0.3, atr_sl_mult=2.0, atr_tp_mult=3.0,
                       max_hold_hours=24, adx_min=15),
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


def _pf_from_trades(trades):
    if trades is None or len(trades) == 0:
        return 0
    w = trades[trades["PnL"] > 0]["PnL"].sum()
    l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
    return w / l if l > 0 else float("inf")


def _run_bt(df, strategy_class, cash, **params):
    bt = Backtest(
        df, strategy_class,
        cash=cash, commission=COMMISSION,
        spread=SPREAD_RATIO, exclusive_orders=True,
    )
    return bt.run(**params), bt


def _equity_series(stats):
    eq = stats["_equity_curve"]["Equity"]
    return eq.resample("D").last().dropna()


def _monthly_pnl(equity):
    monthly = equity.resample("ME").last().dropna()
    return monthly.diff().dropna()


def _max_consec(series, cond):
    m, c = 0, 0
    for v in series:
        if cond(v):
            c += 1
            m = max(m, c)
        else:
            c = 0
    return m


def _evaluate_portfolio(equities_with_alloc):
    """各戦略のエクイティ曲線 × 配分 を合算してメトリクスを計算する。"""
    combined = None
    for eq, alloc in equities_with_alloc:
        scaled = eq * alloc  # eq は絶対額、allocはスケーリング係数
        if combined is None:
            combined = scaled.copy()
        else:
            idx = combined.index.intersection(scaled.index)
            combined = combined.loc[idx] + scaled.loc[idx]

    monthly = _monthly_pnl(combined)
    m_total = len(monthly)
    m_wins = (monthly > 0).sum()
    m_wr = m_wins / m_total * 100 if m_total > 0 else 0
    m_streak = _max_consec(monthly, lambda x: x <= 0)

    peak = combined.cummax()
    dd = (combined - peak) / peak * 100
    mdd = dd.min()

    init_val = combined.iloc[0]
    total_ret = (combined.iloc[-1] / init_val - 1) * 100
    years = (combined.index[-1] - combined.index[0]).days / 365.25
    annual = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    return {
        "monthly_wr": m_wr,
        "monthly_streak": m_streak,
        "mdd": mdd,
        "return_pct": total_ret,
        "annual_pct": annual,
        "final_equity": combined.iloc[-1],
        "combined_equity": combined,
        "monthly_pnl": monthly,
    }


def main():
    print("="*60)
    print("Multi-Strategy Portfolio Search")
    print("="*60)

    print("\nUSD/JPY H1 25年データを読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"データ: {len(df_raw)}本")

    # === 各戦略を個別にバックテストし、エクイティ曲線を取得 ===
    print("\n--- 各戦略の個別バックテスト ---")
    individual_results = {}

    for key, spec in STRATEGIES.items():
        print(f"  {spec['name']}...", end=" ", flush=True)
        df = spec["prep"](df_raw)
        # 100% 資金で実行し、相対的な曲線を取得
        stats, _ = _run_bt(df, spec["strategy"],
                           cash=INITIAL_EQUITY, **spec["params"])
        equity = _equity_series(stats)
        pf = _pf_from_trades(stats.get("_trades"))
        ret = stats["Return [%]"]
        monthly = _monthly_pnl(equity)
        m_wr = (monthly > 0).sum() / len(monthly) * 100 if len(monthly) > 0 else 0

        individual_results[key] = {
            "name": spec["name"],
            "stats": stats,
            "equity": equity,
            "pf": pf,
            "return_pct": ret,
            "mdd": stats["Max. Drawdown [%]"],
            "monthly_wr": m_wr,
            "monthly_streak": _max_consec(monthly, lambda x: x <= 0),
            "trades": stats["# Trades"],
        }

        print(f"Return={ret:+.1f}% PF={pf:.2f} "
              f"MDD={stats['Max. Drawdown [%]']:.1f}% 月次WR={m_wr:.1f}%")

    # === 均等配分のポートフォリオ（全戦略） ===
    print("\n\n--- 均等配分ポートフォリオ ---")

    # 全戦略均等配分
    all_keys = list(STRATEGIES.keys())
    n = len(all_keys)

    # 各戦略の基準エクイティを揃える：初期資金 * (1/n) として計算
    # 戦略Sのエクイティeq_S: 初期100万で実行 → eq_S * (1/n) が配分後の資産
    equities_even = [
        (individual_results[k]["equity"], 1.0 / n)
        for k in all_keys
    ]
    even = _evaluate_portfolio(equities_even)
    print(f"  全戦略均等配分 ({n}戦略×{100/n:.0f}%):")
    print(f"    Return={even['return_pct']:+.1f}% 年利={even['annual_pct']:+.1f}%")
    print(f"    MDD={even['mdd']:.1f}% 月次勝率={even['monthly_wr']:.1f}% "
          f"最大連敗={even['monthly_streak']}ヶ月")

    # === 全2〜4戦略の組み合わせを均等配分で試す ===
    print("\n\n--- 2〜4戦略の組み合わせ探索 ---")

    best_mwr = 0
    best_combo = None
    best_result = None
    all_combos = []

    for r in [2, 3, 4, 5]:
        for combo in itertools.combinations(all_keys, r):
            alloc = 1.0 / r
            eqs = [(individual_results[k]["equity"], alloc) for k in combo]
            result = _evaluate_portfolio(eqs)
            all_combos.append((combo, result))
            if result["monthly_wr"] > best_mwr:
                best_mwr = result["monthly_wr"]
                best_combo = combo
                best_result = result

    # 上位10を月次勝率でソート
    all_combos.sort(key=lambda x: x[1]["monthly_wr"], reverse=True)

    print(f"\n  月次勝率 TOP 10 組み合わせ (均等配分):")
    print(f"  {'順位':<4} {'戦略構成':<24} {'月次WR':>8} {'最大連敗':>10} {'Return':>10} {'MDD':>8}")
    print(f"  {'-'*72}")
    for rank, (combo, res) in enumerate(all_combos[:10], 1):
        combo_str = " + ".join(combo)
        star = " ★" if rank == 1 else ""
        print(f"  {rank:<4} {combo_str:<24} {res['monthly_wr']:>7.1f}% "
              f"{res['monthly_streak']:>8}ヶ月 {res['return_pct']:>+9.1f}% "
              f"{res['mdd']:>7.1f}%{star}")

    # === ベスト組み合わせの詳細 ===
    print(f"\n\n{'='*60}")
    print(f"  BEST ポートフォリオ: {' + '.join(best_combo)}")
    print(f"{'='*60}")
    print(f"\n  配分: 各戦略 {100/len(best_combo):.0f}%")
    print(f"  Return     : {best_result['return_pct']:+.1f}%")
    print(f"  年利       : {best_result['annual_pct']:+.1f}%")
    print(f"  MDD        : {best_result['mdd']:.1f}%")
    print(f"  月次勝率   : {best_result['monthly_wr']:.1f}%")
    print(f"  最大連敗   : {best_result['monthly_streak']}ヶ月")

    # 年次分析
    yearly = best_result["combined_equity"].resample("YE").last().pct_change().dropna() * 100
    yearly_wins = (yearly > 0).sum()
    print(f"\n  年次勝率   : {yearly_wins}/{len(yearly)} ({yearly_wins/len(yearly)*100:.1f}%)")

    print(f"\n  年次パフォーマンス:")
    print(f"  {'年':>6}  {'Return':>10}  判定")
    for year in yearly.index:
        r = yearly.loc[year]
        status = "勝ち" if r > 0 else "負け"
        print(f"  {year.year:>6}  {r:>+9.1f}%  {status}")

    # === レポート生成 ===
    print("\nHTMLレポート生成中...")
    _gen_multi_report(
        "report_multi_portfolio.html",
        individual_results, all_combos[:10], best_combo, best_result,
    )
    print("レポート出力: report_multi_portfolio.html")


def _gen_multi_report(filepath, individual_results, top_combos,
                      best_combo, best_result):
    rows_individual = ""
    for key, r in individual_results.items():
        ret_color = "#16a34a" if r["return_pct"] > 0 else "#dc2626"
        rows_individual += f"""
        <tr>
          <td><strong>{r['name']}</strong> ({key})</td>
          <td style="text-align:right;color:{ret_color};font-weight:700">{r['return_pct']:+.1f}%</td>
          <td style="text-align:right">{r['pf']:.2f}</td>
          <td style="text-align:right">{r['mdd']:.1f}%</td>
          <td style="text-align:right">{r['monthly_wr']:.1f}%</td>
          <td style="text-align:right">{r['monthly_streak']}ヶ月</td>
          <td style="text-align:right">{r['trades']}</td>
        </tr>
        """

    rows_combo = ""
    for rank, (combo, res) in enumerate(top_combos, 1):
        combo_str = " + ".join(combo)
        is_best = rank == 1
        ret_color = "#16a34a" if res["return_pct"] > 0 else "#dc2626"
        mwr_color = "#16a34a" if res["monthly_wr"] > 55 else "#666"
        bg = "background:#e8f5e9;font-weight:700" if is_best else ""
        star = " ★" if is_best else ""
        rows_combo += f"""
        <tr style="{bg}">
          <td style="text-align:center">{rank}{star}</td>
          <td>{combo_str} ({len(combo)}戦略)</td>
          <td style="text-align:right;color:{mwr_color};font-weight:700">{res['monthly_wr']:.1f}%</td>
          <td style="text-align:right">{res['monthly_streak']}ヶ月</td>
          <td style="text-align:right;color:{ret_color}">{res['return_pct']:+.1f}%</td>
          <td style="text-align:right">{res['annual_pct']:+.1f}%</td>
          <td style="text-align:right">{res['mdd']:.1f}%</td>
        </tr>
        """

    best_combo_str = " + ".join(best_combo)

    # 年次テーブル
    yearly = best_result["combined_equity"].resample("YE").last().pct_change().dropna() * 100
    yearly_wins = (yearly > 0).sum()
    yearly_rate = yearly_wins / len(yearly) * 100
    rows_yearly = ""
    for year in yearly.index:
        r = yearly.loc[year]
        status = "勝ち" if r > 0 else "負け"
        color = "#16a34a" if r > 0 else "#dc2626"
        status_bg = "#dcfce7;color:#166534" if r > 0 else "#fef2f2;color:#991b1b"
        rows_yearly += f"""
        <tr>
          <td style="text-align:center">{year.year}</td>
          <td style="text-align:right;color:{color};font-weight:700">{r:+.1f}%</td>
          <td style="text-align:center"><span style="background:{status_bg};padding:1px 8px;border-radius:3px;font-size:11px;font-weight:700">{status}</span></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Multi-Strategy Portfolio Search</title>
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
.highlight-box {{
    background: #fff; border: 2px solid #4dabf7; border-radius: 8px;
    padding: 20px; margin-bottom: 20px; text-align: center;
}}
.highlight-box .big {{ font-size: 36px; font-weight: 800; color: #16a34a; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 700; text-align: left; }}
.back {{ display: inline-block; margin-bottom: 16px; color: #4263eb; text-decoration: none; }}
.back:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<a href="index.html" class="back">← 戦略一覧に戻る</a>

<h1>Multi-Strategy Portfolio Search</h1>
<p style="color:#666">
  複数の戦略を組み合わせて月次勝率を最大化するポートフォリオ分析。
  2〜5戦略の全組み合わせを均等配分で試し、最高の月次勝率を達成する組み合わせを探索。
</p>

<div class="highlight-box">
  <div style="color:#666;font-size:14px">BEST ポートフォリオ (月次勝率最大)</div>
  <div style="font-weight:700;font-size:18px;margin:8px 0">{best_combo_str}</div>
  <div class="big">{best_result['monthly_wr']:.1f}%</div>
  <div style="color:#666;font-size:13px;margin-top:4px">
    月次勝率 | 最大連敗 {best_result['monthly_streak']}ヶ月 |
    Return {best_result['return_pct']:+.1f}% (年利 {best_result['annual_pct']:+.1f}%) |
    MDD {best_result['mdd']:.1f}% |
    年次勝率 {yearly_rate:.1f}% ({yearly_wins}/{len(yearly)}年)
  </div>
</div>

<h2>組み合わせ TOP 10 (月次勝率順)</h2>
<div class="card">
  <table>
    <thead>
      <tr>
        <th style="text-align:center">順位</th>
        <th>戦略構成</th>
        <th style="text-align:right">月次勝率</th>
        <th style="text-align:right">最大連敗</th>
        <th style="text-align:right">Return</th>
        <th style="text-align:right">年利</th>
        <th style="text-align:right">MDD</th>
      </tr>
    </thead>
    <tbody>
      {rows_combo}
    </tbody>
  </table>
  <p style="color:#888;font-size:11px;margin-top:8px">
    ※ すべて均等配分（例：3戦略なら各33%）。組み合わせは2〜5戦略のすべての組み合わせを探索。
  </p>
</div>

<h2>個別戦略の成績 (基準データ)</h2>
<div class="card">
  <table>
    <thead>
      <tr>
        <th>戦略</th>
        <th style="text-align:right">Return</th>
        <th style="text-align:right">PF</th>
        <th style="text-align:right">MDD</th>
        <th style="text-align:right">月次勝率</th>
        <th style="text-align:right">最大連敗</th>
        <th style="text-align:right">トレード</th>
      </tr>
    </thead>
    <tbody>
      {rows_individual}
    </tbody>
  </table>
</div>

<h2>BEST ポートフォリオ 年次パフォーマンス</h2>
<div class="card">
  <table>
    <thead>
      <tr>
        <th style="text-align:center">年</th>
        <th style="text-align:right">Return</th>
        <th style="text-align:center">判定</th>
      </tr>
    </thead>
    <tbody>
      {rows_yearly}
    </tbody>
  </table>
</div>

</body>
</html>
"""
    with open(filepath, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
