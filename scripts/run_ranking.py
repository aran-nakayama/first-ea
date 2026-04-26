#!/usr/bin/env python3
"""全戦略を同一データ・同一指標で実行し、ソート可能な比較ランキングHTMLを出力する。

各戦略について以下の指標を統一して算出:
- トレード数 / 勝率 (per-trade) / トレードPF
- 月次PF (sum positive months / |sum negative months|)
- MDD / Return / 年利 / Sharpe
- 月次勝率 / 年次勝率
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtesting import Backtest
from config.settings import INITIAL_EQUITY, COMMISSION

from strategy.trend_follow import prepare_data as prep_tf, TrendFollowStrategy
from strategy.supertrend import prepare_data as prep_st, SupertrendStrategy
from strategy.macd_momentum import prepare_data as prep_macd, MACDMomentumStrategy
from strategy.ichimoku import prepare_data as prep_ichi, IchimokuStrategy
from strategy.volatility_breakout import prepare_data as prep_vb, VolatilityBreakoutStrategy
from strategy.asian_breakout import prepare_data as prep_ab, AsianBreakoutStrategy
from strategy.rsi_mean_reversion import prepare_data as prep_rsi, RSIMeanReversionStrategy


SPREAD_PIPS = 0.4
AVG_PRICE = 130.0
SPREAD_RATIO = SPREAD_PIPS * 0.01 / AVG_PRICE

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "usdjpy_h1_25years.csv",
)


# ============================================================
# 個別戦略の最適化済みパラメータ
# ============================================================
STRATEGY_SPECS = {
    "tf": {
        "name": "Trend Follow (Donchian)",
        "report": "report_trend_follow.html",
        "color": "#4263eb",
        "prep": lambda df: prep_tf(df.copy(), entry_period=15, exit_period=15),
        "klass": TrendFollowStrategy,
        "params": dict(entry_period=15, exit_period=15, atr_sl_mult=3.0,
                       atr_trail_mult=3.0, use_ma_filter=0, max_hold_hours=72),
    },
    "st": {
        "name": "Supertrend",
        "report": "report_supertrend.html",
        "color": "#e03131",
        "prep": lambda df: prep_st(df.copy(), period=10, multiplier=2.0),
        "klass": SupertrendStrategy,
        "params": dict(st_period=10, st_multiplier=2.0,
                       atr_sl_mult=2.0, atr_trail_mult=3.0, max_hold_hours=72),
    },
    "macd": {
        "name": "MACD Momentum",
        "report": "report_macd.html",
        "color": "#f59f00",
        "prep": lambda df: prep_macd(df.copy(), fast_period=12, slow_period=26, signal_period=12),
        "klass": MACDMomentumStrategy,
        "params": dict(atr_sl_mult=2.5, atr_trail_mult=2.5, use_ma_filter=0),
    },
    "ichi": {
        "name": "Ichimoku Cloud",
        "report": "report_ichimoku.html",
        "color": "#9333ea",
        "prep": lambda df: prep_ichi(df.copy(), tenkan_period=9, kijun_period=26, senkou_b_period=52),
        "klass": IchimokuStrategy,
        "params": dict(atr_sl_mult=2.0, atr_trail_mult=3.0,
                       use_tenkan_kijun=1, max_hold_hours=72),
    },
    "vb": {
        "name": "Volatility Breakout",
        "report": "report_combined.html",  # VB単独レポートは無いのでcombinedへ
        "color": "#16a34a",
        "prep": lambda df: prep_vb(df.copy(), lookback=24, atr_period=14),
        "klass": VolatilityBreakoutStrategy,
        "params": dict(k_factor=0.3, atr_sl_mult=2.0, atr_tp_mult=3.0,
                       max_hold_hours=24, adx_min=15),
    },
    "ab": {
        "name": "Asian Breakout (25年最適化)",
        "report": "report_asian_breakout.html",
        "color": "#0ea5e9",
        "prep": lambda df: prep_ab(df.copy()),
        "klass": AsianBreakoutStrategy,
        "params": dict(sl_pips=80, tp_pips=120, session_start=8,
                       session_end=14, close_hour=21, min_range=0.10, max_range=2.0),
    },
    "rsi": {
        "name": "RSI Mean Reversion",
        "report": None,
        "color": "#a78bfa",
        "prep": lambda df: prep_rsi(df.copy(), rsi_period=14, bb_period=20, bb_std=2.0),
        "klass": RSIMeanReversionStrategy,
        "params": dict(rsi_oversold=25, rsi_overbought=75,
                       atr_sl_mult=2.0, atr_tp_mult=1.5, max_hold_hours=24,
                       vol_filter=2.0, use_bb_tp=1),
    },
}


# ============================================================
# 単一戦略を実行 (cash=フル)
# ============================================================
def run_single(spec, cash=INITIAL_EQUITY):
    df = spec["prep"]
    df = df(pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime"))
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    bt = Backtest(
        df, spec["klass"],
        cash=cash, commission=COMMISSION,
        spread=SPREAD_RATIO, exclusive_orders=True,
    )
    stats = bt.run(**spec["params"])
    return stats


def run_single_with_df(spec, df_raw, cash=INITIAL_EQUITY, margin=1.0):
    df = spec["prep"](df_raw)
    bt = Backtest(
        df, spec["klass"],
        cash=cash, commission=COMMISSION,
        spread=SPREAD_RATIO, exclusive_orders=True, margin=margin,
    )
    return bt.run(**spec["params"])


# ============================================================
# ポートフォリオ実行
# ============================================================
def run_portfolio(df_raw, allocs):
    """allocs: dict {key: ratio}, 合計1.0想定。
    各戦略を cash=INITIAL_EQUITY*ratio で実行し、equityを合算する。
    トレードリストもまとめて返す。
    """
    eq_list = []
    trades_list = []
    for key, ratio in allocs.items():
        spec = STRATEGY_SPECS[key]
        stats = run_single_with_df(spec, df_raw, cash=INITIAL_EQUITY * ratio)
        eq = stats["_equity_curve"]["Equity"].resample("D").last().dropna()
        eq_list.append(eq)
        # trades はそのまま (PnL は ratio比例の絶対額)
        if "_trades" in stats:
            trades_list.append(stats["_trades"])

    # equity合算
    combined = eq_list[0].copy()
    for e in eq_list[1:]:
        idx = combined.index.intersection(e.index)
        combined = combined.loc[idx] + e.loc[idx]

    # trades結合
    all_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    return combined, all_trades


# ============================================================
# 指標計算 (equity series + trades から統一指標)
# ============================================================
def compute_metrics(equity, trades):
    """共通の指標を計算する。equityは日次、tradesはbacktesting.pyのtrades DataFrame。"""
    # Return / 年利
    init = equity.iloc[0]
    final = equity.iloc[-1]
    total_ret = (final / init - 1) * 100
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    annual = ((final / init) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    mdd = dd.min()

    # Sharpe (日次リターンベース、年率化)
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # 月次P&L
    monthly = equity.resample("ME").last().diff().dropna()
    m_total = len(monthly)
    m_wins = int((monthly > 0).sum())
    m_wr = m_wins / m_total * 100 if m_total > 0 else 0
    m_pos = monthly[monthly > 0].sum()
    m_neg = abs(monthly[monthly < 0].sum())
    monthly_pf = (m_pos / m_neg) if m_neg > 0 else float("inf")

    # 年次勝率
    yearly = equity.resample("YE").last().pct_change().dropna() * 100
    y_total = len(yearly)
    y_wins = int((yearly > 0).sum())
    y_wr = y_wins / y_total * 100 if y_total > 0 else 0

    # トレード指標
    n_trades = 0
    trade_wr = 0
    trade_pf = 0
    if trades is not None and len(trades) > 0:
        n_trades = len(trades)
        wins = trades[trades["PnL"] > 0]
        losses = trades[trades["PnL"] <= 0]
        trade_wr = len(wins) / n_trades * 100
        w_sum = wins["PnL"].sum()
        l_sum = abs(losses["PnL"].sum())
        trade_pf = (w_sum / l_sum) if l_sum > 0 else float("inf")

    return {
        "trades": n_trades,
        "trade_wr": trade_wr,
        "trade_pf": trade_pf,
        "monthly_pf": monthly_pf,
        "mdd": mdd,
        "return_pct": total_ret,
        "annual_pct": annual,
        "sharpe": sharpe,
        "monthly_wr": m_wr,
        "yearly_wr": y_wr,
        "m_total": m_total,
        "y_total": y_total,
    }


# ============================================================
# 判定ロジック
# ============================================================
def judge(m):
    if m["return_pct"] <= 0 or m["mdd"] < -50:
        return ("負け", "lose")
    if m["monthly_wr"] >= 58 and m["mdd"] > -15 and m["return_pct"] > 100:
        return ("★最優秀", "best")
    if m["monthly_wr"] >= 55 and m["return_pct"] > 100 and m["mdd"] > -25:
        return ("優良", "win")
    if m["return_pct"] > 50 and m["mdd"] > -30:
        return ("勝ち", "win")
    if m["return_pct"] > 0:
        return ("微妙", "even")
    return ("負け", "lose")


# ============================================================
# 全戦略を実行
# ============================================================
def main():
    print("="*60)
    print("全戦略 統一ランキング 比較")
    print("="*60)

    print("\nデータ読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"  期間: {df_raw.index[0]} ~ {df_raw.index[-1]} ({len(df_raw)}本)")

    results = []

    # === ポートフォリオ ===
    print("\n[1] Multi Portfolio (TF+ST+MACD+ICHI 各25%) 実行中...")
    eq, trd = run_portfolio(df_raw, {"tf": 0.25, "st": 0.25, "macd": 0.25, "ichi": 0.25})
    m = compute_metrics(eq, trd)
    rank, rank_class = judge(m)
    results.append({
        "key": "multi",
        "name": "Multi (TF+ST+MACD+ICHI)",
        "type": "portfolio",
        "report": "report_multi_portfolio.html",
        "color": "#16a34a",
        "params": "4戦略均等配分 (各25%)",
        "rank": rank, "rank_class": rank_class, **m,
    })
    print(f"  Return={m['return_pct']:+.1f}% MDD={m['mdd']:.1f}% 月次WR={m['monthly_wr']:.1f}% PF(月次)={m['monthly_pf']:.2f}")

    print("\n[2] Combined Portfolio (TF65% + VB35%) 実行中...")
    eq, trd = run_portfolio(df_raw, {"tf": 0.65, "vb": 0.35})
    m = compute_metrics(eq, trd)
    rank, rank_class = judge(m)
    results.append({
        "key": "combined",
        "name": "Combined (TF + VB)",
        "type": "portfolio",
        "report": "report_combined.html",
        "color": "#4263eb",
        "params": "TF 65% + VB 35%",
        "rank": rank, "rank_class": rank_class, **m,
    })
    print(f"  Return={m['return_pct']:+.1f}% MDD={m['mdd']:.1f}% 月次WR={m['monthly_wr']:.1f}% PF(月次)={m['monthly_pf']:.2f}")

    # === 単一戦略 ===
    for i, key in enumerate(["tf", "st", "macd", "ichi", "vb", "ab", "rsi"], 3):
        spec = STRATEGY_SPECS[key]
        print(f"\n[{i}] {spec['name']} 実行中...")
        stats = run_single_with_df(spec, df_raw, cash=INITIAL_EQUITY)
        eq = stats["_equity_curve"]["Equity"].resample("D").last().dropna()
        trd = stats.get("_trades", pd.DataFrame())
        m = compute_metrics(eq, trd)
        rank, rank_class = judge(m)
        results.append({
            "key": key,
            "name": spec["name"],
            "type": "single",
            "report": spec["report"],
            "color": spec["color"],
            "params": ", ".join(f"{k}={v}" for k, v in list(spec["params"].items())[:3]),
            "rank": rank, "rank_class": rank_class, **m,
        })
        print(f"  Trades={m['trades']} WR={m['trade_wr']:.1f}% PF={m['trade_pf']:.2f} "
              f"Return={m['return_pct']:+.1f}% MDD={m['mdd']:.1f}% 月次WR={m['monthly_wr']:.1f}%")

    # === HTMLレポート出力 ===
    print("\n\nHTMLランキング生成中...")
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "report_ranking.html",
    )
    _write_ranking_html(output_path, results)
    print(f"  → {output_path}")

    # === コンソールにも表示 ===
    print("\n\n" + "="*100)
    print(f"  {'戦略':<35} {'Trd':>5} {'WR':>6} {'PF':>5} {'M-PF':>5} "
          f"{'Sharpe':>7} {'MDD':>7} {'Return':>9} {'年利':>6} {'M-WR':>6} {'Y-WR':>6} {'判定':>8}")
    print("  " + "-"*98)
    for r in sorted(results, key=lambda x: -x["monthly_wr"]):
        print(f"  {r['name'][:34]:<35} {r['trades']:>5} {r['trade_wr']:>5.1f}% "
              f"{r['trade_pf']:>4.2f} {r['monthly_pf']:>4.2f} "
              f"{r['sharpe']:>6.2f} {r['mdd']:>6.1f}% "
              f"{r['return_pct']:>+8.1f}% {r['annual_pct']:>+5.1f}% "
              f"{r['monthly_wr']:>5.1f}% {r['yearly_wr']:>5.1f}% {r['rank']:>8}")

    print("\n完了")


def _fmt_pf(v):
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


def _write_ranking_html(path, results):
    """ソート可能なJSテーブルでランキングHTMLを生成する。"""
    rows_json = []
    for r in results:
        rows_json.append({
            "name": r["name"],
            "type": r["type"],
            "report": r["report"] or "",
            "params": r["params"],
            "trades": r["trades"],
            "trade_wr": round(r["trade_wr"], 1),
            "trade_pf": round(r["trade_pf"], 2) if r["trade_pf"] != float("inf") else 999,
            "monthly_pf": round(r["monthly_pf"], 2) if r["monthly_pf"] != float("inf") else 999,
            "sharpe": round(r["sharpe"], 2),
            "mdd": round(r["mdd"], 1),
            "return_pct": round(r["return_pct"], 1),
            "annual_pct": round(r["annual_pct"], 1),
            "monthly_wr": round(r["monthly_wr"], 1),
            "yearly_wr": round(r["yearly_wr"], 1),
            "rank": r["rank"],
            "rank_class": r["rank_class"],
        })
    rows_js = json.dumps(rows_json, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strategy Ranking - 全戦略統一比較</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  padding: 24px;
}
h1 { font-size: 22px; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 13px; margin-bottom: 16px; }
.back { display: inline-block; margin-bottom: 12px; color: #4dabf7; text-decoration: none; font-size: 13px; }
.back:hover { text-decoration: underline; }
.legend { color: #888; font-size: 12px; margin-bottom: 12px; line-height: 1.7; }
.legend code { background: #1a1d27; padding: 2px 6px; border-radius: 3px; color: #e0e0e0; }

.controls { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.controls label { font-size: 12px; color: #888; }
.controls select {
  background: #1a1d27; color: #e0e0e0; border: 1px solid #2a2d37;
  padding: 6px 10px; border-radius: 4px; font-size: 13px; cursor: pointer;
}
.controls .reset {
  background: #1a1d27; color: #4dabf7; border: 1px solid #2a2d37;
  padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer;
}
.controls .reset:hover { background: #243043; }

table { width: 100%; border-collapse: collapse; font-size: 13px; background: #1a1d27; border-radius: 8px; overflow: hidden; }
thead { background: #20232f; }
th {
  padding: 10px 8px; text-align: center; font-weight: 600; font-size: 11px;
  text-transform: uppercase; color: #aaa; cursor: pointer; user-select: none;
  border-bottom: 2px solid #2a2d37; white-space: nowrap; position: sticky; top: 0;
}
th:hover { background: #2a2d37; color: #4dabf7; }
th.sorted-asc::after { content: " ▲"; color: #4dabf7; }
th.sorted-desc::after { content: " ▼"; color: #4dabf7; }
td {
  padding: 10px 8px; border-bottom: 1px solid #2a2d37; text-align: right;
  white-space: nowrap;
}
td.name { text-align: left; font-weight: 600; }
td.name a { color: #e0e0e0; text-decoration: none; }
td.name a:hover { color: #4dabf7; text-decoration: underline; }
td.center { text-align: center; }
tbody tr:hover { background: #20232f; }
tbody tr.rank-best { background: rgba(34, 197, 94, 0.15); }
tbody tr.rank-best:hover { background: rgba(34, 197, 94, 0.25); }

.tag {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 700;
}
.tag-best { background: rgba(96, 165, 250, 0.2); color: #60a5fa; }
.tag-win { background: rgba(74, 222, 128, 0.15); color: #4ade80; }
.tag-even { background: rgba(251, 191, 36, 0.15); color: #fbbf24; }
.tag-lose { background: rgba(248, 113, 113, 0.15); color: #f87171; }

.positive { color: #4ade80; }
.negative { color: #f87171; }
.dim { color: #666; }
.type-badge {
  display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-size: 10px; margin-left: 6px; vertical-align: middle;
}
.type-portfolio { background: #1a2a4a; color: #60a5fa; }
.type-single { background: #2a2a2a; color: #888; }

.help-card {
  background: #1a1d27; border: 1px solid #2a2d37; border-radius: 8px;
  padding: 16px; margin-top: 20px; font-size: 12px; line-height: 1.7; color: #aaa;
}
.help-card strong { color: #e0e0e0; }
.help-card ul { margin: 6px 0 6px 20px; }
</style>
</head>
<body>

<a href="index.html" class="back">← 戦略一覧に戻る</a>

<h1>Strategy Ranking - 全戦略統一比較</h1>
<p class="subtitle">USD/JPY H1 / 2000-2024 (24.6年) / 初期資金100万円 / スプレッド0.4pips込み<br>
列ヘッダーをクリックでソート、もう一度クリックで降順。複数戦略を同条件で比較できます。</p>

<div class="controls">
  <label>並び替え:</label>
  <select id="sortPreset">
    <option value="monthly_wr-desc">月次勝率が高い順 (毎月勝ちたい)</option>
    <option value="return_pct-desc">Return が大きい順 (利益最大化)</option>
    <option value="annual_pct-desc">年利が高い順</option>
    <option value="mdd-desc">MDDが小さい順 (リスク最小)</option>
    <option value="trade_pf-desc">トレードPF が高い順</option>
    <option value="monthly_pf-desc">月次PF が高い順</option>
    <option value="sharpe-desc">Sharpe が高い順</option>
    <option value="trade_wr-desc">勝率(トレード)が高い順</option>
    <option value="yearly_wr-desc">年次勝率が高い順</option>
  </select>
  <button class="reset" onclick="filterType('all')">全て</button>
  <button class="reset" onclick="filterType('portfolio')">ポートフォリオのみ</button>
  <button class="reset" onclick="filterType('single')">単一戦略のみ</button>
</div>

<table id="rankingTable">
  <thead>
    <tr>
      <th data-key="name" class="name-col">戦略</th>
      <th data-key="trades">トレード</th>
      <th data-key="trade_wr">勝率</th>
      <th data-key="trade_pf">PF</th>
      <th data-key="monthly_pf">月次PF</th>
      <th data-key="sharpe">Sharpe</th>
      <th data-key="mdd">MDD</th>
      <th data-key="return_pct">Return</th>
      <th data-key="annual_pct">年利</th>
      <th data-key="monthly_wr">月次勝率</th>
      <th data-key="yearly_wr">年次勝率</th>
      <th data-key="rank">判定</th>
    </tr>
  </thead>
  <tbody id="rankingBody"></tbody>
</table>

<div class="help-card">
  <strong>指標の見方:</strong>
  <ul>
    <li><strong>勝率</strong> - トレード単位の勝率。トレンドフォロー系は30-40%でも長期で勝てる(損小利大)</li>
    <li><strong>PF</strong> - Profit Factor (総利益÷総損失)。1.0で損益分岐、1.1以上が許容、1.3以上で優秀</li>
    <li><strong>月次PF</strong> - 月単位の総勝ち額÷総負け額。ポートフォリオ間で比較しやすい統一指標</li>
    <li><strong>Sharpe</strong> - リスク調整後リターン。1.0以上で良好、0.5でも実用</li>
    <li><strong>MDD</strong> - 最大ドローダウン (資産ピークからの最大下落率)。-15%以下を推奨</li>
    <li><strong>月次勝率</strong> - 月次P&Lがプラスだった月の割合。<strong>毎月プラスを目指すならこれが最重要</strong></li>
    <li><strong>年次勝率</strong> - 年次P&Lがプラスだった年の割合</li>
  </ul>
  <p style="margin-top:8px"><strong>判定ルール:</strong>
    <span class="tag tag-best">★最優秀</span> 月次勝率58%以上 + MDD&gt;-15% + Return&gt;100%
    &nbsp; <span class="tag tag-win">優良</span> 月次勝率55%以上 + Return&gt;100%
    &nbsp; <span class="tag tag-win">勝ち</span> Return&gt;50% + MDD&gt;-30%
    &nbsp; <span class="tag tag-even">微妙</span> 利益あり、リスク注意
    &nbsp; <span class="tag tag-lose">負け</span> 損失 or 致命的MDD
  </p>
</div>

<script>
const DATA = __ROWS_JSON__;

let currentSort = { key: 'monthly_wr', dir: 'desc' };
let currentFilter = 'all';

function fmt(v, key) {
  if (v === 999) return '∞';
  if (v === null || v === undefined) return '-';
  if (key === 'trades') return v.toLocaleString();
  if (key === 'trade_wr' || key === 'monthly_wr' || key === 'yearly_wr') return v.toFixed(1) + '%';
  if (key === 'mdd') return v.toFixed(1) + '%';
  if (key === 'return_pct' || key === 'annual_pct') return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
  if (key === 'trade_pf' || key === 'monthly_pf' || key === 'sharpe') return v.toFixed(2);
  return v;
}

function colorClass(v, key) {
  if (key === 'return_pct' || key === 'annual_pct' || key === 'sharpe') {
    return v > 0 ? 'positive' : v < 0 ? 'negative' : '';
  }
  if (key === 'monthly_pf' || key === 'trade_pf') {
    return v >= 1.1 ? 'positive' : v < 1.0 ? 'negative' : '';
  }
  if (key === 'mdd') {
    return v > -15 ? 'positive' : v < -30 ? 'negative' : '';
  }
  if (key === 'monthly_wr') {
    return v >= 55 ? 'positive' : v < 50 ? 'negative' : '';
  }
  if (key === 'yearly_wr') {
    return v >= 75 ? 'positive' : v < 60 ? 'negative' : '';
  }
  if (key === 'trade_wr') {
    return v >= 50 ? 'positive' : '';
  }
  return '';
}

function render() {
  const tbody = document.getElementById('rankingBody');
  let rows = DATA.slice();
  if (currentFilter !== 'all') {
    rows = rows.filter(r => r.type === currentFilter);
  }
  rows.sort((a, b) => {
    let av = a[currentSort.key];
    let bv = b[currentSort.key];
    if (currentSort.key === 'name' || currentSort.key === 'rank') {
      av = String(av); bv = String(bv);
      return currentSort.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    }
    return currentSort.dir === 'asc' ? av - bv : bv - av;
  });

  tbody.innerHTML = rows.map((r, idx) => {
    const trClass = r.rank_class === 'best' ? 'rank-best' : '';
    const reportLink = r.report
      ? `<a href="${r.report}">${r.name}</a>`
      : r.name;
    const typeBadge = r.type === 'portfolio'
      ? '<span class="type-badge type-portfolio">PORTFOLIO</span>'
      : '<span class="type-badge type-single">SINGLE</span>';
    return `<tr class="${trClass}">
      <td class="name">${reportLink}${typeBadge}<br><span class="dim" style="font-size:11px">${r.params}</span></td>
      <td>${fmt(r.trades, 'trades')}</td>
      <td class="${colorClass(r.trade_wr, 'trade_wr')}">${fmt(r.trade_wr, 'trade_wr')}</td>
      <td class="${colorClass(r.trade_pf, 'trade_pf')}">${fmt(r.trade_pf, 'trade_pf')}</td>
      <td class="${colorClass(r.monthly_pf, 'monthly_pf')}">${fmt(r.monthly_pf, 'monthly_pf')}</td>
      <td class="${colorClass(r.sharpe, 'sharpe')}">${fmt(r.sharpe, 'sharpe')}</td>
      <td class="${colorClass(r.mdd, 'mdd')}">${fmt(r.mdd, 'mdd')}</td>
      <td class="${colorClass(r.return_pct, 'return_pct')}">${fmt(r.return_pct, 'return_pct')}</td>
      <td class="${colorClass(r.annual_pct, 'annual_pct')}">${fmt(r.annual_pct, 'annual_pct')}</td>
      <td class="${colorClass(r.monthly_wr, 'monthly_wr')}">${fmt(r.monthly_wr, 'monthly_wr')}</td>
      <td class="${colorClass(r.yearly_wr, 'yearly_wr')}">${fmt(r.yearly_wr, 'yearly_wr')}</td>
      <td class="center"><span class="tag tag-${r.rank_class}">${r.rank}</span></td>
    </tr>`;
  }).join('');

  // ヘッダーのソートアイコン更新
  document.querySelectorAll('th').forEach(th => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === currentSort.key) {
      th.classList.add(currentSort.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
    }
  });
}

// ヘッダークリックでソート
document.querySelectorAll('th').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (currentSort.key === key) {
      currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
      currentSort.key = key;
      // 数値系列はデフォルト降順
      currentSort.dir = (key === 'mdd') ? 'desc' : 'desc';
    }
    document.getElementById('sortPreset').value = `${key}-${currentSort.dir}`;
    render();
  });
});

// プリセット選択
document.getElementById('sortPreset').addEventListener('change', (e) => {
  const [key, dir] = e.target.value.split('-');
  currentSort = { key, dir };
  render();
});

function filterType(t) {
  currentFilter = t;
  render();
}

render();
</script>
</body>
</html>
"""
    html = html.replace("__ROWS_JSON__", rows_js)
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
