#!/usr/bin/env python3
"""Combined Portfolio: Trend Follow + Volatility Breakout
月次勝率を改善するためのポートフォリオ合成分析"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtesting import Backtest
from config.settings import INITIAL_EQUITY, COMMISSION

from strategy.trend_follow import (
    prepare_data as prepare_tf,
    TrendFollowStrategy,
)
from strategy.volatility_breakout import (
    prepare_data as prepare_vb,
    VolatilityBreakoutStrategy,
)


# === Trend Follow パラメータ (既存の最適値) ===
TF_PARAMS = dict(
    entry_period=15, exit_period=15,
    atr_sl_mult=3.0, atr_trail_mult=3.0,
    use_ma_filter=0, max_hold_hours=72,
)

SPREAD_PIPS = 0.4
AVG_PRICE = 130.0
SPREAD_RATIO = SPREAD_PIPS * 0.01 / AVG_PRICE

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "usdjpy_h1_25years.csv",
)


def _pf_from_trades(trades):
    if trades is not None and len(trades) > 0:
        w = trades[trades["PnL"] > 0]["PnL"].sum()
        l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
        return w / l if l > 0 else float("inf")
    return 0


def _run_bt(df, strategy_class, cash, spread=SPREAD_RATIO, **params):
    bt = Backtest(
        df, strategy_class,
        cash=cash, commission=COMMISSION,
        spread=spread, exclusive_orders=True,
    )
    stats = bt.run(**params)
    return stats, bt


def _equity_series(stats):
    equity = stats["_equity_curve"]["Equity"]
    daily = equity.resample("D").last().dropna()
    return daily


def _monthly_pnl(equity_series):
    monthly = equity_series.resample("ME").last().dropna()
    pnl = monthly.diff().dropna()
    return pnl


def _max_consecutive(series, condition_fn):
    max_streak = 0
    current = 0
    for val in series:
        if condition_fn(val):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _print_monthly_analysis(name, monthly_pnl):
    total = len(monthly_pnl)
    wins = (monthly_pnl > 0).sum()
    losses = (monthly_pnl <= 0).sum()
    win_rate = wins / total * 100 if total > 0 else 0
    avg_win = monthly_pnl[monthly_pnl > 0].mean() if wins > 0 else 0
    avg_loss = monthly_pnl[monthly_pnl <= 0].mean() if losses > 0 else 0
    max_win = monthly_pnl.max()
    max_loss = monthly_pnl.min()
    max_losing_streak = _max_consecutive(monthly_pnl, lambda x: x <= 0)
    max_winning_streak = _max_consecutive(monthly_pnl, lambda x: x > 0)

    print(f"\n{'='*60}")
    print(f"  {name} - 月次パフォーマンス分析")
    print(f"{'='*60}")
    print(f"  総月数: {total}")
    print(f"  勝ち月: {wins} ({win_rate:.1f}%)")
    print(f"  負け月: {losses} ({100-win_rate:.1f}%)")
    print(f"  平均勝ち月: {avg_win:+,.0f} 円")
    print(f"  平均負け月: {avg_loss:+,.0f} 円")
    print(f"  最大勝ち月: {max_win:+,.0f} 円")
    print(f"  最大負け月: {max_loss:+,.0f} 円")
    print(f"  最大連勝: {max_winning_streak}ヶ月")
    print(f"  最大連敗: {max_losing_streak}ヶ月")

    return {
        "total": total, "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_win": max_win, "max_loss": max_loss,
        "max_winning_streak": max_winning_streak,
        "max_losing_streak": max_losing_streak,
    }


def optimize_vb(df_raw, cash):
    """Volatility Breakout のパラメータを簡易グリッドサーチで最適化する。"""
    print("\n--- Volatility Breakout パラメータ最適化中... ---")

    param_grid = []
    for k in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for sl in [1.5, 2.0, 2.5]:
            for tp in [1.5, 2.0, 2.5, 3.0]:
                for hold in [8, 12, 24]:
                    for adx in [10, 15, 20]:
                        param_grid.append((k, sl, tp, hold, adx))

    # まず粗いグリッドで絞る
    coarse_grid = []
    for k in [0.3, 0.5, 0.7]:
        for sl in [1.5, 2.0, 2.5]:
            for tp in [1.5, 2.0, 3.0]:
                for hold in [8, 12, 24]:
                    for adx in [10, 15, 20]:
                        coarse_grid.append((k, sl, tp, hold, adx))

    best_score = -999
    best_params = None
    best_stats = None

    total = len(coarse_grid)
    for i, (k, sl, tp, hold, adx) in enumerate(coarse_grid):
        df = prepare_vb(df_raw.copy())
        try:
            s, _ = _run_bt(
                df, VolatilityBreakoutStrategy, cash=cash,
                k_factor=k, atr_sl_mult=sl, atr_tp_mult=tp,
                max_hold_hours=hold, adx_min=adx,
            )
            pf = _pf_from_trades(s.get("_trades"))
            ret = s["Return [%]"]
            trades = s["# Trades"]
            wr = s["Win Rate [%]"]
            mdd = s["Max. Drawdown [%]"]

            # スコア: PF重視 (PF > 1必須) + リターン + 取引数ボーナス
            if pf > 1.0 and ret > 0 and trades > 500:
                score = pf * ret * (1 + min(trades, 5000) / 10000)
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_params = dict(
                    k_factor=k, atr_sl_mult=sl, atr_tp_mult=tp,
                    max_hold_hours=hold, adx_min=adx,
                )
                best_stats = s

            if (i + 1) % 27 == 0 or i == total - 1:
                print(f"  [{i+1:3d}/{total}] 最良: "
                      f"k={best_params['k_factor'] if best_params else '?'} "
                      f"SL={best_params['atr_sl_mult'] if best_params else '?'} "
                      f"TP={best_params['atr_tp_mult'] if best_params else '?'} "
                      f"Hold={best_params['max_hold_hours'] if best_params else '?'} "
                      f"ADX={best_params['adx_min'] if best_params else '?'} "
                      f"→ score={best_score:.1f}")

        except Exception as e:
            pass

    if best_params and best_score > 0:
        pf = _pf_from_trades(best_stats.get("_trades"))
        print(f"\n  最適パラメータ: {best_params}")
        print(f"  Return={best_stats['Return [%]']:+.1f}% PF={pf:.2f} "
              f"WR={best_stats['Win Rate [%]']:.1f}% Trades={best_stats['# Trades']}")
    else:
        print("  有効なパラメータが見つかりませんでした。")
        # フォールバック: セッション無しで再試行
        print("  セッションフィルター緩和で再試行...")
        for k in [0.3, 0.4, 0.5, 0.6]:
            for sl in [1.5, 2.0, 2.5, 3.0]:
                for tp in [2.0, 2.5, 3.0, 4.0]:
                    for hold in [12, 24, 48]:
                        df = prepare_vb(df_raw.copy())
                        try:
                            s, _ = _run_bt(
                                df, VolatilityBreakoutStrategy, cash=cash,
                                k_factor=k, atr_sl_mult=sl, atr_tp_mult=tp,
                                max_hold_hours=hold, adx_min=0,
                                session_start=0, session_end=24,
                            )
                            pf = _pf_from_trades(s.get("_trades"))
                            ret = s["Return [%]"]
                            trades = s["# Trades"]
                            if pf > 1.0 and ret > 0 and trades > 500:
                                score = pf * ret
                                if score > best_score:
                                    best_score = score
                                    best_params = dict(
                                        k_factor=k, atr_sl_mult=sl, atr_tp_mult=tp,
                                        max_hold_hours=hold, adx_min=0,
                                        session_start=0, session_end=24,
                                    )
                                    best_stats = s
                        except:
                            pass
        if best_params and best_score > 0:
            pf = _pf_from_trades(best_stats.get("_trades"))
            print(f"\n  最適パラメータ (緩和): {best_params}")
            print(f"  Return={best_stats['Return [%]']:+.1f}% PF={pf:.2f}")
        else:
            print("  緩和でも見つかりませんでした。デフォルトで続行。")
            best_params = dict(
                k_factor=0.5, atr_sl_mult=2.0, atr_tp_mult=2.0,
                max_hold_hours=12, adx_min=15,
            )

    return best_params, best_stats


def main():
    print("=" * 60)
    print("Combined Portfolio: Trend Follow + Volatility Breakout")
    print("=" * 60)

    # --- データ読み込み ---
    print("\nUSD/JPY H1 25年データを読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"データ: {len(df_raw)}本 ({df_raw.index[0]} ~ {df_raw.index[-1]})")

    # --- まず TF を100% で実行（ベースライン） ---
    print("\n--- Trend Follow 100% (ベースライン) ---")
    df_tf_full = prepare_tf(df_raw.copy(), entry_period=TF_PARAMS["entry_period"],
                            exit_period=TF_PARAMS["exit_period"])
    tf_full_stats, _ = _run_bt(df_tf_full, TrendFollowStrategy,
                                cash=INITIAL_EQUITY, **TF_PARAMS)
    tf_full_equity = _equity_series(tf_full_stats)
    tf_full_monthly = _monthly_pnl(tf_full_equity)
    tf_full_result = _print_monthly_analysis("Trend Follow 100% (ベースライン)", tf_full_monthly)

    # --- VB 最適化 ---
    best_vb_params, _ = optimize_vb(df_raw, cash=INITIAL_EQUITY)

    if best_vb_params is None:
        print("\nVB戦略が機能しないため、ポートフォリオ構築を中止します。")
        return

    # --- VB 単独 100% ---
    print("\n--- Volatility Breakout 100% ---")
    df_vb_full = prepare_vb(df_raw.copy())
    vb_full_stats, _ = _run_bt(df_vb_full, VolatilityBreakoutStrategy,
                                cash=INITIAL_EQUITY, **best_vb_params)
    vb_full_equity = _equity_series(vb_full_stats)
    vb_full_monthly = _monthly_pnl(vb_full_equity)
    vb_full_result = _print_monthly_analysis("Volatility Breakout 100%", vb_full_monthly)

    vb_pf = _pf_from_trades(vb_full_stats.get("_trades"))
    print(f"\n  VB単独: Return={vb_full_stats['Return [%]']:+.1f}% "
          f"PF={vb_pf:.2f} WR={vb_full_stats['Win Rate [%]']:.1f}% "
          f"Trades={vb_full_stats['# Trades']} MDD={vb_full_stats['Max. Drawdown [%]']:.1f}%")

    # --- 資金配分の感応度分析 ---
    print(f"\n\n{'='*60}")
    print(f"  資金配分の感応度分析")
    print(f"{'='*60}")

    best_alloc = None
    best_monthly_wr = 0

    for tf_pct in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        vb_pct = 100 - tf_pct
        df_tf2 = prepare_tf(df_raw.copy(), entry_period=TF_PARAMS["entry_period"],
                            exit_period=TF_PARAMS["exit_period"])
        s_tf, _ = _run_bt(df_tf2, TrendFollowStrategy,
                          cash=INITIAL_EQUITY * tf_pct / 100, **TF_PARAMS)
        df_vb2 = prepare_vb(df_raw.copy())
        s_vb, _ = _run_bt(df_vb2, VolatilityBreakoutStrategy,
                          cash=INITIAL_EQUITY * vb_pct / 100, **best_vb_params)

        eq_tf = _equity_series(s_tf)
        eq_vb = _equity_series(s_vb)
        idx = eq_tf.index.intersection(eq_vb.index)
        eq_combined = eq_tf.loc[idx] + eq_vb.loc[idx]
        m_pnl = _monthly_pnl(eq_combined)
        m_wins = (m_pnl > 0).sum()
        m_total = len(m_pnl)
        m_rate = m_wins / m_total * 100

        total_ret = (eq_combined.iloc[-1] / eq_combined.iloc[0] - 1) * 100
        pk = eq_combined.cummax()
        mdd = ((eq_combined - pk) / pk * 100).min()
        max_lose_streak = _max_consecutive(m_pnl, lambda x: x <= 0)

        star = ""
        if m_rate > best_monthly_wr:
            best_monthly_wr = m_rate
            best_alloc = (tf_pct, vb_pct)
            star = " ★"

        print(f"  TF {tf_pct}% / VB {vb_pct}% → "
              f"月次勝率 {m_rate:.1f}% | "
              f"Return {total_ret:+.1f}% | "
              f"MDD {mdd:.1f}% | "
              f"最大連敗 {max_lose_streak}ヶ月{star}")

    # --- 最適配分でフル分析 ---
    tf_alloc = best_alloc[0] / 100 if best_alloc else 0.7
    vb_alloc = 1.0 - tf_alloc

    print(f"\n\n最適配分: TF {tf_alloc*100:.0f}% / VB {vb_alloc*100:.0f}%")
    print("=" * 60)

    df_tf = prepare_tf(df_raw.copy(), entry_period=TF_PARAMS["entry_period"],
                       exit_period=TF_PARAMS["exit_period"])
    tf_stats, tf_bt = _run_bt(df_tf, TrendFollowStrategy,
                               cash=INITIAL_EQUITY * tf_alloc, **TF_PARAMS)

    df_vb = prepare_vb(df_raw.copy())
    vb_stats, vb_bt = _run_bt(df_vb, VolatilityBreakoutStrategy,
                               cash=INITIAL_EQUITY * vb_alloc, **best_vb_params)

    # ポートフォリオ合成
    tf_equity = _equity_series(tf_stats)
    vb_equity = _equity_series(vb_stats)
    common_idx = tf_equity.index.intersection(vb_equity.index)
    combined_equity = tf_equity.loc[common_idx] + vb_equity.loc[common_idx]

    total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100
    peak = combined_equity.cummax()
    dd = (combined_equity - peak) / peak * 100
    max_dd = dd.min()
    years = (common_idx[-1] - common_idx[0]).days / 365.25
    annual_ret = ((1 + total_return / 100) ** (1 / years) - 1) * 100

    print(f"\n  ポートフォリオ合算結果:")
    print(f"    初期資金: {INITIAL_EQUITY:,.0f} 円")
    print(f"    最終資産: {combined_equity.iloc[-1]:,.0f} 円")
    print(f"    トータルReturn: {total_return:+.1f}%")
    print(f"    年利: {annual_ret:+.1f}%")
    print(f"    最大DD: {max_dd:.1f}%")

    # 月次分析
    tf_monthly = _monthly_pnl(tf_equity)
    vb_monthly = _monthly_pnl(vb_equity)
    combined_monthly = _monthly_pnl(combined_equity)

    tf_result = _print_monthly_analysis(f"Trend Follow ({tf_alloc*100:.0f}%)", tf_monthly)
    vb_result = _print_monthly_analysis(f"Volatility Breakout ({vb_alloc*100:.0f}%)", vb_monthly)
    combined_result = _print_monthly_analysis("Combined Portfolio (合算)", combined_monthly)

    # 相関分析
    print(f"\n{'='*60}")
    print(f"  月次リターン相関分析")
    print(f"{'='*60}")

    tf_m = tf_monthly.reindex(combined_monthly.index).fillna(0)
    vb_m = vb_monthly.reindex(combined_monthly.index).fillna(0)
    corr = tf_m.corr(vb_m)
    print(f"  TF vs VB 月次相関: {corr:.3f}")

    tf_lose_months = tf_m[tf_m <= 0].index
    if len(tf_lose_months) > 0:
        vb_in_tf_lose = vb_m.loc[tf_lose_months]
        vb_win_when_tf_lose = (vb_in_tf_lose > 0).sum()
        print(f"\n  TFが負けた月: {len(tf_lose_months)}ヶ月")
        print(f"  そのうちVBが勝った月: {vb_win_when_tf_lose}ヶ月 "
              f"({vb_win_when_tf_lose/len(tf_lose_months)*100:.1f}%)")

    both_lose = ((tf_m <= 0) & (vb_m <= 0)).sum()
    print(f"  両方負けた月: {both_lose}ヶ月 ({both_lose/len(combined_monthly)*100:.1f}%)")

    # ベースラインとの比較
    print(f"\n{'='*60}")
    print(f"  ベースライン vs Combined 比較")
    print(f"{'='*60}")
    print(f"  {'指標':<16} {'TF 100%':>12} {'Combined':>12} {'改善':>8}")
    print(f"  {'-'*52}")
    print(f"  {'月次勝率':<14} {tf_full_result['win_rate']:>11.1f}% {combined_result['win_rate']:>11.1f}% {combined_result['win_rate']-tf_full_result['win_rate']:>+7.1f}%")
    print(f"  {'最大連敗':<14} {tf_full_result['max_losing_streak']:>10}ヶ月 {combined_result['max_losing_streak']:>10}ヶ月")
    print(f"  {'Return':<16} {tf_full_stats['Return [%]']:>+11.1f}% {total_return:>+11.1f}%")
    print(f"  {'MDD':<16} {tf_full_stats['Max. Drawdown [%]']:>11.1f}% {max_dd:>11.1f}%")

    # 年次パフォーマンス
    print(f"\n{'='*60}")
    print(f"  年次パフォーマンス比較")
    print(f"{'='*60}")
    yearly_tf = tf_equity.resample("YE").last().pct_change().dropna() * 100
    yearly_vb = vb_equity.resample("YE").last().pct_change().dropna() * 100
    yearly_combined = combined_equity.resample("YE").last().pct_change().dropna() * 100

    print(f"  {'年':>6}  {'TF':>8}  {'VB':>8}  {'Combined':>10}  判定")
    print(f"  {'-'*50}")
    for year in yearly_combined.index:
        tf_r = yearly_tf.loc[year] if year in yearly_tf.index else 0
        vb_r = yearly_vb.loc[year] if year in yearly_vb.index else 0
        cb_r = yearly_combined.loc[year]
        status = "勝ち" if cb_r > 0 else "負け"
        print(f"  {year.year:>6}  {tf_r:>+7.1f}%  {vb_r:>+7.1f}%  {cb_r:>+9.1f}%  {status}")

    yearly_wins = (yearly_combined > 0).sum()
    print(f"\n  年次勝率: {yearly_wins}/{len(yearly_combined)} ({yearly_wins/len(yearly_combined)*100:.1f}%)")

    # --- レポート出力 ---
    print("\n\nHTMLレポート生成中...")
    try:
        tf_bt.plot(filename="report_combined.html", open_browser=False,
                   plot_volume=False, show_legend=False)
        _fix_report("report_combined.html",
                    tf_stats, vb_stats, best_vb_params,
                    tf_alloc, vb_alloc,
                    tf_result, vb_result, combined_result,
                    tf_full_result,
                    combined_equity, corr)
        print("レポート出力: report_combined.html")
    except Exception as e:
        print(f"レポート出力スキップ: {e}")
        import traceback
        traceback.print_exc()


def _fix_report(filepath, tf_stats, vb_stats, vb_params,
                tf_alloc, vb_alloc,
                tf_monthly_result, vb_monthly_result, combined_monthly_result,
                tf_full_monthly_result,
                combined_equity, correlation):
    with open(filepath, "r") as f:
        html = f.read()

    tf_pf = _pf_from_trades(tf_stats.get("_trades"))
    vb_pf = _pf_from_trades(vb_stats.get("_trades"))

    total_ret = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100
    final_equity = combined_equity.iloc[-1]
    peak = combined_equity.cummax()
    mdd = ((combined_equity - peak) / peak * 100).min()
    years = (combined_equity.index[-1] - combined_equity.index[0]).days / 365.25
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    tf_wr = tf_monthly_result["win_rate"]
    vb_wr = vb_monthly_result["win_rate"]
    cb_wr = combined_monthly_result["win_rate"]
    base_wr = tf_full_monthly_result["win_rate"]
    improvement = cb_wr - base_wr

    ret_color = "#16a34a" if total_ret > 0 else "#dc2626"
    imp_color = "#16a34a" if improvement > 0 else "#dc2626"

    patch = f"""
<style>
.report-header {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px;
    margin: 20px auto 12px;
}}
.report-header h1 {{
    font-size: 22px;
    margin: 0 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #333;
}}
.report-section {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}}
.report-section .card {{
    flex: 1;
    min-width: 250px;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
}}
.report-section .card-title {{
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
    color: #333;
}}
.report-section table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.report-section td {{
    padding: 2px 4px;
}}
.report-section td:first-child {{
    color: #666;
    white-space: nowrap;
}}
.report-section td:last-child {{
    text-align: right;
    font-weight: 600;
}}
.highlight-box {{
    background: #fff;
    border: 2px solid #4dabf7;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
    text-align: center;
}}
.highlight-box .big {{
    font-size: 28px;
    font-weight: 800;
}}
.compare-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
}}
.compare-grid .card {{
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
}}
.compare-grid .card-title {{
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 6px;
}}
.result-highlight {{
    font-size: 20px;
    font-weight: 700;
    color: {ret_color};
}}
</style>

<div class="report-header">
  <h1>Combined Portfolio: Trend Follow + Volatility Breakout</h1>
  <p style="color:#666;font-size:13px;margin-bottom:16px">
    2つのトレンド系戦略を異なるタイムフレームで組み合わせ、月次勝率を改善するポートフォリオ
  </p>

  <div class="highlight-box">
    <div>月次勝率</div>
    <div class="big" style="color:#16a34a">{cb_wr:.1f}%</div>
    <div style="font-size:13px;color:#666;margin-top:4px">
      TF単独 {base_wr:.1f}% → Combined {cb_wr:.1f}%
      (<span style="color:{imp_color}">{improvement:+.1f}%改善</span>)
      | 最大連敗 {combined_monthly_result['max_losing_streak']}ヶ月
    </div>
  </div>

  <div class="report-section">
    <div class="card" style="flex:2;min-width:400px">
      <div class="card-title">Portfolio Strategy (ポートフォリオ戦略)</div>
      <p style="margin:0 0 8px;font-size:12px;color:#555">
        <strong>Trend Follow</strong>（長期トレンドで大きく勝つ）と
        <strong>Volatility Breakout</strong>（短期モメンタムで小さく頻繁に勝つ）を組み合わせ、
        異なるタイムフレームの値動きを捕捉して月次安定性を改善する。
      </p>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px;margin-bottom:10px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">資金配分</div>
        <table style="font-size:12px">
          <tr><td>Trend Follow</td><td><strong>{tf_alloc*100:.0f}%</strong> ({INITIAL_EQUITY*tf_alloc:,.0f}円)</td></tr>
          <tr><td>Volatility Breakout</td><td><strong>{vb_alloc*100:.0f}%</strong> ({INITIAL_EQUITY*vb_alloc:,.0f}円)</td></tr>
          <tr><td>合計</td><td><strong>{INITIAL_EQUITY:,.0f}円</strong></td></tr>
        </table>
      </div>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px;margin-bottom:10px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">Trend Follow (Donchian Breakout)</div>
        <table style="font-size:12px">
          <tr><td>ロジック</td><td>15H チャネル突破 + ATRトレーリング</td></tr>
          <tr><td>保有期間</td><td>最大72時間 (長期トレンド狙い)</td></tr>
          <tr><td>勝率</td><td>37% (低勝率・高利益型)</td></tr>
        </table>
      </div>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">Volatility Breakout</div>
        <table style="font-size:12px">
          <tr><td>ロジック</td><td>24Hレンジ × k倍のブレイクアウト + モメンタム確認</td></tr>
          <tr><td>保有期間</td><td>最大{vb_params.get('max_hold_hours', 12)}時間 (短期モメンタム狙い)</td></tr>
          <tr><td>パラメータ</td><td>k={vb_params.get('k_factor', 0.5)} SL={vb_params.get('atr_sl_mult', 2.0)} TP={vb_params.get('atr_tp_mult', 2.0)} ADX&gt;{vb_params.get('adx_min', 15)}</td></tr>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Combined Results (合算結果)</div>
      <table>
        <tr><td>期間</td><td>2000 ~ 2024 (24.6年)</td></tr>
        <tr><td>初期資金</td><td>{INITIAL_EQUITY:,.0f} 円</td></tr>
        <tr><td>スプレッド</td><td>{SPREAD_PIPS} pips</td></tr>
      </table>

      <div style="margin-top:14px">
        <div style="text-align:center;margin:4px 0 8px">
          <span class="result-highlight">{total_ret:+.1f}%</span>
          <span style="font-size:13px;color:#666"> ({final_equity:,.0f} 円)</span>
        </div>
        <table>
          <tr><td>年利</td><td>{annual_ret:+.1f}%</td></tr>
          <tr><td>最大DD</td><td>{mdd:.1f}%</td></tr>
          <tr><td>月次勝率</td><td style="color:#16a34a;font-weight:700">{cb_wr:.1f}%</td></tr>
          <tr><td>最大連敗</td><td>{combined_monthly_result['max_losing_streak']}ヶ月</td></tr>
          <tr><td>月次相関</td><td>{correlation:.3f}</td></tr>
        </table>
      </div>

      <div style="margin-top:14px;background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">vs TF単独</div>
        <table>
          <tr><td>月次勝率改善</td><td style="color:{imp_color}">{improvement:+.1f}%</td></tr>
          <tr><td>TF単独月次勝率</td><td>{base_wr:.1f}%</td></tr>
          <tr><td>TF単独最大連敗</td><td>{tf_full_monthly_result['max_losing_streak']}ヶ月</td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="compare-grid">
    <div class="card">
      <div class="card-title" style="color:#4263eb">Trend Follow ({tf_alloc*100:.0f}%)</div>
      <table>
        <tr><td>トレード数</td><td>{tf_stats['# Trades']}</td></tr>
        <tr><td>勝率</td><td>{tf_stats['Win Rate [%]']:.1f}%</td></tr>
        <tr><td>PF</td><td>{tf_pf:.2f}</td></tr>
        <tr><td>Return</td><td>{tf_stats['Return [%]']:+.1f}%</td></tr>
        <tr><td>MDD</td><td>{tf_stats['Max. Drawdown [%]']:.1f}%</td></tr>
        <tr><td>月次勝率</td><td>{tf_wr:.1f}%</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="card-title" style="color:#e03131">Vol Breakout ({vb_alloc*100:.0f}%)</div>
      <table>
        <tr><td>トレード数</td><td>{vb_stats['# Trades']}</td></tr>
        <tr><td>勝率</td><td>{vb_stats['Win Rate [%]']:.1f}%</td></tr>
        <tr><td>PF</td><td>{vb_pf:.2f}</td></tr>
        <tr><td>Return</td><td>{vb_stats['Return [%]']:+.1f}%</td></tr>
        <tr><td>MDD</td><td>{vb_stats['Max. Drawdown [%]']:.1f}%</td></tr>
        <tr><td>月次勝率</td><td>{vb_wr:.1f}%</td></tr>
      </table>
    </div>
    <div class="card" style="border:2px solid #4dabf7">
      <div class="card-title" style="color:#16a34a">Combined (合算)</div>
      <table>
        <tr><td>Return</td><td>{total_ret:+.1f}%</td></tr>
        <tr><td>年利</td><td>{annual_ret:+.1f}%</td></tr>
        <tr><td>MDD</td><td>{mdd:.1f}%</td></tr>
        <tr><td>月次勝率</td><td style="color:#16a34a;font-weight:700">{cb_wr:.1f}%</td></tr>
        <tr><td>最大連敗</td><td>{combined_monthly_result['max_losing_streak']}ヶ月</td></tr>
        <tr><td>平均勝ち月</td><td>{combined_monthly_result['avg_win']:+,.0f}円</td></tr>
        <tr><td>平均負け月</td><td>{combined_monthly_result['avg_loss']:+,.0f}円</td></tr>
      </table>
    </div>
  </div>

  <p style="font-size:11px;color:#888;margin-bottom:8px">
    ※ 下記チャートはTrend Follow側のみの表示です（backtesting.pyの制約上、2戦略同時表示は不可）
  </p>
</div>

<script>
function hideBokehLegends() {{
    if (typeof Bokeh === 'undefined' || !Bokeh.index) return;
    for (const [id, view] of Object.entries(Bokeh.index)) {{
        const doc = (view.model || view).document;
        if (!doc || !doc._all_models) continue;
        for (const [mid, m] of doc._all_models) {{
            if (m.type === 'Legend') m.visible = false;
        }}
    }}
}}
document.addEventListener('DOMContentLoaded', function() {{
    [500, 1500, 3000].forEach(d => setTimeout(hideBokehLegends, d));
}});
if (document.readyState === 'complete') {{
    [100, 500, 1500].forEach(d => setTimeout(hideBokehLegends, d));
}}
</script>
"""

    m = re.search(r'(<div[^>]*display:\s*contents[^>]*>)', html)
    if m:
        html = html[:m.start()] + patch + html[m.start():]
    else:
        html = html.replace("</body>", patch + "</body>")

    with open(filepath, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
