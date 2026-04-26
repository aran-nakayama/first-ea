#!/usr/bin/env python3
"""Supertrend / MACD / Ichimoku を25年データで検証・最適化"""

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


def _pf_from_trades(trades):
    if trades is None or len(trades) == 0:
        return 0
    w = trades[trades["PnL"] > 0]["PnL"].sum()
    l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
    return w / l if l > 0 else float("inf")


def _run_bt(df, strategy_class, cash=INITIAL_EQUITY, spread=SPREAD_RATIO, **params):
    bt = Backtest(
        df, strategy_class,
        cash=cash, commission=COMMISSION,
        spread=spread, exclusive_orders=True,
    )
    stats = bt.run(**params)
    return stats, bt


def _equity_series(stats):
    equity = stats["_equity_curve"]["Equity"]
    return equity.resample("D").last().dropna()


def _monthly_pnl(equity):
    monthly = equity.resample("ME").last().dropna()
    return monthly.diff().dropna()


def _max_consec(series, cond):
    m = 0
    c = 0
    for v in series:
        if cond(v):
            c += 1
            m = max(m, c)
        else:
            c = 0
    return m


def _summarize(name, stats, equity):
    monthly = _monthly_pnl(equity)
    m_wins = (monthly > 0).sum()
    m_total = len(monthly)
    m_wr = m_wins / m_total * 100 if m_total > 0 else 0
    m_streak = _max_consec(monthly, lambda x: x <= 0)

    pf = _pf_from_trades(stats.get("_trades"))
    ret = stats["Return [%]"]
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    annual = ((1 + ret / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "name": name,
        "return_pct": ret,
        "annual_pct": annual,
        "pf": pf,
        "win_rate": stats["Win Rate [%]"],
        "trades": stats["# Trades"],
        "mdd": stats["Max. Drawdown [%]"],
        "sharpe": stats["Sharpe Ratio"] or 0,
        "final_equity": stats["Equity Final [$]"],
        "monthly_wr": m_wr,
        "monthly_streak": m_streak,
        "stats": stats,
    }


def _print_result(r):
    print(f"\n  === {r['name']} ===")
    print(f"  Return      : {r['return_pct']:+.1f}% (年利 {r['annual_pct']:+.1f}%)")
    print(f"  PF          : {r['pf']:.2f}")
    print(f"  勝率        : {r['win_rate']:.1f}%")
    print(f"  トレード数  : {r['trades']}")
    print(f"  MDD         : {r['mdd']:.1f}%")
    print(f"  月次勝率    : {r['monthly_wr']:.1f}%")
    print(f"  月次最大連敗: {r['monthly_streak']}ヶ月")
    print(f"  最終資産    : {r['final_equity']:,.0f} 円")


def optimize_supertrend(df_raw):
    print("\n" + "="*60)
    print("Supertrend 最適化中...")
    print("="*60)

    grid = []
    for period in [10, 14]:
        for mult in [2.0, 3.0, 4.0]:
            for sl in [2.0, 2.5]:
                for trail in [2.0, 2.5, 3.0]:
                    for hold in [72, 120]:
                        grid.append((period, mult, sl, trail, hold))

    best_score = 0
    best_params = None
    best_result = None
    total = len(grid)

    for i, (p, m, sl, tr, h) in enumerate(grid):
        try:
            df = prep_st(df_raw.copy(), period=p, multiplier=m)
            stats, _ = _run_bt(
                df, SupertrendStrategy,
                st_period=p, st_multiplier=m,
                atr_sl_mult=sl, atr_trail_mult=tr, max_hold_hours=h,
            )
            pf = _pf_from_trades(stats.get("_trades"))
            ret = stats["Return [%]"]
            trades = stats["# Trades"]

            if pf > 1.0 and ret > 0 and trades > 100:
                score = pf * ret
                if score > best_score:
                    best_score = score
                    best_params = dict(period=p, multiplier=m,
                                       atr_sl_mult=sl, atr_trail_mult=tr,
                                       max_hold_hours=h)
                    eq = _equity_series(stats)
                    best_result = _summarize(f"Supertrend P={p} M={m}", stats, eq)

            if (i + 1) % 40 == 0:
                best_str = f"score={best_score:.1f}" if best_params else "まだ見つからず"
                print(f"  [{i+1:3d}/{total}] {best_str}")
        except Exception:
            pass

    if best_params:
        print(f"\n  最適: period={best_params['period']} mult={best_params['multiplier']} "
              f"sl={best_params['atr_sl_mult']} trail={best_params['atr_trail_mult']} "
              f"hold={best_params['max_hold_hours']}")
        _print_result(best_result)
    return best_params, best_result


def optimize_macd(df_raw):
    print("\n" + "="*60)
    print("MACD Momentum 最適化中...")
    print("="*60)

    grid = []
    for fast in [8, 12]:
        for slow in [21, 26]:
            for signal in [9, 12]:
                for sl in [2.0, 2.5]:
                    for trail in [2.0, 2.5]:
                        for ma_flt in [0, 1]:
                            grid.append((fast, slow, signal, sl, trail, ma_flt))

    best_score = 0
    best_params = None
    best_result = None
    total = len(grid)

    for i, (f, s, sg, sl, tr, ma) in enumerate(grid):
        if f >= s:
            continue
        try:
            df = prep_macd(df_raw.copy(), fast_period=f, slow_period=s, signal_period=sg)
            stats, _ = _run_bt(
                df, MACDMomentumStrategy,
                atr_sl_mult=sl, atr_trail_mult=tr, use_ma_filter=ma,
            )
            pf = _pf_from_trades(stats.get("_trades"))
            ret = stats["Return [%]"]
            trades = stats["# Trades"]

            if pf > 1.0 and ret > 0 and trades > 100:
                score = pf * ret
                if score > best_score:
                    best_score = score
                    best_params = dict(fast=f, slow=s, signal=sg,
                                       atr_sl_mult=sl, atr_trail_mult=tr,
                                       use_ma_filter=ma)
                    eq = _equity_series(stats)
                    best_result = _summarize(
                        f"MACD({f}/{s}/{sg}) MA={ma}", stats, eq)

            if (i + 1) % 50 == 0:
                best_str = f"score={best_score:.1f}" if best_params else "まだ見つからず"
                print(f"  [{i+1:3d}/{total}] {best_str}")
        except Exception:
            pass

    if best_params:
        print(f"\n  最適: fast={best_params['fast']} slow={best_params['slow']} "
              f"signal={best_params['signal']} sl={best_params['atr_sl_mult']} "
              f"trail={best_params['atr_trail_mult']} MA={best_params['use_ma_filter']}")
        _print_result(best_result)
    return best_params, best_result


def optimize_ichimoku(df_raw):
    print("\n" + "="*60)
    print("Ichimoku 最適化中...")
    print("="*60)

    # 一目のパラメータは伝統的な 9/26/52 が一般的なので軽めに最適化
    grid = []
    for t in [9]:
        for k in [26]:
            for b in [52]:
                for sl in [2.0, 2.5, 3.0]:
                    for trail in [2.0, 2.5, 3.0]:
                        for tk in [0, 1]:
                            for hold in [72, 120, 168]:
                                grid.append((t, k, b, sl, trail, tk, hold))

    best_score = 0
    best_params = None
    best_result = None
    total = len(grid)

    for i, (t, k, b, sl, tr, tk, h) in enumerate(grid):
        if t >= k:
            continue
        try:
            df = prep_ichi(df_raw.copy(),
                           tenkan_period=t, kijun_period=k, senkou_b_period=b)
            stats, _ = _run_bt(
                df, IchimokuStrategy,
                atr_sl_mult=sl, atr_trail_mult=tr,
                use_tenkan_kijun=tk, max_hold_hours=h,
            )
            pf = _pf_from_trades(stats.get("_trades"))
            ret = stats["Return [%]"]
            trades = stats["# Trades"]

            if pf > 1.0 and ret > 0 and trades > 100:
                score = pf * ret
                if score > best_score:
                    best_score = score
                    best_params = dict(tenkan=t, kijun=k, senkou_b=b,
                                       atr_sl_mult=sl, atr_trail_mult=tr,
                                       use_tenkan_kijun=tk, max_hold_hours=h)
                    eq = _equity_series(stats)
                    best_result = _summarize(
                        f"Ichimoku({t}/{k}/{b})", stats, eq)

            if (i + 1) % 50 == 0:
                best_str = f"score={best_score:.1f}" if best_params else "まだ見つからず"
                print(f"  [{i+1:3d}/{total}] {best_str}")
        except Exception:
            pass

    if best_params:
        print(f"\n  最適: tenkan={best_params['tenkan']} kijun={best_params['kijun']} "
              f"senkou_b={best_params['senkou_b']} sl={best_params['atr_sl_mult']} "
              f"trail={best_params['atr_trail_mult']} tk={best_params['use_tenkan_kijun']} "
              f"hold={best_params['max_hold_hours']}")
        _print_result(best_result)
    return best_params, best_result


def _gen_report(filepath, name, desc, params_desc, result):
    """個別戦略のHTMLレポートを生成する。"""
    stats = result["stats"]
    pf = result["pf"]
    ret_color = "#16a34a" if result["return_pct"] > 0 else "#dc2626"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>{name} - Backtest Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1000px;
    margin: 20px auto;
    padding: 20px;
    background: #f5f5f5;
    line-height: 1.6;
}}
h1 {{
    font-size: 22px;
    margin: 0 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #333;
}}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.card {{
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
}}
.card-title {{
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 8px;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td {{ padding: 4px; border-bottom: 1px solid #eee; }}
td:first-child {{ color: #666; }}
td:last-child {{ text-align: right; font-weight: 600; }}
.big {{ font-size: 32px; font-weight: 800; color: {ret_color}; }}
.back {{ display: inline-block; margin-bottom: 16px; color: #4263eb; text-decoration: none; }}
.back:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<a href="index.html" class="back">← 戦略一覧に戻る</a>

<h1>{name} - Backtest Report</h1>

<div class="card">
  <div class="card-title">戦略概要</div>
  <p>{desc}</p>
  <div style="background:#f8f9fa;border-radius:4px;padding:10px;margin-top:10px;font-family:monospace;font-size:12px">
    最適パラメータ: {params_desc}
  </div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-title">結果ハイライト</div>
    <div style="text-align:center;margin:12px 0">
      <div class="big">{result['return_pct']:+.1f}%</div>
      <div style="color:#666">最終資産 {result['final_equity']:,.0f} 円 (年利 {result['annual_pct']:+.1f}%)</div>
    </div>
    <table>
      <tr><td>期間</td><td>{stats['Start'].strftime('%Y/%m/%d')} ~ {stats['End'].strftime('%Y/%m/%d')}</td></tr>
      <tr><td>初期資金</td><td>{INITIAL_EQUITY:,.0f} 円</td></tr>
      <tr><td>スプレッド</td><td>{SPREAD_PIPS} pips</td></tr>
    </table>
  </div>

  <div class="card">
    <div class="card-title">詳細統計</div>
    <table>
      <tr><td>トレード数</td><td>{result['trades']}</td></tr>
      <tr><td>勝率 (トレード)</td><td>{result['win_rate']:.1f}%</td></tr>
      <tr><td>PF</td><td>{result['pf']:.2f}</td></tr>
      <tr><td>Sharpe Ratio</td><td>{result['sharpe']:.2f}</td></tr>
      <tr><td>最大DD</td><td>{result['mdd']:.1f}%</td></tr>
      <tr><td style="color:#4263eb"><strong>月次勝率</strong></td><td style="color:#4263eb"><strong>{result['monthly_wr']:.1f}%</strong></td></tr>
      <tr><td>月次最大連敗</td><td>{result['monthly_streak']}ヶ月</td></tr>
    </table>
  </div>
</div>

<div class="card">
  <div class="card-title">判定</div>
  <p>
    {"✅ 長期運用可能" if result['pf'] > 1.05 and result['return_pct'] > 50 else "⚠️ 改善余地あり" if result['return_pct'] > 0 else "❌ 長期運用不可"}
    (PF={result['pf']:.2f}, Return={result['return_pct']:+.1f}%)
  </p>
</div>

</body>
</html>
"""
    with open(filepath, "w") as f:
        f.write(html)


def main():
    print("="*60)
    print("新戦略検証: Supertrend / MACD / Ichimoku")
    print("="*60)

    print("\nUSD/JPY H1 25年データを読み込み中...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"], index_col="DateTime")
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"データ: {len(df_raw)}本 ({df_raw.index[0]} ~ {df_raw.index[-1]})")

    results = {}

    # --- Supertrend ---
    st_params, st_result = optimize_supertrend(df_raw)
    if st_result:
        results["supertrend"] = (st_params, st_result)
        _gen_report(
            "report_supertrend.html",
            "Supertrend Strategy",
            "ATRベースのトレンドフォロー指標。価格がATR×係数のバンドを突破したらトレンド転換と判定し、転換方向にエントリー。ATRトレーリングストップで利益を伸ばす。",
            f"period={st_params['period']}, multiplier={st_params['multiplier']}, "
            f"ATR_SL={st_params['atr_sl_mult']}, ATR_Trail={st_params['atr_trail_mult']}, "
            f"Hold={st_params['max_hold_hours']}h",
            st_result,
        )
        print("\nレポート出力: report_supertrend.html")

    # --- MACD ---
    macd_params, macd_result = optimize_macd(df_raw)
    if macd_result:
        results["macd"] = (macd_params, macd_result)
        _gen_report(
            "report_macd.html",
            "MACD Momentum Strategy",
            "MACDヒストグラムのゼロクロスで勢いを捕捉するモメンタム戦略。EMA200によるトレンドフィルターで逆張りエントリーを排除。",
            f"fast={macd_params['fast']}, slow={macd_params['slow']}, "
            f"signal={macd_params['signal']}, "
            f"ATR_SL={macd_params['atr_sl_mult']}, ATR_Trail={macd_params['atr_trail_mult']}, "
            f"MA_Filter={'ON' if macd_params['use_ma_filter'] else 'OFF'}",
            macd_result,
        )
        print("\nレポート出力: report_macd.html")

    # --- Ichimoku ---
    ichi_params, ichi_result = optimize_ichimoku(df_raw)
    if ichi_result:
        results["ichimoku"] = (ichi_params, ichi_result)
        _gen_report(
            "report_ichimoku.html",
            "Ichimoku Cloud Strategy",
            "一目均衡表の雲ブレイクアウト戦略。日本発の伝統的指標で、JPYペアに適している。価格が雲を突破 + 転換線/基準線の関係でエントリー判定。",
            f"tenkan={ichi_params['tenkan']}, kijun={ichi_params['kijun']}, "
            f"senkou_b={ichi_params['senkou_b']}, "
            f"ATR_SL={ichi_params['atr_sl_mult']}, ATR_Trail={ichi_params['atr_trail_mult']}, "
            f"TK_Cross={'ON' if ichi_params['use_tenkan_kijun'] else 'OFF'}, "
            f"Hold={ichi_params['max_hold_hours']}h",
            ichi_result,
        )
        print("\nレポート出力: report_ichimoku.html")

    # --- サマリー比較 ---
    print("\n\n" + "="*60)
    print("全戦略サマリー (25年間・初期資金100万円)")
    print("="*60)
    print(f"\n  {'戦略':<22} {'Return':>10} {'PF':>6} {'MDD':>8} {'月次WR':>8} {'月次連敗':>10}")
    print(f"  {'-'*66}")
    print(f"  {'Trend Follow (基準)':<22} {'+224.5%':>10} {'1.12':>6} {'-17.4%':>8} {'59.7%':>8} {'4':>8}ヶ月")
    print(f"  {'Vol Breakout (基準)':<22} {'+122.3%':>10} {'1.08':>6} {'-14.9%':>8} {'56.3%':>8} {'4':>8}ヶ月")
    for key in ["supertrend", "macd", "ichimoku"]:
        if key in results:
            _, r = results[key]
            print(f"  {r['name']:<22} {r['return_pct']:>+9.1f}% {r['pf']:>6.2f} "
                  f"{r['mdd']:>7.1f}% {r['monthly_wr']:>7.1f}% {r['monthly_streak']:>8}ヶ月")

    return results


if __name__ == "__main__":
    main()
