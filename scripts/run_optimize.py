#!/usr/bin/env python3
"""パラメータ最適化 + ウォークフォワード分析スクリプト"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_hourly_data
from strategy.asian_breakout import prepare_data, AsianBreakoutStrategy
from backtest.optimizer import grid_search, walk_forward


def main():
    print("USD/JPY H1 データを取得中...")
    df = fetch_hourly_data()
    df.index = df.index.tz_localize(None)
    df = prepare_data(df)
    print(f"データ: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})\n")

    # === グリッドサーチ ===
    print("=" * 60)
    print("グリッドサーチ実行中...")
    print("=" * 60)

    param_grid = {
        "sl_pips": [20, 30, 40, 50, 60],
        "tp_pips": [30, 50, 75, 100],
        "session_end": [12, 14, 16, 18],
        "min_range": [0.10, 0.15, 0.20],
    }

    results = grid_search(df, AsianBreakoutStrategy, param_grid)

    print(f"\n検証した組み合わせ: {len(results)}")
    print("\n--- Top 10 (Sharpe順) ---")
    top10 = results.head(10)
    for _, row in top10.iterrows():
        print(f"SL={row['sl_pips']:>2.0f}p TP={row['tp_pips']:>3.0f}p Sess<{row['session_end']:.0f} "
              f"MinR={row['min_range']:.2f}: "
              f"T={row['trades']:.0f}, WR={row['win_rate']:.1f}%, "
              f"Ret={row['return_pct']:.1f}%, MDD={row['max_dd']:.1f}%, "
              f"Sharpe={row['sharpe']:.2f}")

    # === ウォークフォワード分析 ===
    if len(results) > 0:
        best = results.iloc[0]
        best_params = {
            "sl_pips": int(best["sl_pips"]),
            "tp_pips": int(best["tp_pips"]),
            "session_end": int(best["session_end"]),
            "min_range": best["min_range"],
        }

        print(f"\n{'=' * 60}")
        print(f"ウォークフォワード分析 (12ヶ月訓練 / 3ヶ月テスト)")
        print(f"最適パラメータ: {best_params}")
        print(f"{'=' * 60}")

        wf = walk_forward(df, AsianBreakoutStrategy, best_params,
                          train_months=12, test_months=3)

        if len(wf) > 0:
            for _, row in wf.iterrows():
                print(f"{row['period_start']} ~ {row['period_end']}: "
                      f"T={row['trades']:.0f}, WR={row['win_rate']:.1f}%, "
                      f"Ret={row['return_pct']:.1f}%, MDD={row['max_dd']:.1f}%, "
                      f"Sharpe={row['sharpe']:.2f}")

            print(f"\n--- ウォークフォワード集計 ---")
            print(f"平均勝率:    {wf['win_rate'].mean():.1f}%")
            print(f"平均リターン: {wf['return_pct'].mean():.1f}%")
            print(f"平均MDD:     {wf['max_dd'].mean():.1f}%")
            print(f"平均Sharpe:  {wf['sharpe'].mean():.2f}")
            print(f"全期間プラス: {(wf['return_pct'] > 0).sum()}/{len(wf)}")
        else:
            print("データが不足しています")


if __name__ == "__main__":
    main()
