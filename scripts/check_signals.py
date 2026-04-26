#!/usr/bin/env python3
"""現在の市場で各戦略が出しているシグナルを確認する (取引はしない)。

毎時バー確定後に実行することで、「いま4戦略がそれぞれ何をしようとしているか」
を表示。実際にトレードする前のチェック用。

使い方:
  python3 -u scripts/check_signals.py                    # yfinanceで最新データ
  python3 -u scripts/check_signals.py --csv data/...csv  # 既存CSVの末尾で評価
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from live.signals import STRATEGIES, Position
from live.feed import PaperFeed, YfinanceFeed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="CSVから読む (省略時はyfinanceから取得)")
    parser.add_argument("--symbol", default="JPY=X", help="yfinanceシンボル")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["DateTime"], index_col="DateTime")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # 最後の500本だけ評価 (warmup含む)
        df = df.tail(500).copy()
        print(f"CSV: {args.csv} の末尾500本を使用")
    else:
        feed = YfinanceFeed(symbol=args.symbol, lookback_days=60)
        df = feed.get_latest_bars()
        print(f"yfinance: {args.symbol} 直近60日H1")

    print(f"期間: {df.index[0]} ~ {df.index[-1]} ({len(df)}本)")
    print(f"現在価格: {df['Close'].iloc[-1]:.4f}")
    print()
    print("="*70)
    print(f"{'戦略':<20} {'現状':<12} {'シグナル':<14} {'SL':<10} {'理由'}")
    print("="*70)

    flat_pos = Position()
    for sid, spec in STRATEGIES.items():
        try:
            df_strat = spec["prepare"](df)
            sig = spec["signal"](df_strat, flat_pos)
            sl_str = f"{sig.sl:.4f}" if sig.sl else "-"
            print(f"{spec['name']:<20} {flat_pos.side:<12} {sig.action:<14} "
                  f"{sl_str:<10} {sig.reason}")
        except Exception as e:
            print(f"{spec['name']:<20} ERROR: {e}")

    print("="*70)
    print("\n注意:")
    print("  これは現在flat (ノーポジ) を仮定したシグナル。")
    print("  実際の取引時は既存ポジションを考慮する必要あり。")


if __name__ == "__main__":
    main()
