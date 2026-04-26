#!/usr/bin/env python3
"""Paper Trading - ヒストリカルデータをリプレイしてアルゴを検証。

実際のお金は動かさない。25年分の過去データを高速再生し、
ライブと同じシグナルロジックでトレードした結果を出力する。

使い方:
  python3 -u scripts/run_paper.py                # 過去25年全体
  python3 -u scripts/run_paper.py --start 2020-01-01    # 期間指定
  python3 -u scripts/run_paper.py --leverage 3          # レバレッジ指定
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.broker import PaperBroker
from live.portfolio import MultiPortfolio
from live.feed import PaperFeed
from live.runner import run_replay


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "usdjpy_h1_25years.csv",
)
STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "live", "state",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--leverage", type=float, default=1.0, help="レバレッジ倍率")
    parser.add_argument("--initial", type=float, default=1_000_000.0,
                        help="初期資金 (円)")
    parser.add_argument("--state", default=None,
                        help="state JSON path (省略時はメモリのみ)")
    parser.add_argument("--reset", action="store_true",
                        help="既存stateをリセットして開始")
    parser.add_argument("--log-every", type=int, default=2000, help="進捗ログ頻度")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    state_path = args.state
    if state_path and args.reset and os.path.exists(state_path):
        os.remove(state_path)
        print(f"既存state削除: {state_path}")

    print("="*70)
    print(f"Paper Trading (Replay モード)")
    print(f"  データ: {DATA_PATH}")
    print(f"  期間: {args.start or '全期間開始'} ~ {args.end or '全期間終了'}")
    print(f"  初期資金: {args.initial:,.0f} 円")
    print(f"  レバレッジ: {args.leverage}x")
    print(f"  state: {state_path or 'メモリのみ (永続化なし)'}")
    print("="*70)

    broker = PaperBroker(
        initial_balance=args.initial, leverage=args.leverage,
        state_path=state_path,
    )
    portfolio = MultiPortfolio(broker, leverage=args.leverage)
    feed = PaperFeed(DATA_PATH, warmup_bars=300,
                     start_date=args.start, end_date=args.end)

    final_stat = run_replay(broker, portfolio, feed, log_every=args.log_every)

    # 結果サマリー
    print("\n" + "="*70)
    print("最終結果")
    print("="*70)
    print(f"初期資金:     {args.initial:,.0f} 円")
    print(f"最終評価額:   {final_stat['equity']:,.0f} 円")
    print(f"リターン:     {final_stat['drawdown_pct']:+.2f}%")
    print(f"総取引数:     {final_stat['n_trades']}")
    print(f"未決済ポジ:   {len(final_stat['positions'])}")

    if broker.trades:
        wins = [t for t in broker.trades if t.pnl > 0]
        losses = [t for t in broker.trades if t.pnl <= 0]
        wr = len(wins) / len(broker.trades) * 100
        total_w = sum(t.pnl for t in wins)
        total_l = abs(sum(t.pnl for t in losses))
        pf = total_w / total_l if total_l > 0 else float("inf")
        print(f"勝率:         {wr:.1f}%")
        print(f"PF:           {pf:.2f}")
        print(f"平均勝ち:     {total_w/len(wins) if wins else 0:+,.0f} 円")
        print(f"平均負け:     {-total_l/len(losses) if losses else 0:+,.0f} 円")

        # 戦略別
        print(f"\n戦略別:")
        for sid in ["TF", "ST", "MACD", "ICHI"]:
            ts = [t for t in broker.trades if t.strategy_id == sid]
            if not ts:
                continue
            tw = [t for t in ts if t.pnl > 0]
            tl = [t for t in ts if t.pnl <= 0]
            sub_pnl = sum(t.pnl for t in ts)
            sub_wr = len(tw) / len(ts) * 100
            sub_pf = sum(t.pnl for t in tw) / abs(sum(t.pnl for t in tl)) \
                if tl and sum(t.pnl for t in tl) < 0 else float("inf")
            print(f"  {sid:6s}: {len(ts):>5}件  勝率{sub_wr:>5.1f}%  "
                  f"PF{sub_pf:>4.2f}  P&L{sub_pnl:>+12,.0f} 円")


if __name__ == "__main__":
    main()
