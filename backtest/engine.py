"""backtesting.py ベースのバックテストエンジン"""

import pandas as pd
from backtesting import Backtest

from config.settings import COMMISSION, INITIAL_EQUITY, SPREAD_RATIO


def run_backtest(df: pd.DataFrame, strategy_class,
                 cash: float = INITIAL_EQUITY, spread: float = SPREAD_RATIO,
                 **strategy_params) -> tuple:
    """バックテストを実行する。

    Args:
        df: OHLCV DataFrame + インジケーター列
        strategy_class: backtesting.py Strategy クラス
        cash: 初期資金
        spread: スプレッド比率 (backtesting.py形式: 0.4pips/152 ≈ 2.6e-5)
        **strategy_params: 戦略パラメータの上書き

    Returns:
        (stats, bt) - 統計情報とBacktestオブジェクト
    """
    bt = Backtest(
        df,
        strategy_class,
        cash=cash,
        commission=COMMISSION,
        spread=spread,
        exclusive_orders=True,
    )

    stats = bt.run(**strategy_params)
    return stats, bt


def print_stats(stats) -> None:
    """バックテスト結果の主要メトリクスを表示する。"""
    print("=" * 60)
    print("バックテスト結果")
    print("=" * 60)
    print(f"期間:          {stats['Start']} ~ {stats['End']}")
    print(f"総トレード数:  {stats['# Trades']}")
    print(f"勝率:          {stats['Win Rate [%]']:.1f}%")
    print(f"平均利益:      {stats['Avg. Trade [%]']:.2f}%")
    print(f"最大ドローダウン: {stats['Max. Drawdown [%]']:.1f}%")
    sharpe = stats['Sharpe Ratio']
    print(f"シャープレシオ:   {sharpe:.2f}" if sharpe else "シャープレシオ:   N/A")
    print(f"最終資産:      {stats['Equity Final [$]']:,.0f}")
    print(f"リターン:      {stats['Return [%]']:.1f}%")
    print(f"Buy & Hold:    {stats['Buy & Hold Return [%]']:.1f}%")

    trades = stats.get("_trades")
    if trades is not None and len(trades) > 0:
        winning = trades[trades["PnL"] > 0]
        losing = trades[trades["PnL"] <= 0]
        if len(winning) > 0 and len(losing) > 0:
            avg_win = winning["PnL"].mean()
            avg_loss = abs(losing["PnL"].mean())
            pf = winning["PnL"].sum() / abs(losing["PnL"].sum()) if losing["PnL"].sum() != 0 else float("inf")
            print(f"プロフィットファクター: {pf:.2f}")
            print(f"平均利益/平均損失: {avg_win:,.0f} / {avg_loss:,.0f}")
            print(f"最大連敗数:    {_max_consecutive_losses(trades)}")
    print("=" * 60)


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    max_streak = 0
    current = 0
    for pnl in trades["PnL"]:
        if pnl <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
