"""パラメータ最適化 (グリッドサーチ + ウォークフォワード分析)"""

import pandas as pd
import numpy as np
from backtesting import Backtest

from config.settings import INITIAL_EQUITY, SPREAD_RATIO


def grid_search(df: pd.DataFrame, strategy_class, param_grid: dict,
                cash: float = INITIAL_EQUITY, spread: float = SPREAD_RATIO,
                min_trades: int = 30) -> pd.DataFrame:
    """パラメータのグリッドサーチを実行する。

    Args:
        df: 前処理済みDataFrame
        strategy_class: Strategy クラス
        param_grid: {パラメータ名: [値リスト]} の辞書
        min_trades: 最低トレード数フィルター

    Returns:
        結果のDataFrame (各パラメータセットの統計)
    """
    bt = Backtest(df, strategy_class, cash=cash, spread=spread, exclusive_orders=True)

    results = []
    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())

    # 全組み合わせを生成
    import itertools
    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        try:
            stats = bt.run(**params)
        except Exception:
            continue

        n_trades = stats["# Trades"]
        if n_trades < min_trades:
            continue

        results.append({
            **params,
            "trades": n_trades,
            "win_rate": stats["Win Rate [%]"],
            "return_pct": stats["Return [%]"],
            "max_dd": stats["Max. Drawdown [%]"],
            "sharpe": stats["Sharpe Ratio"] or 0,
            "avg_trade": stats["Avg. Trade [%]"],
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("sharpe", ascending=False)
    return results_df


def walk_forward(df: pd.DataFrame, strategy_class, params: dict,
                 train_months: int = 12, test_months: int = 3,
                 cash: float = INITIAL_EQUITY, spread: float = SPREAD_RATIO) -> pd.DataFrame:
    """ウォークフォワード分析を実行する。

    訓練期間でパラメータを確認し、テスト期間で検証を繰り返す。

    Args:
        df: 前処理済みDataFrame
        strategy_class: Strategy クラス
        params: 固定パラメータ
        train_months: 訓練期間 (月)
        test_months: テスト期間 (月)

    Returns:
        各テスト期間の結果DataFrame
    """
    results = []
    start = df.index[0]
    end = df.index[-1]

    current = start + pd.DateOffset(months=train_months)

    while current + pd.DateOffset(months=test_months) <= end:
        train_end = current
        test_end = current + pd.DateOffset(months=test_months)

        # テスト期間でバックテスト
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        if len(test_df) < 100:
            current += pd.DateOffset(months=test_months)
            continue

        bt = Backtest(test_df, strategy_class, cash=cash, spread=spread, exclusive_orders=True)
        try:
            stats = bt.run(**params)
        except Exception:
            current += pd.DateOffset(months=test_months)
            continue

        results.append({
            "period_start": train_end.strftime("%Y-%m"),
            "period_end": test_end.strftime("%Y-%m"),
            "trades": stats["# Trades"],
            "win_rate": stats["Win Rate [%]"],
            "return_pct": stats["Return [%]"],
            "max_dd": stats["Max. Drawdown [%]"],
            "sharpe": stats["Sharpe Ratio"] or 0,
        })

        current += pd.DateOffset(months=test_months)

    return pd.DataFrame(results)
