"""追加パフォーマンスメトリクス"""

import numpy as np
import pandas as pd


def profit_factor(trades: pd.DataFrame) -> float:
    """プロフィットファクター = 総利益 / 総損失"""
    winning = trades[trades["PnL"] > 0]["PnL"].sum()
    losing = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
    if losing == 0:
        return float("inf") if winning > 0 else 0.0
    return winning / losing


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Sortino比率 (下方リスクのみ考慮)"""
    excess = returns - risk_free
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return excess.mean() / downside.std() * np.sqrt(252)


def recovery_factor(net_profit: float, max_drawdown: float) -> float:
    """回復係数 = 純利益 / 最大ドローダウン"""
    if max_drawdown == 0:
        return 0.0
    return net_profit / abs(max_drawdown)


def trade_summary(trades: pd.DataFrame) -> dict:
    """トレード統計の要約を返す。"""
    if len(trades) == 0:
        return {"total_trades": 0}

    winning = trades[trades["PnL"] > 0]
    losing = trades[trades["PnL"] <= 0]

    return {
        "total_trades": len(trades),
        "win_rate": len(winning) / len(trades) * 100,
        "profit_factor": profit_factor(trades),
        "avg_win": winning["PnL"].mean() if len(winning) > 0 else 0,
        "avg_loss": losing["PnL"].mean() if len(losing) > 0 else 0,
        "max_win": trades["PnL"].max(),
        "max_loss": trades["PnL"].min(),
        "avg_duration_bars": trades["Duration"].apply(lambda x: x.total_seconds() / 3600).mean() if "Duration" in trades else 0,
    }
