"""テクニカルインジケーター計算関数"""

import numpy as np
import pandas as pd


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands を計算する。

    Returns: (upper, middle, lower)
    """
    middle = close.rolling(period).mean()
    rolling_std = close.rolling(period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) を計算する。"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range) を計算する。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                   rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """DataFrameにすべてのインジケーターを追加する。"""
    df = df.copy()
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"], bb_period, bb_std)
    df["RSI"] = rsi(df["Close"], rsi_period)
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], atr_period)
    df["ATR_MA"] = df["ATR"].rolling(50).mean()
    return df
