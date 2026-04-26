"""トレードフィルター: セッション、ボラティリティ、上位足トレンド"""

import numpy as np
import pandas as pd

from config.settings import (
    ATR_VOL_THRESHOLD,
    H4_RSI_LONG_MIN,
    H4_RSI_SHORT_MAX,
    SESSION_END_UTC,
    SESSION_START_UTC,
)


def session_filter(index: pd.DatetimeIndex) -> pd.Series:
    """アジアセッション(00:00-08:00 UTC)内かどうかを判定する。"""
    hours = index.hour
    return pd.Series(
        (hours >= SESSION_START_UTC) & (hours < SESSION_END_UTC),
        index=index,
    )


def volatility_filter(atr: pd.Series, atr_ma: pd.Series,
                      threshold: float = ATR_VOL_THRESHOLD) -> pd.Series:
    """ATRが平均の threshold倍以下なら True (低ボラ=エントリー許可)。"""
    return atr < atr_ma * threshold


def h4_trend_filter(h4_rsi: pd.Series, h1_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """H4 RSI を H1 インデックスにリサンプルし、ロング/ショート許可フラグを返す。

    Returns: (long_allowed, short_allowed)
    """
    # H4のRSI値をH1のタイムスタンプにforward-fillで展開
    h4_rsi_resampled = h4_rsi.reindex(h1_index, method="ffill")

    long_allowed = h4_rsi_resampled >= H4_RSI_LONG_MIN
    short_allowed = h4_rsi_resampled <= H4_RSI_SHORT_MAX

    return long_allowed, short_allowed
