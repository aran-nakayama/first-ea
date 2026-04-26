"""価格フィード - 過去データの再生(replay) または リアルタイム取得。

PaperFeed: 既存のCSVヒストリカルデータを順番に流す (テスト/シミュレーション用)
YfinanceFeed: yfinanceで最新H1データを取得 (無料、15分遅延あり)
OandaFeed: OANDA API でリアルタイム取得 (要 API キー)
"""

from typing import Optional, Iterator
import pandas as pd
import os
import sys
import time
import logging

logger = logging.getLogger(__name__)


class PaperFeed:
    """CSVファイルから順番にバーを再生する (リプレイモード)。

    バックテストデータをライブシミュレーションとして流せるので、
    アルゴが正しく動くか時間圧縮で検証できる。
    """

    def __init__(self, csv_path: str, warmup_bars: int = 200,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        df = pd.read_csv(csv_path, parse_dates=["DateTime"], index_col="DateTime")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        self.df = df
        self.warmup_bars = warmup_bars
        self._cursor = warmup_bars

    def total_bars(self) -> int:
        return len(self.df) - self.warmup_bars

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """各イテレーションで「これまでのバー全部」を返す (rolling window)。"""
        while self._cursor <= len(self.df):
            yield self.df.iloc[:self._cursor].copy()
            self._cursor += 1

    def current_price(self, df: pd.DataFrame) -> float:
        return float(df["Close"].iloc[-1])

    def current_time(self, df: pd.DataFrame):
        return df.index[-1]


class YfinanceFeed:
    """yfinance で最新H1データを取得する。無料、ただし15分遅延あり。"""

    def __init__(self, symbol: str = "JPY=X", lookback_days: int = 60):
        try:
            import yfinance  # noqa
        except ImportError:
            raise ImportError("yfinance が必要です: pip install --user yfinance")
        self.symbol = symbol
        self.lookback_days = lookback_days
        self._last_bar_time = None

    def get_latest_bars(self) -> pd.DataFrame:
        """最新のH1バーを取得して返す。"""
        import yfinance as yf
        df = yf.download(self.symbol, period=f"{self.lookback_days}d",
                         interval="1h", progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"yfinanceからデータ取得失敗: {self.symbol}")
        # multi-indexの場合はフラット化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                                "Close": "Close", "Volume": "Volume"})
        return df[["Open", "High", "Low", "Close"]].copy()

    def has_new_bar(self) -> bool:
        df = self.get_latest_bars()
        latest = df.index[-1]
        if self._last_bar_time is None or latest > self._last_bar_time:
            self._last_bar_time = latest
            return True
        return False
