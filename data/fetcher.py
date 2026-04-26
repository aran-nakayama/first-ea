"""yfinanceからUSD/JPYヒストリカルデータを取得"""

import pandas as pd
import yfinance as yf

from config.settings import DATA_YEARS, SYMBOL


def fetch_hourly_data(years: int = DATA_YEARS) -> pd.DataFrame:
    """USD/JPY H1データを取得する。

    yfinanceの制限: 1h足は最大730日分取得可能。
    返却するDataFrameのカラム: Open, High, Low, Close, Volume
    インデックス: DatetimeIndex (UTC)
    """
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{years}y", interval="1h")

    if df.empty:
        raise RuntimeError(f"{SYMBOL}のデータ取得に失敗しました")

    # 不要なカラムを削除
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna()

    print(f"取得完了: {SYMBOL} H1 | {len(df)}本 | {df.index[0]} ~ {df.index[-1]}")
    return df


def fetch_h4_data(years: int = DATA_YEARS) -> pd.DataFrame:
    """H1データから H4 データをリサンプルして生成する。

    yfinanceはH4足を直接提供しないため、H1から変換する。
    """
    h1 = fetch_hourly_data(years)
    h4 = h1.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    print(f"H4リサンプル完了: {len(h4)}本")
    return h4


if __name__ == "__main__":
    df = fetch_hourly_data()
    print(df.head())
    print(f"\n統計:\n{df['Close'].describe()}")
