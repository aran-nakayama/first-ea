"""自動売買ランナー - メインループ。

モード:
- replay: ヒストリカルCSVを高速再生してシミュレーション
- live: yfinance/OANDAから1時間ごとに最新バーを取得して実取引

使い方は scripts/run_paper.py / scripts/run_live.py を参照。
"""

from typing import Optional
import logging
import time
import pandas as pd

from .feed import PaperFeed, YfinanceFeed
from .portfolio import MultiPortfolio
from .broker import PaperBroker

logger = logging.getLogger(__name__)


def run_replay(broker: PaperBroker, portfolio: MultiPortfolio,
               feed: PaperFeed, log_every: int = 100,
               progress_callback=None) -> dict:
    """ヒストリカルデータを高速再生してアルゴを検証する。"""
    logger.info(f"=== Replay モード開始 ({feed.total_bars()}本) ===")
    bar_count = 0
    start = time.time()

    for df_window in feed:
        bar_count += 1
        events = portfolio.step(df_window)

        if bar_count % log_every == 0:
            current_price = float(df_window["Close"].iloc[-1])
            stat = portfolio.status(current_price)
            elapsed = time.time() - start
            speed = bar_count / elapsed
            logger.info(
                f"[{df_window.index[-1]}] price={current_price:.3f} "
                f"equity={stat['equity']:,.0f} ({stat['drawdown_pct']:+.2f}%) "
                f"positions={len(stat['positions'])} trades={stat['n_trades']} "
                f"speed={speed:.0f} bars/s"
            )
            if progress_callback:
                progress_callback(bar_count, stat)

    final_price = float(df_window["Close"].iloc[-1])
    final_stat = portfolio.status(final_price)
    elapsed = time.time() - start
    logger.info(f"=== Replay 完了 ({bar_count}本, {elapsed:.1f}秒) ===")
    logger.info(f"最終資産: {final_stat['equity']:,.0f} 円 "
                f"({final_stat['drawdown_pct']:+.2f}%)")
    logger.info(f"総取引数: {final_stat['n_trades']}")
    return final_stat


def run_live(broker: PaperBroker, portfolio: MultiPortfolio,
             feed: YfinanceFeed, poll_interval_sec: int = 60,
             max_iterations: Optional[int] = None):
    """ライブモード - 1時間ごとにバー確定を検出して取引。

    H1足が確定するのは XX:00:00 (UTC)。15分後にyfinanceで取得可能。
    """
    logger.info("=== Live モード開始 ===")
    iteration = 0

    while True:
        iteration += 1
        try:
            if feed.has_new_bar():
                df = feed.get_latest_bars()
                logger.info(f"新バー検出 [{df.index[-1]}] price={df['Close'].iloc[-1]:.3f}")
                events = portfolio.step(df)
                for ev in events:
                    logger.info(f"  EVENT: {ev}")
                stat = portfolio.status(float(df["Close"].iloc[-1]))
                logger.info(
                    f"  STATUS: equity={stat['equity']:,.0f} "
                    f"({stat['drawdown_pct']:+.2f}%) "
                    f"positions={len(stat['positions'])}"
                )
            else:
                logger.debug("まだ新バーなし")

            if max_iterations and iteration >= max_iterations:
                logger.info(f"max_iterations={max_iterations} に到達、終了")
                break

            time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Ctrl+C でランナー停止")
            break
        except Exception as e:
            logger.exception(f"ループ中にエラー: {e}")
            time.sleep(poll_interval_sec)
