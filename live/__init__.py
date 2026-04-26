"""Live trading package for Multi (TF+ST+MACD+ICHI) Portfolio.

主要モジュール:
- signals: 各戦略のエントリー/決済シグナル抽出
- broker: PaperBroker (in-memory) / OandaBroker (実取引)
- feed: 価格データフィード (yfinance / OANDA)
- portfolio: 4戦略×25%のポートフォリオ調整
- runner: メインループ
"""
