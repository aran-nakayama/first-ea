# Live Trading - Multi (TF+ST+MACD+ICHI) Portfolio

バックテストで実証された **Multi 4戦略ポートフォリオ** (月次勝率60%、年利4%@1x / 11%@3x) を
ペーパートレード or 実取引で実行する自動売買ツール。

## 構成

```
live/
├── signals.py     # 4戦略のシグナル抽出 (バックテストロジック純粋関数化)
├── broker.py      # PaperBroker (ペーパー) / OandaBroker (実取引、雛形)
├── feed.py        # PaperFeed (CSV再生) / YfinanceFeed (リアルタイム)
├── portfolio.py   # 4戦略×25%調整
├── runner.py      # メインループ (replay / live)
└── state/         # 永続化されたポジション・取引履歴

scripts/
├── run_paper.py        # ペーパートレード (リプレイ) ← まずはコレで検証
├── check_signals.py    # 現在のシグナル確認 (取引なし)
└── run_live.py         # ライブモード (TODO)
```

## クイックスタート

### 1. ペーパートレード (過去データで動作検証)

```bash
# 2024年だけリプレイ (~1分)
python3 -u scripts/run_paper.py --start 2024-01-01 --end 2024-12-31

# 過去5年でレバ3x検証
python3 -u scripts/run_paper.py --start 2020-01-01 --leverage 3

# 25年フル (~5分)
python3 -u scripts/run_paper.py --leverage 1

# 状態を永続化
python3 -u scripts/run_paper.py --state live/state/paper.json --start 2024-01-01
```

### 2. 現在のシグナル確認 (取引なし)

```bash
# 既存CSVから (オフライン)
python3 -u scripts/check_signals.py --csv data/usdjpy_h1_25years.csv

# yfinance最新データから (要インターネット、15分遅延あり)
python3 -u scripts/check_signals.py
```

これで「いま4戦略がそれぞれ何をしようとしているか」が表示されます。

### 3. 実取引 (OANDA - TODO)

OANDA で実取引するには以下が必要:

1. **OANDAデモアカウント取得** (無料): https://www.oanda.jp/
2. **APIトークン発行**: アカウント設定 → API → トークン生成
3. **依存パッケージ**:
   ```bash
   pip install --user oandapyV20
   ```
4. **環境変数設定**:
   ```bash
   export OANDA_API_KEY="your_token"
   export OANDA_ACCOUNT_ID="your_account_id"
   export OANDA_ENV="practice"   # デモ。本番は "live"
   ```
5. **OandaBroker の実装完成** (現在は `live/broker.py` に雛形のみ):
   - `place_order()` を `oandapyV20.endpoints.orders.OrderCreate` で実装
   - `close_position()` を `oandapyV20.endpoints.trades.TradeClose` で実装
   - `update_sl()` を `TradeCRCDO` で実装

## 実装の特徴

### バックテストとの一貫性

`live/signals.py` の各 `signal_*()` は、対応する `strategy/*.py` の `Strategy.next()` の
ロジックを純粋関数として再実装したもの。

検証: 2024年の paper trade 結果 (+5.5%, WR 37%, PF 1.32) は、
backtest の同期間結果とほぼ一致 (差は position sizing による)。

### スプレッド込み

PaperBroker は OANDA の USD/JPY 平均スプレッド **0.4 pips**(=0.004円) を、
エントリー時 + 決済時 の両方に適用。実取引に近い損益が得られる。

### Stop Loss 自動執行

ATR ベースのトレーリングストップを毎バー更新。価格がストップに到達すると
自動決済 (`broker.check_stops()` が冒頭で実行)。

### 戦略別の独立した状態

4戦略はそれぞれ独立してポジションを持てる。25年backtestと同様の動作。

## 推奨運用フロー

### Phase 1: ペーパートレード検証 (1-2週間)
1. `run_paper.py` で過去5年・10年・25年のリプレイを実行
2. 戦略別P&L、月次P&L、月次連敗回数を確認
3. レバレッジ別 (1x/3x/5x) のリスク許容度を体感

### Phase 2: ライブシミュレーション (1ヶ月)
1. OANDA デモアカウント取得
2. OandaBroker 実装完成
3. デモ口座で1ヶ月間自動運転
4. バックテストとの乖離を検証

### Phase 3: スモール実取引 (3ヶ月)
1. OANDA 本番口座開設、**少額** (10万円程度) で開始
2. レバレッジ **1x or 3x** から
3. 月次レポート作成、想定通り動いているか確認

### Phase 4: 本格運用
1. 資金追加
2. 多通貨ペア化 (EUR/USD, GBP/USD)
3. 月次勝率 70%以上を目指して継続改善

## 安全装置 (今後追加すべき)

現在の実装には未実装の機能が多くあります:

- ❌ **日次損失リミッター** (3%/日でストップ)
- ❌ **ロスカット閾値の自動チェック**
- ❌ **接続切断時の自動再接続**
- ❌ **Slack/LINE通知**
- ❌ **Web ダッシュボード**
- ❌ **緊急停止ボタン**

これらは Phase 2 移行前に必須実装。**現状は paper trading での検証用**として
お使いください。

## 注意事項

⚠️ **本ツールは教育・研究目的です。実取引で使用して損失が発生しても責任は負いません。**

⚠️ **実取引前に必ず:**
- デモ口座で最低1ヶ月の検証
- 余剰資金 (失っても生活に影響しない金額) で開始
- 自動売買を放置せず、定期的に監視
- 想定外の動作をしたら即停止
