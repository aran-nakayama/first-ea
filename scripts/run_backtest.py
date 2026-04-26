#!/usr/bin/env python3
"""バックテスト実行スクリプト"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SPREAD_PIPS, SPREAD_RATIO
from data.fetcher import fetch_hourly_data
from strategy.asian_breakout import prepare_data, AsianBreakoutStrategy
from backtest.engine import run_backtest, print_stats


def main():
    print("USD/JPY H1 データを取得中...")
    df = fetch_hourly_data()
    df.index = df.index.tz_localize(None)

    # Asian Breakout用のデータ前処理
    df = prepare_data(df)

    print(f"\nデータ: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})")
    print("バックテスト実行中...\n")

    # 最適化済みパラメータでバックテスト (基準スプレッド)
    stats, bt = run_backtest(
        df,
        AsianBreakoutStrategy,
        sl_pips=30,
        tp_pips=75,
        session_end=14,
    )
    print_stats(stats)

    # スプレッド感応度分析 (複数スプレッドで比較)
    spread_levels = [0.0, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print("\n--- スプレッド感応度分析 ---")
    spread_results = []
    for sp in spread_levels:
        sp_ratio = sp * 0.01 / 152.0
        s, _ = run_backtest(
            df, AsianBreakoutStrategy, spread=sp_ratio,
            sl_pips=30, tp_pips=75, session_end=14,
        )
        trades = s.get("_trades")
        pf_val = 0
        if trades is not None and len(trades) > 0:
            w = trades[trades["PnL"] > 0]["PnL"].sum()
            l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
            pf_val = w / l if l > 0 else float("inf")
        row = {
            "spread_pips": sp,
            "return_pct": s["Return [%]"],
            "win_rate": s["Win Rate [%]"],
            "pf": pf_val,
            "sharpe": s["Sharpe Ratio"] or 0,
            "mdd": s["Max. Drawdown [%]"],
            "trades": s["# Trades"],
            "final_equity": s["Equity Final [$]"],
        }
        spread_results.append(row)
        mark = " ★ 基準" if sp == SPREAD_PIPS else ""
        status = "勝ち" if row["return_pct"] > 0 else "負け"
        print(f"  {sp:>4.1f} pips → Return {row['return_pct']:+6.1f}% | "
              f"WR {row['win_rate']:.1f}% | PF {pf_val:.2f} | "
              f"MDD {row['mdd']:.1f}% [{status}]{mark}")

    # レバレッジ感応度分析
    from backtesting import Backtest as BT
    from config.settings import COMMISSION, INITIAL_EQUITY as EQUITY
    leverage_levels = [1, 2, 3, 5, 10, 15, 20, 25]
    print("\n--- レバレッジ感応度分析 ---")
    leverage_results = []
    for lev in leverage_levels:
        bt_lev = BT(
            df, AsianBreakoutStrategy,
            cash=EQUITY, commission=COMMISSION, spread=SPREAD_RATIO,
            exclusive_orders=True, margin=1.0 / lev,
        )
        s = bt_lev.run(sl_pips=30, tp_pips=75, session_end=14)
        trades = s.get("_trades")
        pf_val = 0
        if trades is not None and len(trades) > 0:
            w = trades[trades["PnL"] > 0]["PnL"].sum()
            l = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
            pf_val = w / l if l > 0 else float("inf")
        ret = s["Return [%]"]
        annual = ((1 + ret / 100) ** 0.5 - 1) * 100
        row = {
            "leverage": lev,
            "return_pct": ret,
            "annual_pct": annual,
            "win_rate": s["Win Rate [%]"],
            "pf": pf_val,
            "sharpe": s["Sharpe Ratio"] or 0,
            "mdd": s["Max. Drawdown [%]"],
            "trades": s["# Trades"],
            "final_equity": s["Equity Final [$]"],
        }
        leverage_results.append(row)
        print(f"  {lev:>3}x → Return {ret:>+7.1f}% (年利{annual:>+6.1f}%) | "
              f"MDD {s['Max. Drawdown [%]']:.1f}% | {s['Equity Final [$]']:>11,.0f} 円")

    # HTML レポート出力
    try:
        bt.plot(filename="backtest_report.html", open_browser=False,
               plot_volume=False, show_legend=False)
        _fix_report("backtest_report.html", stats, spread_results, leverage_results)
        print("\nレポート出力: backtest_report.html")
    except Exception as e:
        print(f"\nレポート出力スキップ: {e}")


def _build_spread_table(spread_results):
    """スプレッド感応度分析のHTMLテーブルを生成する。"""
    from config.settings import SPREAD_PIPS

    rows = ""
    for r in spread_results:
        is_base = r["spread_pips"] == SPREAD_PIPS
        is_profitable = r["return_pct"] > 0
        highlight = ' style="background:#e8f5e9;font-weight:700"' if is_base else ""
        ret_color = "#16a34a" if is_profitable else "#dc2626"
        status = "勝ち" if is_profitable else "負け"
        status_bg = "#dcfce7;color:#166534" if is_profitable else "#fef2f2;color:#991b1b"
        base_mark = " (基準)" if is_base else ""
        # 業者の目安
        broker = ""
        sp = r["spread_pips"]
        if sp <= 0.3:
            broker = "理論値"
        elif sp <= 0.5:
            broker = "SBI FX, DMM FX 等"
        elif sp <= 1.0:
            broker = "一般的な国内業者"
        elif sp <= 2.0:
            broker = "スプレッド広めの業者"
        elif sp <= 3.0:
            broker = "早朝・指標時"
        else:
            broker = "荒れ相場・年末年始"

        rows += f"""<tr{highlight}>
          <td style="text-align:center">{r['spread_pips']:.1f}{base_mark}</td>
          <td style="color:#888;font-size:11px">{broker}</td>
          <td style="text-align:right;color:{ret_color};font-weight:700">{r['return_pct']:+.1f}%</td>
          <td style="text-align:right">{r['final_equity']:,.0f}</td>
          <td style="text-align:right">{r['win_rate']:.1f}%</td>
          <td style="text-align:right">{r['pf']:.2f}</td>
          <td style="text-align:right">{r['sharpe']:.2f}</td>
          <td style="text-align:right">{r['mdd']:.1f}%</td>
          <td style="text-align:center"><span style="background:{status_bg};padding:1px 8px;border-radius:3px;font-size:11px;font-weight:700">{status}</span></td>
        </tr>\n"""

    return f"""
    <div style="margin-bottom:14px">
      <div class="card" style="min-width:100%">
        <div class="card-title">Spread Sensitivity (スプレッド感応度分析)</div>
        <p style="margin:0 0 8px;color:#555;font-size:12px">
          スプレッド (取引コスト) を変えた場合の結果比較。実際の業者ではスプレッドが変動するため、
          悪条件でも利益が出るか確認する。
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <thead>
            <tr style="border-bottom:2px solid #333;font-weight:700">
              <td style="text-align:center;padding:4px">Spread</td>
              <td style="padding:4px">想定業者</td>
              <td style="text-align:right;padding:4px">Return</td>
              <td style="text-align:right;padding:4px">最終資産</td>
              <td style="text-align:right;padding:4px">勝率</td>
              <td style="text-align:right;padding:4px">PF</td>
              <td style="text-align:right;padding:4px">Sharpe</td>
              <td style="text-align:right;padding:4px">MDD</td>
              <td style="text-align:center;padding:4px">判定</td>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </div>"""


def _build_leverage_table(leverage_results):
    """レバレッジ感応度分析のHTMLテーブルを生成する。"""
    rows = ""
    for r in leverage_results:
        lev = r["leverage"]
        ret = r["return_pct"]
        mdd = r["mdd"]
        annual = r["annual_pct"]
        is_profitable = ret > 0

        # おすすめ度の判定
        if mdd > -10 and is_profitable:
            rank = "おすすめ"
            rank_bg = "#dcfce7;color:#166534"
        elif mdd > -25 and is_profitable:
            rank = "中リスク"
            rank_bg = "#fef9c3;color:#854d0e"
        elif mdd > -40 and is_profitable:
            rank = "注意"
            rank_bg = "#ffedd5;color:#9a3412"
        elif is_profitable:
            rank = "危険"
            rank_bg = "#fef2f2;color:#991b1b"
        else:
            rank = "負け"
            rank_bg = "#fef2f2;color:#991b1b"

        # ロスカットリスク (SL30pips時の1回損失 + フラッシュクラッシュ150pips想定)
        loss_per_trade = lev * 0.2  # SL30pips ≈ 0.2% × レバ
        flash_loss = lev * 1.0  # 150pips ≈ 1.0% × レバ

        highlight = ' style="background:#e8f5e9;font-weight:700"' if lev == 5 else ""
        ret_color = "#16a34a" if is_profitable else "#dc2626"
        star = " (推奨)" if lev == 5 else ""

        rows += f"""<tr{highlight}>
          <td style="text-align:center;font-weight:700">{lev}x{star}</td>
          <td style="text-align:right;color:{ret_color};font-weight:700">{ret:+.1f}%</td>
          <td style="text-align:right">{annual:+.1f}%</td>
          <td style="text-align:right">{r['final_equity']:,.0f}</td>
          <td style="text-align:right">{mdd:.1f}%</td>
          <td style="text-align:right">{loss_per_trade:.1f}%</td>
          <td style="text-align:right">{flash_loss:.1f}%</td>
          <td style="text-align:center"><span style="background:{rank_bg};padding:1px 8px;border-radius:3px;font-size:11px;font-weight:700">{rank}</span></td>
        </tr>\n"""

    return f"""
    <div style="margin-bottom:14px">
      <div class="card" style="min-width:100%">
        <div class="card-title">Leverage Sensitivity (レバレッジ感応度分析)</div>
        <p style="margin:0 0 8px;color:#555;font-size:12px">
          レバレッジ倍率を変えた場合の結果比較。初期資金100万円・2年間。
          リターンとドローダウンはレバレッジに比例して拡大する。
          国内FX業者の上限は25倍。
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <thead>
            <tr style="border-bottom:2px solid #333;font-weight:700">
              <td style="text-align:center;padding:4px">レバレッジ</td>
              <td style="text-align:right;padding:4px">2年Return</td>
              <td style="text-align:right;padding:4px">年利</td>
              <td style="text-align:right;padding:4px">最終資産</td>
              <td style="text-align:right;padding:4px">MDD</td>
              <td style="text-align:right;padding:4px">1回損失</td>
              <td style="text-align:right;padding:4px">暴落時損失</td>
              <td style="text-align:center;padding:4px">判定</td>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        <p style="margin:8px 0 0;color:#888;font-size:11px">
          ※ 1回損失 = SL(-30pips)到達時の資金に対する損失率 / 暴落時損失 = フラッシュクラッシュ(-150pips)でSLを超えて約定した場合の想定損失率<br>
          ※ MDD = 最大ドローダウン（資産ピークからの最大下落率）。この数値分だけ一時的に資金が減る可能性がある
        </p>
      </div>
    </div>"""


def _fix_report(filepath: str, stats, spread_results=None, leverage_results=None):
    """レポートHTMLに戦略説明・検証条件・結果サマリー・グラフ凡例を追加する。"""
    from config.settings import INITIAL_EQUITY, SPREAD_PIPS

    with open(filepath, "r") as f:
        html = f.read()

    # --- 結果データの取得 ---
    trades = stats.get("_trades")
    pf = ""
    avg_win = ""
    avg_loss = ""
    max_consec = ""
    if trades is not None and len(trades) > 0:
        winning = trades[trades["PnL"] > 0]
        losing = trades[trades["PnL"] <= 0]
        if len(winning) > 0 and len(losing) > 0:
            pf = f"{winning['PnL'].sum() / abs(losing['PnL'].sum()):.2f}"
            avg_win = f"{winning['PnL'].mean():,.0f}"
            avg_loss = f"{abs(losing['PnL'].mean()):,.0f}"
            streak = 0
            max_s = 0
            for p in trades["PnL"]:
                if p <= 0:
                    streak += 1
                    max_s = max(max_s, streak)
                else:
                    streak = 0
            max_consec = str(max_s)

    sharpe = stats["Sharpe Ratio"]
    sharpe_str = f"{sharpe:.2f}" if sharpe else "N/A"

    patch = f"""
<style>
.report-header {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px;
    margin: 20px auto 12px;
}}
.report-header h1 {{
    font-size: 22px;
    margin: 0 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #333;
}}
.report-section {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}}
.report-section .card {{
    flex: 1;
    min-width: 250px;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
}}
.report-section .card-title {{
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
    color: #333;
}}
.report-section table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.report-section td {{
    padding: 2px 4px;
}}
.report-section td:first-child {{
    color: #666;
    white-space: nowrap;
}}
.report-section td:last-child {{
    text-align: right;
    font-weight: 600;
}}
.chart-legend {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}}
.chart-legend .card {{
    flex: 1;
    min-width: 200px;
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    line-height: 1.5;
}}
.chart-legend .card-title {{
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 4px;
}}
.chart-legend .item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.chart-legend .swatch {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
}}
.result-highlight {{
    font-size: 20px;
    font-weight: 700;
    color: {"#16a34a" if stats["Return [%]"] > 0 else "#dc2626"};
}}
</style>

<div class="report-header">
  <h1>USD/JPY Asian Range Breakout - Backtest Report</h1>

  <div class="report-section">
    <div class="card" style="flex:2;min-width:400px">
      <div class="card-title">Strategy (戦略概要)</div>
      <p style="margin:0 0 8px"><strong>Asian Range Breakout</strong> - アジア時間のレンジブレイクアウト戦略</p>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px;margin-bottom:10px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">トレードの流れ (1日のサイクル)</div>
        <table style="font-size:12px">
          <tr><td style="color:#4263eb">1. レンジ計測</td><td>00:00〜08:00 UTC (日本時間 9:00〜17:00)<br>アジア時間の高値 (Asian High) と安値 (Asian Low) を記録</td></tr>
          <tr><td colspan="2" style="padding:2px;color:#aaa">---</td></tr>
          <tr><td style="color:#e03131">2. IN (エントリー)</td><td>08:00〜14:00 UTC (日本時間 17:00〜23:00)<br>
            <strong>買い</strong>: 価格がAsian Highを上に突破 → 次の1時間足の始値で買い<br>
            <strong>売り</strong>: 価格がAsian Lowを下に突破 → 次の1時間足の始値で売り</td></tr>
          <tr><td colspan="2" style="padding:2px;color:#aaa">---</td></tr>
          <tr><td style="color:#16a34a">3. OUT (決済)</td><td>以下のいずれか最初に到達した条件で決済:<br>
            <strong>利確</strong>: エントリー価格から +75 pips (= +0.75円) → 勝ち<br>
            <strong>損切</strong>: エントリー価格から -30 pips (= -0.30円) → 負け<br>
            <strong>時間切れ</strong>: 21:00 UTC (日本時間 翌6:00) までに上記未到達 → 強制決済</td></tr>
        </table>
      </div>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px;margin-bottom:10px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">ポジションサイズ (動かす金額)</div>
        <table style="font-size:12px">
          <tr><td>投入額</td><td><strong>全資金の100%</strong> (常にフルポジション)</td></tr>
          <tr><td>同時ポジション</td><td>1つまで (前のトレードが決済されるまで次は入らない)</td></tr>
          <tr><td>1日の上限</td><td>1トレードまで (1日に何度もブレイクしても最初の1回のみ)</td></tr>
          <tr><td>最大損失/回</td><td>-30 pips = 資金の約 0.2% (100万円なら約 2,000円)</td></tr>
          <tr><td>最大利益/回</td><td>+75 pips = 資金の約 0.5% (100万円なら約 5,000円)</td></tr>
        </table>
      </div>

      <div style="background:#fff;border:1px solid #e9ecef;border-radius:4px;padding:10px 12px">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px">エントリーフィルター (見送り条件)</div>
        <table style="font-size:12px">
          <tr><td>レンジ幅</td><td>0.15円未満 or 1.5円超 → トレードしない</td></tr>
          <tr><td>時間帯</td><td>08:00〜14:00 UTC 以外 → トレードしない</td></tr>
          <tr><td>当日取引済</td><td>その日すでに1回取引した → トレードしない</td></tr>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Conditions (検証条件)</div>
      <table>
        <tr><td>通貨ペア</td><td>USD/JPY</td></tr>
        <tr><td>時間足</td><td>1時間足 (H1)</td></tr>
        <tr><td>期間</td><td>{stats['Start'].strftime('%Y/%m/%d')} ~ {stats['End'].strftime('%Y/%m/%d')}</td></tr>
        <tr><td>データ本数</td><td>{stats['Duration'].days * 24 // 1:,} 時間相当</td></tr>
        <tr><td>初期資金</td><td>{INITIAL_EQUITY:,.0f} 円</td></tr>
        <tr><td>スプレッド</td><td>{SPREAD_PIPS} pips</td></tr>
        <tr><td>取引手数料</td><td>0 円 (スプレッドに含む)</td></tr>
        <tr><td>データソース</td><td>yfinance (JPY=X)</td></tr>
      </table>

      <div style="margin-top:14px">
        <div class="card-title">Results (結果)</div>
        <div style="text-align:center;margin:4px 0 8px">
          <span class="result-highlight">{stats['Return [%]']:+.1f}%</span>
          <span style="font-size:13px;color:#666"> ({stats['Equity Final [$]']:,.0f} 円)</span>
        </div>
        <table>
          <tr><td>トレード数</td><td>{stats['# Trades']}</td></tr>
          <tr><td>勝率</td><td>{stats['Win Rate [%]']:.1f}%</td></tr>
          <tr><td>PF (利益/損失比)</td><td>{pf}</td></tr>
          <tr><td>Sharpe Ratio</td><td>{sharpe_str}</td></tr>
          <tr><td>最大ドローダウン</td><td>{stats['Max. Drawdown [%]']:.1f}%</td></tr>
          <tr><td>平均利益 / 平均損失</td><td>{avg_win} / {avg_loss} 円</td></tr>
          <tr><td>最大連敗数</td><td>{max_consec}</td></tr>
          <tr><td>Buy &amp; Hold</td><td>{stats['Buy & Hold Return [%]']:.1f}%</td></tr>
        </table>
      </div>
    </div>
  </div>

  __SPREAD_TABLE__

  __LEVERAGE_TABLE__

  <div class="chart-legend">
    <div class="card">
      <div class="card-title">1. Equity (資産推移)</div>
      <div class="item"><span class="swatch" style="background:#4dabf7"></span> 資産推移 (右肩上がり=順調)</div>
      <div class="item"><span class="swatch" style="background:#ffd43b"></span> ピーク資産</div>
      <div class="item"><span class="swatch" style="background:#fa5252"></span> 最大ドローダウン地点</div>
    </div>
    <div class="card">
      <div class="card-title">2. Profit / Loss (個別トレード損益)</div>
      <div class="item"><span class="swatch" style="background:#51cf66"></span> 勝ちトレード</div>
      <div class="item"><span class="swatch" style="background:#fa5252"></span> 負けトレード</div>
    </div>
    <div class="card">
      <div class="card-title">3. OHLC (ローソク足チャート)</div>
      <div class="item"><span class="swatch" style="background:#51cf66"></span> ローソク足 (USD/JPY)</div>
      <div class="item"><span class="swatch" style="background:#4263eb"></span> Asian High</div>
      <div class="item"><span class="swatch" style="background:#f76707"></span> Asian Low</div>
      <div class="item"><span class="swatch" style="background:#e03131"></span> エントリー/決済ポイント</div>
    </div>
  </div>
</div>

<script>
function hideBokehLegends() {{
    if (typeof Bokeh === 'undefined' || !Bokeh.index) return;
    for (const [id, view] of Object.entries(Bokeh.index)) {{
        const doc = (view.model || view).document;
        if (!doc || !doc._all_models) continue;
        for (const [mid, m] of doc._all_models) {{
            if (m.type === 'Legend') m.visible = false;
        }}
    }}
}}
document.addEventListener('DOMContentLoaded', function() {{
    [500, 1500, 3000].forEach(d => setTimeout(hideBokehLegends, d));
}});
if (document.readyState === 'complete') {{
    [100, 500, 1500].forEach(d => setTimeout(hideBokehLegends, d));
}}
</script>
"""
    # 比較テーブルのプレースホルダーを置換
    spread_html = _build_spread_table(spread_results) if spread_results else ""
    patch = patch.replace("__SPREAD_TABLE__", spread_html)
    leverage_html = _build_leverage_table(leverage_results) if leverage_results else ""
    patch = patch.replace("__LEVERAGE_TABLE__", leverage_html)

    # チャート本体 (Bokeh の div) の直前にパネルを挿入
    import re
    m = re.search(r'(<div[^>]*display:\s*contents[^>]*>)', html)
    if m:
        html = html[:m.start()] + patch + html[m.start():]
    else:
        html = html.replace("</body>", patch + "</body>")

    with open(filepath, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
