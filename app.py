"""ゴールデンクロス スクリーナー（トップページ）"""
import pandas as pd
import streamlit as st

from gc_core import SAMPLE, compute, fetch_all, fetch_market, parse_tickers, render_conditions, signal

st.set_page_config(page_title="ゴールデンクロス検出", page_icon="📈", layout="centered")

st.title("📈 ゴールデンクロス検出")
st.caption("短期線が長期線を上抜け、条件をすべて満たした銘柄を探します。株価はYahoo Finance（遅延・欠損あり）。"
           "条件の効果は左上メニューの「backtest」で過去検証できます。")

with st.expander("① 銘柄を入力", expanded=False):
    text = st.text_area("1行1銘柄（「7203」または「7203,トヨタ」）", value=SAMPLE, height=200)

p = render_conditions(show_days=True)

if st.button("▶ スクリーニング開始", type="primary", width="stretch"):
    items = parse_tickers(text)
    if not items:
        st.warning("銘柄コードを入力してください。")
        st.stop()
    if p["short"] >= p["long"]:
        st.warning("短期線は長期線より小さくしてください。")
        st.stop()

    names = dict(items)
    codes = [c for c, _ in items]
    try:
        data = fetch_all(codes, "2y")
    except Exception as e:
        st.error(f"株価の取得に失敗しました: {e}")
        st.stop()

    mkt = fetch_market("2y")
    if mkt is not None:
        st.write("地合い：日経平均は75日線より " + ("**上** 🟢" if bool(mkt.iloc[-1]) else "**下** 🔴"))
    elif p["market_on"]:
        st.warning("日経平均を取得できなかったため、「日経平均が75日線より上」の条件は外して判定しました。")
        p["market_on"] = False

    rows = []
    for code, df in data.items():
        if len(df) < p["long"] + p["days"] + 2:
            continue
        x = compute(df, p["short"], p["long"], mkt)
        sig = signal(x, p).iloc[-p["days"]:]
        if not sig.any() or not bool(x["above_now"].iloc[-1]):
            continue
        d = sig[sig].index[-1]  # クロス日
        last = x.iloc[-1]
        rows.append({
            "コード": code, "銘柄名": names.get(code, ""), "クロス日": d.strftime("%Y-%m-%d"),
            "終値": round(float(last["close"]), 1),
            "短期線": round(float(last["sma_s"]), 1), "長期線": round(float(last["sma_l"]), 1),
            "RSI": round(float(last["rsi_v"]), 1), "乖離率%": round(float(last["dev_v"]), 1),
            "出来高倍率": round(float(x.loc[d, "vol_x"]), 2) if pd.notna(x.loc[d, "vol_x"]) else None,
            "ATR%": round(float(last["atr_v"]), 1),
            "売買代金(百万円)": round(float(last["value_m"]), 1),
            "チャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
        })

    miss = len(codes) - len(data)
    note = f"（取得できなかった銘柄: {miss}）" if miss else ""
    if not rows:
        st.info(f"該当なし。対象 {len(codes)} 銘柄 {note}\n\n条件を減らすか、「直近N営業日以内」を増やすと見つかる場合があります。")
    else:
        res = pd.DataFrame(rows).sort_values("クロス日", ascending=False)
        st.success(f"該当 {len(res)} 銘柄 / 対象 {len(codes)} 銘柄 {note}")
        st.dataframe(res, hide_index=True, width="stretch",
                     column_config={"チャート": st.column_config.LinkColumn("チャート", display_text="開く")})
        st.download_button("CSVでダウンロード", res.drop(columns="チャート").to_csv(index=False).encode("utf-8-sig"),
                           file_name="golden_cross.csv", mime="text/csv", width="stretch")

st.caption("条件はクロス日の時点で判定しています（過去検証と同じ基準）。投資判断の材料の一つです。利益を保証するものではありません。")
