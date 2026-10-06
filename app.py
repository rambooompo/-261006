"""ゴールデンクロス スクリーナー（トップページ）"""
import pandas as pd
import streamlit as st

from gc_core import SAMPLE, compute, fetch_all, fetch_market, fetch_pbr, parse_tickers, render_conditions, signal

st.set_page_config(page_title="ゴールデンクロス検出", page_icon="📈", layout="centered")

st.title("📈 買いシグナル検出")
st.caption("選んだ「買いのきっかけ」が出て、条件をすべて満たした銘柄を探します。株価はYahoo Finance（遅延・欠損あり）。"
           "きっかけ同士の比較や条件の効果は、左上メニューの「backtest」で過去検証できます。")

with st.expander("① 銘柄を入力", expanded=False):
    text = st.text_area("1行1銘柄（「7203」または「7203,トヨタ」）", value=SAMPLE, height=200)

p = render_conditions(show_days=True)

with st.expander("④ 割安さ（スクリーニングのみ）【新】", expanded=False):
    pbr_on = st.checkbox("PBRが◯倍以下〔推奨: 目安1.0〜1.5倍〕", value=False,
                         help="日本株では割安株が報われる傾向（バリュー効果）が確認されており、トレンド系の手法と組み合わせると"
                              "効果的という研究があります（Fama & French 2012、Asness 2011）。"
                              "現在のPBRしか取れないため、過去検証・自動探索では使えません。")
    pbr_max = float(st.number_input("└ PBR上限（倍）", 0.1, 20.0, 1.5, 0.1, disabled=not pbr_on))
    st.caption("条件を満たした銘柄だけPBRを取得します。PBRが取れない銘柄は「PBR不明」として残します。")

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
        x = compute(df, p["short"], p["long"], mkt, p["pb_pct"], p["rv_rsi"])
        sig = signal(x, p).iloc[-p["days"]:]
        if not sig.any() or (p["trigger"] == "gc" and not bool(x["above_now"].iloc[-1])):
            continue
        d = sig[sig].index[-1]  # シグナル日
        last = x.iloc[-1]
        rows.append({
            "コード": code, "銘柄名": names.get(code, ""), "シグナル日": d.strftime("%Y-%m-%d"),
            "終値": round(float(last["close"]), 1),
            "短期線": round(float(last["sma_s"]), 1), "長期線": round(float(last["sma_l"]), 1),
            "RSI": round(float(last["rsi_v"]), 1), "RSI(3)": round(float(last["rsi3_v"]), 1), "乖離率%": round(float(last["dev_v"]), 1),
            "出来高倍率": round(float(x.loc[d, "vol_x"]), 2) if pd.notna(x.loc[d, "vol_x"]) else None,
            "ATR%": round(float(last["atr_v"]), 1),
            "52週高値比%": round(float(last["near52_v"]), 1) if pd.notna(last["near52_v"]) else None,
            "シグナル日の上昇率%": round(float(x.loc[d, "jump_v"]), 1) if pd.notna(x.loc[d, "jump_v"]) else None,
            "売買代金(百万円)": round(float(last["value_m"]), 1),
            "チャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
        })

    if pbr_on and rows:
        with st.spinner("PBRを取得中..."):
            for r in rows:
                r["PBR"] = fetch_pbr(r["コード"])
        before = len(rows)
        rows = [r for r in rows if r["PBR"] is None or r["PBR"] <= pbr_max]
        unknown = sum(r["PBR"] is None for r in rows)
        st.caption(f"PBR {pbr_max:g}倍以下で絞り込み：{before}銘柄 → {len(rows)}銘柄"
                   + (f"（うちPBR不明 {unknown}銘柄）" if unknown else ""))
        for r in rows:
            r["PBR"] = f"{r['PBR']:.2f}" if r["PBR"] is not None else "不明"

    miss = len(codes) - len(data)
    note = f"（取得できなかった銘柄: {miss}）" if miss else ""
    if not rows:
        st.info(f"該当なし。対象 {len(codes)} 銘柄 {note}\n\n条件を減らすか、「直近N営業日以内」を増やすと見つかる場合があります。")
    else:
        res = pd.DataFrame(rows).sort_values("シグナル日", ascending=False)
        res = res[[c for c in res.columns if c != "チャート"] + ["チャート"]]  # チャートのリンクを右端に
        st.success(f"該当 {len(res)} 銘柄 / 対象 {len(codes)} 銘柄 {note}")
        st.dataframe(res, hide_index=True, width="stretch",
                     column_config={"チャート": st.column_config.LinkColumn("チャート", display_text="開く")})
        st.download_button("CSVでダウンロード", res.drop(columns="チャート").to_csv(index=False).encode("utf-8-sig"),
                           file_name="golden_cross.csv", mime="text/csv", width="stretch")

st.caption("条件はシグナル日の時点で判定しています（過去検証と同じ基準）。投資判断の材料の一つです。利益を保証するものではありません。")
