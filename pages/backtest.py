import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="過去検証", page_icon="🔬", layout="centered")

SAMPLE = """7203,トヨタ自動車
6758,ソニーグループ
9984,ソフトバンクグループ
8306,三菱UFJフィナンシャル
6861,キーエンス
9432,NTT
8035,東京エレクトロン
6501,日立製作所
7974,任天堂
4063,信越化学工業
6098,リクルートHD
9983,ファーストリテイリング
8316,三井住友フィナンシャル
4502,武田薬品工業
6902,デンソー
7267,本田技研工業
8058,三菱商事
2914,日本たばこ産業
6367,ダイキン工業
4519,中外製薬"""

HORIZONS = [5, 10, 20]


def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def signals_and_returns(df, p):
    close, vol, opn = df["Close"], df["Volume"], df["Open"]
    s = close.rolling(p["short"]).mean()
    l = close.rolling(p["long"]).mean()
    m75 = close.rolling(75).mean()
    diff = s - l

    sig = (diff > 0) & (diff.shift(1) <= 0)
    if p["rising"]:
        sig &= l > l.shift(1)
    if p["above75"]:
        sig &= close > m75
    if p["perfect"]:
        sig &= (s > l) & (l > m75)
    if p["vol_on"]:
        sig &= (vol / vol.rolling(20).mean().shift(1)) >= p["vol_ratio"]
    if p["value_on"]:
        sig &= (close * vol).rolling(20).mean() / 1e6 >= p["value_min"]
    if p["rsi_on"]:
        sig &= rsi_wilder(close) <= p["rsi_max"]
    if p["dev_on"]:
        sig &= (close - l) / l * 100 <= p["dev_max"]

    entry = opn.shift(-1)
    out = pd.DataFrame({"sig": sig.fillna(False)})
    for h in HORIZONS:
        out[f"r{h}"] = close.shift(-h) / entry - 1
    return out


def summarize(sig, base, h, cost):
    x = sig[f"r{h}"].dropna() - cost
    b = base[f"r{h}"].dropna() - cost
    n = len(x)
    if n == 0:
        return None
    se = x.std() / np.sqrt(n) if n > 1 else float("nan")
    return {
        "保有日数": f"{h}日", "件数": n,
        "勝率%": round((x > 0).mean() * 100, 1),
        "平均%": round(x.mean() * 100, 2),
        "±(95%)": round(1.96 * se * 100, 2),
        "中央値%": round(x.median() * 100, 2),
        "最悪%": round(x.min() * 100, 1),
        "最良%": round(x.max() * 100, 1),
        "基準平均%": round(b.mean() * 100, 2),
        "基準勝率%": round((b > 0).mean() * 100, 1),
        "差(平均)%": round((x.mean() - b.mean()) * 100, 2),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chunk(codes: tuple, period: str):
    import yfinance as yf

    tickers = [c + ".T" for c in codes]
    raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for c, t in zip(codes, tickers):
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna(subset=["Close", "Open"])
            if not d.empty:
                out[c] = d
        except KeyError:
            pass
    return out


def parse_tickers(text):
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.replace("\t", ",").split(",")]
        code = parts[0].upper().replace(".T", "")
        if code:
            items.append((code, parts[1] if len(parts) > 1 else ""))
    return items


st.title("🔬 過去検証")
st.caption("同じ条件で過去のシグナルを調べ、その後の値動きを集計します。")

with st.expander("① 銘柄", expanded=False):
    text = st.text_area("1行1銘柄", value=SAMPLE, height=180)

with st.expander("② 条件（スクリーニングと同じ）", expanded=True):
    c1, c2, c3 = st.columns(3)
    short = c1.number_input("短期線", 2, 50, 5)
    long_ = c2.number_input("長期線", 5, 200, 25)
    n_days = c3.number_input("検証期間(営業日)", 250, 1500, 1000, 50)
    rising = st.checkbox("長期線が上向き", True)
    above75 = st.checkbox("株価が75日線より上", True)
    perfect = st.checkbox("パーフェクトオーダー", False)
    vol_on = st.checkbox("出来高が平均の◯倍以上", True)
    vol_ratio = st.slider("出来高倍率", 0.5, 10.0, 1.5, 0.5, disabled=not vol_on)
    value_on = st.checkbox("平均売買代金が◯百万円以上", True)
    value_min = st.number_input("売買代金（百万円）", 1, 100000, 50, 10, disabled=not value_on)
    rsi_on = st.checkbox("RSI(14)が◯以下", True)
    rsi_max = st.slider("RSI上限", 30, 100, 70, disabled=not rsi_on)
    dev_on = st.checkbox("長期線との乖離が◯％以下", True)
    dev_max = st.slider("乖離率上限（%）", 1, 50, 10, disabled=not dev_on)
    cost = st.number_input("売買コスト（往復・%）", 0.0, 3.0, 0.2, 0.1) / 100

if st.button("▶ 検証を実行", type="primary", use_container_width=True):
    items = parse_tickers(text)
    if not items or short >= long_:
        st.warning("銘柄を入力し、短期線は長期線より小さくしてください。")
        st.stop()
    p = dict(short=int(short), long=int(long_), rising=rising, above75=above75, perfect=perfect,
             vol_on=vol_on, vol_ratio=vol_ratio, value_on=value_on, value_min=value_min,
             rsi_on=rsi_on, rsi_max=rsi_max, dev_on=dev_on, dev_max=dev_max)
    codes = [c for c, _ in items]
    years = int(np.ceil(n_days / 245)) + 1
    period = f"{min(years, 10)}y"

    bar = st.progress(0, text="株価を取得中...")
    data = {}
    try:
        for i in range(0, len(codes), 100):
            data.update(fetch_chunk(tuple(codes[i:i + 100]), period))
            bar.progress(min((i + 100) / len(codes), 1.0), text="株価を取得中...")
    except Exception as e:
        bar.empty()
        st.error(f"株価の取得に失敗しました: {e}")
        st.stop()
    bar.empty()

    frames = []
    for code, df in data.items():
        if len(df) < 200:
            continue
        o = signals_and_returns(df, p).iloc[-int(n_days):].copy()
        o["code"] = code
        frames.append(o)
    if not frames:
        st.error("検証できる株価データがありませんでした。")
        st.stop()
    allr = pd.concat(frames)
    sigr = allr[allr["sig"]]

    st.subheader("結果")
    st.write(f"対象 {len(frames)} 銘柄 / 期間 {allr.index.min():%Y-%m-%d} 〜 {allr.index.max():%Y-%m-%d} "
             f"/ シグナル {len(sigr)} 件（{sigr.index.nunique()} 日）")
    if len(sigr) == 0:
        st.info("この条件ではシグナルが出ませんでした。条件を減らしてみてください。")
        st.stop()

    rows = [r for h in HORIZONS if (r := summarize(sigr, allr, h, cost))]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("基準 = 同じ銘柄・同じ期間で、毎日買った場合の平均（条件に意味があるかの比較用）。"
               "売買コストは往復で差し引き済み。")

    st.subheader("再現性（時期を分けて確認）")
    h = st.selectbox("保有日数", HORIZONS, index=2, format_func=lambda v: f"{v}日")
    rcol = f"r{h}"
    by = []
    for y, g in sigr.groupby(sigr.index.year):
        b = allr[allr.index.year == y]
        x = g[rcol].dropna() - cost
        bb = b[rcol].dropna() - cost
        if len(x):
            by.append({"年": int(y), "件数": len(x), "勝率%": round((x > 0).mean() * 100, 1),
                       "平均%": round(x.mean() * 100, 2), "基準平均%": round(bb.mean() * 100, 2),
                       "差%": round((x.mean() - bb.mean()) * 100, 2)})
    if by:
        bydf = pd.DataFrame(by)
        st.dataframe(bydf, hide_index=True, use_container_width=True)
        plus = int((bydf["差%"] > 0).sum())
        st.write(f"基準を上回った年: **{plus} / {len(bydf)}**")

    mid = allr.index.min() + (allr.index.max() - allr.index.min()) / 2
    halves = []
    for name, mask_s, mask_a in [("前半", sigr.index <= mid, allr.index <= mid),
                                 ("後半", sigr.index > mid, allr.index > mid)]:
        x = sigr[mask_s][rcol].dropna() - cost
        bb = allr[mask_a][rcol].dropna() - cost
        if len(x):
            halves.append({"期間": name, "件数": len(x), "勝率%": round((x > 0).mean() * 100, 1),
                           "平均%": round(x.mean() * 100, 2), "差%": round((x.mean() - bb.mean()) * 100, 2)})
    if halves:
        st.dataframe(pd.DataFrame(halves), hide_index=True, use_container_width=True)

    st.subheader("銘柄別（件数の多い順）")
    per = []
    for code, g in sigr.groupby("code"):
        x = g[rcol].dropna() - cost
        if len(x):
            per.append({"コード": code, "件数": len(x), "勝率%": round((x > 0).mean() * 100, 1),
                        "平均%": round(x.mean() * 100, 2)})
    if per:
        st.dataframe(pd.DataFrame(per).sort_values("件数", ascending=False), hide_index=True,
                     use_container_width=True)

    st.info(
        "読み方の目安\n\n"
        "- 件数が少ない（目安30件未満）と、結果は偶然かもしれません。\n"
        "- 「差(平均)」が小さい・マイナスなら、この条件に特別な優位性は見えません。\n"
        "- 年ごと・前半後半で結果がバラバラなら、再現性は低いと考えられます。\n"
        "- 同じ日に多数の銘柄でシグナルが出る場合、独立した結果ではありません（相場全体の影響）。\n"
        "- 今の銘柄リストは「今も上場している銘柄」だけなので、実際より成績が良く出やすい（生存者バイアス）。"
    )

st.caption("過去の結果は将来を保証しません。投資判断の材料の一つです。")
