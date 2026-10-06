"""過去検証ページ：同じ条件で過去のシグナルを集計し、条件ごとの効果と再現性を比べる"""
import numpy as np
import pandas as pd
import streamlit as st

from gc_core import (COND_LABEL, CONDS, SAMPLE, compute, fetch_all, fetch_market, masks, parse_tickers,
                     render_conditions, signal)

st.set_page_config(page_title="過去検証", page_icon="🔬", layout="centered")
HORIZONS = [5, 10, 20]


def stat(r):
    r = r.dropna()
    if len(r) == 0:
        return 0, float("nan"), float("nan")
    return len(r), (r > 0).mean() * 100, r.mean() * 100


st.title("🔬 過去検証")
st.caption("買い＝シグナル翌営業日の始値、売り＝5/10/20営業日後の終値。条件はその日までのデータだけで判定しています。")

with st.expander("① 銘柄", expanded=False):
    text = st.text_area("1行1銘柄", value=SAMPLE, height=180)

p = render_conditions(show_days=False)
c1, c2 = st.columns(2)
n_days = int(c1.number_input("検証期間（営業日）", 250, 1500, 1000, 50, help="推奨: 1000日（約4年）。"))
cost = c2.number_input("売買コスト（往復%）", 0.0, 3.0, 0.2, 0.1, help="推奨: 0.2%前後（手数料＋値段のズレ）。") / 100

if st.button("▶ 検証を実行", type="primary", width="stretch"):
    items = parse_tickers(text)
    if not items or p["short"] >= p["long"]:
        st.warning("銘柄を入力し、短期線は長期線より小さくしてください。")
        st.stop()
    codes = [c for c, _ in items]
    period = f"{min(int(np.ceil(n_days / 245)) + 2, 10)}y"  # 200日線の準備期間を足す
    try:
        data = fetch_all(codes, period)
    except Exception as e:
        st.error(f"株価の取得に失敗しました: {e}")
        st.stop()
    mkt = fetch_market(period)
    if mkt is None:
        st.warning("日経平均を取得できなかったため、地合いの条件は使わずに検証します。")
        p["market_on"] = False

    frames = []
    for code, df in data.items():
        if len(df) < 260:
            continue
        x = compute(df, p["short"], p["long"], mkt)
        m = masks(x, p)
        t = pd.DataFrame(m)
        t["cross"] = x["cross"]
        t["sig"] = signal(x, p, m)
        entry = df["Open"].shift(-1)
        for h in HORIZONS:
            t[f"r{h}"] = df["Close"].shift(-h) / entry - 1 - cost
        t["code"] = code
        frames.append(t.iloc[-n_days:])
    if not frames:
        st.error("検証できる株価データがありませんでした。")
        st.stop()
    st.session_state["bt"] = dict(T=pd.concat(frames), p=dict(p), mkt_ok=mkt is not None, n_codes=len(frames))

if "bt" not in st.session_state:
    st.stop()

bt = st.session_state["bt"]
T, pp = bt["T"], bt["p"]
S = T[T["sig"]]
mid = T.index.min() + (T.index.max() - T.index.min()) / 2
st.divider()
st.write(f"対象 {bt['n_codes']} 銘柄 / 期間 {T.index.min():%Y-%m-%d} 〜 {T.index.max():%Y-%m-%d} / "
         f"今の設定のシグナル **{len(S)} 件**（{S.index.nunique()} 日）")

# ---- 1. 今の設定の成績
st.subheader("1. 今の設定の成績")
rows = []
for h in HORIZONS:
    n, w, mu = stat(S[f"r{h}"])
    _, bw, bmu = stat(T[f"r{h}"])
    _, cw, cmu = stat(T[T["cross"]][f"r{h}"])
    se = S[f"r{h}"].dropna().std() / np.sqrt(n) * 100 if n > 1 else float("nan")
    rows.append({"保有": f"{h}日", "件数": n, "勝率%": round(w, 1), "平均%": round(mu, 2), "±(95%)": round(1.96 * se, 2),
                 "クロスのみ平均%": round(cmu, 2), "毎日買った平均%": round(bmu, 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("「平均%」が「クロスのみ」や「毎日買った」を上回り、その差が「±(95%)」より大きければ、偶然ではない可能性が高まります。")

h = st.selectbox("以下の表の保有日数", HORIZONS, index=2, format_func=lambda v: f"{v}日")
rc = f"r{h}"

# ---- 2. 条件ごとの効果（推奨の追加項目を探す）
st.subheader("2. 条件ごとの効果と再現性")
st.caption("「クロスのみ」に条件を1つだけ足したら成績がどう変わるか。期間を前半・後半に分け、両方で改善した条件を「有望」としています。")

base = T[T["cross"]]
_, _, b_all = stat(base[rc])
_, _, b1 = stat(base[base.index <= mid][rc])
_, _, b2 = stat(base[base.index > mid][rc])


def noise(r):
    """平均リターンの誤差の目安（95%・%ポイント）。"""
    r = r.dropna()
    return 1.96 * r.std() / np.sqrt(len(r)) * 100 if len(r) > 1 else float("nan")


def judge(n, d, d1, d2, err):
    if n < 30:
        return "件数不足"
    if d > 0 and d1 > 0 and d2 > 0 and d > err:
        return "◎ 有望"
    if d > 0:
        return "△ 誤差の範囲"
    return "× 効果なし"


def cmp_row(label, g):
    n, w, mu = stat(g[rc])
    _, _, m1 = stat(g[g.index <= mid][rc])
    _, _, m2 = stat(g[g.index > mid][rc])
    d, d1, d2, err = mu - b_all, m1 - b1, m2 - b2, noise(g[rc])
    return {"条件": label, "件数": n, "勝率%": round(w, 1), "平均%": round(mu, 2), "改善": round(d, 2),
            "誤差±": round(err, 2), "前半改善": round(d1, 2), "後半改善": round(d2, 2), "判定": judge(n, d, d1, d2, err)}


cmp_rows = [{"条件": "クロスのみ（基準）", "件数": len(base[rc].dropna()), "勝率%": round(stat(base[rc])[1], 1),
             "平均%": round(b_all, 2), "改善": 0.0, "誤差±": round(noise(base[rc]), 2),
             "前半改善": 0.0, "後半改善": 0.0, "判定": "—"}]
for c in CONDS:
    k = c["key"]
    if k == "market" and not bt["mkt_ok"]:
        continue
    label = COND_LABEL[k] + ("（ON中）" if pp.get(k + "_on") else "")
    if "val" in c:
        label += f" ※{pp[k + '_val']:g}"
    cmp_rows.append(cmp_row(label, T[T["cross"] & T[k]]))
cmp_rows.append(cmp_row("今の設定（ON全部）", S))
cmp = pd.DataFrame(cmp_rows)
st.dataframe(cmp, hide_index=True, width="stretch")
good = [r["条件"] for r in cmp_rows[1:-1] if r["判定"] == "◎ 有望"]
st.write("**有望な条件:** " + ("、".join(good) if good else "なし（この期間・銘柄では、はっきり効く条件は見つかりませんでした）"))
st.caption("改善＝その条件を足したときの平均リターン − クロスのみの平均（%ポイント）。"
           "「◎ 有望」は、前半・後半ともに改善し、かつ改善が「誤差±」より大きいもの。"
           "値を使う条件は、上で設定した値で判定しています。")

# ---- 3. 年ごとの再現性
st.subheader("3. 今の設定の年ごとの成績")
by = []
for y, g in S.groupby(S.index.year):
    n, w, mu = stat(g[rc])
    _, _, bmu = stat(T[T.index.year == y][rc])
    if n:
        by.append({"年": int(y), "件数": n, "勝率%": round(w, 1), "平均%": round(mu, 2),
                   "毎日買った平均%": round(bmu, 2), "差%": round(mu - bmu, 2)})
if by:
    bydf = pd.DataFrame(by)
    st.dataframe(bydf, hide_index=True, width="stretch")
    st.write(f"毎日買った場合を上回った年: **{int((bydf['差%'] > 0).sum())} / {len(bydf)}**")

# ---- 4. 銘柄別
st.subheader("4. 銘柄別（件数の多い順）")
per = [{"コード": code, "件数": stat(g[rc])[0], "勝率%": round(stat(g[rc])[1], 1), "平均%": round(stat(g[rc])[2], 2)}
       for code, g in S.groupby("code") if stat(g[rc])[0]]
if per:
    st.dataframe(pd.DataFrame(per).sort_values("件数", ascending=False), hide_index=True, width="stretch")

st.info(
    "読み方の注意\n\n"
    "- 件数が30件未満の結果は、偶然の可能性が高いです。\n"
    "- 条件をたくさん試すと、偶然よく見えるものが混ざります。「前半・後半の両方で改善」を重視してください。\n"
    "- 同じ日に多数の銘柄でシグナルが出ると、相場全体の動きに結果が左右されます。\n"
    "- 今も上場している銘柄だけで調べているため、実際より成績が良く出やすいです（生存者バイアス）。"
)
st.caption("過去の結果は将来を保証しません。投資判断の材料の一つです。")
