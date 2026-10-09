"""過去検証ページ：同じ条件で過去のシグナルを集計し、条件ごとの効果と再現性を比べる"""
import numpy as np
import pandas as pd
import streamlit as st

from gc_core import (COND_LABEL, CONDS, TRIGGERS, WIN_BASES, clustered_se, compute, fetch_all, fetch_market,
                     fetch_market_ohlc, fwd_returns, is_seq, make_template, masks, render_conditions,
                     render_template_saver, render_universe, render_win_basis, resolve_items, signal,
                     trigger_label, trigger_series)

st.set_page_config(page_title="過去検証", page_icon="🔬", layout="centered")
WIN_THR = 0.0  # 勝ちの基準（リターン、小数。下の入力欄で上書き）


def stat(r):
    """(件数, 勝率%, 平均%)。勝ち＝リターン（コスト引き後）が基準以上。"""
    r = r.dropna()
    if len(r) == 0:
        return 0, float("nan"), float("nan")
    return len(r), (r >= WIN_THR).mean() * 100, r.mean() * 100


st.title("🔬 過去検証")
st.caption("買い＝シグナル（きっかけ）の翌営業日の始値。売り＝購入日から数えて指定した営業日後の終値（損切り・利確なし）。"
           "条件はその日までのデータだけで判定しています。")

kind, text = render_universe()

p = render_conditions(show_days=False)
with st.expander("④ 勝ちの条件・期間", expanded=True):
    basis = render_win_basis("bt_win_basis")
    c1, c2 = st.columns(2)
    n_hold = int(c1.number_input("保有日数（購入後◯営業日）", 1, 250, 15, 1,
                                 help="購入日から数えて何営業日後に売るか。推奨: 15日（約3週間）。20営業日≒1か月。"))
    win_pct = c2.number_input("勝ちの基準（◯%以上）", -20.0, 50.0, 0.0, 0.5,
                              help="売買コストを引いたあとのリターンがこの値以上なら「勝ち」。推奨: 0%（プラスなら勝ち）。"
                                   "「+3%以上で勝ち」にしたいときは3を入れます。")
    c3, c4 = st.columns(2)
    n_days = int(c3.number_input("検証期間（営業日）", 250, 1500, 1000, 50, help="推奨: 1000日（約4年）。"))
    cost = c4.number_input("売買コスト（往復%）", 0.0, 3.0, 0.2, 0.1,
                           help="推奨: 0.2%前後（手数料＋値段のズレ）。") / 100
    st.caption(f"今の設定：購入後 **{n_hold}営業日** 時点で、"
               + ("リターン" if basis == "abs" else "日経平均との差") + f"が **{win_pct:g}%以上** なら勝ち")
WIN_THR = win_pct / 100

if st.button("▶ 検証を実行", type="primary", width="stretch"):
    items = resolve_items(kind, text)
    if not items or p["short"] >= p["long"]:
        st.warning("銘柄を入力し、短期線は長期線より小さくしてください。")
        st.stop()
    codes = [c for c, _ in items]
    horizons = sorted({5, 10, 20, 40, n_hold})  # 保有日数ごとの比較用に、いくつか一緒に計算
    period = f"{min(int(np.ceil((n_days + n_hold) / 245)) + 2, 10)}y"  # 200日線の準備期間を足す
    try:
        data = fetch_all(codes, period)
    except Exception as e:
        st.error(f"株価の取得に失敗しました: {e}")
        st.stop()
    mkt = fetch_market(period)
    if mkt is None:
        st.warning("日経平均を取得できなかったため、地合いの条件は使わずに検証します。")
        p["market_on"] = False
    mkt_px = fetch_market_ohlc(period) if basis == "rel" else None
    if basis == "rel" and mkt_px is None:
        st.warning("日経平均を取得できなかったため、「リターンがプラスなら勝ち」で検証します。")
        basis = "abs"

    frames = []
    for code, df in data.items():
        if len(df) < 260:
            continue
        x = compute(df, p["short"], p["long"], mkt, p["pb_pct"], p["rv_rsi"])
        m = masks(x, p)
        t = pd.DataFrame(m)
        t["cross"] = trigger_series(x, p["trigger"])  # 選んだきっかけ（以下「基準のきっかけ」）
        for k in TRIGGERS:
            t["trig_" + k] = x["trig_" + k]
        t["sig"] = signal(x, p, m)
        # 買い＝シグナル翌営業日の始値、売り＝購入日からh営業日後の終値（「日経平均に勝ったら」は日経平均との差）
        for h, r in fwd_returns(df, horizons, cost, 0, mkt_px if basis == "rel" else None).items():
            t[f"r{h}"] = r
        t["code"] = code
        frames.append(t.iloc[-n_days:])
    if not frames:
        st.error("検証できる株価データがありませんでした。")
        st.stop()
    st.session_state["bt"] = dict(T=pd.concat(frames), p=dict(p), mkt_ok=mkt is not None, n_codes=len(frames),
                                  horizons=horizons, n_hold=n_hold, universe=kind, basis=basis)

if "bt" not in st.session_state:
    st.stop()

bt = st.session_state["bt"]
T, pp = bt["T"], bt["p"]
S = T[T["sig"]]
mid = T.index.min() + (T.index.max() - T.index.min()) / 2
st.divider()
st.write(f"対象 {bt['n_codes']} 銘柄 / 期間 {T.index.min():%Y-%m-%d} 〜 {T.index.max():%Y-%m-%d} / "
         f"今の設定のシグナル **{len(S)} 件**（{S.index.nunique()} 日）")
BASIS = bt.get("basis", "abs")
if BASIS == "rel":
    st.info("勝ち負けと平均%は「日経平均との差」です（同じ期間の日経平均のリターンを差し引いた値）。"
            "プラスなら日経平均より上がった、という意味です。")

# ---- 0. きっかけ別の比較
st.subheader("0. 買いのきっかけ別の比較")
st.caption(f"同じ保有日数（購入後{bt['n_hold']}日）・同じ勝ちの基準で、きっかけを比べています。"
           "「絞り込みなし」はきっかけだけ、「絞り込みあり」は③でONにした条件をすべて足した場合です。")
cond_all = np.ones(len(T), dtype=bool)
for c in CONDS:
    if pp.get(c["key"] + "_on") and c["key"] in T:
        cond_all &= T[c["key"]].to_numpy(bool)
rc0 = f"r{bt['n_hold']}"
_, bw0, bmu0 = stat(T[rc0])
trig_rows = []
_keys = list(TRIGGERS) + ([pp["trigger"]] if is_seq(pp["trigger"]) else [])
for k in _keys:
    _col = "cross" if k == pp["trigger"] else "trig_" + k
    g0 = T[T[_col].to_numpy(bool)]
    g1 = T[T[_col].to_numpy(bool) & cond_all]
    n0, w0, m0 = stat(g0[rc0])
    n1, w1, m1 = stat(g1[rc0])
    _, wa, _ = stat(g0[g0.index <= mid][rc0])
    _, wb, _ = stat(g0[g0.index > mid][rc0])
    trig_rows.append({"きっかけ": trigger_label(k) + ("◀選択中" if k == pp["trigger"] else ""),
                      "件数": n0, "勝率%": round(w0, 1), "平均%": round(m0, 2),
                      "前半勝率%": round(wa, 1), "後半勝率%": round(wb, 1),
                      "絞り込みあり 件数": n1, "絞り込みあり 勝率%": round(w1, 1), "絞り込みあり 平均%": round(m1, 2)})
st.dataframe(pd.DataFrame(trig_rows), hide_index=True, width="stretch")
st.caption(f"参考：毎日買った場合 勝率 {bw0:.1f}%・平均 {bmu0:.2f}%。"
           "勝率だけでなく平均%も見てください（勝率が低くても、勝つときの利幅が大きいきっかけもあります）。")

# ---- 1. 今の設定の成績
st.subheader("1. 今の設定の成績")
HZ, NH = bt["horizons"], bt["n_hold"]
if WIN_THR != 0:
    st.caption(f"勝ちの基準: リターン {WIN_THR * 100:g}% 以上（コスト引き後）")
rows = []
for h in HZ:
    n, w, mu = stat(S[f"r{h}"])
    _, bw, bmu = stat(T[f"r{h}"])
    _, cw, cmu = stat(T[T["cross"]][f"r{h}"])
    se = clustered_se(S[f"r{h}"], S.index) * 100  # 同じ月の取引どうしの連動を考慮した誤差
    rows.append({"保有（購入後）": f"{h}日" + ("◀設定" if h == NH else ""), "件数": n, "勝率%": round(w, 1),
                 "平均%": round(mu, 2), "±(95%)": round(1.96 * se, 2),
                 "きっかけのみ勝率%": round(cw, 1), "きっかけのみ平均%": round(cmu, 2),
                 "毎日買った勝率%": round(bw, 1), "毎日買った平均%": round(bmu, 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("勝率は、リターン（売買コスト引き後）が勝ちの基準以上だった割合。"
           "「平均%」が「きっかけのみ」や「毎日買った」を上回り、その差が「±(95%)」より大きければ、偶然ではない可能性が高まります。"
           "（◀設定＝上で指定した保有日数。5・10・20・40日は比較用です）")

with st.expander("💾 この条件をテンプレとして保存", expanded=False):
    _n, _w, _mu = stat(S[f"r{NH}"])
    _label = trigger_label(pp["trigger"])
    _name = st.text_input("テンプレの名前", value=f"{_label} 勝率{_w:.0f}%（{pd.Timestamp.now():%m/%d}検証）",
                          key="_tpl_name_bt")
    _memo = (f"{_label}／購入後{NH}日・" + ("リターン" if BASIS == "abs" else "日経平均との差")
             + f"{WIN_THR * 100:g}%以上で勝ち／"
             f"過去検証：件数{_n}・勝率{_w:.1f}%・平均{_mu:.2f}%（{pd.Timestamp.now():%Y-%m-%d}）")
    render_template_saver(make_template(_name, {**pp, "universe": bt.get("universe", "topix500")}, _memo))

h = st.selectbox("以下の表の保有日数", HZ, index=HZ.index(NH), format_func=lambda v: f"購入後{v}日")
rc = f"r{h}"

# ---- 2. 条件ごとの効果（推奨の追加項目を探す）
st.subheader("2. 条件ごとの効果と再現性")
st.caption("「きっかけのみ」に条件を1つだけ足したら成績がどう変わるか。期間を前半・後半に分け、両方で改善した条件を「有望」としています。")

base = T[T["cross"]]
_, _, b_all = stat(base[rc])
_, _, b1 = stat(base[base.index <= mid][rc])
_, _, b2 = stat(base[base.index > mid][rc])


def noise(r):
    """平均リターンの誤差の目安（95%・%ポイント）。同じ月の取引どうしの連動を考慮。"""
    return 1.96 * clustered_se(r, r.index) * 100


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


cmp_rows = [{"条件": "きっかけのみ（基準）", "件数": len(base[rc].dropna()), "勝率%": round(stat(base[rc])[1], 1),
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
st.caption("改善＝その条件を足したときの平均リターン − きっかけのみの平均（%ポイント）。"
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
    "- 今も上場している銘柄だけで調べているため、実際より成績が良く出やすいです（生存者バイアス）。\n"
    "- TOPIX500や小型株500は「今の」構成銘柄です。過去に成長して入った銘柄も含むため、成績がよく出やすい点に注意してください。\n"
    "- 「±」の誤差は、同じ月の取引どうしが連動しやすいことを考慮して計算しています。"
)
st.caption("過去の結果は将来を保証しません。投資判断の材料の一つです。")
