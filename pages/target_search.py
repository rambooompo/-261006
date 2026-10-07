"""目標勝率サーチ：勝率◯%以上（既定65%）になる「きっかけ×保有日数×条件」を自動で探し、後半の期間と偶然の水準で確かめる"""
from itertools import combinations, product

import numpy as np
import pandas as pd
import streamlit as st

from gc_core import (TRIGGERS, compute, fetch_all, fetch_market, make_template, render_template_saver,
                     render_universe, resolve_items)

st.set_page_config(page_title="目標勝率サーチ", page_icon="🎯", layout="centered")

BINARY = [
    ("rising", "長期線が上向き"), ("above75", "株価が75日線より上"), ("above200", "株価が200日線より上"),
    ("perfect", "パーフェクトオーダー"), ("market", "日経平均が75日線より上"), ("bullish", "シグナル日が陽線"),
    ("breakout", "直近20日の高値更新"), ("macd", "MACDがシグナルより上"),
]
VALUED = [  # (ID, 列名, 表示, 比較, 試す値)
    ("vol", "vol_x", "出来高が平均の{}倍以上", ">=", [1.5, 2.0, 3.0]),
    ("volmax", "vol_x", "出来高が平均の{}倍以下", "<=", [2.0, 3.0, 5.0]),
    ("value", "value_m", "平均売買代金{}百万円以上", ">=", [100, 300, 1000]),
    ("rsi", "rsi_v", "RSIが{}以下", "<=", [60, 70, 75]),
    ("dev", "dev_v", "長期線との乖離{}%以下", "<=", [5, 10, 15]),
    ("atr", "atr_v", "ATR(値動き)が株価の{}%以下", "<=", [3, 5, 7]),
    ("near52", "near52_v", "52週高値の{}%以上", ">=", [80, 90, 95]),
    ("jump", "jump_v", "シグナル日の上昇率{}%以下", "<=", [3, 5, 8]),
]


# ============================================================ 探索ロジック（ここから）
def build_pools(data, mkt, short, long_, holds, cost, n_days, triggers, pb_pct, rv_rsi):
    """きっかけごとに、シグナルが出た日（銘柄×日）の条件の値と、保有日数ごとのリターンを集める。"""
    cols = [k for k, _ in BINARY] + sorted({c for _, c, *_ in VALUED})
    frames = {k: [] for k in triggers}
    base = {h: [] for h in holds}  # 毎日買った場合（目標の難しさを示す参考値）
    for code, df in data.items():
        if len(df) < 260:
            continue
        x = compute(df, short, long_, mkt, pb_pct, rv_rsi)
        t = x[cols].copy()
        entry = df["Open"].shift(-1)
        for h in holds:
            t[f"r{h}"] = df["Close"].shift(-(h + 1)) / entry - 1 - cost
        t = t.iloc[-n_days:]
        for h in holds:
            base[h].append(t[f"r{h}"].dropna().to_numpy())
        for k in triggers:
            frames[k].append(t[x["trig_" + k].iloc[-n_days:].to_numpy(bool)])
    pools = {k: pd.concat(v).sort_index() for k, v in frames.items() if v}
    base = {h: np.concatenate(v) if v else np.array([]) for h, v in base.items()}
    return pools, base


def build_combos(pool, use_market, max_k):
    atoms = {}
    for key, label in BINARY:
        if key == "market" and not use_market:
            continue
        atoms[key] = [(label, pool[key].to_numpy(bool), (key, None))]
    for cid, col, fmt, op, vals in VALUED:
        arr = pool[col].to_numpy(float)
        atoms[cid] = [(fmt.format(v), (arr >= v) if op == ">=" else (arr <= v), (cid, v)) for v in vals]
    labels, masks, specs, seen = [], [], [], set()
    for k in range(1, max_k + 1):
        for ks in combinations(list(atoms), k):
            for opt in product(*[atoms[a] for a in ks]):
                m = opt[0][1].copy()
                for o in opt[1:]:
                    m &= o[1]
                key = m.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                labels.append(" ＋ ".join(o[0] for o in opt))
                masks.append(m)
                specs.append([o[2] for o in opt])
    return labels, np.array(masks), specs


def wilson_lb(p, n, z=1.96):
    n = np.maximum(n, 1)
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


class Half:
    """前半または後半の行について、組み合わせごとの件数・勝率・平均を行列計算でまとめて出す。"""

    def __init__(self, M, ret):
        self.M = M
        self.valid = (~np.isnan(ret)).astype(np.float32)
        self.ret0 = np.nan_to_num(ret).astype(np.float32)
        self.n = M @ self.valid

    def stats(self, ret0, thr, rows=None):
        M = self.M if rows is None else self.M[rows]
        n = self.n if rows is None else self.n[rows]
        nn = np.maximum(n, 1)
        win = ((ret0 >= thr) * self.valid).astype(np.float32)
        return (M @ win) / nn, (M @ (ret0 * self.valid)) / nn


def hunt(pools, holds, target, thr, min_tr, min_te, max_k, use_market, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    groups, rows = [], []
    n_combos = n_cand = 0
    for trig, pool in pools.items():
        if len(pool) < 100:
            continue
        mid = pool.index.min() + (pool.index.max() - pool.index.min()) / 2
        tr = np.asarray(pool.index <= mid)
        labels, masks, specs = build_combos(pool, use_market, max_k)
        q = pd.qcut(pool.index.to_series().rank(method="first"), 4, labels=False).to_numpy()
        for h in holds:
            ret = pool[f"r{h}"].to_numpy(np.float32)
            A = Half(masks[:, tr].astype(np.float32), ret[tr])
            B = Half(masks[:, ~tr].astype(np.float32), ret[~tr])
            wr_a, mu_a = A.stats(A.ret0, thr)
            elig = (A.n >= min_tr) & (B.n >= min_te)  # 件数の条件（値動きは見ていない）
            cand = np.where(elig & (wr_a >= target))[0]
            n_combos += len(labels)
            n_cand += len(cand)
            wr_b, mu_b = B.stats(B.ret0, thr, cand) if len(cand) else (np.array([]), np.array([]))
            ok = (wr_b >= target) & (mu_b > 0)
            vb = B.valid.astype(bool)
            base_b = float(((B.ret0 >= thr)[vb]).mean()) if vb.any() else float("nan")
            groups.append(dict(trig=trig, h=h, A=A, B=B, elig=elig, n_rows=len(pool), base_b=base_b,
                               pass_n=int(ok.sum())))
            allwin = (ret >= thr) & ~np.isnan(ret)
            for i, w2, m2, good in zip(cand, wr_b, mu_b, ok):
                if not good:
                    continue
                m = masks[i]
                beat = sum(1 for b in range(4)
                           if (sel := m & (q == b) & ~np.isnan(ret)).sum() >= 8 and allwin[sel].mean() >= target)
                rows.append({
                    "きっかけ": TRIGGERS[trig]["label"], "保有": f"{h}日", "条件の組み合わせ": labels[i],
                    "前半 件数": int(A.n[i]), "前半 勝率%": round(float(wr_a[i]) * 100, 1),
                    "後半 件数": int(B.n[i]), "後半 勝率%": round(float(w2) * 100, 1),
                    "後半 平均%": round(float(m2) * 100, 2), "4期間で目標以上": f"{beat}/4",
                    "_trig": trig, "_h": h, "_spec": specs[i],
                    "_score": float(wilson_lb(np.array(w2), np.array(B.n[i]))),
                })
    actual = sum(g["pass_n"] for g in groups)

    # 偶然の目安：前半・後半それぞれの中で値動きを入れ替え、同じ手順で「合格」が何件出るか
    null = []
    for _ in range(n_perm):
        cnt = 0
        for g in groups:
            A, B = g["A"], g["B"]
            a = A.ret0.copy()
            va = A.valid.astype(bool)
            a[va] = rng.permutation(a[va])
            b = B.ret0.copy()
            vb = B.valid.astype(bool)
            b[vb] = rng.permutation(b[vb])
            wr_a, _ = A.stats(a, thr)
            cand = np.where(g["elig"] & (wr_a >= target))[0]
            if len(cand):
                wr_b, mu_b = B.stats(b, thr, cand)
                cnt += int(((wr_b >= target) & (mu_b > 0)).sum())
        null.append(cnt)
    null = np.array(null)
    p = float((np.sum(null >= actual) + 1) / (len(null) + 1)) if len(null) else float("nan")

    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["_score"], ascending=False).reset_index(drop=True)
    base_tbl = pd.DataFrame([{"きっかけ": TRIGGERS[g["trig"]]["label"], "保有": f"{g['h']}日",
                              "シグナル件数": g["n_rows"], "後半の勝率%（きっかけのみ）": round(g["base_b"] * 100, 1),
                              "合格した組み合わせ": g["pass_n"]} for g in groups])
    return dict(table=table, base_tbl=base_tbl, actual=actual, null=null, p=p,
                n_combos=n_combos, n_cand=n_cand)
# ============================================================ 探索ロジック（ここまで）


st.title("🎯 目標勝率サーチ")
st.caption("「買いのきっかけ × 保有日数 × 条件の組み合わせ」を自動で試し、勝率が目標（既定65%）以上になる条件を探します。"
           "前半の期間で見つけた条件が、後半の期間でも目標を超えたものだけを「合格」にします。")

kind, text = render_universe()

with st.expander("② 目標と探す範囲", expanded=True):
    c1, c2 = st.columns(2)
    target_pct = c1.number_input("目標の勝率（%）", 50.0, 95.0, 65.0, 1.0,
                                 help="前半・後半の両方で、この勝率以上になる条件を探します。")
    win_pct = c2.number_input("勝ちの基準（リターン◯%以上）", -20.0, 50.0, 0.0, 0.5,
                              help="推奨: 0%。下げると勝率は上がりますが、意味のある勝ちではなくなります。")
    holds = st.multiselect("保有日数（購入後◯営業日）の候補", [5, 10, 15, 20, 40], default=[10, 15, 20],
                           help="候補を増やすほど、偶然の合格も増えます。推奨: 3つ程度。")
    trigs = st.multiselect("買いのきっかけ", list(TRIGGERS), default=list(TRIGGERS),
                           format_func=lambda k: TRIGGERS[k]["label"])

with st.expander("③ 細かい設定", expanded=False):
    c3, c4 = st.columns(2)
    min_tr = int(c3.number_input("前半の最小件数", 10, 1000, 30, 5, help="推奨: 30件以上。少ないと偶然の65%が増えます。"))
    min_te = int(c4.number_input("後半の最小件数", 5, 1000, 30, 5, help="推奨: 30件以上。"))
    c5, c6 = st.columns(2)
    max_k = int(c5.slider("同時に使う条件の最大数", 1, 3, 2, help="推奨: 2。増やすほど偶然の合格が増えます。"))
    n_perm = int(c6.slider("偶然チェックの回数", 10, 200, 30, 10, help="推奨: 30回以上（多いほど時間がかかります）。"))
    c7, c8, c9 = st.columns(3)
    n_days = int(c7.number_input("検証期間（営業日）", 250, 1500, 1000, 50))
    cost = c8.number_input("売買コスト（往復%）", 0.0, 3.0, 0.2, 0.1) / 100
    short = int(c9.number_input("短期線（日）", 2, 50, 5))
    c10, c11, c12 = st.columns(3)
    long_ = int(c10.number_input("長期線（日）", 5, 200, 25))
    pb_pct = float(c11.number_input("押し目の深さ（%）", 1.0, 20.0, 5.0, 0.5))
    rv_rsi = float(c12.number_input("逆張りRSI(3)基準", 1.0, 50.0, 20.0, 1.0))

if st.button("▶ 目標勝率サーチを実行", type="primary", width="stretch"):
    if not holds or not trigs:
        st.warning("保有日数ときっかけを1つ以上選んでください。")
        st.stop()
    items = resolve_items(kind, text)
    codes = [c for c, _ in items]
    period = f"{min(int(np.ceil((n_days + max(holds)) / 245)) + 2, 10)}y"
    try:
        data = fetch_all(codes, period)
    except Exception as e:
        st.error(f"株価の取得に失敗しました: {e}")
        st.stop()
    mkt = fetch_market(period)
    with st.spinner("条件の組み合わせを試しています（1〜数分かかることがあります）..."):
        pools, base = build_pools(data, mkt, short, long_, holds, cost, n_days, trigs, pb_pct, rv_rsi)
        R = hunt(pools, holds, target_pct / 100, win_pct / 100, min_tr, min_te, max_k, mkt is not None, n_perm)
    R.update(target=target_pct, win_pct=win_pct, n_codes=len(data), short=short, long=long_, pb_pct=pb_pct,
             rv_rsi=rv_rsi, universe=kind,
             base_daily={h: round(float((v >= win_pct / 100).mean()) * 100, 1) if len(v) else None
                         for h, v in base.items()})
    st.session_state["hunt"] = R

if "hunt" not in st.session_state:
    st.stop()

R = st.session_state["hunt"]
st.divider()
st.write(f"対象 {R['n_codes']} 銘柄 / 試した組み合わせ **{R['n_combos']:,} 通り**"
         f"（きっかけ×保有日数×条件）/ 前半で勝率{R['target']:g}%以上：{R['n_cand']:,} 通り")

st.subheader("1. 結果のまとめ")
null = R["null"]
c1, c2, c3 = st.columns(3)
c1.metric("合格（後半でも目標以上）", f"{R['actual']} 通り")
c2.metric("偶然でも合格する数（平均）", f"{null.mean():.1f}" if len(null) else "-")
c3.metric("偶然の上位5%", f"{np.percentile(null, 95):.0f}" if len(null) else "-")
reliable = len(null) > 0 and R["actual"] > 0 and R["p"] < 0.05
_hits = int((null >= R["actual"]).sum()) if len(null) else 0
_chance = f"偶然チェック{len(null)}回のうち、同じ数以上の合格が出たのは{_hits}回"
if R["actual"] == 0:
    st.warning(f"後半でも勝率{R['target']:g}%以上を保った条件はありませんでした。"
               "目標を下げる・保有日数を変える・銘柄の範囲を変えるなどで、もう一度試してください。")
elif reliable:
    st.success(f"合格の数が、偶然で出る数を上回っています（{_chance}）。"
               "合格した条件には、一定の再現性が期待できます。")
else:
    st.warning(f"合格の数は、偶然でも出る範囲です（{_chance}）。"
               "合格した条件も、たまたま勝率が高く見えている可能性が高いです。")
st.caption("値動きをランダムに入れ替えたデータで同じ探索を繰り返し、「偶然でも何通りくらい合格するか」と比べています。")

st.subheader("2. 目標の難しさ（参考）")
bt = R["base_tbl"].copy()
bt["毎日買った場合の勝率%"] = [R["base_daily"].get(int(s[:-1])) for s in bt["保有"]]
st.dataframe(bt, hide_index=True, width="stretch")
st.caption("きっかけだけで買った場合や、毎日買った場合の勝率です。これが目標に近いほど、目標は達成しやすくなります"
           "（相場全体が上昇していた時期は、何を買っても勝率が高く出やすい点に注意）。")

T = R["table"]
st.subheader(f"3. 合格した条件（{len(T)} 通り）")
if len(T) == 0:
    st.stop()
show = T[[c for c in T.columns if not c.startswith("_")]].head(30).copy()
show.insert(0, "No.", range(1, len(show) + 1))  # 下の「保存する条件」の番号と同じ
st.dataframe(show, hide_index=True, width="stretch")
st.caption("後半の成績で並べています（件数が少ないものは割り引いて評価）。"
           "「4期間で目標以上」は、全期間を4つに分けたうち、目標の勝率を超えた期間の数です。4/4に近いほど安定しています。")

st.subheader("4. テンプレとして保存")
if not reliable:
    st.warning("今回の結果は「偶然でも出る範囲」です。保存はできますが、使うときは注意してください。")
_i = st.selectbox("保存する条件", list(range(min(len(T), 30))),
                  format_func=lambda i: f"No.{i + 1}  {T.iloc[i]['きっかけ']}・{T.iloc[i]['保有']}：{T.iloc[i]['条件の組み合わせ']}",
                  key="_tpl_pick_hunt")
_row = T.iloc[_i]
_settings = {"trigger": _row["_trig"], "short": R["short"], "long": R["long"], "pb_pct": R["pb_pct"],
             "rv_rsi": R["rv_rsi"], "universe": R["universe"]}
for _k, _v in _row["_spec"]:
    _settings[_k + "_on"] = True
    if _v is not None:
        _settings[_k + "_val"] = float(_v)
# 選んだ条件ごとに入力欄を分けるので、条件を選び直すと名前も自動で切り替わる
_name = st.text_input("テンプレの名前（自由に変更できます）",
                      value=f"{_row['きっかけ']}・{_row['保有']}：{_row['条件の組み合わせ']}（勝率{_row['後半 勝率%']:.0f}%）"[:80],
                      key=f"_tpl_name_hunt_{_i}")
_memo = (f"{_row['きっかけ']}／購入後{_row['_h']}日・リターン{R['win_pct']:g}%以上で勝ち／目標勝率サーチの後半："
         f"件数{_row['後半 件数']}・勝率{_row['後半 勝率%']}%・平均{_row['後半 平均%']}%・4期間{_row['4期間で目標以上']}"
         f"（{pd.Timestamp.now():%Y-%m-%d}）")
render_template_saver(make_template(_name, _settings, _memo))
st.caption(f"保有日数（購入後{_row['_h']}日）はテンプレに含まれないので、売るタイミングの目安として覚えておいてください。")

st.info(
    "読み方の注意\n\n"
    "- 勝率が高くても、負けたときの損が大きいと全体ではマイナスになります。「後半 平均%」も必ず見てください。\n"
    "- 「偶然でも出る範囲」と出たときは、合格した条件も使わないほうが安全です。\n"
    "- 今も上場している銘柄だけで調べているため、実際より成績が良く出やすいです（生存者バイアス）。\n"
    "- 結果は期間や銘柄で変わります。銘柄の範囲を変えて、同じ条件が何度も合格するかを確かめてください。"
)
st.caption("過去の結果は将来を保証しません。投資判断の材料の一つです。")
