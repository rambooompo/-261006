"""共通部品：指標の計算・条件・入力画面・株価取得（app.py と pages/backtest.py から使う）"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _dv(key, **default):
    """テンプレで値が入っている欄は既定値を渡さない（Streamlitの二重設定の警告を防ぐ）。"""
    return {} if key in st.session_state else default

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
4519,中外製薬
1332,ニッスイ
1801,大成建設
1802,大林組
1803,清水建設
1812,鹿島建設
1925,大和ハウス工業
1928,積水ハウス
2002,日清製粉グループ本社
2269,明治ホールディングス
2282,日本ハム
2501,サッポロホールディングス
2502,アサヒグループHD
2503,キリンホールディングス
2587,サントリー食品インターナショナル
2801,キッコーマン
2802,味の素
2897,日清食品ホールディングス
3086,J.フロント リテイリング
3099,三越伊勢丹ホールディングス
3382,セブン&アイ・ホールディングス
3402,東レ
3405,クラレ
3407,旭化成
3436,SUMCO
3659,ネクソン
3861,王子ホールディングス
4005,住友化学
4021,日産化学
4042,東ソー
4151,協和キリン
4183,三井化学
4188,三菱ケミカルグループ
4204,積水化学工業
4452,花王
4503,アステラス製薬
4507,塩野義製薬
4523,エーザイ
4543,テルモ
4568,第一三共
4578,大塚ホールディングス
4661,オリエンタルランド
4684,オービック
4704,トレンドマイクロ
4755,楽天グループ
4901,富士フイルムHD
4911,資生堂
5019,出光興産
5020,ENEOSホールディングス
5101,横浜ゴム
5108,ブリヂストン
5201,AGC
5333,日本碍子
5401,日本製鉄
5411,JFEホールディングス
5713,住友金属鉱山
5802,住友電気工業
6103,オークマ
6113,アマダ
6178,日本郵政
6301,コマツ
6326,クボタ
6471,日本精工
6503,三菱電機
6504,富士電機
6506,安川電機
6594,ニデック
6645,オムロン
6702,富士通
6723,ルネサスエレクトロニクス
6752,パナソニックHD
6762,TDK
6857,アドバンテスト
6920,レーザーテック
6954,ファナック
6971,京セラ
6981,村田製作所
7011,三菱重工業
7201,日産自動車
7261,マツダ
7269,スズキ
7270,SUBARU
7733,オリンパス
7741,HOYA
7751,キヤノン
7832,バンダイナムコHD
7911,TOPPANホールディングス
8001,伊藤忠商事
8002,丸紅
8031,三井物産
8053,住友商事
8113,ユニ・チャーム
8267,イオン
8309,三井住友トラスト・HD
8411,みずほフィナンシャルG
8591,オリックス
8601,大和証券グループ本社
8604,野村ホールディングス
8725,MS&ADインシュアランスG
8766,東京海上ホールディングス
8801,三井不動産"""

# ------------------------------------------------------------ 買いのきっかけ（エントリー）
TRIGGERS = {
    "gc": dict(label="ゴールデンクロス", desc="短期線が長期線を下から上に抜けた日。"),
    "pullback": dict(label="押し目買い", desc="上昇トレンド中（長期線＞75日線、75日線が上向き）に、株価が長期線より"
                                             "◯%以上下に沈んだ最初の日。上昇中の一時的な下げを拾います。"),
    "breakout": dict(label="52週高値ブレイク", desc="終値が過去約1年の最高値（前日までの52週高値）を初めて上回った日。"),
    "reversal": dict(label="短期逆張り", desc="長期の上昇トレンド中（株価＞200日線）に、短期RSI(3)が◯以下まで"
                                            "売られた最初の日。上昇中の短い売られすぎを拾います。"),
}

# ------------------------------------------------------------ 条件の定義
# rec = 画面に出す推奨値（一般的な目安）。help = 「?」を押すと出る説明。
CONDS = [
    # --- トレンド
    dict(key="rising", group="トレンド", label="長期線が上向き", on=True, rec="推奨: ON",
         help="長期線（25日線）が前日より上がっていること。下降トレンド中の一時的な反発によるクロスを除けます。"),
    dict(key="above75", group="トレンド", label="株価が75日線より上", on=True, rec="推奨: ON",
         help="中期のトレンドが上向きの銘柄だけに絞ります。"),
    dict(key="above200", group="トレンド", label="株価が200日線より上【新】", on=False, rec="推奨: 検証で判断",
         help="長期（約1年）の上昇トレンド中の銘柄だけに絞ります。75日線と重なる部分が多いので、件数とのバランスで判断。"),
    dict(key="perfect", group="トレンド", label="パーフェクトオーダー（短期＞長期＞75日）", on=False, rec="推奨: OFF",
         help="移動平均線が上から順に並ぶ強いトレンド。クロス直後は成立しにくく、件数が大きく減るためOFF推奨。"),
    dict(key="near52", group="トレンド", label="株価が52週高値の◯%以上【新】", on=False, rec="推奨: 検証で判断（目安90%）",
         val=dict(label="52週高値に対する割合（%）", default=90.0, min=50.0, max=100.0, step=1.0),
         help="終値が過去約1年の最高値にどれだけ近いか。52週高値に近い株ほど、その後のリターンが高い傾向があるという研究"
              "（George & Hwang 2004、米国株）に基づく条件。日本株での効果は自動探索で確認してください。"),
    # --- 出来高・流動性
    dict(key="vol", group="出来高・流動性", label="シグナル日の出来高が平均の◯倍以上", on=True, rec="推奨: 1.5〜2.0倍",
         val=dict(label="出来高倍率（倍）", default=1.5, min=0.5, max=10.0, step=0.5),
         help="シグナル日の出来高が直近20日平均より多いこと。買いが本当に入っているかの確認です。"),
    dict(key="volmax", group="出来高・流動性", label="シグナル日の出来高が平均の◯倍以下【新】", on=False,
         rec="推奨: 検証で判断（目安3倍）",
         val=dict(label="出来高倍率の上限（倍）", default=3.0, min=1.0, max=20.0, step=0.5),
         help="出来高が極端に多い日を除きます。売買が多い銘柄ほど、その後のリターンが低く、上昇の反転も早いという研究"
              "（Lee & Swaminathan 2000）に基づく条件。「◯倍以上」と組み合わせると、出来高の範囲を指定できます。"),
    dict(key="value", group="出来高・流動性", label="平均売買代金が◯百万円以上", on=True, rec="推奨: 100百万円以上",
         val=dict(label="売買代金（百万円）", default=100.0, min=1.0, max=100000.0, step=10.0),
         help="1日あたりの売買代金（20日平均）。少ない銘柄は値が飛びやすく、思った値段で売買できないことがあります。"),
    # --- 過熱感
    dict(key="rsi", group="過熱感", label="RSI(14)が◯以下", on=True, rec="推奨: 70",
         val=dict(label="RSI上限", default=70.0, min=30.0, max=100.0, step=5.0),
         help="RSIは買われすぎ度合いの指標（0〜100）。70超は過熱とされ、クロス後すぐ反落しやすいと言われます。"),
    dict(key="dev", group="過熱感", label="長期線との乖離が◯％以下", on=True, rec="推奨: 5〜10%",
         val=dict(label="乖離率上限（%）", default=10.0, min=1.0, max=50.0, step=1.0),
         help="株価が長期線からどれだけ離れているか。大きいほど、すでに上がりきった後のクロスです。"),
    dict(key="jump", group="過熱感", label="シグナル日の上昇率が◯%以下【新】", on=False, rec="推奨: 検証で判断（目安5%）",
         val=dict(label="前日比の上限（%）", default=5.0, min=0.0, max=30.0, step=0.5),
         help="急騰した日のクロスを除きます。日本株では1日に大きく上がった株が直後の数日で下がりやすいという研究"
              "（Pham 2007）があります。ただし影響は短期で、主な効果は「高値で買わない」ことです。"),
    dict(key="atr", group="過熱感", label="値動きの荒さ（ATR）が◯％以下【新】", on=False, rec="推奨: 検証で判断（目安5%）",
         val=dict(label="ATR上限（株価比 %）", default=5.0, min=1.0, max=20.0, step=0.5),
         help="1日の平均的な値幅（14日）が株価の何％か。大きい銘柄はだましで大きく損をしやすいです。"),
    # --- 地合い・タイミング
    dict(key="market", group="地合い・タイミング", label="日経平均が75日線より上【新】", on=True, rec="推奨: ON",
         help="相場全体が下落基調のときは、個別株のクロスも失敗しやすいと言われます。相場全体の向きで絞ります。"),
    dict(key="bullish", group="地合い・タイミング", label="シグナル日が陽線（終値＞始値）【新】", on=False, rec="推奨: 検証で判断",
         help="クロスした日にしっかり買われて引けたか。弱い形のクロスを除きます。"),
    dict(key="breakout", group="地合い・タイミング", label="終値が直近20日の最高値【新】", on=False, rec="推奨: 検証で判断",
         help="クロスと同時に直近の高値を抜けたか（ブレイクアウト）。強い動きだけに絞りますが、件数は減ります。"),
    dict(key="macd", group="地合い・タイミング", label="MACDがシグナルより上【新】", on=False, rec="推奨: 検証で判断",
         help="別の勢いの指標（MACD）でも上向きかを確認します。移動平均と似た性質なので、効果は限定的なことも。"),
]
COND_LABEL = {c["key"]: c["label"].replace("【新】", "") for c in CONDS}


def render_conditions(show_days=True, key_prefix=""):
    """条件の入力画面を出して、設定値の辞書を返す。"""
    p = {}
    with st.expander("② 買いのきっかけ", expanded=True):
        p["trigger"] = st.radio("きっかけ", list(TRIGGERS), format_func=lambda k: TRIGGERS[k]["label"],
                                key=key_prefix + "trigger",
                                help="どのタイミングで買うか。過去検証ページでは、4つのきっかけを同じ条件で比べられます。")
        st.caption(TRIGGERS[p["trigger"]]["desc"])
        cols = st.columns(3 if show_days else 2)
        p["short"] = int(cols[0].number_input("短期線（日）", min_value=2, max_value=50, step=1, key=key_prefix + "short",
                                              **_dv(key_prefix + "short", value=5),
                                              help="推奨: 5日。ゴールデンクロスで使います。"))
        p["long"] = int(cols[1].number_input("長期線（日）", min_value=5, max_value=200, step=1, key=key_prefix + "long",
                                             **_dv(key_prefix + "long", value=25),
                                             help="推奨: 25日。ゴールデンクロスと押し目買いで使います。"))
        if show_days:
            p["days"] = int(cols[2].number_input("直近N営業日以内", min_value=1, max_value=20, step=1, key=key_prefix + "days",
                                                 **_dv(key_prefix + "days", value=1),
                                                 help="推奨: 1〜3日。大きくすると、少し前のシグナルも拾います。"))
        c4, c5 = st.columns(2)
        p["pb_pct"] = float(c4.number_input("押し目の深さ（長期線から◯%下）", min_value=1.0, max_value=20.0, step=0.5,
                                            key=key_prefix + "pb_pct", **_dv(key_prefix + "pb_pct", value=5.0),
                                            help="押し目買いで使います。推奨: 5%前後。深いほど件数は減りますが、"
                                                 "反発が大きい傾向があるという検証があります。"))
        p["rv_rsi"] = float(c5.number_input("逆張りのRSI(3)基準（◯以下）", min_value=1.0, max_value=50.0, step=1.0,
                                            key=key_prefix + "rv_rsi", **_dv(key_prefix + "rv_rsi", value=20.0),
                                            help="短期逆張りで使います。推奨: 20前後（10〜20）。小さいほど強い売られすぎです。"))
        st.caption("推奨: 短期5日・長期25日・直近1〜3日・押し目5%・RSI(3)20以下。"
                   "きっかけを変えたら、③の絞り込み（特に出来高・乖離の条件）も見直してください。")

    groups = []
    for c in CONDS:
        if c["group"] not in groups:
            groups.append(c["group"])
    st.markdown("**③ 絞り込み**（ONにしたものだけ適用。〔 〕内は一般的な推奨値）")
    for g in groups:
        with st.expander(g, expanded=False):
            for c in [c for c in CONDS if c["group"] == g]:
                on = st.checkbox(f"{c['label']}〔{c['rec']}〕", help=c["help"], key=key_prefix + c["key"] + "_on",
                                 **_dv(key_prefix + c["key"] + "_on", value=c["on"]))
                p[c["key"] + "_on"] = on
                if "val" in c:
                    v = c["val"]
                    p[c["key"] + "_val"] = float(st.number_input(
                        f"└ {v['label']}", min_value=float(v["min"]), max_value=float(v["max"]), step=float(v["step"]),
                        disabled=not on, key=key_prefix + c["key"] + "_val",
                        **_dv(key_prefix + c["key"] + "_val", value=float(v["default"]))))
    return p


# ------------------------------------------------------------ 計算
def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def compute(df, short, long_, mkt_ok=None, pb_pct=5.0, rv_rsi=20.0):
    """各日の指標・各条件・各きっかけを計算（その日までのデータだけを使う）。"""
    c, o, v = df["Close"], df["Open"], df["Volume"]
    h = df["High"] if "High" in df else c
    lo = df["Low"] if "Low" in df else c
    s = c.rolling(short).mean()
    l = c.rolling(long_).mean()
    m75 = c.rolling(75).mean()
    m200 = c.rolling(200).mean()
    diff = s - l
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)

    x = pd.DataFrame(index=df.index)
    x["close"], x["sma_s"], x["sma_l"] = c, s, l
    x["cross"] = (diff > 0) & (diff.shift(1) <= 0)
    x["above_now"] = diff > 0
    x["vol_x"] = v / v.rolling(20).mean().shift(1)
    x["value_m"] = (c * v).rolling(20).mean() / 1e6
    x["rsi_v"] = rsi_wilder(c)
    x["dev_v"] = (c - l) / l * 100
    x["atr_v"] = tr.rolling(14).mean() / c * 100
    x["near52_v"] = c / h.rolling(245, min_periods=200).max() * 100  # 52週（約245営業日）高値に対する割合
    x["jump_v"] = (c / c.shift(1) - 1) * 100  # 前日比（%）
    x["rising"] = l > l.shift(1)
    x["above75"] = c > m75
    x["above200"] = c > m200
    x["perfect"] = (s > l) & (l > m75)
    x["bullish"] = c > o
    x["breakout"] = c >= c.rolling(20).max()
    x["macd"] = macd > macd.ewm(span=9, adjust=False).mean()
    if mkt_ok is not None:
        x["market"] = mkt_ok.reindex(x.index, method="ffill").fillna(False).astype(bool)
    else:
        x["market"] = True

    # --- 買いのきっかけ（どれも「条件を満たした最初の日」だけを1回のシグナルとする）
    x["trig_gc"] = x["cross"]
    up = (l > m75) & (m75 > m75.shift(5))
    deep = x["dev_v"] <= -pb_pct
    x["trig_pullback"] = up & deep & ~deep.shift(1, fill_value=False)
    brk = c > h.shift(1).rolling(245, min_periods=200).max()
    x["trig_breakout"] = brk & ~brk.shift(1, fill_value=False)
    rsi3 = rsi_wilder(c, 3)
    low = rsi3 <= rv_rsi
    x["trig_reversal"] = (c > m200) & low & ~low.shift(1, fill_value=False)
    x["rsi3_v"] = rsi3
    return x


def masks(x, p):
    """条件ごとに True/False の列を返す（値を使う条件は p の値で判定）。"""
    m = {
        "rising": x["rising"], "above75": x["above75"], "above200": x["above200"], "perfect": x["perfect"],
        "vol": x["vol_x"] >= p["vol_val"], "value": x["value_m"] >= p["value_val"],
        "rsi": x["rsi_v"] <= p["rsi_val"], "dev": x["dev_v"] <= p["dev_val"], "atr": x["atr_v"] <= p["atr_val"],
        "near52": x["near52_v"] >= p["near52_val"], "volmax": x["vol_x"] <= p["volmax_val"],
        "jump": x["jump_v"] <= p["jump_val"],
        "market": x["market"], "bullish": x["bullish"], "breakout": x["breakout"], "macd": x["macd"],
    }
    return {k: v.fillna(False).astype(bool) for k, v in m.items()}


def signal(x, p, m=None):
    m = m or masks(x, p)
    sig = x["trig_" + p.get("trigger", "gc")].copy()
    for c in CONDS:
        if p.get(c["key"] + "_on"):
            sig &= m[c["key"]]
    return sig


# ------------------------------------------------------------ 株価の取得
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


# ------------------------------------------------------------ 銘柄の範囲（JPX公式の上場銘柄一覧から選ぶ）
JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_FILE = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
UNIVERSES = {
    "topix500": "TOPIX500（大型・中型株 約500銘柄）",
    "small500": "TOPIX Small 500（小型株 約500銘柄）",
    "prime": "プライム市場からランダムに500銘柄",
    "growth": "グロース市場からランダムに最大500銘柄",
    "manual": "手入力（下の欄の銘柄）",
}


@st.cache_data(ttl=86400, show_spinner=False)
def load_jpx_list():
    """東証の上場銘柄一覧（毎月更新）を読み込み、内国株式だけを返す。列: code, name, market, size"""
    import io
    import re
    from urllib.parse import urljoin

    import requests

    hdr = {"User-Agent": "Mozilla/5.0"}
    url = JPX_FILE
    try:  # ページからファイルの最新の場所を探す（見つからなければ既定の場所）
        html = requests.get(JPX_PAGE, headers=hdr, timeout=20).text
        m = re.search(r'href="([^"]*data_j\.xlsx?)"', html)
        if m:
            url = urljoin(JPX_PAGE, m.group(1))
    except Exception:
        pass
    res = requests.get(url, headers=hdr, timeout=60)
    res.raise_for_status()
    df = pd.read_excel(io.BytesIO(res.content), dtype=str)

    def col(key):
        return key if key in df.columns else next(c for c in df.columns if key in str(c))

    out = pd.DataFrame({
        "code": df[col("コード")].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
        "name": df[col("銘柄名")].astype(str).str.strip(),
        "market": df[col("市場・商品区分")].fillna("").astype(str),
        "size": df[col("規模区分")].fillna("").astype(str).str.strip(),
    })
    return out[out["market"].str.contains("内国株式")].reset_index(drop=True)


def universe_items(kind, n=500, seed=0):
    """銘柄の範囲から (コード, 銘柄名) のリストを返す。ランダムは毎回同じ銘柄になるよう固定。"""
    df = load_jpx_list()
    if kind == "topix500":
        sel = df[df["size"].isin(["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"])]
    elif kind == "small500":
        sel = df[df["size"] == "TOPIX Small 1"]
    elif kind == "prime":
        sel = df[df["market"].str.startswith("プライム")]
    elif kind == "growth":
        sel = df[df["market"].str.startswith("グロース")]
    else:
        return []
    if len(sel) > n:
        sel = sel.sample(n, random_state=seed)
    return list(zip(sel["code"], sel["name"]))


def render_universe(key_prefix=""):
    """「① 銘柄」の入力欄。(範囲の種類, 手入力の文字) を返す。"""
    with st.expander("① 銘柄", expanded=False):
        kind = st.selectbox("対象にする銘柄", list(UNIVERSES), format_func=lambda k: UNIVERSES[k],
                            key=key_prefix + "universe",
                            help="東証の公式の上場銘柄一覧（毎月更新）から自動で選びます。"
                                 "推奨: 検証はTOPIX500か小型株500で。小型株のほうが、値動きの癖が残りやすいと言われます。")
        st.caption("500銘柄だと、株価の取得に数分かかることがあります（同じ設定なら1時間は再取得しません）。")
        text = st.text_area("手入力の銘柄（1行1銘柄。「手入力」を選んだときに使います）", value=SAMPLE, height=150,
                            key=key_prefix + "manual_text")
    return kind, text


def resolve_items(kind, text):
    """実行時に、対象の (コード, 銘柄名) を決める。一覧が取れなければ手入力の銘柄で代用。"""
    if kind == "manual":
        return parse_tickers(text)
    try:
        items = universe_items(kind)
        if items:
            return items
        st.warning("銘柄一覧に該当する銘柄がありませんでした。手入力の銘柄で実行します。")
    except Exception as e:
        st.warning(f"東証の銘柄一覧を読み込めなかったため、手入力の銘柄で実行します（{type(e).__name__}）。")
    return parse_tickers(text)


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
            d = d[~d.index.duplicated(keep="last")].sort_index()  # 同じ日付の重複行を除く
            if not d.empty:
                out[c] = d
        except KeyError:
            pass
    return out


def fetch_all(codes, period):
    bar = st.progress(0, text="株価を取得中...")
    data = {}
    try:
        for i in range(0, len(codes), 100):
            data.update(fetch_chunk(tuple(codes[i:i + 100]), period))
            bar.progress(min((i + 100) / len(codes), 1.0), text="株価を取得中...")
    finally:
        bar.empty()
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pbr(code: str):
    """現在のPBR（取れなければ None）。過去の値は取れないので、スクリーニングでのみ使う。"""
    import yfinance as yf

    try:
        v = yf.Ticker(code + ".T").info.get("priceToBook")
        return float(v) if v is not None and v > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market(period: str):
    """日経平均が75日線より上か（True/False の列）。取れなければ日経225連動ETF(1321)で代用。"""
    import yfinance as yf

    for t in ("^N225", "1321.T"):
        try:
            d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1) if "Close" in d.columns.get_level_values(0) else d[t]
            c = d["Close"].dropna()
            if len(c) > 100:
                return c > c.rolling(75).mean()
        except Exception:
            continue
    return None


# ------------------------------------------------------------ テンプレ（条件の保存・読み込み）
TEMPLATE_FILE = Path(__file__).with_name("templates.json")


def template_keys():
    keys = ["trigger", "short", "long", "days", "pb_pct", "rv_rsi", "universe"]
    for c in CONDS:
        keys.append(c["key"] + "_on")
        if "val" in c:
            keys.append(c["key"] + "_val")
    return keys


def _plain(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def make_template(name, settings, memo=""):
    keys = set(template_keys())
    return {"name": name, "memo": memo, "settings": {k: _plain(v) for k, v in settings.items() if k in keys}}


def encode_template(t):
    return base64.urlsafe_b64encode(json.dumps(t, ensure_ascii=False).encode("utf-8")).decode().rstrip("=")


def decode_template(code):
    return json.loads(base64.urlsafe_b64decode(code + "=" * (-len(code) % 4)).decode("utf-8"))


def load_repo_templates():
    """GitHubの templates.json に書かれたテンプレ（ずっと残る）。"""
    try:
        data = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
        return [t for t in data if isinstance(t, dict) and "settings" in t]
    except Exception:
        return []


# --- クラウド保存（GitHubのGist）。Streamlitの Secrets に GITHUB_TOKEN があるときだけ使う
GIST_FILE = "gc_templates.json"
GH_API = "https://api.github.com"


def _gh_token():
    try:
        return st.secrets.get("GITHUB_TOKEN") or None
    except Exception:  # Secrets が未設定
        return None


def cloud_enabled():
    return _gh_token() is not None


def _gh(method, path, **kw):
    import requests

    res = requests.request(method, GH_API + path, timeout=20, **kw,
                           headers={"Authorization": f"Bearer {_gh_token()}",
                                    "Accept": "application/vnd.github+json"})
    res.raise_for_status()
    return res.json()


def _find_gist():
    """テンプレ保存用のGist（ファイル名 gc_templates.json）を探す。なければ None。"""
    if st.session_state.get("_gist_id"):
        return st.session_state["_gist_id"]
    for page in range(1, 6):
        gists = _gh("GET", f"/gists?per_page=100&page={page}")
        for g in gists:
            if GIST_FILE in g.get("files", {}):
                st.session_state["_gist_id"] = g["id"]
                return g["id"]
        if len(gists) < 100:
            break
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_cloud(token_hint: str):  # 引数はキャッシュの区別用（トークンが変わったら読み直す）
    gid = _find_gist()
    if not gid:
        return []
    content = _gh("GET", f"/gists/{gid}")["files"][GIST_FILE]["content"]
    data = json.loads(content or "[]")
    return [t for t in data if isinstance(t, dict) and "settings" in t]


def load_cloud_templates():
    if not cloud_enabled():
        return []
    try:
        return _load_cloud(_gh_token()[-6:])
    except Exception as e:
        st.session_state["_tpl_msg"] = ("error", f"クラウドのテンプレを読み込めませんでした（{type(e).__name__}）。")
        return []


def _write_cloud(templates):
    body = {"files": {GIST_FILE: {"content": json.dumps(templates, ensure_ascii=False, indent=2)}}}
    gid = _find_gist()
    if gid:
        _gh("PATCH", f"/gists/{gid}", json=body)
    else:
        g = _gh("POST", "/gists", json={"description": "株価アプリのテンプレ（自動作成）", "public": False, **body})
        st.session_state["_gist_id"] = g["id"]
    _load_cloud.clear()


def save_cloud_template(t):
    try:
        cur = [x for x in load_cloud_templates() if x.get("name") != t["name"]]  # 同じ名前は上書き
        _write_cloud(cur + [t])
        st.session_state["_tpl_msg"] = ("success", f"「{t['name']}」を保存しました。再読み込みしても消えません。")
    except Exception as e:
        st.session_state["_tpl_msg"] = ("error", f"保存できませんでした（{type(e).__name__}）。"
                                                 "GITHUB_TOKEN の設定を確認してください。")


def delete_cloud_template(name):
    try:
        _write_cloud([x for x in load_cloud_templates() if x.get("name") != name])
        st.session_state["_tpl_msg"] = ("success", f"「{name}」を削除しました。")
        st.session_state.pop("_tpl_pick", None)
    except Exception as e:
        st.session_state["_tpl_msg"] = ("error", f"削除できませんでした（{type(e).__name__}）。")


def _show_tpl_msg():
    msg = st.session_state.pop("_tpl_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])


def all_templates():
    """☁️クラウド保存 ＋ 📁templates.json ＋ 🕘このセッション中だけ のテンプレ。"""
    out = {f"☁️ {t.get('name', '名前なし')}": t for t in load_cloud_templates()}
    out.update({f"📁 {t.get('name', '名前なし')}": t for t in load_repo_templates()})
    for name, t in st.session_state.get("templates", {}).items():
        out[f"🕘 {name}"] = t
    return out


def apply_template(t, key_prefix=""):
    """テンプレの値を画面の入力欄に入れる（ボタンの on_click から呼ぶ）。"""
    keys = set(template_keys())
    for c in CONDS:  # テンプレにない条件はOFFにする（前の設定が残らないように）
        if c["key"] + "_on" not in t["settings"]:
            st.session_state[key_prefix + c["key"] + "_on"] = False
    for k, v in t["settings"].items():
        if k in keys:
            st.session_state[key_prefix + k] = v
    st.session_state["_tpl_loaded"] = t.get("name", "")


def save_session_template(t):
    st.session_state.setdefault("templates", {})[t["name"]] = t
    st.session_state["_tpl_msg"] = ("success", f"「{t['name']}」をこのセッションに保存しました（再読み込みで消えます）。")


SETUP_HELP = (
    "**テンプレをずっと残す設定（最初に1回だけ）**\n\n"
    "1. GitHub で右上のアイコン →「Settings」→「Developer settings」→「Personal access tokens」→"
    "「Tokens (classic)」→「Generate new token (classic)」を開く。\n"
    "2. Note に「株価アプリ」、Expiration は「No expiration」、権限は **gist だけ** にチェックして作成し、"
    "表示された文字（ghp_ で始まる）をコピー。\n"
    "3. Streamlit のアプリ画面右下「Manage app」→「⋮」→「Settings」→「Secrets」に、次の1行を貼って保存：  \n"
    "`GITHUB_TOKEN = \"ghp_ここにコピーした文字\"`\n\n"
    "この文字は合言葉と同じなので、人に見せたりGitHubのファイルに書いたりしないでください。"
)


def render_template_saver(t):
    """テンプレの保存欄（過去検証・自動探索・目標勝率サーチの結果の下に出す）。"""
    code = encode_template(t)
    _show_tpl_msg()
    st.markdown(f"**{t['name']}**  \n{t['memo']}")
    if cloud_enabled():
        st.button("☁️ このテンプレを保存（ずっと残る）", on_click=save_cloud_template, args=(t,),
                  key="_save_" + code[:24], type="primary", width="stretch")
    else:
        st.button("🕘 このテンプレを保存（このセッション中だけ）", on_click=save_session_template, args=(t,),
                  key="_save_" + code[:24], width="stretch")
        with st.expander("再読み込みしても消えないようにするには", expanded=False):
            st.markdown(SETUP_HELP)
    st.markdown(f"[🔗 シグナル検出をこの条件で開く](/?embed=true&tpl={code})")
    st.caption("このリンクをホーム画面に追加すると、その条件専用のアプリとしても使えます。")


def render_template_loader(key_prefix=""):
    """シグナル検出の画面の上に出す、テンプレの読み込み欄。"""
    code = st.query_params.get("tpl")
    if code and st.session_state.get("_tpl_code") != code:  # リンクから開いたとき（最初の1回だけ反映）
        st.session_state["_tpl_code"] = code
        try:
            apply_template(decode_template(code), key_prefix)
        except Exception:
            st.warning("テンプレのリンクを読み込めませんでした。")
    tpls = all_templates()
    _show_tpl_msg()
    with st.expander("⭐ テンプレ（保存した条件）", expanded=not st.session_state.get("_tpl_loaded")):
        if st.session_state.get("_tpl_loaded"):
            st.success(f"テンプレ「{st.session_state['_tpl_loaded']}」の条件を読み込んでいます。")
        if not tpls:
            st.caption("まだテンプレがありません。過去検証・自動探索・目標勝率サーチの結果の下から保存できます。")
            if not cloud_enabled():
                with st.expander("再読み込みしても消えないようにするには", expanded=False):
                    st.markdown(SETUP_HELP)
            return
        name = st.selectbox("テンプレ", list(tpls), key="_tpl_pick")
        t = tpls[name]
        if t.get("memo"):
            st.caption(t["memo"])
        st.button("この条件を読み込む", on_click=apply_template, args=(t, key_prefix), type="primary", width="stretch")
        if name.startswith("☁️"):
            st.button("このテンプレを削除", on_click=delete_cloud_template, args=(t.get("name"),), width="stretch")
        st.caption("☁️＝クラウドに保存（ずっと残る）、📁＝テンプレ一覧ファイル、🕘＝このセッション中だけ（再読み込みで消えます）")
        if not cloud_enabled():
            with st.expander("再読み込みしても消えないようにするには", expanded=False):
                st.markdown(SETUP_HELP)
