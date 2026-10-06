"""共通部品：指標の計算・条件・入力画面・株価取得（app.py と pages/backtest.py から使う）"""
import pandas as pd
import streamlit as st

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
    # --- 出来高・流動性
    dict(key="vol", group="出来高・流動性", label="クロス日の出来高が平均の◯倍以上", on=True, rec="推奨: 1.5〜2.0倍",
         val=dict(label="出来高倍率（倍）", default=1.5, min=0.5, max=10.0, step=0.5),
         help="クロス日の出来高が直近20日平均より多いこと。買いが本当に入っているかの確認です。"),
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
    dict(key="atr", group="過熱感", label="値動きの荒さ（ATR）が◯％以下【新】", on=False, rec="推奨: 検証で判断（目安5%）",
         val=dict(label="ATR上限（株価比 %）", default=5.0, min=1.0, max=20.0, step=0.5),
         help="1日の平均的な値幅（14日）が株価の何％か。大きい銘柄はだましで大きく損をしやすいです。"),
    # --- 地合い・タイミング
    dict(key="market", group="地合い・タイミング", label="日経平均が75日線より上【新】", on=True, rec="推奨: ON",
         help="相場全体が下落基調のときは、個別株のクロスも失敗しやすいと言われます。相場全体の向きで絞ります。"),
    dict(key="bullish", group="地合い・タイミング", label="クロス日が陽線（終値＞始値）【新】", on=False, rec="推奨: 検証で判断",
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
    with st.expander("② クロスの条件", expanded=True):
        cols = st.columns(3 if show_days else 2)
        p["short"] = int(cols[0].number_input("短期線（日）", 2, 50, 5, key=key_prefix + "short",
                                              help="推奨: 5日。短いほど早くクロスしますが、だましも増えます。"))
        p["long"] = int(cols[1].number_input("長期線（日）", 5, 200, 25, key=key_prefix + "long",
                                             help="推奨: 25日。日本では5日×25日の組み合わせが一般的です。"))
        if show_days:
            p["days"] = int(cols[2].number_input("直近N営業日以内", 1, 20, 1, key=key_prefix + "days",
                                                 help="推奨: 1〜3日。大きくすると、少し前のクロスも拾います。"))
        st.caption("推奨: 短期5日・長期25日・直近1〜3日")

    groups = []
    for c in CONDS:
        if c["group"] not in groups:
            groups.append(c["group"])
    st.markdown("**③ 絞り込み**（ONにしたものだけ適用。〔 〕内は一般的な推奨値）")
    for g in groups:
        with st.expander(g, expanded=False):
            for c in [c for c in CONDS if c["group"] == g]:
                on = st.checkbox(f"{c['label']}〔{c['rec']}〕", value=c["on"], help=c["help"],
                                 key=key_prefix + c["key"] + "_on")
                p[c["key"] + "_on"] = on
                if "val" in c:
                    v = c["val"]
                    p[c["key"] + "_val"] = float(st.number_input(
                        f"└ {v['label']}", float(v["min"]), float(v["max"]), float(v["default"]), float(v["step"]),
                        disabled=not on, key=key_prefix + c["key"] + "_val"))
    return p


# ------------------------------------------------------------ 計算
def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def compute(df, short, long_, mkt_ok=None):
    """各日の指標と、各条件を満たしたかを計算（その日までのデータだけを使う）。"""
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
    return x


def masks(x, p):
    """条件ごとに True/False の列を返す（値を使う条件は p の値で判定）。"""
    m = {
        "rising": x["rising"], "above75": x["above75"], "above200": x["above200"], "perfect": x["perfect"],
        "vol": x["vol_x"] >= p["vol_val"], "value": x["value_m"] >= p["value_val"],
        "rsi": x["rsi_v"] <= p["rsi_val"], "dev": x["dev_v"] <= p["dev_val"], "atr": x["atr_v"] <= p["atr_val"],
        "market": x["market"], "bullish": x["bullish"], "breakout": x["breakout"], "macd": x["macd"],
    }
    return {k: v.fillna(False).astype(bool) for k, v in m.items()}


def signal(x, p, m=None):
    m = m or masks(x, p)
    sig = x["cross"].copy()
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
