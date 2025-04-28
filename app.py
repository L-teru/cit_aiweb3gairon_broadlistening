import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from gspread_dataframe import get_as_dataframe
import matplotlib.pyplot as plt
import textwrap
import shutil
import os
from google.auth import default
from sentence_transformers import SentenceTransformer
import numpy as np

# フォルダパス
ORIGINAL_DIR = "data/original/"
WORKING_DIR = "data/working/"

def ensure_working_dir():
    if not os.path.exists(WORKING_DIR) or not os.listdir(WORKING_DIR):
        # フォルダが無いか、中身が空ならoriginalからコピー
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)  # 中途半端な空フォルダは削除
        shutil.copytree(ORIGINAL_DIR, WORKING_DIR)

def save_working_to_original():
    if os.path.exists(WORKING_DIR):
        # working → originalに上書き
        for filename in os.listdir(WORKING_DIR):
            src = os.path.join(WORKING_DIR, filename)
            dst = os.path.join(ORIGINAL_DIR, filename)
            shutil.copy2(src, dst)

def step1_generate_args_csv():
    uploaded_file = st.file_uploader("📝 初期アイデアリスト（CSV）をアップロードしてください", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        df_input = df_input.rename(columns={"番号": "comment-id", "アイデア": "comment-body"})

        # arg-id と argument を作成
        df_args = df_input.copy()
        df_args["arg-id"] = df_args["comment-id"].apply(lambda cid: f"A{cid}_0")
        df_args["argument"] = df_args["comment-body"]
        df_args = df_args[["arg-id", "comment-id", "argument"]]

        # workingディレクトリに保存
        working_dir = "data/working"
        os.makedirs(working_dir, exist_ok=True)
        df_args.to_csv(os.path.join(working_dir, "args.csv"), index=False)

        st.success("✅ アイデアリストを取り込みました！")
    else:
        st.info("📂 CSVファイルをアップロードしてください。")

def step2_generate_embeddings():
    st.info("📚 テキストをベクトルに変換中... 少々お待ちください。")

    input_path = os.path.join(WORKING_DIR, "args.csv")
    output_path = os.path.join(WORKING_DIR, "embeddings.pkl")

    if not os.path.exists(input_path):
        st.error("❌ args.csv が存在しません。ステップ1を実行してください。")
        return

    df = pd.read_csv(input_path)
    model = SentenceTransformer("cl-nagoya/sup-simcse-ja-large")

    texts = df["argument"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    df_embeds = pd.DataFrame({
        "arg-id": df["arg-id"],
        "embedding": embeddings.tolist()
    })

    df_embeds.to_pickle(output_path)
    st.success("✅ embeddings.pkl を保存しました！")

def discard_working_changes():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    shutil.copytree(ORIGINAL_DIR, WORKING_DIR)

# ページ設定
st.set_page_config(
    page_title="アイデアクラスタリングアプリ",
    page_icon="🎯",
    layout="wide",  # ←これが大事！（横幅を広く使う）
    initial_sidebar_state="collapsed",
)

st.title("🎯 アイデアクラスタリングアプリ")


# タブ作成
tab_summary, tab_analysis, tab_list, tab_settings = st.tabs(["📊 サマリ", "📖 解説", "📑 一覧", "⚙️ 設定（実装中）"])

# --- スプレッドシートから args.csv を生成（Groqなし版） ---
@st.cache_data
def generate_args_from_spreadsheet(spreadsheet_id: str, worksheet_name: str = "Sheet1"):
    creds, _ = default()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet(worksheet_name)
    df_input = get_as_dataframe(worksheet, evaluate_formulas=True)

    # 必要なカラムだけリネーム
    df_input = df_input.rename(columns={"番号": "comment-id", "アイデア": "comment-body"})
    df_args = df_input.copy()
    df_args["arg-id"] = df_args["comment-id"].apply(lambda cid: f"A{cid}_0")
    df_args["argument"] = df_args["comment-body"]

    # 必要な列だけ残す
    df_args = df_args[["arg-id", "comment-id", "argument"]]

    # 保存
    df_args.to_csv(os.path.join(WORKING_DIR, "args.csv"), index=False)
    st.success("✅ スプレッドシートから args.csv を生成しました！")

# --- データ読み込み ---
@st.cache_data
def load_data():
    ensure_working_dir()
    # CSV読み込み
    df_clusters = pd.read_csv(os.path.join(WORKING_DIR, "clusters.csv"))
    df_takeaways = pd.read_csv(os.path.join(WORKING_DIR, "takeaways.csv"))
    df_labels = pd.read_csv(os.path.join(WORKING_DIR, "labels.csv"))
    df_ideas = pd.read_csv(os.path.join(WORKING_DIR, "clustered_ideas.csv"))
    df_args = pd.read_csv(os.path.join(WORKING_DIR, "args.csv"))

    def matplotlib_color_to_hex(color_tuple):
        r, g, b, a = color_tuple
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

    # カラーマップをここで動的生成
    cluster_ids = sorted(df_clusters["cluster-id"].unique())
    cmap = plt.get_cmap("tab10")
    cluster_color_map = {str(cid): matplotlib_color_to_hex(cmap(i % 10)) for i, cid in enumerate(cluster_ids)}

    return df_clusters, df_takeaways, df_labels, df_ideas, df_args, cluster_color_map

# ここも数合わせる！（6個返ってくる）
df_clusters, df_takeaways, df_labels, df_ideas, df_args, cluster_color_map = load_data()

# --- データ整形 ---
df_clusters = df_clusters.merge(df_labels, on="cluster-id", how="left")
df_clusters = df_clusters.merge(df_takeaways, on="cluster-id", how="left")
df_clusters = df_clusters.merge(df_args[["arg-id", "argument"]], on="arg-id", how="left")

# --- サマリタブ ---
with tab_summary:
    st.header("📊 クラスタリング結果")
    # 最大表示件数
    MAX_POINTS = 1000  # 最大表示件数

    # 必要ならサンプリング
    if len(df_clusters) > MAX_POINTS:
        df_plot = df_clusters.sample(MAX_POINTS, random_state=42)
    else:
        df_plot = df_clusters

    # グラフ作成
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=df_plot["cluster-id"].astype(str),  # クラスタIDを文字列に変換
        color_discrete_map=cluster_color_map,  # カラーマップも文字列キー
        hover_data=["argument", "label"],
        title="クラスタリング結果（マウスオーバーで詳細表示）"
    )

    # ✨ここが追加・修正ポイント
    fig.update_layout(
        autosize=True,
        width=800,  # 初期サイズ 4:3
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(scaleanchor="x", scaleratio=1),  # 縦横比固定
    )

    # Streamlitに描画
    st.plotly_chart(fig, use_container_width=True)

    # --- ステップ6の概要表示 ---
    try:
        with open(os.path.join(WORKING_DIR, "overview.txt"), "r", encoding="utf-8") as f:
            overview_text = f.read()
            st.markdown(overview_text)
    except FileNotFoundError:
        st.warning("⚠️ overview.txt が見つかりませんでした。")


# --- 解説タブ ---
with tab_analysis:
    st.header("📖 クラスタ別詳細")

    for cluster_id in sorted(df_clusters["cluster-id"].unique()):
        cluster_data = df_clusters[df_clusters["cluster-id"] == cluster_id]
        label = cluster_data["label"].values[0]
        takeaway = cluster_data["takeaways"].values[0]

        count = len(cluster_data)
        percent = round((count / len(df_clusters)) * 100, 1)
        st.subheader(f"🟦 クラスタ {cluster_id}「{label}」（{count}件 / {percent}%）")

        # ▶️ マウスオーバーできる可視化を表示
        plot_df = df_clusters.copy()
        plot_df["highlight_color"] = plot_df["cluster-id"].apply(
            lambda cid: cluster_color_map[str(cid)] if cid == cluster_id else "#d3d3d3"
        )
        plot_df["hover_text"] = "クラスタ " + plot_df["cluster-id"].astype(str) + "<br>" + plot_df["argument"]

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="highlight_color",
            color_discrete_map="identity",
            hover_name="hover_text",
            title=f"Cluster {cluster_id}: {label}",
            width=800,   # ←ここ 800に変更！（4:3）
            height=600,  # ←ここ 600に変更！（4:3）
        )
        fig.update_traces(marker=dict(size=8, opacity=0.9), selector=dict(mode='markers'))
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(scaleanchor="x", scaleratio=1)  # 縦横比を固定！
        )
        st.plotly_chart(fig, use_container_width=True)

        # ▶️ 要約をテキストで表示
        st.markdown(f"**要約:** {takeaway}")
        st.divider()

# --- 一覧タブ ---
with tab_list:
    st.header("📑 クラスタ別バグリスト一覧")

    # 🔥 クラスタID or ラベルでフィルタできるようにする
    cluster_options = df_ideas["cluster-id"].unique()
    selected_cluster = st.selectbox("表示するクラスタを選択", options=["全て"] + sorted(cluster_options.tolist()))

    # 🔥 選択に応じてデータを絞る
    if selected_cluster == "全て":
        df_display = df_ideas.copy()
    else:
        df_display = df_ideas[df_ideas["cluster-id"] == selected_cluster]

    # 🔥 データフレームの表示カスタマイズ
    st.dataframe(
        df_display.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',   # 長文でも折り返す
        }),
        hide_index=True,
        use_container_width=True,
        column_config={
            "label": st.column_config.TextColumn(width="large"),    # ラベル列を広げる
            "argument": st.column_config.TextColumn(width="large"), # バグリスト列を広げる
        }
    )

    st.markdown("---")
    st.subheader("🔍 類似バグリスト検索＆LLM回答")

    input_text = st.text_input("💬 あなたのバグリストを入力してください")
    top_k = st.slider("🔢 何件まで候補を出すか？", min_value=3, max_value=20, value=5)

    if st.button("🚀 類似バグリストを検索"):
        if input_text.strip() == "":
            st.warning("⚠️ 入力してください。")
        else:
            with st.spinner("🔍 類似バグリストを検索中..."):
                from sklearn.metrics.pairwise import cosine_similarity
                from sentence_transformers import SentenceTransformer
                from openai import OpenAI

                # モデル選択UI
                model_choice = st.selectbox("🔍 使用するベクトル化モデルを選択", [
                    "OpenAI API (text-embedding-3-small)",
                    "SentenceTransformer (multilingual-e5-small)",
                    "SentenceTransformer (sup-simcse-ja-large)",
                ])

                if model_choice == "OpenAI API (text-embedding-3-small)":
                    client = OpenAI(api_key=st.secrets["openai_api_key"])
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[input_text]
                    )
                    # OpenAI returns a list of embeddings
                    input_vec = np.array([record.embedding for record in response.data])
                elif model_choice == "SentenceTransformer (multilingual-e5-small)":
                    model = SentenceTransformer("intfloat/multilingual-e5-small")
                    input_vec = model.encode([input_text])
                elif model_choice == "SentenceTransformer (sup-simcse-ja-large)":
                    model = SentenceTransformer("cl-nagoya/sup-simcse-ja-large")
                    input_vec = model.encode([input_text])
                else:
                    st.error("❌ モデル選択エラー")
                    st.stop()

                # 既存ベクトル読み込み
                df_embed = pd.read_pickle(os.path.join(WORKING_DIR, "embeddings.pkl"))
                embeddings = np.vstack(df_embed["embedding"].values)

                # コサイン類似度計算
                similarities = cosine_similarity(input_vec, embeddings)[0]

                # 類似度Top Kを取得
                top_indices = similarities.argsort()[-top_k:][::-1]

                similar_args = df_args.iloc[top_indices].copy()
                similar_args["similarity"] = similarities[top_indices]

                # クラスタ情報をマージ
                similar_args = similar_args.merge(
                    df_clusters[["arg-id", "cluster-id", "label"]],
                    on="arg-id",
                    how="left"
                )

                # 表示用整形
                df_show = similar_args[["similarity", "cluster-id", "label", "argument"]]
                df_show = df_show.rename(columns={"similarity": "類似度", "cluster-id": "クラスタID", "label": "クラスタラベル", "argument": "バグリスト"})
                df_show = df_show.sort_values(by="類似度", ascending=False)

                st.success(f"✅ 類似度トップ{top_k}件を表示します。")
                st.dataframe(
                    df_show.style.set_properties(**{
                        'text-align': 'left',
                        'white-space': 'pre-wrap',
                    }),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "クラスタラベル": st.column_config.TextColumn(width="large"),
                        "バグリスト": st.column_config.TextColumn(width="large"),
                    }
                )

            if st.button("🤖 LLMに最も似たバグリストを選ばせる"):
                with st.spinner("🤖 LLMが考え中..."):
                    import requests
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
                        "Content-Type": "application/json"
                    }

                    # プロンプトを組み立て
                    prompt = "以下のバグリストの中で、あなたのバグリスト「{}」に最も意味が近いものを選んでください。\n\n".format(input_text)
                    for idx, row in similar_args.iterrows():
                        prompt += f"- ({row['comment-id']}) {row['argument']}\n"

                    prompt += "\n最も近いものの (番号) を教えてください。"

                    system_message = {
                        "role": "system",
                        "content": "あなたは与えられたリストから、最も意味が近いものを選ぶAIアシスタントです。"
                    }

                    response = requests.post(url, headers=headers, json={
                        "model": "llama3-70b-8192",
                        "messages": [
                            system_message,
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0
                    })

                    if response.status_code == 200:
                        res_json = response.json()
                        answer = res_json["choices"][0]["message"]["content"].strip()
                        st.success(f"🤖 LLMの回答: {answer}")
                    else:
                        st.error("❌ LLMリクエストに失敗しました。")

# --- 設定タブ ---
with tab_settings:
    st.header("⚙️ 設定メニュー（実装中）")
    
    # 🔥 公開版では設定を封印
    if "developer_mode" in st.secrets and st.secrets["developer_mode"]:
        # 開発者だけが設定機能を見れる（st.secrets["developer_mode"] = true なら表示）

        with st.expander("🔄 データ再取得と再実行"):
            st.write("データ再取得や再クラスタリングを行う場合は、以下のボタンを使ってください。")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🚀 データ再取得＋一括実行"):
                    st.session_state["action"] = "reload_all"

            with col2:
                if st.button("🌀 クラスター数だけ再設定"):
                    st.session_state["action"] = "recluster_only"

            with col3:
                if st.button("📝 プロンプト変更 → ラベル再生成"):
                    st.session_state["action"] = "relabel_only"

        st.subheader("プロンプト設定")
        new_prompt = st.text_area("📝 ラベリング用プロンプト", value="（ここに初期プロンプトを入れる）", height=200)
        if st.button("💾 プロンプトを保存"):
            st.success("✅ プロンプトを更新しました！（仮保存）")

        st.subheader("保存・破棄メニュー")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 このデータで保存（次回起動時に反映）", type="primary"):
                save_working_to_original()
                st.success("✅ 保存しました！アプリを再起動すると反映されます。")
                st.experimental_rerun()

        with col2:
            if st.button("🗑️ 変更を破棄（起動時状態に戻す）"):
                discard_working_changes()
                st.warning("⚠️ 変更を破棄しました。アプリを再起動します。")
                st.experimental_rerun()

    else:
        st.info("🚧 このセクションは開発中です。近日公開予定。")

# 新しい関数を追加
def step3_clustering(n_clusters: int):
    import numpy as np
    from sklearn.cluster import SpectralClustering
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk

    # nltkのstopwordsダウンロード（済みならスキップされる）
    nltk.download('stopwords')

    # データ読み込み
    df_args = pd.read_csv(os.path.join(WORKING_DIR, "args.csv"))
    df_embed = pd.read_pickle(os.path.join(WORKING_DIR, "embeddings.pkl"))

    # マージ
    df = pd.merge(df_args, df_embed, on="arg-id")

    docs = df["argument"].tolist()
    embeddings = np.vstack(df["embedding"].values)

    # 次元圧縮（UMAP）
    umap_model = UMAP(random_state=42, n_components=2)
    umap_embeds = umap_model.fit_transform(embeddings)

    # クラスタリング（HDBSCAN）
    hdbscan_model = HDBSCAN(min_cluster_size=2)

    # ベクトル化（日本語対応なのでstopwordsなし）
    vectorizer_model = CountVectorizer(stop_words=None)

    # BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="japanese",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # スペクトラルクラスタリング
    spectral_model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=min(10, len(embeddings) - 1),
        random_state=42
    )
    cluster_labels = spectral_model.fit_predict(umap_embeds)

    # 情報まとめ
    doc_info = topic_model.get_document_info(docs)
    doc_info["arg-id"] = df["arg-id"]
    doc_info["comment-id"] = df["comment-id"]
    doc_info["x"] = umap_embeds[:, 0]
    doc_info["y"] = umap_embeds[:, 1]
    doc_info["cluster-id"] = cluster_labels

    # 保存
    doc_info[["arg-id", "comment-id", "x", "y", "cluster-id"]].to_csv(
        os.path.join(WORKING_DIR, "clusters.csv"), index=False
    )
#    print("✅ clusters.csv を保存しました！")
    print("✅ clusters.csv を保存しました！")

# --- New functions added after step3_clustering ---
import requests
import time

# ステップ4用：プロンプト組み立て
def build_prompt(question, inside_list, outside_list):
    example = """
【例：Brexitに関する議論のクラスタラベリング】

コンサルテーションの質問：「英国のEU離脱決定の影響は何だと思いますか？」

関心のあるクラスター以外の論点の例：
 * エラスムス・プログラムからの除外により、教育・文化交流の機会が制限された。
 * 環境基準における協力が減少し、気候変動と闘う努力が妨げられた。
 * 家族の居住権や市民権の申請が複雑になった。
 * 医療協定の中断、患者への影響。
 * EUの文化助成プログラムからの除外により、創造的なプロジェクトの制限。

クラスター内部での議論の例：
 * サプライチェーンの混乱によるコスト増と納期遅延。
 * 投資不安・退職金の価値変動。
 * 輸出業者の関税対応と利益減少。
 * ポンド安と生活費の上昇。

このクラスタのカテゴリラベル例：経済的影響
"""

    inside_text = '\n * ' + '\n * '.join(inside_list)
    outside_text = '\n * ' + '\n * '.join(outside_list)

    return f"""
{example}

----

【本題】

コンサルテーションの質問：「{question}」

関心のあるクラスター以外の論点の例：
{outside_text}

クラスター内部での議論の例：
{inside_text}

このクラスタのカテゴリラベルを日本語で1つ、他のクラスタと意味が被らないようなものを簡潔に出力してください。
"""

# ステップ4用：Groq API呼び出し
def call_groq(prompt, max_retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['groq_api_key']}",
        "Content-Type": "application/json"
    }

    system_message = {
        "role": "system",
        "content": (
            "あなたは、与えられたクラスタ内・外の意見を比較し、クラスタを最も特徴づける"
            "カテゴリラベルを日本語の名詞句で1つ返すアシスタントです。\n"
            "出力はラベルのみ。10語以内で。\n"
            "ラベルは抽象語（例：不便さ、問題）だけでなく、"
            "他のクラスターと比べ具体的に分類の差がわかるようにしてください（例：学校生活の不便、コミュニケーション上の不満など）。\n"
            "解釈が広すぎる表現（日常や日常生活）や意味が同じクラスタとならないように注意してください。"
        )
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json={
                "model": "llama3-70b-8192",
                "messages": [
                    system_message,
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            })

            if response.status_code == 200:
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"].strip()
            else:
                print(f"⚠ ステータスコード: {response.status_code}, 再試行")
                time.sleep(2)
        except Exception as e:
            print("❌ Error:", e)
            time.sleep(2)
    return "ラベル未取得"

# ステップ4：ラベル生成
def step4_generate_labels(sample_size=10):
    st.info("🏷️ クラスタにラベルを付けています... 少々お待ちください。")

    input_args = os.path.join(WORKING_DIR, "args.csv")
    input_clusters = os.path.join(WORKING_DIR, "clusters.csv")
    output_labels = os.path.join(WORKING_DIR, "labels.csv")

    if not os.path.exists(input_args) or not os.path.exists(input_clusters):
        st.error("❌ 必要なデータが存在しません。")
        return

    df_args = pd.read_csv(input_args)
    df_clusters = pd.read_csv(input_clusters)

    question = "Discordに投稿された「生活上の困りごと・不満・改善提案」を、種類や状況に応じて分類してください。"
    cluster_ids = sorted(df_clusters["cluster-id"].unique())
    labels = []

    for cluster_id in cluster_ids:
        inside_ids = df_clusters[df_clusters["cluster-id"] == cluster_id]["arg-id"]
        outside_ids = df_clusters[df_clusters["cluster-id"] != cluster_id]["arg-id"]

        inside_sample = df_args[df_args["arg-id"].isin(inside_ids)].sample(min(sample_size, len(inside_ids)))
        outside_sample = df_args[df_args["arg-id"].isin(outside_ids)].sample(min(sample_size, len(outside_ids)))

        prompt = build_prompt(
            question,
            inside_sample["argument"].tolist(),
            outside_sample["argument"].tolist()
        )

        label = call_groq(prompt)
        labels.append({"cluster-id": cluster_id, "label": label})
        print(f"✅ cluster {cluster_id}: {label}")
        time.sleep(1.0)

    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(output_labels, index=False)
    st.success("✅ labels.csv を保存しました！")

# フッター
st.markdown("---")
st.caption("© 2025 Web3Gairon Broad Listening Project")