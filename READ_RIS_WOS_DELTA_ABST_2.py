import math
import os
import re
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# 실행 환경 및 라이브러리 설정
# =============================================================================
RUNNING_IN_SPYDER = any(
    key.startswith("SPYDER") or key.startswith("SPY_") for key in os.environ
)
SHOW_PLOTS = "--show" in sys.argv or RUNNING_IN_SPYDER

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

try:
    import openai
except ImportError:
    openai = None


# =============================================================================
# 입력·출력 경로 및 분석 설정
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_OUTPUT_DIR = "E:/Text_Mining/Fig/WoS/Ver.7"
OUTPUT_CLUSTER_CSV = os.path.join(SCRIPT_DIR, "wos_clustered_abstracts.csv")
OUTPUT_WORDCLOUD_PNG = os.path.join(FIGURE_OUTPUT_DIR, "cluster_wordclouds_wos.png")
OUTPUT_NETWORK_PNG = os.path.join(FIGURE_OUTPUT_DIR, "cluster_network_wos.png")

RIS_FILES = [
    "E:/Text_Mining/Data/WoS/savedrecs_1_cleaned.ris",
    "E:/Text_Mining/Data/WoS/savedrecs_2_cleaned.ris",
    "E:/Text_Mining/Data/WoS/savedrecs_3_cleaned.ris",
    "E:/Text_Mining/Data/WoS/savedrecs_4_cleaned.ris",
    "E:/Text_Mining/Data/WoS/savedrecs_5_cleaned.ris",
]

MAX_YEAR = 2025
DEFAULT_N_TOPICS = 5

FIGURE_BG = "#eef4fb"
PANEL_BG = "#f8fbff"
TEXT_COLOR = "#17324d"
MUTED_TEXT = "#63778d"
GRID_COLOR = "#cad7e6"
EDGE_COLOR = "#95a8be"
ANNOTATION_BG = "#ffffff"
CLUSTER_COLORS = [
    "#35518d",
    "#2f7a95",
    "#3ca271",
    "#c9822e",
    "#a14e9d",
    "#d8576b",
    "#6f63c2",
    "#6a9f30",
]


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="white", context="talk")


base_stop_words = set(ENGLISH_STOP_WORDS)
custom_stop_words = {
    "study",
    "studies",
    "result",
    "results",
    "method",
    "methods",
    "using",
    "used",
    "use",
    "based",
    "analysis",
    "data",
    "paper",
    "papers",
    "research",
    "new",
    "show",
    "shows",
    "shown",
    "however",
    "also",
    "can",
    "may",
    "one",
    "two",
    "three",
    "within",
    "among",
    "across",
}
stop_words = list(base_stop_words.union(custom_stop_words))


# =============================================================================
# 시각화 공통 유틸리티
# =============================================================================
def cluster_color(cluster_id):
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def make_cluster_colormap(base_color):
    return LinearSegmentedColormap.from_list(
        "cluster_map",
        ["#ffffff", base_color],
    )


# =============================================================================
# RIS 초록 데이터 추출
# =============================================================================
def extract_ris_abstracts(file_paths):
    records = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as file:
                lines = file.readlines()

        print(file_path)

        current_year = None
        current_record = {"TI": "", "AB": ""}

        for raw_line in lines:
            line = raw_line.strip()

            if line.startswith("TI  -"):
                current_record["TI"] = line.replace("TI  -", "").strip()
            elif line.startswith("AB  -"):
                current_record["AB"] = line.replace("AB  -", "").strip()

            if line.startswith("PY  -") or line.startswith("Y1  -"):
                match = re.search(r"\d{4}", line)
                if match:
                    current_year = int(match.group())
            elif line.startswith("ER  -"):
                if current_year and current_year <= MAX_YEAR and current_record["AB"]:
                    records.append(current_record.copy())

                current_record = {"TI": "", "AB": ""}
                current_year = None

    return pd.DataFrame(records)


# =============================================================================
# 텍스트 벡터화 및 군집 분석
# =============================================================================
def find_optimal_clusters(tfidf_matrix, range_n_clusters=(2, 10)):
    best_score = -1
    best_n = range_n_clusters[0]

    for n in range(range_n_clusters[0], range_n_clusters[1] + 1):
        kmeans = KMeans(n_clusters=n, random_state=1, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        print(f"Clusters: {n}, silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_n = n

    print(f"Optimal number of clusters: {best_n} (score: {best_score:.4f})")
    return best_n


def vectorize_abstracts(df):
    cleaned = (
        df["AB"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_df=0.95,
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(cleaned)
    return tfidf, vectorizer


def cluster_abstracts(tfidf_matrix, n_topics):
    nmf = NMF(n_components=n_topics, random_state=1)
    topic_matrix = nmf.fit_transform(tfidf_matrix)

    kmeans = KMeans(n_clusters=n_topics, random_state=1, n_init=10)
    labels = kmeans.fit_predict(topic_matrix)
    return labels, nmf


def extract_top_keywords_per_cluster(nmf_model, tfidf_vectorizer, top_n=5):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    cluster_keywords = []

    for topic in nmf_model.components_:
        top_words = [feature_names[index] for index in topic.argsort()[:-top_n - 1 : -1]]
        cluster_keywords.append(top_words)

    return cluster_keywords


# =============================================================================
# GPT 기반 토픽명 생성(선택 기능)
# =============================================================================
def gpt_label_topics(cluster_keywords):
    if openai is None:
        raise ImportError("openai package is not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    openai.api_key = api_key
    topic_labels = []

    for keywords in cluster_keywords:
        prompt = (
            "Suggest a concise academic topic name (2-4 words) for the following keywords:\n"
            + ", ".join(keywords)
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=50,
        )
        topic_labels.append(response.choices[0].message["content"].strip())

    return topic_labels


# =============================================================================
# 전체 군집 분석 파이프라인
# =============================================================================
def run_clustering_pipeline(ris_files, use_gpt=False, n_topics=None):
    df = extract_ris_abstracts(ris_files)
    if df.empty:
        raise ValueError("No abstracts were found in the RIS files.")

    tfidf_matrix, vectorizer = vectorize_abstracts(df)

    if n_topics is None:
        n_topics = find_optimal_clusters(tfidf_matrix)

    labels, nmf_model = cluster_abstracts(tfidf_matrix, n_topics=n_topics)
    cluster_keywords = extract_top_keywords_per_cluster(nmf_model, vectorizer)

    df["Cluster"] = labels

    print("\nCluster representative keywords:")
    for cluster_id, keywords in enumerate(cluster_keywords):
        print(f"Cluster {cluster_id}: {', '.join(keywords)}")

    if use_gpt:
        topic_labels = gpt_label_topics(cluster_keywords)
        df["Topic"] = [topic_labels[label] for label in df["Cluster"]]
    else:
        df["Topic"] = [f"Cluster {label}" for label in df["Cluster"]]

    df.to_csv(OUTPUT_CLUSTER_CSV, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {OUTPUT_CLUSTER_CSV}")
    return df, cluster_keywords


# =============================================================================
# 군집 결과 시각화
# =============================================================================
def style_axis_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_cluster_wordclouds(df, cluster_keywords, cluster_col="Cluster", text_col="AB"):
    cluster_ids = sorted(df[cluster_col].dropna().unique())
    n_clusters = len(cluster_ids)
    ncols = min(3, max(1, n_clusters))
    nrows = math.ceil(n_clusters / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, max(8, nrows * 4.8)))
    fig.patch.set_facecolor(FIGURE_BG)
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        "Abstract Keyword Landscapes by Cluster",
        fontsize=22,
        fontweight="bold",
        color=TEXT_COLOR,
        x=0.055,
        ha="left",
        y=0.955,
    )

    for ax in axes:
        style_axis_panel(ax)
        ax.set_xticks([])
        ax.set_yticks([])

    for axis_index, cluster_id in enumerate(cluster_ids):
        ax = axes[axis_index]
        cluster_texts = df.loc[df[cluster_col] == cluster_id, text_col].dropna()
        full_text = " ".join(cluster_texts)
        if not full_text.strip():
            ax.set_visible(False)
            continue

        wordcloud = WordCloud(
            width=900,
            height=520,
            background_color=PANEL_BG,
            colormap=make_cluster_colormap(cluster_color(cluster_id)),
            stopwords=set(stop_words),
            collocations=False,
            prefer_horizontal=0.9,
            random_state=42,
        ).generate(full_text)

        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(
            f"Cluster {cluster_id}",
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR,
            loc="left",
            pad=10,
        )

    for axis_index in range(n_clusters, len(axes)):
        axes[axis_index].set_visible(False)

    plt.tight_layout(rect=(0.02, 0.03, 0.98, 1))
    fig.savefig(OUTPUT_WORDCLOUD_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {OUTPUT_WORDCLOUD_PNG}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def build_and_plot_cluster_network_with_labels(df, cluster_keywords, cluster_col="Cluster", title_col="TI"):
    G = nx.Graph()

    for idx, row in df.iterrows():
        G.add_node(idx, title=row[title_col], cluster=row[cluster_col])

    for cluster_id in df[cluster_col].unique():
        nodes = df[df[cluster_col] == cluster_id].index.tolist()
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])

    # NetworkX 3.6 defaults to the new energy-based solver for graphs with
    # 500+ nodes. For this dense cluster graph that solver collapses each
    # cluster into an almost single point. Force the legacy solver used to
    # generate the original network layout.
    pos = nx.spring_layout(G, seed=42, k=0.6, method="force")
    # Keep the original numeric node-color rendering so NetworkX draws the
    # complete network consistently, but replace Cluster 3's gray with orange.
    node_colors = [int(df.loc[n, cluster_col]) for n in G.nodes]
    network_cluster_cmap = ListedColormap(
        [
            "#1f77b4",  # Cluster 0: blue
            "#2ca02c",  # Cluster 1: green
            "#8c564b",  # Cluster 2: brown
            "#f28e2b",  # Cluster 3: orange
            "#17becf",  # Cluster 4: cyan
        ]
    )

    plt.figure(figsize=(12, 10))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        cmap=network_cluster_cmap,
        vmin=0,
        vmax=4,
        with_labels=False,
        node_size=100,
        edge_color="gray",
        alpha=0.5,
    )

    y_values = [coord[1] for coord in pos.values()]
    y_span = max(y_values) - min(y_values)
    manual_label_offsets = {
        4: (0.0, -y_span * 0.10),
    }

    for cluster_id in df[cluster_col].unique():
        cluster_nodes = df[df[cluster_col] == cluster_id].index.tolist()
        if not cluster_nodes:
            continue

        x_mean = sum(pos[n][0] for n in cluster_nodes) / len(cluster_nodes)
        y_mean = sum(pos[n][1] for n in cluster_nodes) / len(cluster_nodes)

        keywords = cluster_keywords[cluster_id]
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        label_text = f"Cluster {cluster_id}:\n{keywords}"
        offset_x, offset_y = manual_label_offsets.get(int(cluster_id), (0.0, 0.0))
        plt.text(
            x_mean + offset_x,
            y_mean + offset_y,
            label_text,
            fontsize=20,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.0,
                boxstyle="round",
            ),
        )

    plt.title("Cluster Network with Topic Labels", fontsize=30)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(OUTPUT_NETWORK_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved chart: {OUTPUT_NETWORK_PNG}")

    if SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================================================================
# 메인 실행 함수
# =============================================================================
def main():
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    df_result, cluster_keywords = run_clustering_pipeline(
        RIS_FILES,
        use_gpt=False,
        n_topics=DEFAULT_N_TOPICS,
    )

    plot_cluster_wordclouds(
        df_result,
        cluster_keywords,
        cluster_col="Cluster",
        text_col="AB",
    )

    build_and_plot_cluster_network_with_labels(
        df_result,
        cluster_keywords,
        cluster_col="Cluster",
        title_col="TI",
    )

    cluster_counts = df_result["Cluster"].value_counts().sort_index()
    print("\nNumber of papers in each cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} papers")

    print("done.")


if __name__ == "__main__":
    main()
