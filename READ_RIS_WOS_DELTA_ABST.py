import os
from pathlib import Path
import re
import sys
import tempfile

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
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# =============================================================================
# 입력·출력 경로 및 분석 설정
# =============================================================================
# Run this saved Python file; paths are relative to its project root.
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "Data" / "WoS"
FIG = BASE / "Fig" / "WoS" / 'Ver.8'
OUT = BASE / "Output" / "WoS" / 'Ver.8'
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_OUTPUT_DIR = FIG
OUTPUT_NETWORK_PNG = FIGURE_OUTPUT_DIR / "cluster_network_wos_abst3.png"
RIS_FILES = [DATA / f"savedrecs_{i}_cleaned.ris" for i in range(1, 6)]

MAX_YEAR = 2025


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


# =============================================================================
# 군집 결과 시각화
# =============================================================================


def build_and_plot_cluster_network_with_labels(df, cluster_keywords, cluster_col="Cluster", title_col="TI", output_path=None):
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
    print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges. Computing layout...", flush=True)
    # Older NetworkX releases use the force solver without a method argument.
    import inspect
    layout_options = {"seed": 42, "k": 0.6}
    if "method" in inspect.signature(nx.spring_layout).parameters:
        layout_options["method"] = "force"
    pos = nx.spring_layout(G, **layout_options)
    print("Layout complete. Drawing and saving network...", flush=True)
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

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path or OUTPUT_NETWORK_PNG, dpi=600, bbox_inches="tight")
    print(f"Saved chart: {output_path or OUTPUT_NETWORK_PNG}")

    if not SHOW_PLOTS:
        plt.close()


# =============================================================================
# 메인 실행 함수
# =============================================================================


def extract_contrastive_keywords(tfidf, labels, terms, k, top_n=15):
    """Rank positive within-minus-rest mean TF-IDF differences per cluster."""
    labels = np.asarray(labels)
    total = np.asarray(tfidf.sum(axis=0)).ravel()
    rows = []
    for cluster in range(k):
        mask = labels == cluster
        count = int(mask.sum())
        if count == 0 or count == len(labels):
            raise ValueError("Each cluster and its complement must contain documents")
        inside_sum = np.asarray(tfidf[mask].sum(axis=0)).ravel()
        inside = inside_sum / count
        outside = (total - inside_sum) / (len(labels) - count)
        difference = inside - outside
        candidates = np.flatnonzero(difference > 0)
        order = candidates[np.argsort(-difference[candidates], kind="stable")][:top_n]
        for rank, index in enumerate(order, 1):
            rows.append({"k": k, "cluster": cluster, "rank": rank,
                         "term": terms[index], "mean_tfidf": float(inside[index]),
                         "rest_mean_tfidf": float(outside[index]),
                         "contrastive_score": float(difference[index])})
    return rows


def evaluate_silhouette(tfidf):
    """Fit NMF once; compare k=2..10 and return the original k=5 labels."""
    features = NMF(n_components=5, random_state=1).fit_transform(tfidf)
    rows = []
    labels_k5 = None
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, random_state=1, n_init=10).fit_predict(features)
        score = float(silhouette_score(features, labels, metric="euclidean"))
        rows.append({"k": k, "nmf_components": 5, "n_documents": len(labels),
                     "silhouette": score})
        if k == 5:
            labels_k5 = labels.copy()
        print(f"k={k}: silhouette={score:.6f}", flush=True)
    return pd.DataFrame(rows), labels_k5


def plot_silhouette_comparison(results):
    best = results.loc[results.silhouette.idxmax()]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(results.k, results.silhouette, "o-", color="#286497", linewidth=2, markersize=6)
        ax.scatter([best.k], [best.silhouette], color="#c55a34", s=80, zorder=4)
        ax.annotate(f"k = {int(best.k)}: {best.silhouette:.4f}",
                    xy=(best.k, best.silhouette), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=12, color="#9c3e20")
        ax.set(xlabel="Number of clusters (k)", ylabel="Mean silhouette score",
               xticks=results.k.tolist())
        ax.set_ylim(min(.20, results.silhouette.min()-.02), max(.36, results.silhouette.max()+.035))
        ax.grid(axis="y", alpha=.22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)
#        fig.text(.5, .025, "NMF; n_components = 5, random_state = 1",
#                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .055, 1, 1))
        path = FIG / "silhouette_score_abst3.png"
        # Finish writing a separate PNG before replacing the existing file.
        # This avoids opening the existing PNG in Pillow's w+b mode.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.stem + "_", suffix=".png", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            fig.savefig(temporary_path, dpi=600, format="png")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Saved chart: {path}")
        if not SHOW_PLOTS:
            plt.close(fig)


def main():
    """Generate silhouette validation and the k=5 cluster network."""
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    output = OUT
    with threadpool_limits(limits=4):
        df = extract_ris_abstracts(RIS_FILES).reset_index(drop=True)
        tfidf, vectorizer = vectorize_abstracts(df)
        scores, labels = evaluate_silhouette(tfidf)
        scores.to_csv(OUT / "silhouette_scores_abst3.csv", index=False, encoding="utf-8-sig")
        plot_silhouette_comparison(scores)
        df["Cluster"] = labels
        rows = extract_contrastive_keywords(tfidf, labels, vectorizer.get_feature_names_out(), 5)
        pd.DataFrame(rows).to_csv(output / "cluster_keywords_abst3.csv", index=False, encoding="utf-8-sig")
        keywords = [[r["term"] for r in rows if r["cluster"] == c and r["rank"] <= 5] for c in range(5)]
    df.to_csv(output / "wos_clustered_abstracts_abst3.csv", index=False, encoding="utf-8-sig")
    print("Generating Fig. 8 network; the full layout may take several minutes.", flush=True)
    build_and_plot_cluster_network_with_labels(df, keywords, output_path=FIG / "cluster_network_wos_abst3.png")
    print(f"Both figures saved to: {FIG}", flush=True)
    print(f"Results: {output}", flush=True)
    # Show only after both files have been generated; a blocking window cannot
    # prevent the network from being calculated and saved.
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
