import collections
import math
import os
import re
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

try:
    import rispy
except ImportError:
    rispy = None


# =============================================================================
# 입력·출력 경로 및 실행 설정
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_OUTPUT_DIR = "E:/Text_Mining/Fig/WoS/Ver.7"
RIS_DIRECTORY = "E:/Text_Mining/Data/WoS"
RIS_FILENAMES = [
    "savedrecs_1_cleaned.ris",
    "savedrecs_2_cleaned.ris",
    "savedrecs_3_cleaned.ris",
    "savedrecs_4_cleaned.ris",
    "savedrecs_5_cleaned.ris",
]
MAX_YEAR = 2025
MIN_COUNTRY_TREND_YEAR = 2018
TOP_COUNTRY_TREND_COUNT = 5
MIN_COUNTRY_BUBBLE_YEAR = 2017
LINE_PLOT_START_YEAR = 2019
TOP_COUNTRY_BUBBLE_COUNT = 10
RUNNING_IN_SPYDER = any(
    key.startswith("SPYDER") or key.startswith("SPY_") for key in os.environ
)
SHOW_PLOTS = "--show" in sys.argv or RUNNING_IN_SPYDER
USE_EXISTING_CSV = "--use-existing-csv" in sys.argv

os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"
plt.rcParams["savefig.transparent"] = False
sns.set_theme(style="whitegrid", context="talk")


# =============================================================================
# 국가명 표준화 설정
# =============================================================================
COUNTRY_MAPPING = {
    "USA": "United States",
    "U.S.A.": "United States",
    "United States of America": "United States",
    "US": "United States",
    "U.S": "United States",
    "U.S.": "United States",
    "England": "United Kingdom",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "China, Peoples R": "China",
    "P.R. China": "China",
    "PRC": "China",
    "Peoples R China": "China",
    "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "S. Korea": "South Korea",
    "Germany, Fed Rep": "Germany",
    "Deutschland": "Germany",
    "Japan": "Japan",
    "France": "France",
    "Canada": "Canada",
    "Australia": "Australia",
    "India": "India",
    "Italy": "Italy",
    "Spain": "Spain",
    "Brazil": "Brazil",
    "Russia": "Russia",
    "Russian Federation": "Russia",
}

# =============================================================================
# 요약 데이터 및 차트 설정
# =============================================================================
SUMMARY_FILES = {
    "country_counts": "country_paper_counts_top10.csv",
    "country_citations": "country_total_citations_top10.csv",
    "country_average_citations": "country_average_citations_top10.csv",
    "country_all_metrics": "country_all_metrics.csv",
    "country_year_counts": "country_year_counts_2018_2025.csv",
    "country_year_metrics": "country_year_metrics_2018_2025.csv",
    "journal_counts": "journal_paper_counts_top10.csv",
    "journal_citations": "journal_total_citations_top10.csv",
    "journal_average_citations": "journal_average_citations_top10.csv",
    "most_cited_papers": "most_cited_papers_top10.csv",
}
JOURNAL_METRICS_LOOKUP_FILE = "journal_metrics_lookup.csv"

CHART_CONFIGS = [
    {
        "data_key": "country_counts",
        "label_column": "Country",
        "value_column": "Paper_Count",
        "title": "Top Countries by Paper Count",
        "subtitle": f"Web of Science records up to {MAX_YEAR}",
        "xlabel": "Number of Papers",
        "palette": "crest",
        "filename": "Paper_counts_by_country.png",
    },
    {
        "data_key": "country_citations",
        "label_column": "Country",
        "value_column": "Total_Citations",
        "title": "Top Countries by Total Citations",
        "subtitle": f"Web of Science records up to {MAX_YEAR}",
        "xlabel": "Total Citations",
        "palette": "flare",
        "filename": "Total_Citations_by_country.png",
    },
    {
        "data_key": "country_average_citations",
        "label_column": "Country",
        "value_column": "Average_Citations",
        "title": "Top Countries by Average Citations per Paper",
        "subtitle": f"Web of Science records up to {MAX_YEAR}",
        "xlabel": "Average Citations",
        "palette": "rocket_r",
        "filename": "Average_Citations_by_country.png",
    },
]

MINOR_TITLE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}

JOURNAL_ACRONYMS = {
    "ACM",
    "AI",
    "AQI",
    "IEEE",
    "IJERPH",
    "JMIR",
    "MHEALTH",
    "UHEALTH",
    "MDPI",
}

PUBLISHER_NAME_OVERRIDES = {
    "AMER CHEMICAL SOC": "American Chemical Society",
    "ELSEVIER": "Elsevier",
    "ELSEVIER SCI LTD": "Elsevier",
    "MDPI": "MDPI",
    "NATURE PORTFOLIO": "Nature Portfolio",
    "PERGAMON-ELSEVIER SCIENCE LTD": "Elsevier",
    "TURKISH NATL COMMITTEE AIR POLLUTION RES & CONTROL-TUNCAP": "Elsevier",
}

# =============================================================================
# 문자열·수치 표기 공통 유틸리티
# =============================================================================
def format_metric(value):
    if pd.isna(value):
        return ""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def axis_formatter(value, _position):
    if abs(value - round(value)) < 1e-8:
        return f"{int(value):,}"
    return f"{value:,.1f}"


def as_text(value):
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def format_author_display(author_value):
    if isinstance(author_value, list):
        authors = [str(item).strip() for item in author_value if str(item).strip()]
    else:
        text = str(author_value).strip()
        if not text:
            authors = []
        elif "; " in text:
            authors = [item.strip() for item in text.split("; ") if item.strip()]
        else:
            authors = [text]

    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


def format_journal_title(journal_value):
    journal_text = str(journal_value).strip()
    if not journal_text:
        return "N/A"

    journal_text = re.sub(r"\s+", " ", journal_text)
    words = journal_text.split(" ")
    formatted_words = []

    for index, word in enumerate(words):
        if not word:
            continue

        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9&./-]+)([^A-Za-z0-9]*)$", word)
        if not match:
            formatted_words.append(word)
            continue

        prefix, core, suffix = match.groups()
        upper_core = core.upper()
        lower_core = core.lower()

        if upper_core in JOURNAL_ACRONYMS or any(char.isdigit() for char in core):
            formatted_core = upper_core
        elif index != 0 and index != len(words) - 1 and lower_core in MINOR_TITLE_WORDS:
            formatted_core = lower_core
        else:
            formatted_core = core.title()

        formatted_words.append(f"{prefix}{formatted_core}{suffix}")

    return " ".join(formatted_words)


def format_publisher_name(publisher_value):
    publisher_text = str(publisher_value).strip()
    if not publisher_text or publisher_text.lower() == "nan":
        return ""

    publisher_text = re.sub(r"\s+", " ", publisher_text)
    upper_text = publisher_text.upper()
    if upper_text in PUBLISHER_NAME_OVERRIDES:
        return PUBLISHER_NAME_OVERRIDES[upper_text]

    if upper_text == publisher_text:
        return publisher_text.title()
    return publisher_text


# =============================================================================
# RIS 보조 정보 추출 및 표준화
# =============================================================================
def extract_publisher(entry):
    return format_publisher_name(
        entry.get("publisher")
        or entry.get("publishing_house")
        or entry.get("publication_place")
        or ""
    )


def load_journal_publisher_lookup_from_ris():
    publisher_counter = collections.defaultdict(collections.Counter)
    for filename in RIS_FILENAMES:
        file_path = os.path.join(RIS_DIRECTORY, filename)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as ris_file:
            raw_text = ris_file.read()

        for record_text in re.split(r"\nER\s+-", raw_text):
            journal_match = (
                re.search(r"^T2\s+-\s+(.+)$", record_text, re.MULTILINE)
                or re.search(r"^JO\s+-\s+(.+)$", record_text, re.MULTILINE)
                or re.search(r"^JF\s+-\s+(.+)$", record_text, re.MULTILINE)
            )
            publisher_match = re.search(r"^PU\s+-\s+(.+)$", record_text, re.MULTILINE)
            if not journal_match or not publisher_match:
                continue

            journal = format_journal_title(journal_match.group(1))
            publisher = format_publisher_name(publisher_match.group(1))
            if journal and publisher:
                publisher_counter[journal][publisher] += 1

    return {
        journal: counter.most_common(1)[0][0]
        for journal, counter in publisher_counter.items()
        if counter
    }


def extract_citations(entry):
    citations = 0
    notes = entry.get("notes")
    if not notes:
        return citations

    note_items = [notes] if isinstance(notes, str) else notes
    for item in note_items:
        match = re.search(r"Total Times Cited:\s*(\d+)", str(item))
        if match:
            return int(match.group(1))
    return citations


def extract_affiliation_text(entry):
    return as_text(
        entry.get("author_address")
        or entry.get("custom1")
        or entry.get("custom3")
        or ""
    )


def extract_and_standardize_countries(affiliation_string):
    if not isinstance(affiliation_string, str) or not affiliation_string.strip():
        return []

    found_countries = set()
    for raw_country, standard_country in COUNTRY_MAPPING.items():
        if re.search(r"\b" + re.escape(raw_country) + r"\b", affiliation_string, re.IGNORECASE):
            found_countries.add(standard_country)

    potential_countries = re.findall(
        r"[,\.]\s*([A-Z][a-zA-Z\s]+?)(?:[\.\d\s]|$)",
        affiliation_string,
    )
    for candidate in potential_countries:
        candidate = candidate.strip()
        for raw_country, standard_country in COUNTRY_MAPPING.items():
            if candidate.lower() == raw_country.lower():
                found_countries.add(standard_country)
                break
        if candidate in COUNTRY_MAPPING.values():
            found_countries.add(candidate)

    return sorted(found_countries)


# =============================================================================
# RIS 데이터 로드 및 분석용 데이터프레임 생성
# =============================================================================
def build_dataframe_from_ris():
    if rispy is None:
        raise ImportError("rispy is required to parse RIS files.")

    ris_file_paths = [os.path.join(RIS_DIRECTORY, filename) for filename in RIS_FILENAMES]
    all_records = []

    for file_path in ris_file_paths:
        print(f"Reading file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as bibliography_file:
                entries = rispy.load(bibliography_file)
        except FileNotFoundError:
            print(f"Error: file not found at {file_path}")
            continue
        except Exception as error:
            print(f"An unexpected error occurred while reading {file_path}: {error}")
            continue

        for entry in entries:
            authors = entry.get("authors", [])
            affiliation_text = extract_affiliation_text(entry)
            all_records.append(
                {
                    "title": entry.get("title", "N/A"),
                    "year": entry.get("year", "N/A"),
                    "journal": entry.get("secondary_title", entry.get("journal_name", "N/A")),
                    "publisher": extract_publisher(entry),
                    "authors": "; ".join(authors) if isinstance(authors, list) else as_text(authors),
                    "citations": extract_citations(entry),
                    "affiliations": affiliation_text,
                    "country": affiliation_text,
                }
            )

    df_wos = pd.DataFrame(all_records)
    if df_wos.empty:
        raise ValueError("No RIS records could be loaded.")

    df_wos.drop_duplicates(subset=["title", "year"], inplace=True)
    df_wos["year"] = pd.to_numeric(df_wos["year"], errors="coerce")
    df_wos.dropna(subset=["year"], inplace=True)
    df_wos["year"] = df_wos["year"].astype(int)
    df_wos["citations"] = pd.to_numeric(df_wos["citations"], errors="coerce").fillna(0).astype(int)
    df_wos["journal"] = df_wos["journal"].fillna("N/A").replace("", "N/A").apply(format_journal_title)
    df_wos["publisher"] = df_wos["publisher"].fillna("").apply(format_publisher_name)

    print(
        f"Loaded {len(df_wos):,} unique papers "
        f"from {df_wos['year'].min()} to {df_wos['year'].max()}."
    )

    df_filtered = df_wos[df_wos["year"] <= MAX_YEAR].copy()
    if df_filtered.empty:
        raise ValueError(f"No records were found up to {MAX_YEAR}.")

    df_filtered["countries"] = df_filtered["country"].apply(extract_and_standardize_countries)

    print(
        f"Using {len(df_filtered):,} papers "
        f"from {df_filtered['year'].min()} to {df_filtered['year'].max()}."
    )

    return df_filtered


# =============================================================================
# 국가·저널·인용 지표 집계
# =============================================================================
def build_summary_frames(df_filtered):
    all_countries_counts = collections.Counter()
    for countries_list in df_filtered["countries"]:
        for country in countries_list:
            all_countries_counts[country] += 1

    country_counts_df = pd.DataFrame(
        all_countries_counts.most_common(10),
        columns=["Country", "Paper_Count"],
    )

    country_citations_sum = collections.Counter()
    for _, row in df_filtered.iterrows():
        for country in row["countries"]:
            country_citations_sum[country] += row["citations"]

    country_citations_df = pd.DataFrame(
        country_citations_sum.most_common(10),
        columns=["Country", "Total_Citations"],
    )

    average_citations_data = []
    for country, count in all_countries_counts.items():
        if count > 0:
            average_citations_data.append(
                {
                    "Country": country,
                    "Average_Citations": country_citations_sum[country] / count,
                }
            )

    average_citations_df = (
        pd.DataFrame(average_citations_data)
        .sort_values(by="Average_Citations", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    country_all_metrics_df = (
        pd.DataFrame(
            [
                {
                    "Country": country,
                    "Paper_Count": count,
                    "Total_Citations": country_citations_sum[country],
                    "Average_Citations": country_citations_sum[country] / count if count else 0,
                }
                for country, count in all_countries_counts.items()
            ]
        )
        .sort_values(by="Paper_Count", ascending=False)
        .reset_index(drop=True)
    )

    df_country_year = (
        df_filtered[["year", "countries"]]
        .explode("countries")
        .dropna(subset=["countries"])
        .rename(columns={"countries": "Country", "year": "Year"})
    )
    country_year_counts_df = (
        df_country_year[df_country_year["Year"] >= MIN_COUNTRY_TREND_YEAR]
        .groupby(["Year", "Country"], as_index=False)
        .size()
        .rename(columns={"size": "Paper_Count"})
        .sort_values(["Year", "Paper_Count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    df_country_year_citations = (
        df_filtered[["year", "countries", "citations"]]
        .explode("countries")
        .dropna(subset=["countries"])
        .rename(columns={"countries": "Country", "year": "Year", "citations": "Citation"})
    )
    country_year_metrics_df = (
        df_country_year_citations[df_country_year_citations["Year"] >= MIN_COUNTRY_TREND_YEAR]
        .groupby(["Year", "Country"], as_index=False)
        .agg(Paper_Count=("Citation", "size"), Total_Citations=("Citation", "sum"))
    )
    country_year_metrics_df["Average_Citations"] = (
        country_year_metrics_df["Total_Citations"] / country_year_metrics_df["Paper_Count"]
    )
    country_year_metrics_df = country_year_metrics_df.sort_values(
        ["Year", "Paper_Count"],
        ascending=[True, False],
    ).reset_index(drop=True)

    journal_counts = (
        df_filtered["journal"]
        .fillna("N/A")
        .replace("", "N/A")
        .value_counts()
        .rename_axis("Journal")
        .reset_index(name="Paper_Count")
    )
    top_journals_by_count = journal_counts.head(10).copy()

    journal_citations_sum = (
        df_filtered.groupby("journal", dropna=False)["citations"]
        .sum()
        .reset_index()
        .rename(columns={"journal": "Journal", "citations": "Total_Citations"})
        .sort_values(by="Total_Citations", ascending=False)
    )
    journal_citations_sum["Journal"] = journal_citations_sum["Journal"].fillna("N/A").replace("", "N/A")
    top_journals_by_total_citations = journal_citations_sum.head(10).copy()

    journal_stats = pd.merge(journal_counts, journal_citations_sum, on="Journal", how="left")
    journal_stats["Average_Citations"] = journal_stats["Total_Citations"] / journal_stats["Paper_Count"]
    journal_publishers = (
        df_filtered[["journal", "publisher"]]
        .dropna()
        .query("publisher != ''")
        .groupby(["journal", "publisher"], as_index=False)
        .size()
        .sort_values(["journal", "size"], ascending=[True, False])
        .drop_duplicates("journal")
        .rename(columns={"journal": "Journal", "publisher": "Publisher"})
        [["Journal", "Publisher"]]
    )
    journal_stats = journal_stats.merge(journal_publishers, on="Journal", how="left")
    top_journals_by_avg_citations = (
        journal_stats.sort_values(by="Average_Citations", ascending=False).head(10).copy()
    )
    top_journals_by_count_with_stats = (
        journal_stats.sort_values(by="Paper_Count", ascending=False).head(10).copy()
    )

    top_n_papers_display = (
        df_filtered.sort_values(by="citations", ascending=False)
        .head(10)[["title", "authors", "journal", "year", "citations"]]
        .rename(
            columns={
                "title": "Title",
                "authors": "Authors",
                "journal": "Journal",
                "year": "Year",
                "citations": "Citation",
            }
        )
        .copy()
    )
    top_n_papers_display["Authors"] = top_n_papers_display["Authors"].apply(format_author_display)

    return {
        "country_counts": country_counts_df,
        "country_citations": country_citations_df,
        "country_average_citations": average_citations_df,
        "country_all_metrics": country_all_metrics_df,
        "country_year_counts": country_year_counts_df,
        "country_year_metrics": country_year_metrics_df,
        "journal_counts": top_journals_by_count_with_stats,
        "journal_citations": top_journals_by_total_citations,
        "journal_average_citations": top_journals_by_avg_citations,
        "most_cited_papers": top_n_papers_display,
        "top_journals_by_count": top_journals_by_count,
    }


# =============================================================================
# 요약 CSV 저장·불러오기 및 결과 출력
# =============================================================================
def save_summary_frames(summary_frames):
    for key, filename in SUMMARY_FILES.items():
        output_path = os.path.join(SCRIPT_DIR, filename)
        summary_frames[key].to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_path}")


def load_summary_frames():
    summary_frames = {}
    for key, filename in SUMMARY_FILES.items():
        input_path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(input_path):
            if key in {"country_all_metrics", "country_year_metrics"}:
                continue
            raise FileNotFoundError(
                f"Summary file is missing: {input_path}. "
                "Run once in a Python environment with rispy installed "
                "or add --use-existing-csv only when those CSV files already exist."
            )
        summary_frames[key] = pd.read_csv(input_path)

    if "country_all_metrics" not in summary_frames:
        country_counts = summary_frames["country_counts"].copy()
        country_citations = summary_frames["country_citations"].copy()
        country_averages = summary_frames["country_average_citations"].copy()

        fallback_metrics = country_counts.merge(country_citations, on="Country", how="left").merge(
            country_averages,
            on="Country",
            how="left",
        )
        fallback_metrics["Average_Citations"] = fallback_metrics["Average_Citations"].fillna(
            fallback_metrics["Total_Citations"] / fallback_metrics["Paper_Count"]
        )
        fallback_metrics["Total_Citations"] = fallback_metrics["Total_Citations"].fillna(
            fallback_metrics["Average_Citations"] * fallback_metrics["Paper_Count"]
        )
        summary_frames["country_all_metrics"] = fallback_metrics.sort_values(
            by="Paper_Count",
            ascending=False,
        ).reset_index(drop=True)

    if "country_year_metrics" not in summary_frames and "country_year_counts" in summary_frames:
        fallback_year_metrics = summary_frames["country_year_counts"].copy()
        fallback_year_metrics["Total_Citations"] = np.nan
        fallback_year_metrics["Average_Citations"] = np.nan
        summary_frames["country_year_metrics"] = fallback_year_metrics[
            ["Year", "Country", "Paper_Count", "Total_Citations", "Average_Citations"]
        ].copy()

    summary_frames["top_journals_by_count"] = summary_frames["journal_counts"].copy()
    return summary_frames


def print_summary_tables(summary_frames):
    print("\n### Top countries by paper count")
    print(summary_frames["country_counts"])

    print("\n### Top countries by total citations")
    print(summary_frames["country_citations"])

    print("\n### Top countries by average citations")
    print(summary_frames["country_average_citations"])

    print("\n### Top countries by combined metrics")
    print(summary_frames["country_all_metrics"].head(10))

    print("\n### Country paper counts by year (2018+)")
    print(summary_frames["country_year_counts"].head(15))

    print("\n### Country metrics by year (2018+)")
    print(summary_frames["country_year_metrics"].head(15))

    print("\n### Top journals by paper count")
    print(summary_frames["top_journals_by_count"])

    print("\n### Top journals by total citations")
    print(summary_frames["journal_citations"])

    print("\n### Top journals by average citations")
    print(summary_frames["journal_average_citations"])

    print("\n### Most cited papers")
    print(summary_frames["most_cited_papers"])


# =============================================================================
# 기본 순위 막대그래프
# =============================================================================
def create_ranked_bar_chart(dataframe, label_column, value_column, title, subtitle, xlabel, palette, filename):
    chart_df = dataframe[[label_column, value_column]].dropna().copy()
    chart_df[value_column] = pd.to_numeric(chart_df[value_column], errors="coerce")
    chart_df.dropna(subset=[value_column], inplace=True)
    chart_df.sort_values(by=value_column, ascending=True, inplace=True)

    if chart_df.empty:
        print(f"Skipped chart because no data was available: {filename}")
        return

    max_value = chart_df[value_column].max()
    right_margin = max(max_value * 0.12, 1)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    background_color = "white"
    text_color = "#000000"

    fig.patch.set_facecolor(background_color)
    fig.patch.set_alpha(1)
    ax.set_facecolor(background_color)
    ax.patch.set_alpha(1)

    colors = sns.color_palette(palette, n_colors=len(chart_df))
    bars = ax.barh(
        chart_df[label_column],
        chart_df[value_column],
        color=colors,
        edgecolor="none",
        height=0.68,
    )

    ax.set_xlim(0, max_value + right_margin)
    ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.set_xlabel(xlabel, fontsize=20, color=text_color, labelpad=14)
    ax.set_ylabel("Country", fontsize=20, color=text_color, labelpad=14)
    panel_label_map = {
        "Paper_counts_by_country.png": "(a)",
        "Total_Citations_by_country.png": "(b)",
        "Average_Citations_by_country.png": "(c)",
    }
    fig.text(
        0.015,
        0.985,
        panel_label_map.get(filename, ""),
        ha="left",
        va="top",
        fontsize=28,
        color=text_color,
    )

    ax.tick_params(axis="x", labelsize=20, colors=text_color)
    ax.tick_params(axis="y", labelsize=20, colors=text_color)
    ax.grid(axis="x", linestyle="--", linewidth=1.2, alpha=0.8, color="#8ea6bf")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(text_color)
        spine.set_linewidth(1.5)

    for bar, value in zip(bars, chart_df[value_column]):
        ax.text(
            value + max(right_margin * 0.12, 0.15),
            bar.get_y() + bar.get_height() / 2,
            format_metric(value),
            va="center",
            ha="left",
            fontsize=18,
            fontweight="bold",
            color=text_color,
        )

    plt.tight_layout()
    numbered_filename_map = {
        "Paper_counts_by_country.png": "1.Paper_counts_by_country.png",
        "Total_Citations_by_country.png": "2.Total_Citations_by_country.png",
        "Average_Citations_by_country.png": "3.Average_Citations_by_country.png",
    }
    output_path = os.path.join(FIGURE_OUTPUT_DIR, numbered_filename_map.get(filename, filename))
    fig.savefig(
        output_path,
        dpi=300,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    print(f"Saved chart: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# 국가별 흐름·추세 시각화
# =============================================================================
def add_flow_band(ax, x_left, x_right, top_left, bottom_left, top_right, bottom_right, color):
    control_offset = (x_right - x_left) * 0.36
    vertices = [
        (x_left, top_left),
        (x_left + control_offset, top_left),
        (x_right - control_offset, top_right),
        (x_right, top_right),
        (x_right, bottom_right),
        (x_right - control_offset, bottom_right),
        (x_left + control_offset, bottom_left),
        (x_left, bottom_left),
        (x_left, top_left),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(vertices, codes),
        facecolor=color,
        edgecolor="none",
        alpha=0.88,
        zorder=1,
    )
    ax.add_patch(patch)


def round_percentages_to_100(share_map):
    raw_percentages = {key: float(value) * 100 for key, value in share_map.items()}
    rounded_down = {key: int(math.floor(value)) for key, value in raw_percentages.items()}
    remainder = 100 - sum(rounded_down.values())
    fractional_order = sorted(
        raw_percentages,
        key=lambda key: raw_percentages[key] - rounded_down[key],
        reverse=True,
    )
    for key in fractional_order[:remainder]:
        rounded_down[key] += 1
    return rounded_down


def create_country_combined_sankey(summary_frames):
    paper_top_countries = summary_frames["country_counts"]["Country"].head(10).tolist()
    average_top_countries = summary_frames["country_average_citations"]["Country"].head(10).tolist()
    combined_countries = list(dict.fromkeys(paper_top_countries + average_top_countries))

    combined_df = summary_frames["country_all_metrics"][
        summary_frames["country_all_metrics"]["Country"].isin(combined_countries)
    ].copy()
    combined_df["Paper_Count"] = pd.to_numeric(combined_df["Paper_Count"], errors="coerce")
    combined_df["Total_Citations"] = pd.to_numeric(combined_df["Total_Citations"], errors="coerce")
    combined_df["Average_Citations"] = pd.to_numeric(combined_df["Average_Citations"], errors="coerce")
    combined_df.dropna(subset=["Paper_Count", "Total_Citations", "Average_Citations"], inplace=True)
    combined_df = combined_df[combined_df["Paper_Count"] > 0]

    if combined_df.empty:
        print("Skipped combined sankey chart because no data was available.")
        return

    paper_total = combined_df["Paper_Count"].sum()
    average_citations_total = combined_df["Average_Citations"].sum()

    if paper_total <= 0 or average_citations_total <= 0:
        print("Skipped combined sankey chart because paper or citation totals were zero.")
        return

    combined_df["Paper_Count_Share"] = combined_df["Paper_Count"] / paper_total
    combined_df["Average_Citations_Share"] = combined_df["Average_Citations"] / average_citations_total

    background_color = "#ffffff"
    text_color = "black"
    muted_color = "#5c6f82"
    node_fill = "#edf4fb"
    node_edge = "#c7d8ea"

    fig_height = max(6.8, len(combined_df) * 0.84)
    fig, ax = plt.subplots(figsize=(14.2, fig_height))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.045, 0.94, "Total Paper", transform=ax.transAxes, fontsize=20, color="black", ha="left", fontweight="bold")
    ax.text(0.77, 0.94, "Avg. Citations per Paper", transform=ax.transAxes, fontsize=20, color="black", ha="left", fontweight="bold")

    top_margin = 0.12
    bottom_margin = 0.06
    gap = 0.014
    usable_height = 1 - top_margin - bottom_margin - gap * (len(combined_df) - 1)

    left_height_map = {
        row["Country"]: row["Paper_Count_Share"] * usable_height
        for _, row in combined_df.iterrows()
    }
    right_height_map = {
        row["Country"]: row["Average_Citations_Share"] * usable_height
        for _, row in combined_df.iterrows()
    }
    left_share_map = round_percentages_to_100(
        {row["Country"]: row["Paper_Count_Share"] for _, row in combined_df.iterrows()}
    )
    right_share_map = round_percentages_to_100(
        {row["Country"]: row["Average_Citations_Share"] for _, row in combined_df.iterrows()}
    )

    left_positions = {}
    current_top = 1 - top_margin
    for _, row in combined_df.sort_values(by="Paper_Count", ascending=False).iterrows():
        height = left_height_map[row["Country"]]
        top = current_top
        bottom = top - height
        left_positions[row["Country"]] = (top, bottom)
        current_top = bottom - gap

    right_positions = {}
    current_top = 1 - top_margin
    for _, row in combined_df.sort_values(by="Average_Citations", ascending=False).iterrows():
        height = right_height_map[row["Country"]]
        top = current_top
        bottom = top - height
        right_positions[row["Country"]] = (top, bottom)
        current_top = bottom - gap

    left_x = 0.08
    right_x = 0.81
    node_width = 0.06
    paper_ranked = combined_df.sort_values("Paper_Count", ascending=True).reset_index(drop=True)
    contrast_colors = [
        "#1f4e9d",
        "#0087a8",
        "#00a676",
        "#5b8c00",
        "#6f4bb2",
        "#b33c9f",
        "#d55e00",
        "#f28e2b",
        "#d62728",
        "#8b0000",
    ]
    color_positions = np.linspace(0, len(contrast_colors) - 1, len(paper_ranked)).round().astype(int)
    color_map = {
        row["Country"]: contrast_colors[color_positions[index]]
        for index, row in paper_ranked.iterrows()
    }

    for _, row in combined_df.iterrows():
        country = row["Country"]
        left_top, left_bottom = left_positions[country]
        right_top, right_bottom = right_positions[country]
        left_mid_y = left_bottom + (left_top - left_bottom) / 2
        right_mid_y = right_bottom + (right_top - right_bottom) / 2

        add_flow_band(
            ax=ax,
            x_left=left_x + node_width,
            x_right=right_x,
            top_left=left_top,
            bottom_left=left_bottom,
            top_right=right_top,
            bottom_right=right_bottom,
            color=color_map[country],
        )

        ax.add_patch(
            Rectangle(
                (left_x, left_bottom),
                node_width,
                left_top - left_bottom,
                facecolor=node_fill,
                edgecolor=node_edge,
                linewidth=1.1,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (right_x, right_bottom),
                node_width,
                right_top - right_bottom,
                facecolor=node_fill,
                edgecolor=node_edge,
                linewidth=1.1,
                zorder=3,
            )
        )

        ax.text(
            left_x - 0.015,
            left_mid_y,
            f"{country} ({left_share_map[country]}%)",
            ha="right",
            va="center",
            fontsize=15,
#            fontweight="bold",
            color=text_color,
            zorder=4,
        )
        ax.text(
            right_x + node_width + 0.015,
            right_mid_y,
            f"{country} ({right_share_map[country]}%)",
            ha="left",
            va="center",
            fontsize=15,
 #           fontweight="bold",
            color=text_color,
            zorder=4,
        )

    output_path = os.path.join(FIGURE_OUTPUT_DIR, "4.Paper_Count_to_Average_Citations_Sankey_by_country.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def build_top_country_year_pivot(summary_frames):
    top_countries = summary_frames["country_counts"]["Country"].head(TOP_COUNTRY_TREND_COUNT).tolist()
    year_counts_df = summary_frames["country_year_counts"].copy()
    year_counts_df["Year"] = pd.to_numeric(year_counts_df["Year"], errors="coerce").astype(int)
    year_counts_df["Paper_Count"] = pd.to_numeric(year_counts_df["Paper_Count"], errors="coerce").fillna(0)
    year_counts_df = year_counts_df[year_counts_df["Country"].isin(top_countries)]

    all_years = list(range(MIN_COUNTRY_TREND_YEAR, MAX_YEAR + 1))
    pivot_df = (
        year_counts_df.pivot(index="Year", columns="Country", values="Paper_Count")
        .reindex(index=all_years, columns=top_countries)
        .fillna(0)
        .astype(int)
    )
    return pivot_df


def create_country_trend_line_chart(summary_frames):
    pivot_df = build_top_country_year_pivot(summary_frames)
    if pivot_df.empty:
        print("Skipped yearly country trend line chart because no data was available.")
        return

    background_color = "white"
    text_color = "black"
    muted_color = "black"
    grid_color = "#b8cadc"
    line_colors = ["#234f9b", "#d1495b", "#2a9d8f", "#7a52a5", "#e08a28"]

    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax2 = ax.twinx()
    ax2.set_facecolor(background_color)

    # Draw a complete border around the plotting area. The primary axis
    # supplies the left, top, and bottom edges; the secondary axis supplies
    # the right edge.
    for spine_name in ("left", "top", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("black")
        ax.spines[spine_name].set_linewidth(1.2)
    ax.spines["right"].set_visible(False)

    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("black")
    ax2.spines["right"].set_linewidth(1.2)
    ax.grid(axis="y", linestyle="--", linewidth=1.1, alpha=0.8, color=grid_color)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax2.grid(False)

    china_country = "China" if "China" in pivot_df.columns else pivot_df.columns[0]
    other_countries = [country for country in pivot_df.columns if country != china_country]
    country_axes = {china_country: ax}
    country_axes.update({country: ax2 for country in other_countries})
    color_map = {country: color for color, country in zip(line_colors, pivot_df.columns)}

    handles = []
    labels = []
    labels_below_points = {
        ("United States", 2022),
        ("South Korea", 2023),
        ("India", 2021),
        ("India", 2022),
        ("Italy", 2020),
        ("Italy", 2021),
    }
    for country in pivot_df.columns:
        plot_ax = country_axes[country]
        line, = plot_ax.plot(
            pivot_df.index,
            pivot_df[country],
            marker="o",
            linewidth=2.8,
            markersize=7.5,
            color=color_map[country],
            label=country,
        )
        handles.append(line)
        labels.append(country)

        label_offset = 8 if country == china_country else 3
        for year, value in pivot_df[country].items():
            if value <= 0:
                continue

            label_y = value + label_offset
            label_va = "bottom"
            if (country, year) in labels_below_points:
                if country == "Italy":
                    label_y = value - 3
                elif country == "India" and year == 2021:
                    label_y = value - 2
                else:
                    label_y = value - 5
                label_va = "top"
            elif country == "India" and year == 2020:
                label_y = value + 1

            plot_ax.text(
                year,
                label_y,
                format_metric(value),
                ha="center",
                va=label_va,
                fontsize=18,
                color=color_map[country],
                fontweight="bold",
            )

    ax.set_xlabel("Publication Year", fontsize=20, color=text_color, labelpad=16)
    ax.set_ylabel("Number of Papers (China)", fontsize=20, color=text_color, labelpad=18)
    ax2.set_ylabel("Number of Papers (Other Countries)", fontsize=20, color=text_color, labelpad=18)
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.065, 0.5)
    ax2.yaxis.set_label_coords(1.065, 0.5)
    ax.set_xticks(pivot_df.index)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right", color=muted_color)
    ax.set_ylim(0, 700)
    ax2.set_ylim(0, 175)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax2.yaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.tick_params(axis="both", colors=muted_color, labelsize=16)
    ax2.tick_params(axis="y", colors=muted_color, labelsize=16)
    ax.legend(handles, labels, frameon=False, ncol=3, fontsize=18, loc="upper left")

    plt.tight_layout()
    output_path = os.path.join(FIGURE_OUTPUT_DIR, "6.Country_Paper_Trends_2018_2025.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def create_country_trend_heatmap(summary_frames):
    pivot_df = build_top_country_year_pivot(summary_frames)
    if pivot_df.empty:
        print("Skipped yearly country heatmap because no data was available.")
        return

    background_color = "#f7fbff"
    text_color = "#16324f"
    muted_color = "#5c6f82"

    fig, ax = plt.subplots(figsize=(13.6, 5.2))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    heatmap_df = pivot_df.T
    sns.heatmap(
        heatmap_df,
        cmap="YlGnBu",
        annot=True,
        fmt="d",
        linewidths=1.0,
        linecolor="#f7fbff",
        cbar_kws={"label": "Number of Papers"},
        ax=ax,
    )

    ax.set_title(
        "Heatmap of Annual Paper Counts by Top 5 Countries",
        fontsize=22,
        fontweight="bold",
        color=text_color,
        loc="left",
        pad=24,
    )
    ax.text(
        0,
        1.05,
        f"Publication intensity by country and year from {MIN_COUNTRY_TREND_YEAR} to {MAX_YEAR}",
        transform=ax.transAxes,
        fontsize=11,
        color=muted_color,
        ha="left",
    )
    ax.set_xlabel("Publication Year", fontsize=13, color=text_color, labelpad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, colors=muted_color)
    ax.tick_params(axis="y", labelrotation=0, colors=text_color)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.label.set_color(text_color)
    colorbar.ax.tick_params(colors=muted_color)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_OUTPUT_DIR, "Country_Paper_Heatmap_2018_2025.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def create_country_citation_animation(summary_frames):
    metrics_df = summary_frames["country_year_metrics"].copy()
    if metrics_df.empty:
        print("Skipped country citation animation because no data was available.")
        return

    metrics_df["Year"] = pd.to_numeric(metrics_df["Year"], errors="coerce")
    metrics_df["Paper_Count"] = pd.to_numeric(metrics_df["Paper_Count"], errors="coerce")
    metrics_df["Total_Citations"] = pd.to_numeric(metrics_df["Total_Citations"], errors="coerce")
    metrics_df.dropna(subset=["Year", "Paper_Count", "Total_Citations"], inplace=True)

    top_countries = summary_frames["country_counts"]["Country"].head(TOP_COUNTRY_TREND_COUNT).tolist()
    metrics_df = metrics_df[metrics_df["Country"].isin(top_countries)].copy()
    metrics_df = metrics_df[metrics_df["Year"] >= MIN_COUNTRY_TREND_YEAR]
    metrics_df["Year"] = metrics_df["Year"].astype(int)

    if metrics_df.empty:
        print("Skipped country citation animation because yearly citation data was unavailable.")
        return

    years = sorted(metrics_df["Year"].unique())
    background_color = "#f7fbff"
    text_color = "#16324f"
    muted_color = "#5c6f82"
    grid_color = "#b8cadc"
    color_map = {
        "China": "#234f9b",
        "United States": "#d1495b",
        "India": "#2a9d8f",
        "South Korea": "#7a52a5",
        "United Kingdom": "#e08a28",
    }

    y_max = max(metrics_df["Total_Citations"].max() * 1.08, 100)
    bubble_sizes = 40 + np.sqrt(metrics_df["Paper_Count"]) * 18
    metrics_df["Bubble_Size"] = bubble_sizes

    fig, ax = plt.subplots(figsize=(14.8, 8.8))
    fig.patch.set_facecolor(background_color)

    def draw_frame(frame_year):
        ax.clear()
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="both", linestyle="--", linewidth=1.0, alpha=0.75, color=grid_color)
        ax.set_axisbelow(True)

        cumulative_df = metrics_df[metrics_df["Year"] <= frame_year].copy()
        current_df = metrics_df[metrics_df["Year"] == frame_year].copy()

        for country in top_countries:
            country_cumulative = cumulative_df[cumulative_df["Country"] == country]
            if country_cumulative.empty:
                continue

            ax.scatter(
                country_cumulative["Year"],
                country_cumulative["Total_Citations"],
                s=country_cumulative["Bubble_Size"],
                color=color_map.get(country, "#4c78a8"),
                alpha=0.24,
                edgecolors="none",
                zorder=2,
            )

        for country in top_countries:
            country_current = current_df[current_df["Country"] == country]
            if country_current.empty:
                continue

            x_value = country_current["Year"].iloc[0]
            y_value = country_current["Total_Citations"].iloc[0]
            bubble_size = country_current["Bubble_Size"].iloc[0]

            ax.scatter(
                [x_value],
                [y_value],
                s=[bubble_size],
                color=color_map.get(country, "#4c78a8"),
                alpha=0.88,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
                label=country,
            )

        ax.set_title(
            "Animated Bubble Plot of Total Citations by Country",
            fontsize=21,
            fontweight="bold",
            color=text_color,
            loc="left",
            pad=24,
        )
        ax.text(
            0,
            1.02,
            "x = year, y = total citations, bubble size = number of papers",
            transform=ax.transAxes,
            fontsize=11,
            color=muted_color,
            ha="left",
        )
        ax.set_xlabel("Publication Year", fontsize=13, color=text_color, labelpad=12)
        ax.set_ylabel("Total Citations", fontsize=13, color=text_color, labelpad=12)
        ax.set_xlim(MIN_COUNTRY_TREND_YEAR - 0.4, MAX_YEAR + 0.6)
        ax.set_ylim(0, y_max)
        ax.set_xticks(list(range(MIN_COUNTRY_TREND_YEAR, MAX_YEAR + 1)))
        ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.tick_params(axis="x", colors=muted_color, rotation=45)
        ax.tick_params(axis="y", colors=muted_color)

        handles, labels = ax.get_legend_handles_labels()
        dedup_handles = []
        dedup_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)
        ax.legend(dedup_handles, dedup_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10.5)

    fig.subplots_adjust(left=0.11, right=0.83, top=0.88, bottom=0.16)
    animation = FuncAnimation(fig, draw_frame, frames=years, interval=1200, repeat=True)
    output_path = os.path.join(FIGURE_OUTPUT_DIR, "Country_Total_Citations_Animation_2018_2025.gif")
    animation.save(output_path, writer=PillowWriter(fps=1), dpi=120)
    print(f"Saved chart: {output_path}")
    plt.close(fig)


def create_country_paper_count_animation(summary_frames):
    metrics_df = summary_frames["country_year_metrics"].copy()
    if metrics_df.empty:
        print("Skipped country paper count animation because no data was available.")
        return

    metrics_df["Year"] = pd.to_numeric(metrics_df["Year"], errors="coerce")
    metrics_df["Paper_Count"] = pd.to_numeric(metrics_df["Paper_Count"], errors="coerce")
    metrics_df["Total_Citations"] = pd.to_numeric(metrics_df["Total_Citations"], errors="coerce")
    metrics_df.dropna(subset=["Year", "Paper_Count", "Total_Citations"], inplace=True)

    top_countries = summary_frames["country_counts"]["Country"].head(TOP_COUNTRY_TREND_COUNT).tolist()
    metrics_df = metrics_df[metrics_df["Country"].isin(top_countries)].copy()
    metrics_df = metrics_df[metrics_df["Year"] >= MIN_COUNTRY_TREND_YEAR]
    metrics_df["Year"] = metrics_df["Year"].astype(int)

    if metrics_df.empty:
        print("Skipped country paper count animation because yearly paper count data was unavailable.")
        return

    years = sorted(metrics_df["Year"].unique())
    background_color = "#f7fbff"
    text_color = "#16324f"
    muted_color = "#5c6f82"
    grid_color = "#b8cadc"
    color_map = {
        "China": "#234f9b",
        "United States": "#d1495b",
        "India": "#2a9d8f",
        "South Korea": "#7a52a5",
        "United Kingdom": "#e08a28",
    }

    y_max = max(metrics_df["Paper_Count"].max() * 1.08, 10)
    bubble_sizes = 55 + np.sqrt(metrics_df["Total_Citations"].clip(lower=0)) * 4.2
    metrics_df["Bubble_Size"] = bubble_sizes

    fig, ax = plt.subplots(figsize=(14.8, 8.8))
    fig.patch.set_facecolor(background_color)

    def draw_frame(frame_year):
        ax.clear()
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="both", linestyle="--", linewidth=1.0, alpha=0.75, color=grid_color)
        ax.set_axisbelow(True)

        cumulative_df = metrics_df[metrics_df["Year"] <= frame_year].copy()
        current_df = metrics_df[metrics_df["Year"] == frame_year].copy()

        for country in top_countries:
            country_cumulative = cumulative_df[cumulative_df["Country"] == country]
            if country_cumulative.empty:
                continue

            ax.scatter(
                country_cumulative["Year"],
                country_cumulative["Paper_Count"],
                s=country_cumulative["Bubble_Size"],
                color=color_map.get(country, "#4c78a8"),
                alpha=0.24,
                edgecolors="none",
                zorder=2,
            )

        for country in top_countries:
            country_current = current_df[current_df["Country"] == country]
            if country_current.empty:
                continue

            x_value = country_current["Year"].iloc[0]
            y_value = country_current["Paper_Count"].iloc[0]
            bubble_size = country_current["Bubble_Size"].iloc[0]

            ax.scatter(
                [x_value],
                [y_value],
                s=[bubble_size],
                color=color_map.get(country, "#4c78a8"),
                alpha=0.88,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
                label=country,
            )

        ax.set_title(
            "Animated Bubble Plot of Total Paper Counts by Country",
            fontsize=21,
            fontweight="bold",
            color=text_color,
            loc="left",
            pad=24,
        )
        ax.text(
            0,
            1.02,
            "x = year, y = total paper count, bubble size = total citations",
            transform=ax.transAxes,
            fontsize=11,
            color=muted_color,
            ha="left",
        )
        ax.set_xlabel("Publication Year", fontsize=13, color=text_color, labelpad=12)
        ax.set_ylabel("Paper Count", fontsize=13, color=text_color, labelpad=12)
        ax.set_xlim(MIN_COUNTRY_TREND_YEAR - 0.4, MAX_YEAR + 0.6)
        ax.set_ylim(0, y_max)
        ax.set_xticks(list(range(MIN_COUNTRY_TREND_YEAR, MAX_YEAR + 1)))
        ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.tick_params(axis="x", colors=muted_color, rotation=45)
        ax.tick_params(axis="y", colors=muted_color)

        handles, labels = ax.get_legend_handles_labels()
        dedup_handles = []
        dedup_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)
        ax.legend(dedup_handles, dedup_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10.5)

    fig.subplots_adjust(left=0.11, right=0.83, top=0.88, bottom=0.16)
    animation = FuncAnimation(fig, draw_frame, frames=years, interval=1200, repeat=True)
    output_path = os.path.join(FIGURE_OUTPUT_DIR, "Country_Total_Paper_Count_Animation_2018_2025.gif")
    animation.save(output_path, writer=PillowWriter(fps=1), dpi=120)
    print(f"Saved chart: {output_path}")
    plt.close(fig)


# =============================================================================
# 국가별 연도 버블 차트 및 애니메이션
# =============================================================================
def prepare_top10_country_year_bubble_data(summary_frames):
    metrics_df = summary_frames["country_year_metrics"].copy()
    if metrics_df.empty:
        return metrics_df, []

    for column in ["Year", "Paper_Count", "Total_Citations", "Average_Citations"]:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")

    metrics_df.dropna(subset=["Year", "Country", "Paper_Count", "Total_Citations"], inplace=True)
    metrics_df = metrics_df[metrics_df["Year"] >= MIN_COUNTRY_BUBBLE_YEAR].copy()
    metrics_df["Year"] = metrics_df["Year"].astype(int)
    metrics_df["Average_Citations"] = metrics_df["Average_Citations"].fillna(
        metrics_df["Total_Citations"] / metrics_df["Paper_Count"].replace(0, np.nan)
    ).fillna(0)

    top_countries = (
        metrics_df.groupby("Country")["Paper_Count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_COUNTRY_BUBBLE_COUNT)
        .index.tolist()
    )

    yearly_totals = metrics_df.groupby("Year")["Paper_Count"].transform("sum")
    metrics_df["Year_Percent"] = np.where(
        yearly_totals > 0,
        metrics_df["Paper_Count"] / yearly_totals * 100,
        0,
    )
    metrics_df.sort_values(["Country", "Year"], inplace=True)
    metrics_df["Cumulative_Total_Citations"] = metrics_df.groupby("Country")["Total_Citations"].cumsum()
    if "country_all_metrics" in summary_frames:
        citation_totals = summary_frames["country_all_metrics"].copy()
        citation_totals["Total_Citations"] = pd.to_numeric(
            citation_totals["Total_Citations"],
            errors="coerce",
        )
        target_total_map = citation_totals.set_index("Country")["Total_Citations"].dropna().to_dict()
        current_total_map = metrics_df.groupby("Country")["Cumulative_Total_Citations"].max().to_dict()
        citation_scale_map = {
            country: target_total / current_total
            for country, target_total in target_total_map.items()
            for current_total in [current_total_map.get(country, 0)]
            if current_total > 0
        }
        metrics_df["Cumulative_Total_Citations"] = metrics_df.apply(
            lambda row: row["Cumulative_Total_Citations"] * citation_scale_map.get(row["Country"], 1),
            axis=1,
        )
    metrics_df = metrics_df[metrics_df["Country"].isin(top_countries)].copy()
    metrics_df.sort_values(["Year", "Paper_Count"], ascending=[True, False], inplace=True)
    return metrics_df, top_countries


def draw_bubble_size_legend(ax, size_values, size_function, title, suffix, text_color):
    ax.text(1.03, 0.76, title, transform=ax.transAxes, fontsize=12, fontweight="bold", color=text_color)
    y_position = 0.68
    for value in size_values:
        ax.scatter(
            [1.055],
            [y_position],
            s=[size_function(value)],
            transform=ax.transAxes,
            facecolors="none",
            edgecolors=text_color,
            linewidths=0.8,
            clip_on=False,
        )
        ax.text(
            1.115,
            y_position,
            f"{value:,.0f} {suffix}",
            transform=ax.transAxes,
            fontsize=9,
            color=text_color,
            va="center",
        )
        y_position -= 0.085


def create_top10_country_year_percent_bubble_animation(summary_frames):
    metrics_df, top_countries = prepare_top10_country_year_bubble_data(summary_frames)
    if metrics_df.empty:
        print("Skipped top 10 country percent bubble animation because no data was available.")
        return

    output_dir = FIGURE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(
        os.path.join(output_dir, "country_year_percent_bubble_top10_2017_2025.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    years = sorted(metrics_df["Year"].unique())
    background_color = "white"
    text_color = "black"
    muted_color = "#5c6f82"
    grid_color = "#d8e4ef"
    palette = sns.color_palette("tab10", n_colors=len(top_countries)).as_hex()
    color_map = dict(zip(top_countries, palette))
    size_function = lambda value: 45 + max(value, 0) * 0.035

    x_max = max(metrics_df["Year_Percent"].max() * 1.12, 10)
    y_max = max(metrics_df["Paper_Count"].max() * 1.12, 10)
    metrics_df["Bubble_Size"] = metrics_df["Cumulative_Total_Citations"].apply(size_function)

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(background_color)

    def draw_frame(frame_year):
        ax.clear()
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
            spine.set_linewidth(0.9)
        ax.grid(axis="both", linewidth=0.7, alpha=0.8, color=grid_color)
        ax.set_axisbelow(True)

        current_df = metrics_df[metrics_df["Year"] == frame_year].copy()
        for country in top_countries:
            country_row = current_df[current_df["Country"] == country]
            if country_row.empty:
                continue
            ax.scatter(
                country_row["Year_Percent"],
                country_row["Paper_Count"],
                s=country_row["Bubble_Size"],
                color=color_map[country],
                alpha=0.5,
                edgecolors=text_color,
                linewidths=0.7,
                label=country,
            )
            ax.text(
                country_row["Year_Percent"].iloc[0] + x_max * 0.012,
                country_row["Paper_Count"].iloc[0],
                country,
                fontsize=8.5,
                color=text_color,
                va="center",
            )

        ax.text(
            0.96,
            0.93,
            str(frame_year),
            transform=ax.transAxes,
            fontsize=36,
            fontweight="bold",
            color="#d8e1ea",
            ha="right",
        )
        ax.set_title(
            f"Top 10 Country Bubble Plot ({frame_year})",
            fontsize=40,
            fontweight="bold",
            color=text_color,
            loc="left",
            pad=18,
        )
        ax.text(
            0,
            1.02,
            "x = yearly country share of papers, y = yearly paper count, bubble size = cumulative total citations",
            transform=ax.transAxes,
            fontsize=10.5,
            color=muted_color,
            ha="left",
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0f}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.set_xlabel("Country share of papers within each year (%)", fontsize=12, color=text_color, labelpad=12)
        ax.set_ylabel("Yearly paper count", fontsize=12, color=text_color, labelpad=12)
        ax.tick_params(axis="both", colors=text_color, labelsize=9.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Top 10 countries", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.03), fontsize=9, title_fontsize=10)
        legend_values = [1000, 10000, 30000, 55000]
        draw_bubble_size_legend(ax, legend_values, size_function, "Bubble size", "citations", text_color)

    fig.subplots_adjust(left=0.08, right=0.78, top=0.86, bottom=0.13)
    animation = FuncAnimation(fig, draw_frame, frames=years, interval=1200, repeat=True)
    output_path = os.path.join(output_dir, "country_year_percent_bubble_top10_cumulative_total_citations_2017_2025.gif")
    animation.save(output_path, writer=PillowWriter(fps=1), dpi=110)
    print(f"Saved chart: {output_path}")
    plt.close(fig)


def create_top10_country_year_timeline_bubble_animation(summary_frames):
    metrics_df, top_countries = prepare_top10_country_year_bubble_data(summary_frames)
    if metrics_df.empty:
        print("Skipped top 10 country year-axis bubble animation because no data was available.")
        return

    output_dir = FIGURE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(
        os.path.join(output_dir, "country_year_timeline_bubble_top10_2017_2025.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    years = sorted(metrics_df["Year"].unique())
    background_color = "white"
    text_color = "black"
    grid_color = "#d8e4ef"
    palette = sns.color_palette("tab10", n_colors=len(top_countries)).as_hex()
    color_map = dict(zip(top_countries, palette))
    min_bubble_size = 180
    annual_total_citation_size_scale = 0.55

    def size_function(value, scale):
        return min_bubble_size + max(float(value), 0) * scale

    y_max = max(metrics_df["Paper_Count"].max() * 1.12, 10)
    from matplotlib.lines import Line2D

    def save_timeline_animation(size_column, size_scale, output_filename):
        plot_df = metrics_df.copy()
        plot_df["Bubble_Size"] = plot_df[size_column].fillna(0).apply(lambda value: size_function(value, size_scale))

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor(background_color)

        def draw_frame(frame_year):
            ax.clear()
            ax.set_facecolor(background_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
                spine.set_linewidth(0.9)
            ax.grid(axis="both", linewidth=0.7, alpha=0.8, color=grid_color)
            ax.set_axisbelow(True)

            shown_df = plot_df[plot_df["Year"] <= frame_year].copy()
            current_df = plot_df[plot_df["Year"] == frame_year].copy()

            for country in top_countries:
                country_history = shown_df[shown_df["Country"] == country]
                if country_history.empty:
                    continue
                ax.scatter(
                    country_history["Year"],
                    country_history["Paper_Count"],
                    s=country_history["Bubble_Size"],
                    color=color_map[country],
                    alpha=0.25,
                    edgecolors="none",
                )

                country_current = current_df[current_df["Country"] == country]
                if country_current.empty:
                    continue
                ax.scatter(
                    country_current["Year"],
                    country_current["Paper_Count"],
                    s=country_current["Bubble_Size"],
                    color=color_map[country],
                    alpha=0.5,
                    edgecolors=text_color,
                    linewidths=0.7,
                    label=country,
                )
            ax.text(
                0.96,
                0.93,
                str(frame_year),
                transform=ax.transAxes,
                fontsize=36,
                fontweight="bold",
                color="#d8e1ea",
                ha="right",
            )
            ax.set_title(
                f"Top 10 Country Bubble Plot by Year ({frame_year})",
                fontsize=20,
                fontweight="bold",
                color=text_color,
                loc="left",
                pad=18,
            )
            ax.set_xlim(MIN_COUNTRY_BUBBLE_YEAR - 0.4, MAX_YEAR + 0.7)
            ax.set_ylim(0, y_max)
            ax.set_xticks(list(range(MIN_COUNTRY_BUBBLE_YEAR, MAX_YEAR + 1)))
            ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
            ax.set_xlabel("Publication year", fontsize=20, color=text_color, labelpad=12)
            ax.set_ylabel("Yearly paper count", fontsize=20, color=text_color, labelpad=12)
            ax.tick_params(axis="both", colors=text_color, labelsize=20)

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=14,
                    markerfacecolor=color_map[country],
                    markeredgecolor=text_color,
                    markeredgewidth=0.5,
                    label=country,
                )
                for country in top_countries
            ]
            ax.legend(
                handles=legend_handles,
                title="Top 10 countries",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.03),
                fontsize=20,
                title_fontsize=20,
            )

        fig.subplots_adjust(left=0.08, right=0.72, top=0.86, bottom=0.13)
        animation = FuncAnimation(fig, draw_frame, frames=years, interval=1200, repeat=True)
        output_path = os.path.join(output_dir, output_filename)
        animation.save(output_path, writer=PillowWriter(fps=1), dpi=110)
        print(f"Saved chart: {output_path}")
        plt.close(fig)

    def save_timeline_line_plot(value_column, output_filename, title, y_label, use_log_y=False):
        plot_df = metrics_df.copy()
        plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce").fillna(0)

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        for spine in ax.spines.values():
            spine.set_color(text_color)
            spine.set_linewidth(0.9)

        ax.grid(axis="both", linewidth=0.7, alpha=0.8, color=grid_color)
        ax.set_axisbelow(True)

        for country in top_countries:
            country_df = plot_df[plot_df["Country"] == country].sort_values("Year")
            if country_df.empty:
                continue
            ax.plot(
                country_df["Year"],
                country_df[value_column],
                marker="o",
                linewidth=2.8,
                markersize=7.5,
                color=color_map[country],
                label=country,
            )

        ax.set_title(title, fontsize=24, fontweight="bold", color=text_color, loc="left", pad=18)
        ax.set_xlim(LINE_PLOT_START_YEAR - 0.4, MAX_YEAR + 0.7)
        ax.set_xticks(list(range(LINE_PLOT_START_YEAR, MAX_YEAR + 1)))
        ax.set_xlabel("Publication year", fontsize=20, color=text_color, labelpad=12)
        ax.set_ylabel(y_label, fontsize=20, color=text_color, labelpad=12)
        ax.tick_params(axis="both", colors=text_color, labelsize=18)

        if use_log_y:
            positive_values = plot_df.loc[plot_df[value_column] > 0, value_column]
            if not positive_values.empty:
                ax.set_yscale("log")
                ax.set_ylim(max(positive_values.min() * 0.8, 0.1), positive_values.max() * 1.25)
        else:
            ax.set_ylim(0, max(plot_df[value_column].max() * 1.15, 1))

        if value_column == "Average_Citations":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.1f}"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))

        ax.legend(
            title="Top 10 countries",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.03),
            fontsize=15,
            title_fontsize=16,
        )

        fig.subplots_adjust(left=0.08, right=0.78, top=0.86, bottom=0.13)
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved chart: {output_path}")
        plt.close(fig)

    save_timeline_animation(
        "Total_Citations",
        annual_total_citation_size_scale,
        "country_year_timeline_bubble_top10_cumulative_total_citations_2017_2025.gif",
    )
    save_timeline_line_plot(
        "Total_Citations",
        "country_year_timeline_line_top10_annual_total_citations_2017_2025.png",
        "Top 10 Countries by Annual Total Citations",
        "Annual total citations",
    )
    save_timeline_line_plot(
        "Average_Citations",
        "country_year_timeline_line_top10_average_citations_2017_2025.png",
        "Top 10 Countries by Average Citations per Paper",
        "Average citations per paper",
    )


# =============================================================================
# 논문·저널 순위표 출력
# =============================================================================
def create_most_cited_papers_table_outputs(summary_frames):
    papers_df = summary_frames["most_cited_papers"].copy()
    if papers_df.empty:
        print("Skipped most cited papers table because no data was available.")
        return

    papers_df.rename(
        columns={
            "title": "Title",
            "authors": "Authors",
            "journal": "Journal",
            "year": "Year",
            "citations": "Citation",
        },
        inplace=True,
    )

    for column in ["Year", "Citation"]:
        papers_df[column] = pd.to_numeric(papers_df[column], errors="coerce").fillna(0).astype(int)

    csv_output_path = os.path.join(SCRIPT_DIR, "most_cited_papers_top10.csv")
    papers_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {csv_output_path}")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Skipped most cited papers table image because Pillow is not installed.")
        return

    background_color = "#ffffff"
    text_color = "black"
    header_color = "#eeeeee"
    line_color = "#000000"

    image_width = 8400
    image_height = 4620
    margin_x = 28
    title_y = 20
    table_top = 300
    table_bottom = image_height - 40
    header_height = 390
    row_height = (table_bottom - table_top - header_height) / len(papers_df)
    table_left = margin_x
    table_right = image_width - margin_x
    table_width = table_right - table_left
    col_width_fracs = [0.45, 0.15, 0.26, 0.06, 0.08]
    col_edges = [table_left]
    for frac in col_width_fracs:
        col_edges.append(col_edges[-1] + table_width * frac)
    col_edges[-1] = table_right

    font_dir = "C:/Windows/Fonts"

    def load_font(font_name, size):
        font_path = os.path.join(font_dir, font_name)
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
        return ImageFont.load_default()

    title_font = load_font("arialbd.ttf", 150)
    header_font = load_font("arialbd.ttf", 120)
    title_cell_font = load_font("arial.ttf", 120)
    body_font = load_font("arial.ttf", 105)
    journal_font = load_font("arial.ttf", 103)
    numeric_font = load_font("arial.ttf", 105)

    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    def text_width(text, font):
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return bbox[2] - bbox[0]

    def text_height(text, font):
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return bbox[3] - bbox[1]

    def wrap_text_by_width(text, font, max_width):
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(candidate, font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def draw_wrapped_text(text, box, font, align="left", line_spacing=1.05):
        x0, y0, x1, y1 = box
        max_width = x1 - x0
        lines = wrap_text_by_width(text, font, max_width)
        line_height = max(text_height("Ag", font) * line_spacing, 1)
        total_height = line_height * len(lines)
        y = y0 + max((y1 - y0 - total_height) / 2, 0)
        for line in lines:
            width = text_width(line, font)
            if align == "center":
                x = x0 + (max_width - width) / 2
            else:
                x = x0
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_height

    draw.text((margin_x, title_y), "Top 10 Most Cited Papers", fill=text_color, font=title_font)
    draw.rectangle([table_left, table_top, table_right, table_bottom], fill=background_color, outline=line_color, width=4)
    draw.rectangle([table_left, table_top, table_right, table_top + header_height], fill=header_color, outline=line_color, width=4)

    row_edges = [table_top, table_top + header_height]
    for index in range(len(papers_df)):
        row_edges.append(table_top + header_height + row_height * (index + 1))
    row_edges[-1] = table_bottom

    for x in col_edges:
        draw.line([(x, table_top), (x, table_bottom)], fill=line_color, width=4)
    for y in row_edges:
        draw.line([(table_left, y), (table_right, y)], fill=line_color, width=4)

    headers = list(papers_df.columns)
    for col, header in enumerate(headers):
        draw_wrapped_text(
            header,
            (col_edges[col] + 45, table_top + 20, col_edges[col + 1] - 45, table_top + header_height - 20),
            header_font,
            align="center",
        )

    for row_index, (_, row) in enumerate(papers_df.iterrows(), start=1):
        y0 = row_edges[row_index]
        y1 = row_edges[row_index + 1]
        values = [
            str(row["Title"]),
            str(row["Authors"]),
            str(row["Journal"]),
            f"{int(row['Year'])}",
            f"{int(row['Citation']):,}",
        ]
        fonts = [title_cell_font, body_font, journal_font, numeric_font, numeric_font]
        aligns = ["left", "left", "left", "center", "center"]
        pads = [360, 120, 220, 40, 40]
        for col, value in enumerate(values):
            draw_wrapped_text(
                value,
                (col_edges[col] + pads[col], y0 + 20, col_edges[col + 1] - pads[col], y1 - 20),
                fonts[col],
                align=aligns[col],
            )

    png_output_path = os.path.join(FIGURE_OUTPUT_DIR, "most_cited_papers_top10.png")
    image.save(png_output_path)
    print(f"Saved chart: {png_output_path}")


def load_journal_metrics_lookup():
    lookup_path_candidates = [
        os.path.join(SCRIPT_DIR, JOURNAL_METRICS_LOOKUP_FILE),
        os.path.join("C:/Text_Mining/output", JOURNAL_METRICS_LOOKUP_FILE),
    ]
    lookup_path = next((path for path in lookup_path_candidates if os.path.exists(path)), None)
    if lookup_path is None:
        return pd.DataFrame(columns=["Journal", "IF", "CS"])

    metrics_df = pd.read_csv(lookup_path)
    metrics_df = metrics_df.rename(columns=lambda column: str(column).strip())
    required_columns = {"Journal", "IF", "CS"}
    if not required_columns.issubset(metrics_df.columns):
        print(f"Skipped journal metrics lookup because required columns are missing: {lookup_path}")
        return pd.DataFrame(columns=["Journal", "IF", "CS"])

    metrics_df["Journal"] = (
        metrics_df["Journal"]
        .astype(str)
        .apply(format_journal_title)
        .str.strip()
    )
    return metrics_df[["Journal", "IF", "CS"]].copy()


def create_top_journals_table_outputs(summary_frames):
    journals_df = summary_frames["journal_counts"].copy()
    if journals_df.empty:
        print("Skipped top journals table because no data was available.")
        return

    if not {"Journal", "Paper_Count", "Total_Citations", "Average_Citations"}.issubset(journals_df.columns):
        journal_counts_df = summary_frames["top_journals_by_count"].copy()
        journal_citations_df = summary_frames["journal_citations"].copy()
        journal_average_df = summary_frames["journal_average_citations"].copy()

        journals_df = journal_counts_df.merge(
            journal_citations_df[["Journal", "Total_Citations"]],
            on="Journal",
            how="left",
        ).merge(
            journal_average_df[["Journal", "Average_Citations"]],
            on="Journal",
            how="left",
        )

    journals_df = journals_df[["Journal", "Paper_Count", "Total_Citations", "Average_Citations"]].copy()
    journals_df["Journal"] = (
        journals_df["Journal"]
        .astype(str)
        .apply(format_journal_title)
        .str.strip()
    )
    journals_df["Paper_Count"] = pd.to_numeric(journals_df["Paper_Count"], errors="coerce").fillna(0).astype(int)
    journals_df["Total_Citations"] = pd.to_numeric(journals_df["Total_Citations"], errors="coerce").fillna(0).astype(int)
    journals_df["Average_Citations"] = pd.to_numeric(journals_df["Average_Citations"], errors="coerce").fillna(0.0)

    metrics_df = load_journal_metrics_lookup()
    journals_df = journals_df.merge(metrics_df, on="Journal", how="left")
    journals_df["IF"] = journals_df["IF"].fillna("N/A")
    journals_df["CS"] = journals_df["CS"].fillna("N/A")
    journals_df["Journal_Base"] = journals_df["Journal"]

    if "Publisher" not in journals_df.columns:
        journals_df["Publisher"] = ""
    publisher_lookup = load_journal_publisher_lookup_from_ris()
    journals_df["Publisher"] = (
        journals_df.apply(
            lambda row: (
                row["Publisher"]
                if str(row["Publisher"]).strip()
                and str(row["Publisher"]).strip().lower() != "nan"
                else publisher_lookup.get(row["Journal_Base"], "")
            ),
            axis=1,
        )
        .fillna("")
        .apply(format_publisher_name)
    )
    journals_df["Journal"] = journals_df.apply(
        lambda row: (
            f"{row['Journal_Base']} ({row['Publisher']})"
            if row["Publisher"]
            else row["Journal_Base"]
        ),
        axis=1,
    )
    journals_df.drop(columns=["Publisher", "Journal_Base"], inplace=True)

    csv_df = journals_df.rename(
        columns={
            "Paper_Count": "Paper count",
            "Total_Citations": "Total citations",
            "Average_Citations": "Average citations",
        }
    ).copy()
    csv_output_path = os.path.join(SCRIPT_DIR, "top_journals_table_top10.csv")
    csv_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {csv_output_path}")

    display_df = csv_df.copy()
    display_df["Journal"] = display_df["Journal"].astype(str).apply(lambda value: textwrap.fill(value, width=46))
    display_df["Paper count"] = display_df["Paper count"].map(lambda value: f"{int(value):,}")
    display_df["Total citations"] = display_df["Total citations"].map(lambda value: f"{int(value):,}")
    display_df["Average citations"] = display_df["Average citations"].map(lambda value: f"{value:.1f}")
    display_df["IF"] = display_df["IF"].map(lambda value: f"{float(value):.1f}" if str(value) != "N/A" else "N/A")
    display_df["CS"] = display_df["CS"].map(lambda value: f"{float(value):.1f}" if str(value) != "N/A" else "N/A")

    fig_height = max(13.5, 1.8 + len(display_df) * 1.15)
    fig, ax = plt.subplots(figsize=(26, fig_height))
    background_color = "#ffffff"
    text_color = "black"
    header_color = "#eeeeee"
    line_color = "#000000"

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.axis("off")

    ax.text(
        0.0,
        1.015,
        "Top 10 Journals by Paper Count",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=40,
        fontweight="bold",
        color=text_color,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.44, 0.13, 0.14, 0.17, 0.06, 0.06],
        bbox=[0, 0, 1, 0.985],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1, 3.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(line_color)
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=text_color, ha="center", va="center", fontsize=27)
        else:
            cell.set_facecolor("#ffffff")
            if col == 0:
                cell.set_text_props(color=text_color, ha="left", va="center", fontsize=22)
            else:
                cell.set_text_props(color=text_color, ha="center", va="center", fontsize=22)

    png_output_path = os.path.join(FIGURE_OUTPUT_DIR, "8.top_journals_table_top10.png")
    fig.savefig(png_output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {png_output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# 전체 차트 생성 및 메인 실행 함수
# =============================================================================
def create_country_charts(summary_frames):
    for chart_config in CHART_CONFIGS:
        create_ranked_bar_chart(
            dataframe=summary_frames[chart_config["data_key"]],
            label_column=chart_config["label_column"],
            value_column=chart_config["value_column"],
            title=chart_config["title"],
            subtitle=chart_config["subtitle"],
            xlabel=chart_config["xlabel"],
            palette=chart_config["palette"],
            filename=chart_config["filename"],
        )

    create_country_combined_sankey(summary_frames)
    create_country_trend_line_chart(summary_frames)
    create_country_trend_heatmap(summary_frames)
    create_country_citation_animation(summary_frames)
    create_country_paper_count_animation(summary_frames)
    create_top10_country_year_percent_bubble_animation(summary_frames)
    create_top10_country_year_timeline_bubble_animation(summary_frames)
    create_most_cited_papers_table_outputs(summary_frames)
    create_top_journals_table_outputs(summary_frames)


def main():
    should_load_from_csv = USE_EXISTING_CSV or rispy is None

    if should_load_from_csv:
        if rispy is None:
            print("rispy is not installed. Reusing existing CSV summaries to rebuild the charts.")
        else:
            print("Using existing CSV summaries because --use-existing-csv was provided.")
        summary_frames = load_summary_frames()
    else:
        df_filtered = build_dataframe_from_ris()
        summary_frames = build_summary_frames(df_filtered)
        save_summary_frames(summary_frames)

    print_summary_tables(summary_frames)
    create_country_charts(summary_frames)
    print("done !!")


if __name__ == "__main__":
    main()
