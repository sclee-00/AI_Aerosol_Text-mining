import glob
import math
import os
import re
import sys
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

# =============================================================================
# 실행 환경 및 시각화 기본 설정
# =============================================================================
RUNNING_IN_SPYDER = any(
    key.startswith("SPYDER") or key.startswith("SPY_") for key in os.environ
)
SHOW_PLOTS = "--show" in sys.argv or RUNNING_IN_SPYDER

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")


# =============================================================================
# 입력·출력 경로 및 분석 설정
# =============================================================================
RIS_FOLDER = "E:/Text_Mining/Data/WoS"
RIS_FILENAMES = [f"savedrecs_{file_number}_cleaned.ris" for file_number in range(1, 6)]
OUTPUT_DIR = "E:/Text_Mining/output"
FIGURE_OUTPUT_DIR = "E:/Text_Mining/Fig/WoS/Ver.7"
MAX_YEAR = 2025

FIGURE_BG = "#ffffff"
PANEL_BG = "#ffffff"
TEXT_COLOR = "#000000"
MUTED_TEXT = "#000000"
GRID_COLOR = "#bed0e0"
BAR_PRIMARY = "#36558f"
BAR_SECONDARY = "#4aa3a2"
BAR_ACCENT = "#cf7f2f"
LINE_COLOR = "#d1495b"
HIGHLIGHT_BOX = "#ffffff"
TOP_COUNTRIES = ["China", "United States", "India", "South Korea", "Italy"]
COUNTRY_FILENAME_MAP = {
    "China": "china",
    "United States": "united_states",
    "India": "india",
    "South Korea": "south_korea",
    "Italy": "italy",
}


# =============================================================================
# 출력 및 시각화 공통 유틸리티
# =============================================================================
def safe_preview(text, limit=220):
    if text is None:
        return ""
    cleaned = str(text).replace("\ufeff", "").replace("\x00", "").strip()
    cleaned = cleaned.encode("cp949", errors="replace").decode("cp949", errors="replace")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def format_metric(value, decimals=0):
    if pd.isna(value):
        return ""
    if decimals == 0 or float(value).is_integer():
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def axis_formatter(value, _position):
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def style_axis(ax, format_x=True, format_y=True):
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="both", visible=False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED_TEXT)
    if format_x:
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
    if format_y:
        ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))


def save_figure(fig, output_path):
    ver5_filename_map = {
        "yearly_count_wos.png": "10.yearly_count_wos.png",
        "top_authors_table_wos.png": "top_authors_table_wos.png",
        "top_authors_wos.png": "9.top_authors_wos.png",
        "top_affiliations_wos.png": "11.top_affiliations_wos.png",
        "top_10_institutions_united_states_wos.png": "12.top_10_institutions_united_states_wos.png",
        "top_10_institutions_china_wos.png": "12.top_10_institutions_china_wos.png",
        "top_10_institutions_india_wos.png": "13.top_10_institutions_india_wos.png",
        "top_10_institutions_south_korea_wos.png": "14.top_10_institutions_south_korea_wos.png",
        "top_10_institutions_italy_wos.png": "15.top_10_institutions_italy_wos.png",
    }
    output_name = os.path.basename(output_path)
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    ver5_output_path = os.path.join(FIGURE_OUTPUT_DIR, ver5_filename_map.get(output_name, output_name))
    bbox_inches = None if getattr(fig, "_fixed_panel_layout", False) else "tight"
    fig.savefig(ver5_output_path, dpi=300, bbox_inches=bbox_inches, facecolor=fig.get_facecolor())
    print(f"Saved chart: {ver5_output_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def ranked_bar_colors(base_color, count):
    if count <= 1:
        return [base_color]
    return sns.blend_palette(
        [sns.light_palette(base_color, n_colors=6)[2], base_color],
        n_colors=count,
    )


# =============================================================================
# RIS 데이터 추출
# =============================================================================
def extract_data_from_multiple_ris(files):
    authors = []
    affiliations = []
    years = []
    unknown_records = []
    file_tag_summary = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as file:
                lines = file.readlines()

        print(file_path)

        c1_count = 0
        c2_count = 0
        ad_count = 0
        er_count = 0

        current_year = None
        current_authors = []
        current_affiliations = []
        raw_record_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            raw_record_lines.append(raw_line.rstrip("\n"))

            if line.startswith("AU  -"):
                author = line.replace("AU  -", "").strip()
                if author:
                    current_authors.append(author)

            if line.startswith("C1  -") or line.startswith("C2  -") or line.startswith("AD  -"):
                if line.startswith("C1  -"):
                    c1_count += 1
                elif line.startswith("C2  -"):
                    c2_count += 1
                else:
                    ad_count += 1

                aff_line = line.split("  -", 1)[-1].strip()
                if aff_line:
                    current_affiliations.append(aff_line)

            if line.startswith("PY  -") or line.startswith("Y1  -"):
                match = re.search(r"\d{4}", line)
                if match:
                    current_year = int(match.group())

            if line.startswith("ER  -"):
                er_count += 1
                if current_year and current_year <= MAX_YEAR:
                    years.append(str(current_year))
                    authors.append(current_authors if current_authors else ["Unknown"])

                    if current_affiliations:
                        affiliations.append(current_affiliations)
                    else:
                        affiliations.append(["Unknown"])
                        unknown_records.append(
                            {
                                "file": os.path.basename(file_path),
                                "year": current_year,
                                "authors": current_authors if current_authors else ["Unknown"],
                                "raw_record": "\n".join(raw_record_lines),
                            }
                        )

                current_year = None
                current_authors = []
                current_affiliations = []
                raw_record_lines = []

        file_tag_summary.append(
            {
                "file": os.path.basename(file_path),
                "C1_count": c1_count,
                "C2_count": c2_count,
                "AD_count": ad_count,
                "ER_count": er_count,
            }
        )

    return years, authors, affiliations, unknown_records, file_tag_summary


# =============================================================================
# 소속기관명 정규화 및 국가 판별
# =============================================================================
def normalize_wos_affiliation(name):
    cleaned_name = re.sub(r"\[.*?\]", "", str(name)).strip()
    cleaned_name = cleaned_name.split(";")[0].strip()
    parts = [part.strip() for part in cleaned_name.split(",") if part.strip()]
    if parts:
        institution = parts[0]
        if institution.lower() in {
            "dept",
            "department",
            "school",
            "college",
            "faculty",
            "lab",
            "laboratory",
            "state key laboratory",
            "key laboratory",
            "research center",
            "center",
            "centre",
            "institute",
        } and len(parts) > 1:
            institution = parts[1]
    else:
        institution = cleaned_name

    name = institution.lower()

    if "chin" in name and "acad" in name and "sci" in name:
        return "CAS"
    if "tsinghua" in name:
        return "Tsinghua Univ"
    if "peking" in name:
        return "Peking Univ"
    if "anhui univ" in name:
        return "Anhui Univ."
    if "china univ geosci" in name:
        return "China Univ. Geosci."
    if "china univ min" in name:
        return "China Univ. Min."
    if "china univ petr" in name:
        return "China Univ. Petr."
    if "chongqing univ" in name:
        return "Chongqing Univ."
    if "guangdong univ" in name:
        return "Guangdong Univ."
    if "guizhou univ" in name:
        return "Guizhou Univ."
    if "harbin univ" in name:
        return "Harbin Univ."
    if "hebei univ" in name:
        return "Hebei Univ."
    if "henan univ" in name:
        return "Henan Univ."
    if "hubei univ" in name:
        return "Hubei Univ."
    if "hunan univ" in name:
        return "Hunan Univ."
    if "jiangsu univ" in name:
        return "Jiangsu Univ."
    if "jiangxi univ" in name:
        return "Jiangxi Univ."
    if "jilin univ" in name:
        return "Jilin Univ."
    if "lanzhou univ" in name:
        return "Lanzhou Univ."
    if "shandong univ" in name:
        return "Shandong Univ."
    if "northeastern univ" in name:
        return "Northeastern Univ."
    if "indian inst technol" in name:
        return "Indian Inst Technol"
    if "noaa" in name:
        return "NOAA"
    if name.strip() == "mit" or "massachusetts inst" in name:
        return "MIT"
    if "oxford" in name:
        return "Univ. of Oxford"
    if "csiro" in name:
        return "CSIRO"
    return name.strip().title()


def detect_country_from_affiliation(affiliation):
    lower = str(affiliation).lower()

    if any(token in lower for token in ["south korea", "republic of korea", "korea"]):
        return "South Korea"
    if any(token in lower for token in ["united states", " u.s.", " usa", "u.s.a", "america"]):
        return "United States"
    if any(
        token in lower
        for token in [
            " maryland",
            " california",
            " massachusetts",
            " texas",
            " washington",
            " virginia",
            " florida",
            " georgia",
            " illinois",
            " colorado",
            " arizona",
            " ohio",
            " michigan",
            " wisconsin",
            " north carolina",
            " new york",
            " harvard",
            " mit",
            " stanford",
            " princeton",
            " yale",
            " noaa",
            " nasa",
        ]
    ):
        return "United States"
    if any(token in lower for token in ["england", "united kingdom", " uk", "great britain", "scotland", "wales"]):
        return "United Kingdom"
    if any(token in lower for token in ["italy", "italia", "italian"]):
        return "Italy"
    if "india" in lower:
        return "India"
    if any(token in lower for token in ["china", "beijing", "shanghai", "nanjing", "wuhan", "guangzhou", "peking", "tsinghua"]):
        return "China"
    return None


def normalize_usa_aff(aff):
    aff = aff.lower()
    if "harvard" in aff:
        return "Harvard Univ"
    if "mit" in aff or "massachusetts inst" in aff:
        return "MIT"
    if "stanford" in aff:
        return "Stanford Univ"
    if "princeton" in aff:
        return "Princeton Univ."
    if "yale" in aff:
        return "Yale University"
    if "univ calif" in aff:
        return "Univ of California"
    if "california inst of tech" in aff:
        return "Caltech"
    if "cornell" in aff:
        return "Cornell Univ."
    if "columbia" in aff:
        return "Columbia Univ."
    if "chicago" in aff:
        return "Univ. of Chicago"
    return aff.strip().title()


# =============================================================================
# 연도별 논문 수 모델링 및 시각화
# =============================================================================
def exp_model_centered(x, a, b):
    x0 = x.mean()
    return a * np.exp(b * (x - x0))


def plot_yearly_counts(years_array, counts_array, fitted_values, cagr, fit_params, output_path):
    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.patch.set_facecolor(FIGURE_BG)
    style_axis(ax, format_x=False, format_y=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    bars = ax.bar(
        years_array,
        counts_array,
        width=0.84,
        color=BAR_PRIMARY,
        edgecolor="none",
        alpha=0.94,
        zorder=3,
    )
    ax.plot(
        years_array,
        fitted_values,
        color=LINE_COLOR,
        linewidth=2.6,
        linestyle="--",
        zorder=4,
    )


    ax.set_xlabel("Publication Year", fontsize=25, color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Number of Papers", fontsize=25, color=TEXT_COLOR, labelpad=12)
    ax.set_xticks(years_array)
    ax.set_xticklabels(years_array, rotation=45, ha="right", color=TEXT_COLOR, fontsize=20)
    ax.tick_params(axis="y", labelsize=20, colors=TEXT_COLOR)

    max_y = max(float(max(counts_array)), float(max(fitted_values)))
    ax.set_ylim(0, max_y * 1.15)

    for bar, value in zip(bars, counts_array):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_y * 0.018,
            format_metric(value),
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=BAR_PRIMARY, edgecolor="none", label="Annual Paper Count"),
        Line2D([0], [0], color=LINE_COLOR, lw=2.6, linestyle="--", label="Exponential Fit"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=22)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fit_a, fit_b = fit_params
    pivot_year = years_array.mean()
    equation_text = f"y = {fit_a:.2f} e^({fit_b:.2f}(x - {pivot_year:.1f}))"
    ax.text(
        0.02,
        0.78,
        equation_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        color=TEXT_COLOR,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=HIGHLIGHT_BOX, edgecolor="#d7e1eb", alpha=0.96),
    )

    if cagr is not None:
        ax.text(
            0.02,
            0.69,
            f"CAGR  {cagr * 100:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            color=TEXT_COLOR,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=HIGHLIGHT_BOX, edgecolor="#d7e1eb"),
        )

    plt.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# 저자·소속기관 순위 시각화
# =============================================================================
def plot_ranked_bars(
    counter_data,
    title,
    subtitle,
    xlabel,
    output_path,
    bar_color,
    top_n=None,
    panel_label=None,
):
    if not counter_data:
        print(f"Skipped empty chart: {title}")
        return

    records = counter_data[:top_n] if top_n is not None else counter_data
    labels, values = zip(*records)
    chart_df = pd.DataFrame({"label": labels, "value": values}).sort_values("value", ascending=True)

    if panel_label:
        # Scale the complete country panel down proportionally from (13.0, 5.2).
        # Matplotlib font sizes are specified independently of figsize, so the
        # panel-specific fonts and line widths below are reduced as well.
        figure_size = (13.0, 5.5)
    else:
        figure_size = (13.6, max(6.8, len(chart_df) * 0.58))
    fig, ax = plt.subplots(figsize=figure_size)
    if panel_label:
        fig._fixed_panel_layout = True
    fig.patch.set_facecolor(FIGURE_BG)
    style_axis(ax, format_x=True, format_y=False)
    if panel_label:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)
    grid_linewidth = 0.8 if panel_label else 1.0
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=grid_linewidth, alpha=0.55)
    ax.grid(axis="y", visible=False)
    bar_colors = ranked_bar_colors(bar_color, len(chart_df))

    bars = ax.barh(
        chart_df["label"],
        chart_df["value"],
        height=0.68 if panel_label else 0.72,
        color=bar_colors,
        edgecolor="none",
        alpha=0.95,
        zorder=3,
    )

    max_value = chart_df["value"].max()
    x_limit_multiplier = 1.10 if panel_label else 1.12
    ax.set_xlim(0, max_value * x_limit_multiplier if max_value > 0 else 1)
    if title:
        ax.set_title(title, fontsize=30, fontweight="bold", color=TEXT_COLOR, loc="left", pad=22)
    internal_fontsize = 22 if panel_label else 22
    value_fontsize = 20 if panel_label else 20
    xlabel_pad = 14 if panel_label else 14
    ax.set_xlabel(xlabel, fontsize=internal_fontsize, color=TEXT_COLOR, labelpad=xlabel_pad)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=internal_fontsize, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=internal_fontsize, colors=TEXT_COLOR)

    for bar, value in zip(bars, chart_df["value"]):
        ax.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            format_metric(value),
            ha="left",
            va="center",
            fontsize=value_fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    if panel_label:
        fig.text(
            0.01,
            0.975,
            panel_label,
            ha="left",
            va="top",
            fontsize=30,
            color=TEXT_COLOR,
        )
        # Keep the vertical axis title clear of long institution names
        # (for example, the Italian panel's longest label).
        institution_x = 0.012
        fig.text(
            institution_x,
            0.56,
            "Institution",
            ha="center",
            va="center",
            rotation=90,
            fontsize=internal_fontsize,
            color=TEXT_COLOR,
        )

    if panel_label:
        # Keep the plot rectangle identical across all five country panels,
        # regardless of the varying institution-name lengths.
        fig.subplots_adjust(left=0.46, right=0.965, bottom=0.17, top=0.95)
    else:
        plt.tight_layout()
    save_figure(fig, output_path)


def plot_country_institution_panels(
    country_rankings,
    country_colors,
    panel_labels,
    output_path,
):
    """Draw the five country institution rankings in one vertical figure."""
    countries = [country for country in TOP_COUNTRIES if country_rankings.get(country)]
    if not countries:
        print(f"Skipped empty combined chart: {output_path}")
        return

    fig, axes = plt.subplots(len(countries), 1, figsize=(10.0, 20.0))
    if len(countries) == 1:
        axes = [axes]

    fig._fixed_panel_layout = True
    fig.patch.set_facecolor(FIGURE_BG)
    internal_fontsize = 17
    value_fontsize = 15

    for panel_index, (ax, country) in enumerate(zip(axes, countries)):
        rankings = country_rankings[country]
        labels, values = zip(*rankings)
        chart_df = pd.DataFrame({"label": labels, "value": values}).sort_values(
            "value", ascending=True
        )

        style_axis(ax, format_x=True, format_y=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)

        ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)
        ax.grid(axis="y", visible=False)
        bars = ax.barh(
            chart_df["label"],
            chart_df["value"],
            height=0.68,
            color=ranked_bar_colors(country_colors[country], len(chart_df)),
            edgecolor="none",
            alpha=0.95,
            zorder=3,
        )

        max_value = chart_df["value"].max()
        ax.set_xlim(0, max_value * 1.10 if max_value > 0 else 1)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=internal_fontsize, colors=TEXT_COLOR)
        ax.tick_params(axis="y", labelsize=internal_fontsize, colors=TEXT_COLOR)

        # Show the common x-axis title only below the final panel.
        if panel_index == len(countries) - 1:
            ax.set_xlabel(
                "Number of Papers",
                fontsize=internal_fontsize,
                color=TEXT_COLOR,
                labelpad=9,
            )
        else:
            ax.set_xlabel("")

        for bar, value in zip(bars, chart_df["value"]):
            ax.text(
                value + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                format_metric(value),
                ha="left",
                va="center",
                fontsize=value_fontsize,
                fontweight="bold",
                color=TEXT_COLOR,
            )

        ax.text(
            -0.90,
            1.04,
            panel_labels[country],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=25,
            color=TEXT_COLOR,
            clip_on=False,
        )

    # Use one shared plot geometry so long institution names align consistently.
    fig.subplots_adjust(left=0.46, right=0.965, bottom=0.055, top=0.985, hspace=0.30)
    save_figure(fig, output_path)


# =============================================================================
# 순위표 생성 및 저장
# =============================================================================
def save_rank_table(counter_data, output_path, top_n=10):
    if not counter_data:
        print(f"Skipped empty table: {output_path}")
        return pd.DataFrame(columns=["Rank", "Author", "Number of Papers"])

    records = counter_data[:top_n]
    table_df = pd.DataFrame(records, columns=["Author", "Number of Papers"])
    ranks = []
    previous_count = None
    previous_rank = None
    for index, count in enumerate(table_df["Number of Papers"], start=1):
        if count == previous_count:
            ranks.append(previous_rank)
        else:
            ranks.append(index)
            previous_rank = index
            previous_count = count

    table_df.insert(0, "Rank", ranks)
    table_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {output_path}")
    return table_df


def save_rank_table_png(table_df, output_path, title):
    if table_df.empty:
        print(f"Skipped empty PNG table: {output_path}")
        return

    fig_height = max(4.8, 1.15 + len(table_df) * 0.52)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    fig.patch.set_facecolor(FIGURE_BG)
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.0,
        1.01,
        "Ranked by number of papers",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color=MUTED_TEXT,
    )

    display_df = table_df.copy()
    display_df["Rank"] = display_df["Rank"].astype(str)
    display_df["Number of Papers"] = display_df["Number of Papers"].map(lambda value: f"{int(value):,}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6e0eb")
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor("#dfeaf6")
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f7fbff")
            if col == 1:
                cell.set_text_props(color=TEXT_COLOR, ha="left")
            else:
                cell.set_text_props(color=TEXT_COLOR)

    save_figure(fig, output_path)


# =============================================================================
# 국가별 소속기관 순위 집계
# =============================================================================
def build_country_affiliation_rankings(raw_affiliations, target_countries):
    country_counters = {country: Counter() for country in target_countries}

    for aff in raw_affiliations:
        country = detect_country_from_affiliation(aff)
        if country not in country_counters:
            continue
        normalized_affiliation = normalize_wos_affiliation(aff)
        if country == "United States":
            normalized_affiliation = normalize_usa_aff(normalized_affiliation)
        country_counters[country][normalized_affiliation] += 1

    return {
        country: counter.most_common(10)
        for country, counter in country_counters.items()
        if counter
    }


# =============================================================================
# 메인 실행 함수
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ris_files = [os.path.join(RIS_FOLDER, filename) for filename in RIS_FILENAMES]
    print(f"총 {len(ris_files)}개의 RIS 파일을 읽습니다.")

    years, authors, affiliations, unknown_records, file_tag_summary = extract_data_from_multiple_ris(ris_files)

    df = pd.DataFrame({"Years": years, "Authors": authors, "Affiliations": affiliations})

    print("\n================ 파일별 태그 요약 ================")
    df_tag_summary = pd.DataFrame(file_tag_summary)
    print(df_tag_summary)
    df_tag_summary.to_csv(os.path.join(OUTPUT_DIR, "wos_file_tag_summary.csv"), index=False, encoding="utf-8-sig")
    print("파일별 태그 요약 저장 완료: wos_file_tag_summary.csv")

    print("\n================ Unknown 레코드 요약 ================")
    print("Unknown 개수:", len(unknown_records))
    for index, rec in enumerate(unknown_records[:5], 1):
        print(f"\n[Unknown 샘플 {index}]")
        print("파일:", rec["file"])
        print("연도:", rec["year"])
        print("저자:", rec["authors"])
        print("원문 레코드 미리보기:")
        print(safe_preview(rec["raw_record"]))
        print("-" * 80)

    df_unknown = pd.DataFrame(
        [
            {
                "file": rec["file"],
                "year": rec["year"],
                "authors": "; ".join(rec["authors"]),
                "raw_record": rec["raw_record"],
            }
            for rec in unknown_records
        ]
    )
    df_unknown.to_csv(os.path.join(OUTPUT_DIR, "wos_unknown_records.csv"), index=False, encoding="utf-8-sig")
    print("Unknown 레코드 저장 완료: wos_unknown_records.csv")

    year_counts = Counter(years)
    total = sum(year_counts.values())
    print(f"Total number of WoS papers: {total:,}")

    sorted_years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in sorted_years]
    x = np.array([int(year) for year in sorted_years])
    y = np.array(counts)

    df_count = pd.DataFrame({"Year": x, "Paper_Count": y})
    df_count.to_csv(os.path.join(OUTPUT_DIR, "wos_yearly_counts.csv"), index=False, encoding="utf-8-sig")
    print(f"연도별 카운트 CSV 저장 완료: {os.path.join(OUTPUT_DIR, 'wos_yearly_counts.csv')}")

    absolute_increase = counts[-1] - counts[0]
    print(f"Absolute increase from {y[0]} to {y[-1]}: {absolute_increase}")

    start_year = x[0]
    end_year = x[-1]
    n = end_year - start_year
    print(start_year, end_year, n)

    if y[0] > 0 and n > 0:
        cagr = (y[-1] / y[0]) ** (1 / n) - 1
        print(f"CAGR: {cagr * 100:.2f}%")
    else:
        cagr = None
        print("CAGR could not be computed.")

    scaled_years = x - x.min()
    log_y = np.log(y + 1e-5)
    slope, intercept = np.polyfit(scaled_years, log_y, 1)
    p0 = [np.exp(intercept), slope]

    params, _ = curve_fit(exp_model_centered, scaled_years, y, p0=p0, maxfev=20000)
    y_fit = exp_model_centered(scaled_years, *params)

    flat_affiliations = [aff for sublist in affiliations for aff in sublist]
    flat_authors = [author for sublist in authors for author in sublist]

    top_authors = Counter(flat_authors).most_common(10)
    normalized_affiliations = [normalize_wos_affiliation(aff) for aff in flat_affiliations]
    top_affiliations = Counter(normalized_affiliations).most_common(10)
    country_affiliation_rankings = build_country_affiliation_rankings(flat_affiliations, TOP_COUNTRIES)
    top_authors_table = save_rank_table(
        top_authors,
        os.path.join(OUTPUT_DIR, "top_authors_table_wos.csv"),
        top_n=10,
    )
    save_rank_table_png(
        top_authors_table,
        os.path.join(OUTPUT_DIR, "top_authors_table_wos.png"),
        title="Top 10 Authors in Web of Science",
    )

    print("\nTop 10 authors")
    for author, count in top_authors:
        print(f"{author}: {count}")

    print("\nTop 10 authors table")
    print(top_authors_table)

    print("\nTop affiliations")
    for aff, count in top_affiliations:
        print(f"{aff}: {count}")

    total_top_affiliations = sum(count for _, count in top_affiliations)
    print("Top affiliations total:", total_top_affiliations)
    print("Top affiliations share %:", total_top_affiliations / len(affiliations) * 100)
    print("Unique affiliation count:", len(set(normalized_affiliations)))

    panel_labels = {
        "China": "(a)",
        "United States": "(b)",
        "India": "(c)",
        "South Korea": "(d)",
        "Italy": "(e)",
    }
    for country in TOP_COUNTRIES:
        rankings = country_affiliation_rankings.get(country, [])
        print(f"\nTop 10 institutions in {country}")
        for rank, (institution, count) in enumerate(rankings, 1):
            print(f"{rank}. {institution}: {count}")

    plot_yearly_counts(
        x,
        y,
        y_fit,
        cagr,
        params,
        os.path.join(OUTPUT_DIR, "yearly_count_wos.png"),
    )

    plot_ranked_bars(
        top_authors,
        title="Top 10 Authors in Web of Science",
        subtitle="Authors ranked by number of papers",
        xlabel="Number of Papers",
        output_path=os.path.join(OUTPUT_DIR, "top_authors_wos.png"),
        bar_color=BAR_SECONDARY,
    )

    plot_ranked_bars(
        top_affiliations,
        title="Top 10 Affiliations in Web of Science",
        subtitle="Institution names normalized before ranking",
        xlabel="Number of Papers",
        output_path=os.path.join(OUTPUT_DIR, "top_affiliations_wos.png"),
        bar_color=BAR_ACCENT,
    )

    country_colors = {
        "China": "#bf5f2f",
        "United States": "#4f73c4",
        "India": "#2e9d8f",
        "South Korea": "#c65b7c",
        "Italy": "#7b62b3",
    }
    for country in TOP_COUNTRIES:
        rankings = country_affiliation_rankings.get(country, [])
        if not rankings:
            continue
        plot_ranked_bars(
            rankings,
            title=None,
            subtitle="Institution names normalized and filtered by country-level affiliation matches",
            xlabel="Number of Papers",
            output_path=os.path.join(
                OUTPUT_DIR,
                f"top_10_institutions_{COUNTRY_FILENAME_MAP[country]}_wos.png",
            ),
            bar_color=country_colors[country],
            panel_label=panel_labels[country],
        )

    plot_country_institution_panels(
        country_rankings=country_affiliation_rankings,
        country_colors=country_colors,
        panel_labels=panel_labels,
        output_path=os.path.join(
            OUTPUT_DIR,
            "12-15.top_10_institutions_combined_wos.png",
        ),
    )

    print("done.")


if __name__ == "__main__":
    main()
