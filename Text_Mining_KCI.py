import re
import os
from pathlib import Path

# Run this saved Python file; paths are relative to its project root.
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "Data" / "KCI"
FIG = BASE / "Fig" / "KCI"
OUT = BASE / "Output" / "KCI"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).resolve().parent
import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


# =============================================================================
# 한글 폰트 설정 (운영체제에 맞게 변경)
# =============================================================================

plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
# plt.rcParams['font.family'] = 'AppleGothic'   # macOS
# plt.rcParams['font.family'] = 'NanumGothic'   # Linux

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False



# =============================================================================
# data 추출 함수
# =============================================================================
def extract_data_from_ris(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    years   = []
    authors = []
    affiliations = []
    
    current_year = None
    current_authors = []
    current_affiliations = []
    
    
    
    for line in lines:

        line = line.strip()
        
        
        # 저자: AU - 홍길동, 이서방 등등 모든 저자
        if line.startswith("AU  -"):
            author = line.replace("AU  -", "").strip()
            current_authors.append(author)


        # 기관: C1  - 한국연구재단, ...
        if line.startswith("C1  -") or line.startswith("C2  -") or line.startswith("C3  -"):
            #aff_line = line.replace("C1  -", "").strip()
            aff_line = line.split("  -", 1)[-1].strip()
            
            # 여러 기관이 연결되어 있을 수 있음: 쪼개기
            aff_parts = re.split(r'[.;]', aff_line)            
            if aff_parts:
                current_affiliations.append(aff_parts[0].strip())
                
            #affiliations.append(aff_parts[0]) 
#            print(aff_parts[0])
            
#            for aff in aff_parts:
#                aff = aff.strip()
#                print(aff)
#                if len(aff) > 0:  # 너무 짧은 문자열 제외
#                    affiliations.append(aff)

        # 출판 연도
        if line.startswith("PY  -") or line.startswith("Y1  -"):
            match = re.search(r'\d{4}', line)
            if match:
                current_year = int(match.group())



        # 한 논문 레코드가 끝남
        if line.startswith("ER  -"):
            if current_year and current_year <= 2025:
                years.append(str(current_year))
                authors.append(current_authors if current_authors else ["Unknown"])
                affiliations.append(current_affiliations if current_affiliations else ["Unknown"])

            # 초기화
            current_year = None
            current_authors = []
            current_affiliations = []

                    
    return years, authors, affiliations




# =============================================================================
# 기관명 변경
# =============================================================================
def normalize_kci_affiliation(name):
    name = name.lower()

    if '서울대' in name or 'seoul nat' in name:
        return 'Seoul National Univ.'

    elif '고려' in name or 'korea univ' in name:
        return 'Korea University'
    elif '연세' in name or 'yonsei' in name:
        return 'Yonsei Univ.'
    elif '카이스트' in name or 'kaist' in name:
        return 'KAIST'
    elif '울산과학기술원' in name or 'unist' in name:
        return 'UNIST'
    elif '성균관' in name or 'sungkyunkwan' in name:
        return 'Sungkyunkwan Univ.'
    elif '동국' in name or 'dongkuk' in name:
        return 'Dongguk Univ.'
    elif '건국' in name or 'konkuk' in name:
        return 'Konkuk Univ.'

    if '서울시립' in name or 'seoul nat' in name:
        return 'Univ. of Seoul'

    if '홍익' in name or 'hongkik' in name:
        return 'Hongik Univ'
    
    elif '경희' in name or 'kyunghee univ' in name:
        return 'Kyunghee Univ.'
    elif '부산' in name:
        return 'Pusan National Univ.'
    elif '경북' in name and '대' in name:
        return 'Kyungpook National Univ.'
    elif '전남' in name:
        return 'Chonnam National Univ.'
    elif '부경' in name:
        return 'Pukyong National Univ.'

    elif '한국건설' in name:
        return 'KICT'

    elif '강릉원주' in name:
        return 'Gangneung Wonju National Univ.'

    elif '안양' in name:
        return 'Anyang Univ.'
    
    elif '아주' in name:
        return 'Ajou Univ.'
    
    elif '강원' in name:
        return 'Kangwon Univ.'
    
    elif '중앙' in name:
        return 'Chungang Univ.'
    
    elif '공주' in name:
        return 'Kongju National Univ.'
    
    elif '한국기술교육' in name:
        return 'KOREATECH.'
    
    elif '세종' in name:
        return 'Sejong Univ.'
    
    else:
        return name.strip().title()  # 기타는 단순 정리
    
    
    

# =============================================================================
# 연도 데이터 수집 
# =============================================================================

ris_file_path = DATA / "KCI_ris_102_final.ris" 

# data 추출
years,authors,affiliations = extract_data_from_ris(ris_file_path)

# df 변환
df = pd.DataFrame({
    'Year': years,
    'Affiliations': affiliations
})



# =============================================================================
# 연도별 논문 수
# =============================================================================

year_counts = Counter(years)

total = sum(year_counts.values())
print(f"Total number of KCI papers: {total}")  

# 연도 오름차순 정렬
sorted_years = sorted(year_counts.keys())
counts = [year_counts[year] for year in sorted_years]

# 정수형 연도와 numpy 배열 변환
x_years = np.array([int(y) for y in sorted_years])
y_counts = np.array(counts)


# DataFrame 생성
df_count = pd.DataFrame({
                'Year': x_years,
                'Paper_Count': y_counts
            })

# CSV 저장
csv_output_path = OUT / "kci_yearly_counts.csv"
df_count.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

print(f'x, y 데이터가 CSV 파일로 저장되었습니다: {csv_output_path}')



# =============================================================================
# 절대 증가량 계산 CAGR: Compound Annual Growth Rate
# =============================================================================
absolute_increase = y_counts[-1] - y_counts[0]
print(f"절대 증가량: {y_counts[0]} → {y_counts[-1]} → 증가량: {absolute_increase}편")


# =============================================================================
# CAGR 계산
# =============================================================================
start_year = x_years[0]
end_year   = x_years[-1]
year_range = list(range(start_year, end_year + 1))

n = end_year - start_year

print(start_year,end_year, n)

if y_counts[0] > 0 and n > 0:
    cagr = (y_counts[-1] / y_counts[0])**(1 / n) - 1
    print(f"연평균 증감률 (CAGR): {cagr*100:.2f}%")
else:
    cagr = None
    print("CAGR 계산 불가")


# =============================================================================
# 이동 평균 (2년) 계산
# =============================================================================
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

ma_counts = moving_average(y_counts, window_size=2)


# =============================================================================
# 연도별 출판 그래프 시각화 (인포그래픽 스타일)
# =============================================================================
def plot_publication_activity(x_years, y_counts, ma_counts, cagr, output_file):
    dpi = 300
    target_width_px = 3948
    target_height_px = 2298
    plt.figure(
        figsize=(target_width_px / dpi, target_height_px / dpi),
        facecolor="white",
    )
    ax = plt.gca()
    ax.set_facecolor("white")

    ax.fill_between(x_years, y_counts, color="#fdba74", alpha=0.45, zorder=1)
    ax.plot(
        x_years,
        y_counts,
        color="#c2410c",
        linewidth=3.2,
        marker="o",
        markersize=8,
        label="Annual Paper Count",
        zorder=3
    )
    ax.plot(
        x_years,
        ma_counts,
        color="#0f766e",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=4,
        label="2-Year Moving Average",
        zorder=4
    )

    peak_idx = int(np.argmax(y_counts))
    peak_year = int(x_years[peak_idx])
    peak_value = int(y_counts[peak_idx])

    ax.scatter([peak_year], [peak_value], s=200, color="#7f1d1d", edgecolor="white", linewidth=1.5, zorder=5)

    for year, count in zip(x_years, y_counts):
        ax.annotate(
            f"{int(count)}",
            xy=(year, count),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color="black",
            zorder=6,
        )

    ax.set_ylim(0, max(y_counts) * 1.18)

    ax.set_xlabel('Publication Year', fontsize=25, color="black")
    ax.set_ylabel('Number of Papers', fontsize=25, color="black")
    ax.set_xticks(year_range)
    ax.tick_params(axis='x', rotation=45, labelsize=20, colors="black")
    ax.tick_params(axis='y', labelsize=20, colors="black")
    ax.grid(False, axis='x')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    if cagr is not None:
        cagr_handle = Line2D([], [], linestyle='None')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(cagr_handle)
        labels.append(f'CAGR: {cagr*100:.2f}%')
        legend = ax.legend(handles, labels, frameon=False, fontsize=22, loc='upper left')
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_texts()[-1].set_color('#166534')
    else:
        ax.legend(frameon=False, fontsize=22, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, facecolor="white")
    plt.show()


plot_publication_activity(
    x_years,
    y_counts,
    ma_counts,
    cagr,
    FIG / "yearly_count_kci_infographic.png"
)

print("연도별 논문수 인포그래픽 그래프 저장 완료 ✅")




# =============================================================================
# 기관 통계
# =============================================================================

# affiliations: list of lists
flat_affiliations = [aff for sublist in affiliations for aff in sublist]
affiliation_counts = Counter(flat_affiliations)


# 상위 10명 저자
flat_authors = [aut for sublist in authors for aut in sublist]

author_counts = Counter(flat_authors)
top_authors   = author_counts.most_common(10)


# 정규화 적용
normalized_affiliations = [normalize_kci_affiliation(aff) for aff in flat_affiliations]

# 최빈 기관 계산
top_affiliations = Counter(normalized_affiliations).most_common(11)



# 출력
print("📌 상위 10명 저자:")
for author, count in top_authors:
    print(f"{author}: {count}")

print("\n🏢 상위 10개 기관:")
for aff, count in top_affiliations:
    print(f"{aff}: {count}")


total_top_affiliations = sum(count for _, count in top_affiliations)
print("Top 10 기관 논문 수 총합:", total_top_affiliations)
print("Top 10 기관 논문 수 %:", total_top_affiliations/len(affiliations)*100)


unique_affiliation_count = len(set(normalized_affiliations))
print("고유 기관 수:", unique_affiliation_count)



# =============================================================================
# 시각화 (저자 기준)
# =============================================================================

def plot_top_items(counter_data, title, xlabel, ylabel, output_file):
    if not counter_data:
        print(f"{title} → 데이터 없음")
        return
    
    # 폰트 크기 설정
    title_fontsize = 25
    label_fontsize = 20
    tick_fontsize  = 16

    labels, values = zip(*counter_data)
    
    plt.figure(figsize=(12, 7))
    
    plt.barh(labels, values, color='skyblue')
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.gca().invert_yaxis()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_top_authors_infographic(counter_data, xlabel, ylabel, output_file):
    if not counter_data:
        print("Top authors infographic → 데이터 없음")
        return

    labels, values = zip(*counter_data)
    labels = list(labels)[::-1]
    values = list(values)[::-1]

    base_colors = ["#fed7aa", "#fdba74", "#fb923c", "#f97316", "#ea580c", "#c2410c",
                   "#9a3412", "#7c2d12", "#7f1d1d", "#431407"]
    colors = base_colors[-len(labels):]

    plt.figure(figsize=(12, 7), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    bars = ax.barh(labels, values, color=colors, edgecolor='none')

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va='center',
            fontsize=20,
            fontweight="bold",
            color="black"
        )

    ax.set_title("TOP 10 Authors (KCI)", loc="left", fontsize=30, fontweight="bold", color="black")

    ax.set_xlabel(xlabel, fontsize=25, color="black")
    ax.set_ylabel(ylabel, fontsize=25, color="black")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=20, colors="black")
    ax.tick_params(axis='y', labelsize=20, colors="black")
    ax.grid(False, axis='y')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor="white")
    plt.tight_layout()
    plt.show()


def plot_top_affiliations_infographic(counter_data, xlabel, ylabel, output_file):
    if not counter_data:
        print("Top affiliations infographic ???곗씠???놁쓬")
        return

    labels, values = zip(*counter_data)
    labels = list(labels)[::-1]
    values = list(values)[::-1]

    base_colors = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb",
                   "#1d4ed8", "#1e40af", "#1e3a8a", "#172554", "#0f172a"]
    colors = base_colors[-len(labels):]

    plt.figure(figsize=(13, 7.5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    bars = ax.barh(labels, values, color=colors, edgecolor='none')

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va='center',
            fontsize=20,
            fontweight="bold",
            color="black"
        )

    ax.set_title("TOP Affiliations (KCI)", loc="left", fontsize=30, fontweight="bold", color="black")

    ax.set_xlabel(xlabel, fontsize=25, color="black")
    ax.set_ylabel(ylabel, fontsize=25, color="black")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=20, colors="black")
    ax.tick_params(axis='y', labelsize=20, colors="black")
    ax.grid(False, axis='y')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor="white")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 그래프 출력
# =============================================================================

kci_fig_dir = FIG
os.makedirs(kci_fig_dir, exist_ok=True)

plot_top_authors_infographic(top_authors,
                             'Number of Papers',
                             'Authors',
                             os.path.join(kci_fig_dir, 'top_authors_kci_before.png'))

plot_top_affiliations_infographic(top_affiliations,
                                  'Number of Papers',
                                  'Affiliation',
                                  os.path.join(kci_fig_dir, 'top_affiliations_kci_before.png'))




