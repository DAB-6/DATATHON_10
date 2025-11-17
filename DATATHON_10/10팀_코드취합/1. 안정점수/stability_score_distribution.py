# stable_index_overall_distribution.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 로드
df = pd.read_csv("df3.csv")

# StableIndex 컬럼 자동 인식
col_si = [c for c in df.columns if c.lower() == "stableindex"][0]

# 1️⃣ 히스토그램 + 밀도그래프
plt.figure(figsize=(8,6))
sns.histplot(df[col_si], bins=40, kde=True, color="skyblue", alpha=0.8)
plt.title("안정지수 분포", fontsize=14, weight="bold")
plt.xlabel("StableIndex")
plt.ylabel("빈도수(Frequency)")
plt.tight_layout()
plt.savefig("stableindex_overall_hist.png", dpi=300)
plt.show()

# 2️⃣ 커브 중심 밀도 그래프 (보조)
plt.figure(figsize=(8,6))
sns.kdeplot(df[col_si], fill=True, color="steelblue", alpha=0.5)
plt.title("안정지수 밀도 분포", fontsize=14, weight="bold")
plt.xlabel("StableIndex")
plt.ylabel("밀도(Density)")
plt.tight_layout()
plt.savefig("stableindex_overall_kde.png", dpi=300)
plt.show()
