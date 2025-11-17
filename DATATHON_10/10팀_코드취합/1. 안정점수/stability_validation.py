# -*- coding: utf-8 -*-
"""
stability_validation_df9.py

df7/df9 기반 StableIndex 검증 + 시각화
1) store_status 4그룹 ANOVA + boxplot (중앙값 숫자 표시)
2) close(=close_real/close_external) vs open 로지스틱 회귀 + p-value/오즈비 + 시각화
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# ⚠️ 필요에 따라 df7.csv / df9.csv로 바꿔 쓰세요
PATH_CSV   = "df7.csv"
STAB_COL   = "stability_score"     # 안정지수 컬럼명
STATUS_COL = "store_status"

# 한글 폰트 (윈도우 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sns.set(style="whitegrid")


def load_df(path=PATH_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' 파일이 없습니다.")
    print(f"📦 {path} 불러오는 중...")
    return pd.read_csv(path)


def main():
    df = load_df()

    # 필요한 컬럼만
    needed = [STAB_COL, STATUS_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 없음: {missing}")

    df = df[[STAB_COL, STATUS_COL]].dropna().copy()

    # store_status 4개만 사용 (혹시 다른 값 섞여 있으면 제거)
    valid_status = ["close_real", "close_external", "open_new", "open_old"]
    df = df[df[STATUS_COL].isin(valid_status)].copy()
    df[STATUS_COL] = pd.Categorical(df[STATUS_COL], categories=valid_status, ordered=True)

    print(f"\n✅ 유효 행 수: {len(df):,}")
    print(f"store_status 고유값: {df[STATUS_COL].unique()}")

    # ============================================================
    # [1] ANOVA + Boxplot (Store Status별 StableIndex)
    # ============================================================
    print("\n=== [1] store_status별 StableIndex 기술 통계 ===")
    print(df.groupby(STATUS_COL)[STAB_COL].describe().round(3))

    print("\n=== [2] 일원분산분석(ANOVA): StableIndex ~ store_status ===")
    model = ols(f"{STAB_COL} ~ C({STATUS_COL})", data=df).fit()
    aov_table = anova_lm(model)
    print(aov_table)

    # Boxplot 그리기
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x=STATUS_COL, y=STAB_COL)

    # 중앙값 표기 (원하면 mean으로 변경 가능)
    stats = df.groupby(STATUS_COL)[STAB_COL].median()

    for i, status in enumerate(valid_status):
        if status not in stats.index:
            continue
        y_val = stats[status]
        ax.text(
            i,
            y_val,
            f"{y_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black"
        )

    ax.set_title("Store Status별 StableIndex 분포", fontsize=16)
    ax.set_xlabel("Store Status", fontsize=12)
    ax.set_ylabel("StableIndex", fontsize=12)

    plt.tight_layout()
    plt.savefig("stability_boxplot_store_status.png", dpi=150)
    plt.show()
    print("📁 저장 완료: stability_boxplot_store_status.png")

    # ============================================================
    # [2] Logistic: close vs open (2분류)
    # ============================================================
    print("\n=== [3] 로지스틱 회귀: (close_real/close_external=1) vs (open_new/open_old=0) ===")

    # close = 1, open = 0
    close_set = {"close_real", "close_external"}
    df["is_closed"] = df[STATUS_COL].isin(close_set).astype(int)

    y = df["is_closed"]
    X = sm.add_constant(df[[STAB_COL]])

    try:
        logit_model = sm.Logit(y, X, missing="drop").fit(disp=False)
        print(logit_model.summary())

        # === p-value / 계수 / 오즈비 요약 ===
        params = logit_model.params
        pvalues = logit_model.pvalues
        odds_ratios = np.exp(params)

        print("\n=== [3-1] 로지스틱 회귀 주요 통계 ===")
        print(f"계수(β): {params[STAB_COL]:.4f}")
        print(f"오즈비(Exp(β)): {odds_ratios[STAB_COL]:.4f}")
        print(f"p-value: {pvalues[STAB_COL]:.6f}")

        # 해석용 간단 요약
        if pvalues[STAB_COL] < 0.001:
            signif = "⭐⭐⭐ (p<0.001, 매우 유의)"
        elif pvalues[STAB_COL] < 0.01:
            signif = "⭐⭐ (p<0.01, 유의)"
        elif pvalues[STAB_COL] < 0.05:
            signif = "⭐ (p<0.05, 약간 유의)"
        else:
            signif = "❌ (유의하지 않음)"
        print(f"→ 해석: StableIndex는 폐업 확률에 {signif}한 영향을 미침")

        # 예측 곡선용 grid
        x_min, x_max = df[STAB_COL].min(), df[STAB_COL].max()
        x_grid = np.linspace(x_min, x_max, 200)
        X_grid = sm.add_constant(pd.DataFrame({STAB_COL: x_grid}))
        y_pred = logit_model.predict(X_grid)

        # 시각화
        plt.figure(figsize=(10, 6))

        # 실제 값 산점도 (0/1에 jitter)
        jitter = (np.random.rand(len(df)) - 0.5) * 0.05
        plt.scatter(
            df[STAB_COL],
            df["is_closed"] + jitter,
            s=10,
            alpha=0.3,
            label="실제 데이터 (0=운영, 1=폐업)"
        )

        # 로지스틱 곡선
        plt.plot(
            x_grid,
            y_pred,
            linewidth=2,
            label="예측 폐업확률 P(close=1)"
        )

        plt.title("StableIndex에 따른 폐업확률 (close_real + close_external)", fontsize=16)
        plt.xlabel("StableIndex", fontsize=12)
        plt.ylabel("P(폐업=1)", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig("logistic_closed_vs_open_stableindex.png", dpi=150)
        plt.show()
        print("📁 저장 완료: logistic_closed_vs_open_stableindex.png")

    except Exception as e:
        print(f"로지스틱 회귀 실패: {e}")
        return


if __name__ == "__main__":
    main()
