# -*- coding: utf-8 -*-
"""
cat_df11_score_one_with_shap.py
- cat_df11_close_real_bestF1.py 로 학습한 모델을 이용해
  특정 식당 1곳의 폐업 확률을 예측하고(프로바) 로컬 SHAP로 주요 요인 Top-N을 설명합니다.
- 조건 예: 평점(stars) 낮고, neighbor_density 높고, stability_score 낮은 매장 하나 선정
- open 매장만 대상으로 필터링 + 카테고리(샌드위치) 필터 추가
- 결과:
  1) 콘솔 출력(확률/threshold/판정/Top SHAP + %포인트 영향)
  2) CSV 저장: one_store_explain.csv (SHAP + 확률 변화 %p 포함)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import shap

# =========
# 0. 입력
# =========
DF_PATH = "df11.csv"
MODEL_PATH = "catboost_df11_close_real_bestF1.cbm"
BEST_THR = None   # 예: 0.23 (학습 스크립트에서 출력된 best_thr를 넣으면 일관성↑)
TARGET_BUSINESS_ID = None  # 예: "abc123" (직접 지정 시)

# 🔽 추가: 카테고리 필터 이름 (원핫 컬럼 접두어 c_)
CATEGORY_NAME = "american_new"   # df11에 보통 c_american_new 로 존재 (다르면 자동 감지 시도)

# =========
# 1. 데이터 로드 및 전처리(학습과 동일)
# =========
df = pd.read_csv(DF_PATH)
assert "store_status" in df.columns, "df11에 store_status가 필요합니다."

# 학습 시 사용했던 bool/매핑 전처리(간단화: 이미 df11 저장 시 반영되어 있다고 가정)
bool_cols = [
    "a_outdoor_seating", "a_good_for_group", "a_good_for_kids",
    "a_has_tv", "a_happy_hour"
]
for c in bool_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype(int)

noise_mapping = {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}
if "a_noise_level" in df.columns:
    df["a_noise_level"] = df["a_noise_level"].map(noise_mapping)

if "a_alcohol" in df.columns and "c_nightlife" in df.columns:
    df["a_alcohol"] = np.where(
        df["a_alcohol"].isna(),
        np.where(df["c_nightlife"] == 1, "full_bar", "none"),
        df["a_alcohol"]
    )
if "a_ambience" in df.columns:
    df["a_ambience"] = df["a_ambience"].fillna("unknown")

# get_dummies (store_status 포함: 타깃 더미 생성되지만 X에는 미사용)
df_dum = pd.get_dummies(
    df.copy(),
    columns=["store_status", "a_alcohol", "a_ambience"],
    prefix=["status", "alcohol", "ambience"],
)

TARGET_COL = "status_close_real"
if TARGET_COL not in df_dum.columns:
    raise ValueError("status_close_real 더미가 없습니다. get_dummies 결과를 확인하세요.")

# 학습 시 제외했던 컬럼과 동일하게 feature cols 구성
exclude_cols = [
    "business_id", "name", "address", "city", "attributes", "categories", "hours",
    "is_open", "latitude", "longitude", TARGET_COL,
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "review_count"
]
status_cols = [c for c in df_dum.columns if c.startswith("status_")]
exclude_cols += [c for c in status_cols if c not in exclude_cols]

feature_cols = [c for c in df_dum.columns if c not in exclude_cols]
X_all = df_dum[feature_cols].copy()

cat_cols = [c for c in ["state", "postal_code"] if c in X_all.columns]
for col in X_all.columns:
    if col in cat_cols:
        X_all[col] = X_all[col].astype(str).fillna("missing")
    else:
        X_all[col] = X_all[col].fillna(0)

cat_feature_indices = [X_all.columns.get_loc(c) for c in cat_cols]

# =========
# 2. 타깃 매장 선택 로직 (샌드위치 카테고리 필터 포함)
# =========
# c_american_new 컬럼명 자동 탐지 (예: c_american_new / c_american_newes 등)
cand_cat_cols = [c for c in df.columns if c.startswith("c_") and CATEGORY_NAME in c]
if len(cand_cat_cols) == 0:
    # american_new 관련 원핫이 없으면 경고 없이 전체에서 진행
    cat_col = None
else:
    # american_new 키워드 포함 가장 짧은 컬럼명 선택
    cat_col = sorted(cand_cat_cols, key=len)[0]

if TARGET_BUSINESS_ID is not None and "business_id" in df.columns:
    row = df.loc[df["business_id"] == TARGET_BUSINESS_ID]
    if row.empty:
        raise ValueError("해당 business_id가 df11에 없습니다.")
    idx = row.index[0]
else:
    # 조건: open 매장 + (카테고리=샌드위치) + 별점 낮음 + 경쟁 높음 + 안정 낮음
    conds = []
    conds.append(df["is_open"] == 1)  # ✅ 오픈 매장만
    conds.append(df["review_count"] > 100)

    # ✅ 카테고리(샌드위치) 필터
    if cat_col is not None:
        conds.append(df[cat_col] == 1)

    if "stars" in df.columns:
        stars_p30 = df["stars"].quantile(0.3)
        conds.append(df["stars"] <= stars_p30)
    if "neighbor_density" in df.columns:
        nd_p70 = df["neighbor_density"].quantile(0.7)
        conds.append(df["neighbor_density"] >= nd_p70)
    if "stability_score" in df.columns:
        stab_p30 = df["stability_score"].quantile(0.3)
        conds.append(df["stability_score"] <= stab_p30)

    if conds:
        mask = np.logical_and.reduce(conds)
        candidates = df[mask]
        if candidates.empty:
            # 조건 완화: (샌드위치 유지) open 매장 중 안정지수 낮은 200개에서 우선 선택
            base = df[(df["is_open"] == 1)]
            if cat_col is not None:
                base = base[base[cat_col] == 1]
            if "stability_score" in base.columns and len(base) > 0:
                candidates = base.nsmallest(min(200, len(base)), "stability_score")
            else:
                candidates = base.sample(min(200, len(base)), random_state=42)
    else:
        base = df[(df["is_open"] == 1)]
        if cat_col is not None:
            base = base[base[cat_col] == 1]
        candidates = base.sample(min(200, len(base)), random_state=42)

    # 후보 중에서 “현재 모델 기준 예측 위험 확률”이 가장 높은 1개 선택
    model_tmp = CatBoostClassifier()
    model_tmp.load_model(MODEL_PATH)

    X_cand = X_all.loc[candidates.index]
    proba_cand = model_tmp.predict_proba(X_cand)[:, 1]
    idx = X_cand.index[np.argmax(proba_cand)]

# =========
# 3. 로딩 & 개별 예측 + 로컬 SHAP
# =========
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

x_row = X_all.loc[[idx]].copy()
pool_row = Pool(x_row, cat_features=cat_feature_indices)

proba = float(model.predict_proba(pool_row)[:, 1][0])

# threshold 설정
if BEST_THR is None:
    thr = 0.5  # 학습에서 찾은 best_thr를 넣으면 일관성↑
else:
    thr = float(BEST_THR)

pred = int(proba >= thr)

# ---------- 로컬 SHAP 계산 ----------
# CatBoost ShapValues: shape = (n_samples, n_features + 1) ; 마지막 열이 base value(logit bias)
shap_vals_full = model.get_feature_importance(type="ShapValues", data=pool_row)
base_logit = float(shap_vals_full[0, -1])  # bias term (log-odds 기준)
shap_vals = shap_vals_full[:, :-1]         # 마지막 bias 제거
contrib = pd.Series(shap_vals[0], index=x_row.columns).sort_values(ascending=False)

top_pos = contrib.head(8)      # 폐업↑로 밀어올린 요인
top_neg = contrib.tail(8)      # 폐업↓로 눌러준 요인

# ---------- SHAP → 확률 변화(%포인트) 변환 ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

base_prob = sigmoid(base_logit)

def shap_to_pct_point(s):
    new_prob = sigmoid(base_logit + s)
    return (new_prob - base_prob) * 100.0  # percentage points

impact_pct = contrib.apply(shap_to_pct_point)

# 원본 식당 메타
meta_cols = ["business_id", "name", "state", "city", "stars", "review_count",
             "stability_score", "loyalty_score", "reliability_score",
             "neighbor_density"]
meta_cols = [c for c in meta_cols if c in df.columns]
meta = df.loc[idx, meta_cols].to_dict()

# =========
# 4. 출력
# =========
print("\n==== [개별 식당 폐업 위험 예측] ====")
print(f"- 대상 인덱스: {idx}")
for k, v in meta.items():
    print(f"  {k}: {v}")
print(f"- 예측 폐업 확률: {proba:.4f}")
print(f"- 판정(thr={thr:.2f}): {'폐업(1)' if pred==1 else '비폐업(0)'}")

print("\n[로컬 SHAP 상위 기여 요인 (폐업↑)]")
for k, v in top_pos.items():
    print(f"  {k:30s} {v:+.5f}  (impact ~ {shap_to_pct_point(v):+.2f}%p)")

print("\n[로컬 SHAP 하위 기여 요인 (폐업↓)]")
for k, v in top_neg.items():
    print(f"  {k:30s} {v:+.5f}  (impact ~ {shap_to_pct_point(v):+.2f}%p)")

# =========
# 5. CSV 저장 (SHAP + 확률 변화 %p 포함)
# =========
out = pd.DataFrame({
    "feature": contrib.index,
    "shap_value": contrib.values,
    "impact_pct_point": impact_pct.values
})
out.insert(0, "business_index", idx)
out.insert(1, "pred_proba", proba)
out.insert(2, "pred_label", pred)
out.insert(3, "base_logit", base_logit)
out.insert(4, "base_prob", base_prob)

for k, v in meta.items():
    out[k] = v

out.to_csv("one_store_explain.csv", index=False, encoding="utf-8-sig")
print("\n✅ 저장 완료: one_store_explain.csv")

# (선택) 상위 10개 영향도 표 간단 출력
print("\n[확률 변화 기준 Top 10 요인 (%p)]")
tmp = out[["feature","impact_pct_point"]].copy().sort_values("impact_pct_point", ascending=False).head(10)
for _, r in tmp.iterrows():
    print(f"  {r['feature']:30s} {r['impact_pct_point']:+.2f}%p")
