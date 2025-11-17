# cat_df11_close_real_bestF1.py
# - df10 기반 store_status 재정의 → df11 저장
# - CatBoost (class_weights=[1,20]) + F1 기준 best threshold 탐색/평가

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


# =========================
# 0. df10 로드 & store_status 재정의 → df11 저장
# =========================
df = pd.read_csv("df10.csv")
print(f"총 행 수: {len(df):,}")
print(f"총 열 수: {df.shape[1]}")

required_cols = ["is_open", "review_count", "stars"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"다음 컬럼이 df10에 없습니다: {missing}")

# 별점 하위 70% 기준
q70 = df["stars"].quantile(0.7)
print(f"\n⭐ 별점 하위 70% 기준값(q70): {q70:.3f}")

def define_store_status(row, q70_val):
    if row["is_open"] == 1:
        return "open"
    else:
        # 폐업 매장 중: 리뷰 수 < 50 또는 별점 <= q70 이면 close_real
        if (row["review_count"] < 50) or (row["stars"] <= q70_val):
            return "close_real"
        else:
            return "close_external"

df["store_status"] = df.apply(define_store_status, axis=1, q70_val=q70)

print("\n[store_status 분포 (새 기준)]")
print(df["store_status"].value_counts())
print("\n[store_status 비율]")
print((df["store_status"].value_counts(normalize=True) * 100).round(2))

# df11 저장
df.to_csv("df11.csv", index=False, encoding="utf-8-sig")
print("\n✅ df11.csv 저장 완료 (새 store_status 반영)")


# =========================
# 1. 전처리 (df9/df10 공통 로직)
# =========================
df = df.copy()

bool_cols = [
    "a_outdoor_seating", "a_good_for_group", "a_good_for_kids",
    "a_has_tv", "a_happy_hour"
]

def str_to_int_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "true":
        return 1
    elif s == "false":
        return 0
    else:
        return np.nan

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].apply(str_to_int_bool).astype("Int64")

noise_mapping = {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}
if "a_noise_level" in df.columns:
    df["a_noise_level"] = df["a_noise_level"].map(noise_mapping)

df[bool_cols] = df[bool_cols].fillna(0).astype(int)

# a_alcohol 결측 처리 (nightlife면 full_bar, 아니면 none)
if "a_alcohol" in df.columns and "c_nightlife" in df.columns:
    df["a_alcohol"] = np.where(
        df["a_alcohol"].isna(),
        np.where(df["c_nightlife"] == 1, "full_bar", "none"),
        df["a_alcohol"]
    )

# a_ambience 결측 처리
if "a_ambience" in df.columns:
    df["a_ambience"] = df["a_ambience"].fillna("unknown")

# one-hot encoding
df = pd.get_dummies(
    df,
    columns=["store_status", "a_alcohol", "a_ambience"],
    prefix=["status", "alcohol", "ambience"],
)
print("\n[get_dummies 이후 컬럼 수]:", df.shape[1])


# =========================
# 2. 타깃 정의 (status_close_real)
# =========================
TARGET_COL = "status_close_real"
if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} 컬럼을 찾을 수 없습니다. (get_dummies 결과 확인 필요)")

y = df[TARGET_COL].astype(int)

print("\n[타깃 분포]")
print(y.value_counts())
print(f"\nclose_real 비율: {y.mean():.4f} (약 {y.mean()*100:.2f}%)")


# =========================
# 3. 피처 선택
# =========================
exclude_cols = [
    "business_id", "name", "address", "city", "attributes", "categories", "hours",
    "is_open", "latitude", "longitude", TARGET_COL,
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "review_count"
]
status_cols = [c for c in df.columns if c.startswith("status_")]
exclude_cols += [c for c in status_cols if c not in exclude_cols]

feature_cols = [c for c in df.columns if c not in exclude_cols]
print("\n[사용할 피처 컬럼 수]")
print(len(feature_cols))
print(feature_cols[:40], "...")

X = df[feature_cols].copy()
cat_cols = [c for c in ["state", "postal_code"] if c in X.columns]

for col in X.columns:
    if col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")
    else:
        X[col] = X[col].fillna(0)

cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]
print("\n[최종 X shape]:", X.shape)
print("[범주형 피처]:", cat_cols)
print("[범주형 인덱스]:", cat_feature_indices)


# =========================
# 4. 데이터 분할 (Train / Val / Test)
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)
print(f"\nTrain 크기: {len(X_train):,}, Valid 크기: {len(X_val):,}, Test 크기: {len(X_test):,}")


# =========================
# 5. CatBoost 설정 (class_weights=[1,20])
# =========================
neg = (y == 0).sum()
pos = (y == 1).sum()
print(f"\n[클래스 비율] 음성:{neg}, 양성:{pos}, 비율(neg/pos)≈{neg/pos:.1f}")

model = CatBoostClassifier(
    depth=5,
    iterations=400,
    learning_rate=0.05,
    subsample=0.8,
    rsm=0.8,
    class_weights=[1.0, 20.0],
    loss_function="Logloss",
    eval_metric="F1",
    random_state=42,
    verbose=100,
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    cat_features=cat_feature_indices,
    verbose=100,
)

model.save_model("catboost_df11_close_real_bestF1.cbm")
print("\n✅ 모델 저장 완료: catboost_df11_close_real_bestF1.cbm")


# =========================
# 6. F1 기준 Best Threshold 탐색 (Validation 기준)
# =========================
print("\n==============================")
print("▶ Validation Set에서 F1 기준 Best Threshold 탐색")
print("==============================")

y_val_proba = model.predict_proba(X_val)[:, 1]

best_thr = 0.5
best_f1 = -1
records = []

for thr in np.arange(0.05, 0.96, 0.01):
    y_val_pred = (y_val_proba >= thr).astype(int)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    records.append((thr, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\n[Best Threshold 탐색 결과]")
print(f"- Best Threshold (val 기준 F1 최대): {best_thr:.2f}")
print(f"- Best F1 (Validation): {best_f1:.4f}")


# =========================
# 7. Test 평가 (best_thr 사용)
# =========================
print("\n==============================")
print(f"▶ Test Set 평가 (best threshold={best_thr:.2f})")
print("==============================")

y_test_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_thr).astype(int)

acc = accuracy_score(y_test, y_test_pred)
precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average="binary", pos_label=1, zero_division=0
)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision_1:.4f}")
print(f"Recall   : {recall_1:.4f}")
print(f"F1(best) : {f1_1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

# =========================
# 8. 폐업 예측 정답 개수 출력
# =========================
true_positives = np.sum((y_test == 1) & (y_test_pred == 1))
false_negatives = np.sum((y_test == 1) & (y_test_pred == 0))
total_positives = np.sum(y_test == 1)

print(f"\n[폐업 예측 정답 개수]")
print(f"맞춘 폐업 매장 수 (TP): {true_positives}")
print(f"놓친 폐업 매장 수 (FN): {false_negatives}")
print(f"전체 실제 폐업 매장 수: {total_positives}")
print(f"정답 비율 (TPR/Recall): {true_positives}/{total_positives} = {true_positives/total_positives:.2%}")


# =========================
# 9. Feature Importance
# =========================
print("\n==============================")
print("▶ Feature Importance (상위 20개)")
print("==============================")

importances = model.get_feature_importance(type="FeatureImportance")
feat_imp = list(zip(X.columns, importances))
feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)

for name, val in feat_imp_sorted[:20]:
    print(f"{name:30s} : {val:.4f}")

print("\n▶ 안정/충성/신뢰 지수 중요도")
for col in ["stability_score", "loyalty_score", "reliability_score"]:
    if col in X.columns:
        val = dict(feat_imp).get(col, 0)
        print(f"{col:20s} : {val:.4f}")
    else:
        print(f"{col:20s} : (X에 없음)")

print("\n✅ CatBoost df11 close_real F1-best threshold 평가 완료")
