# -*- coding: utf-8 -*-
"""
one_store_full_explain_nlp.py

✔ 목적
- (1) CatBoost 분류 모델로 개별 매장의 폐업 위험 확률 예측
- (2) 로컬 SHAP으로 주요 기여 요인 Top-N 설명 및 %포인트 영향량 산출
- (3) 전체 분포 대비 해당 매장의 위험 백분위(상위 몇 %) 계산
- (4) 단일 식당 NLP 파이프라인(감성/로그오즈/측면/요약카드) 실행

입력
- df11.csv (메타/카테고리/지수 포함)
- review_filtered.parquet (리뷰 텍스트)
- catboost_df11_close_real_bestF1.cbm (학습된 모델)

출력
- one_store_explain.csv (SHAP + 확률 변화 %p + 메타)
- one_store_distribution.csv (전체 예측 확률 분포 + 백분위)
- nlp_out/ (월별 감성, 부정 bi/tri-gram 로그오즈, 측면 점수, 예시 문장, 요약카드)

사용법(예)
- BUSINESS_ID 지정:
    BUSINESS_ID = "5Md0YaxD5HiOoBmsnmIu7A"
- 또는 자동 선택(별점↓, 경쟁도↑, 안정지수↓ 조건으로 후보→최고 위험 1개)

메모
- BEST_THR를 학습 스크립트의 best_thr로 지정하면 판정 일관성이 올라갑니다.
- SCORE 스케일(선택): 단순형(100×(1−p)), 로그형(100×(1−√p)) 동시에 계산해 저장합니다.
"""

import os
import re
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# 0) 설정
# =========================
DF_PATH      = "df11.csv"
MODEL_PATH   = "catboost_df11_close_real_bestF1.cbm"
REVIEWS_PATH = "review_filtered.parquet"
OUT_DIR_NLP  = Path("nlp_out")

# 개별 실행 옵션
BUSINESS_ID = None  # 예: "5Md0YaxD5HiOoBmsnmIu7A" (None이면 자동 선택 로직)
BEST_THR    = 0.87  # 예: 0.23 (None이면 0.50)
TOP_N       = 8     # SHAP 상/하위 표시 개수

# 🔀 후보 선택 모드
# - "argmax": 가장 위험(확률 최대) 1개
# - "random_topk": 상위 TOPK 중 무작위 1개
# - "weighted_softmax": 확률 가중 소프트맥스 샘플링(확률 높을수록 선택↑)
# - "pure_random": 필터 통과 후보 중 완전 랜덤
PICK_MODE    = "random_topk"
TOPK         = 50         # random_topk에서 사용하는 상위 개수
TEMPERATURE  = 0.7        # weighted_softmax에서 온도(↓이면 상위에 더 집중)
RANDOM_STATE = 42         # 재현성용 시드(원하면 None)

# 🔁 중복 방지: 이전에 뽑힌 매장 제외용 로컬 파일
EXCLUDE_IDS_PATH = Path("picked_ids.txt")

# 분포 백분위 계산 옵션 (전체 df11 대상 예측 수행)
COMPUTE_GLOBAL_DIST = True

# 분포 백분위 계산 옵션 (전체 df11 대상 예측 수행)
COMPUTE_GLOBAL_DIST = True

# =========================
# 1) 공통 유틸
# =========================

def ensure_cols(df: pd.DataFrame, need: set, name: str):
    miss = need - set(df.columns.astype(str))
    if miss:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(miss)}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_scores(p: float) -> dict:
    """확률→점수 스케일 변환(단순형/로그형 둘 다)"""
    p = float(p)
    score_simple = 40 + 60 * (1 - p) ** (0.5) 
    score_log    = (1 - math.sqrt(p)) * 100.0
    return {
        "score_simple": score_simple,
        "score_log": score_log,
    }


# =========================
# 2) 데이터 로드 & 전처리(학습과 동일)
# =========================

df = pd.read_csv(DF_PATH)
ensure_cols(df, {"store_status"}, "df11")

# 학습 시 반영했던 간단 전처리(필요 컬럼만 안전 적용)
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

# get_dummies (store_status 포함: 타깃 더미 생성되지만 X에는 제외)
df_dum = pd.get_dummies(
    df.copy(),
    columns=["store_status", "a_alcohol", "a_ambience"],
    prefix=["status", "alcohol", "ambience"],
)

TARGET_COL = "status_close_real"
if TARGET_COL not in df_dum.columns:
    raise ValueError("status_close_real 더미가 없습니다. get_dummies 결과를 확인하세요.")

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

# CatBoost 범주형 처리(문자형 유지)
cat_cols = [c for c in ["state", "postal_code"] if c in X_all.columns]
for col in X_all.columns:
    if col in cat_cols:
        X_all[col] = X_all[col].astype(str).fillna("missing")
    else:
        X_all[col] = X_all[col].fillna(0)

cat_feature_indices = [X_all.columns.get_loc(c) for c in cat_cols]

# =========================
# 3) 모델 로드
# =========================
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# =========================
# 4) 타깃 매장 선택(지정 or 자동)
# =========================
if BUSINESS_ID is not None and "business_id" in df.columns:
    row_sel = df.loc[df["business_id"] == BUSINESS_ID]
    if row_sel.empty:
        raise ValueError("해당 business_id가 df11에 없습니다.")
    idx = row_sel.index[0]
else:
    # 자동 선택: 오픈매장 & 별점↓ & 경쟁도↑ & 안정지수↓ 조건
    conds = [df["is_open"] == 1] if "is_open" in df.columns else []
    if "stars" in df.columns:
        conds.append(df["stars"] <= df["stars"].quantile(0.3))
    if "neighbor_density" in df.columns:
        conds.append(df["neighbor_density"] >= df["neighbor_density"].quantile(0.7))
    if "stability_score" in df.columns:
        conds.append(df["stability_score"] <= df["stability_score"].quantile(0.3))
    # ✅ 리뷰 100개 이상 필터
    if "review_count" in df.columns:
        conds.append(df["review_count"] >= 100)

    candidates = df[np.logical_and.reduce(conds)] if conds else df.copy()
    if candidates.empty:
        candidates = df[(df.get("is_open", 1) == 1) & (df.get("review_count", 0) >= 100)].nsmallest(50, "stability_score") if "stability_score" in df.columns else df[(df.get("is_open", 1) == 1) & (df.get("review_count", 0) >= 100)].sample(50, random_state=42)

    X_cand  = X_all.loc[candidates.index]
pool_cd = Pool(X_cand, cat_features=cat_feature_indices)
proba_cand = model.predict_proba(pool_cd)[:, 1]

# ===== 선택 전략 =====
np.random.seed(RANDOM_STATE if RANDOM_STATE is not None else None)

# 이전에 선택한 business_id 제외(가능할 때)
exclude_ids = set()
if EXCLUDE_IDS_PATH.exists():
    try:
        with open(EXCLUDE_IDS_PATH, "r", encoding="utf-8") as f:
            exclude_ids = set([line.strip() for line in f if line.strip()])
    except Exception:
        exclude_ids = set()

cand_index = X_cand.index
if "business_id" in df.columns and len(exclude_ids) > 0:
    keep_mask = ~df.loc[cand_index, "business_id"].astype(str).isin(exclude_ids)
    if keep_mask.any():
        cand_index = cand_index[keep_mask]
        proba_cand = proba_cand[keep_mask.values]

if len(cand_index) == 0:
    cand_index = X_cand.index  # 모두 제외됐다면 원복

if PICK_MODE == "argmax":
    sel_pos = int(np.argmax(proba_cand))
    idx = int(cand_index[sel_pos])
elif PICK_MODE == "random_topk":
    # 상위 TOPK 안에서 무작위 선택(후보 수가 TOPK보다 작으면 가능한 범위)
    order = np.argsort(-proba_cand)
    k = min(TOPK, len(order))
    top_idx = order[:k]
    sel_pos = np.random.choice(top_idx)
    idx = int(cand_index[sel_pos])
elif PICK_MODE == "weighted_softmax":
    # 소프트맥스 가중치 샘플링
    # 온도 T: p_i ∝ exp(logit_i / T); 여기서는 logit 대신 확률을 사용
    # 확률이 큰 항목의 선택 확률을 높임
    logits = proba_cand / max(TEMPERATURE, 1e-6)
    # 안정적 softmax
    m = np.max(logits)
    w = np.exp(logits - m)
    w = w / (w.sum() + 1e-12)
    sel_pos = np.random.choice(np.arange(len(cand_index)), p=w)
    idx = int(cand_index[sel_pos])
elif PICK_MODE == "pure_random":
    idx = int(np.random.choice(cand_index))
else:
    # 알 수 없는 모드면 안전하게 argmax
    sel_pos = int(np.argmax(proba_cand))
    idx = int(cand_index[sel_pos])

# 선택한 business_id 기록(중복 방지용)
try:
    if "business_id" in df.columns:
        picked_id = str(df.loc[idx, "business_id"])  # type: ignore
        with open(EXCLUDE_IDS_PATH, "a", encoding="utf-8") as f:
            f.write(picked_id + "\n")

except Exception:
    pass

# =========================
# 5) 개별 예측 + SHAP
# =========================
x_row = X_all.loc[[idx]].copy()
pool_row = Pool(x_row, cat_features=cat_feature_indices)
proba = float(model.predict_proba(pool_row)[:, 1][0])

thr = 0.5 if BEST_THR is None else float(BEST_THR)
pred = int(proba >= thr)

# CatBoost ShapValues: (n_samples, n_features+1) 마지막열 base logit
shap_vals_full = model.get_feature_importance(type="ShapValues", data=pool_row)
base_logit = float(shap_vals_full[0, -1])
shap_vals  = shap_vals_full[:, :-1]
contrib    = pd.Series(shap_vals[0], index=x_row.columns).sort_values(ascending=False)

# 확률 변화(%포인트) 근사
base_prob = sigmoid(base_logit)

def shap_to_pct_point(s):
    new_prob = sigmoid(base_logit + s)
    return (new_prob - base_prob) * 100.0

impact_pct = contrib.apply(shap_to_pct_point)

# 메타 수집
meta_cols = [
    "business_id", "name", "state", "city", "stars", "review_count",
    "stability_score", "loyalty_score", "reliability_score", "neighbor_density"
]
meta_cols = [c for c in meta_cols if c in df.columns]
meta = df.loc[idx, meta_cols].to_dict()

# 점수 스케일 추가
score_dict = to_scores(proba)

# =========================
# 6) 전체 분포 예측 → 백분위(상위 몇 %) 계산
# =========================
percentile_val = None
rank_pct = None

if COMPUTE_GLOBAL_DIST:
    pool_all = Pool(X_all, cat_features=cat_feature_indices)
    all_proba = model.predict_proba(pool_all)[:, 1]
    dist_df = df[["business_id"]].copy() if "business_id" in df.columns else pd.DataFrame(index=df.index)
    dist_df["pred_proba"] = all_proba
    # 백분율 랭크(낮음→높음)
    dist_df["risk_percentile"] = dist_df["pred_proba"].rank(pct=True) * 100.0
    # 현재 idx 위치 값
    rank_pct = float(dist_df.loc[idx, "risk_percentile"]) if idx in dist_df.index else None
    percentile_val = 100.0 - rank_pct if rank_pct is not None else None  # 상위 x% 해석용(큰 확률이 상위)

    # 점수 스케일도 같이 저장
    dist_df["score_simple"] = (1 - dist_df["pred_proba"]) * 100.0
    dist_df["score_log"]    = (1 - np.sqrt(dist_df["pred_proba"])) * 100.0

    dist_df.to_csv("one_store_distribution.csv", index=False, encoding="utf-8-sig")

# =========================
# 7) 결과 CSV 저장 (개별 식당 기준 테이블)
# =========================
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

# 점수/백분위 메타 열 추가
out.insert(5, "score_simple", score_dict["score_simple"])  # 100×(1−p)
out.insert(6, "score_log", score_dict["score_log"])        # 100×(1−√p)
if rank_pct is not None:
    out.insert(7, "risk_percentile", rank_pct)              # 큰 값일수록 위험 상위
    out.insert(8, "risk_top_percent", 100.0 - rank_pct)     # "상위 몇 %"(작을수록 위험 상위)

# 메타 추가
for k, v in meta.items():
    out[k] = v

out.to_csv("one_store_explain.csv", index=False, encoding="utf-8-sig")

# 간단 콘솔 요약
print("\n==== [개별 식당 폐업 위험 예측] ====")
print(f"- 대상 인덱스: {idx}")
for k, v in meta.items():
    print(f"  {k}: {v}")
print(f"- 예측 폐업 확률: {proba:.4f}  (점수: simple={score_dict['score_simple']:.1f}, log={score_dict['score_log']:.1f})")
print(f"- 판정(thr={thr:.2f}): {'폐업(1)' if pred==1 else '비폐업(0)'}")
if rank_pct is not None:
    print(f"- 위험 백분위 랭크: {rank_pct:.2f}%  → 상위 {100.0-rank_pct:.2f}% 위험")

print("\n[로컬 SHAP 상위 기여 요인 (폐업↑)]")
for k, v in contrib.head(TOP_N).items():
    print(f"  {k:30s} {v:+.5f}  (impact ~ {shap_to_pct_point(v):+.2f}%p)")

print("\n[로컬 SHAP 하위 기여 요인 (폐업↓)]")
for k, v in contrib.tail(TOP_N).items():
    print(f"  {k:30s} {v:+.5f}  (impact ~ {shap_to_pct_point(v):+.2f}%p)")

print("\n✅ 저장 완료: one_store_explain.csv")
if COMPUTE_GLOBAL_DIST:
    print("✅ 저장 완료: one_store_distribution.csv (전체 분포 + 백분위)")

# (선택) 상위 10개 영향도 표 간단 출력
print("\n[확률 변화 기준 Top 10 요인 (%p)]")
tmp = out[["feature","impact_pct_point"]].copy().sort_values("impact_pct_point", ascending=False).head(10)
for _, r in tmp.iterrows():
    print(f"  {r['feature']:30s} {r['impact_pct_point']:+.2f}%p")

# =========================
# 8) NLP 파이프라인 (단일 식당)
# =========================

OUT_DIR_NLP.mkdir(exist_ok=True, parents=True)

# 대상 business_id 확정
biz_id = meta.get("business_id") if "business_id" in meta else (df.loc[idx, "business_id"] if "business_id" in df.columns else None)
if biz_id is None:
    raise ValueError("df11에 business_id 컬럼이 없어 NLP 파이프라인을 실행할 수 없습니다.")

# 메타 & 카테고리
c_cols = [c for c in df.columns if str(c).startswith("c_")]
state  = df.loc[idx, "state"] if "state" in df.columns else None

# 리뷰 로드
reviews = pd.read_parquet(REVIEWS_PATH)
ensure_cols(reviews, {"business_id", "date", "text"}, "reviews")
reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
reviews = reviews.dropna(subset=["date", "text"]).copy()
reviews["text"] = reviews["text"].astype(str)

# 피어 선정: 같은 주 + 카테고리 1개 이상 겹침
meta_simple = df[["business_id", "state"] + c_cols].copy()
row_c = df.loc[idx]
match_vec = [1 if row_c.get(c, 0) == 1 else 0 for c in c_cols]
peers = meta_simple[
    (meta_simple["state"] == state) &
    (meta_simple[c_cols].mul(match_vec, axis=1).sum(axis=1) >= 1) &
    (meta_simple["business_id"] != biz_id)
]["business_id"].tolist()

target_df = reviews[reviews["business_id"] == biz_id].copy()
peer_df   = reviews[reviews["business_id"].isin(peers)].copy()

if target_df.empty:
    raise ValueError("해당 business_id의 리뷰가 없습니다 (reviews 확인).")
if peer_df.empty:
    if state and "state" in df.columns:
        other_ids = df[(df["state"] == state) & (df["business_id"] != biz_id)]["business_id"].tolist()
        peer_df = reviews[reviews["business_id"].isin(other_ids)].copy()
    if peer_df.empty:
        peer_df = reviews.sample(min(len(reviews), 5000), random_state=42).copy()

# 저장
target_df.to_csv(OUT_DIR_NLP / "target_reviews.csv", index=False, encoding="utf-8-sig")
peer_df.to_csv(OUT_DIR_NLP / "peer_reviews.csv", index=False, encoding="utf-8-sig")
df.loc[[idx]].to_csv(OUT_DIR_NLP / "target_meta.csv", index=False, encoding="utf-8-sig")

# 감성 + 길이 + 월별 트렌드
analyzer = SentimentIntensityAnalyzer()

def vsent(s):
    d = analyzer.polarity_scores(str(s))
    return d["compound"]

for dfx in (target_df, peer_df):
    dfx["sentiment"] = dfx["text"].apply(vsent)
    dfx["review_len"] = dfx["text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split()))
    dfx["ym"] = pd.to_datetime(dfx["date"]).dt.to_period("M").astype(str)

trend = (target_df.groupby("ym")
         .agg(avg_sentiment=("sentiment","mean"),
              avg_length=("review_len","mean"),
              n=("text","count"))
         .reset_index())
trend.to_csv(OUT_DIR_NLP / "monthly_trend.csv", index=False, encoding="utf-8-sig")

# 부정 리뷰만 n-gram → 로그오즈

def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    toks = [t for t in s.split() if 3 <= len(t) <= 20]
    return toks

def ngram_counts(texts, n=2):
    cnt = Counter()
    for t in texts:
        toks = tokenize(t)
        grams = zip(*[toks[i:] for i in range(n)])
        cnt.update([" ".join(g) for g in grams])
    return cnt

def log_odds(target_cnt: Counter, peer_cnt: Counter, k=1.0) -> pd.DataFrame:
    vocab = set(target_cnt) | set(peer_cnt)
    t_total = sum(target_cnt.values()) + k * len(vocab)
    p_total = sum(peer_cnt.values()) + k * len(vocab)
    rows = []
    for w in vocab:
        t = target_cnt.get(w, 0) + k
        p = peer_cnt.get(w, 0) + k
        lo = math.log((t/(t_total - t)) / (p/(p_total - p)))
        rows.append((w, t - k, p - k, lo))
    df_lo = pd.DataFrame(rows, columns=["ngram","count_target","count_peer","log_odds"])\
            .sort_values("log_odds", ascending=False)
    return df_lo

neg_thr = -0.05
nt = target_df[target_df["sentiment"] <= neg_thr]
np_ = peer_df[peer_df["sentiment"] <= neg_thr]

bi_t  = ngram_counts(nt["text"], n=2)
bi_p  = ngram_counts(np_["text"], n=2)
tri_t = ngram_counts(nt["text"], n=3)
tri_p = ngram_counts(np_["text"], n=3)

bi_lo  = log_odds(bi_t, bi_p, k=1.0)
tri_lo = log_odds(tri_t, tri_p, k=1.0)

bi_lo.to_csv(OUT_DIR_NLP / "logodds_bigram_neg.csv",   index=False, encoding="utf-8-sig")
tri_lo.to_csv(OUT_DIR_NLP / "logodds_trigram_neg.csv", index=False, encoding="utf-8-sig")

# 측면(Aspect) 점수
aspects = {
    "service": ["service","server","staff","wait","rude","attitude","slow","attentive","friendly"],
    "taste":   ["taste","flavor","bland","salty","sweet","soggy","fresh","overcooked","undercooked","seasoning"],
    "price":   ["price","expensive","overpriced","cheap","worth","value","portion"],
    "clean":   ["clean","dirty","smell","sticky","sanitary","restroom","hair"],
    "speed":   ["slow","fast","wait","line","delay","quick","time"],
}
rows_aspect = []
for asp, kws in aspects.items():
    pattern = "|".join([re.escape(k) for k in kws])
    mask = target_df["text"].str.contains(pattern, case=False, na=False)
    sub = target_df[mask]
    rows_aspect.append({
        "aspect": asp,
        "n_reviews": len(sub),
        "avg_sentiment": sub["sentiment"].mean() if len(sub) else np.nan,
        "example": sub["text"].iloc[0][:200].replace("\n", " ") if len(sub) else ""
    })
aspect_df = pd.DataFrame(rows_aspect).sort_values("avg_sentiment")
aspect_df.to_csv(OUT_DIR_NLP / "aspect_scores.csv", index=False, encoding="utf-8-sig")

# 예시 문장 (부정/긍정 각 10개)
ex_neg = target_df.sort_values("sentiment").head(10)[["date","sentiment","text"]]
ex_pos = target_df.sort_values("sentiment").tail(10)[["date","sentiment","text"]]
ex_neg.to_csv(OUT_DIR_NLP / "examples_negative.csv", index=False, encoding="utf-8-sig")
ex_pos.to_csv(OUT_DIR_NLP / "examples_positive.csv", index=False, encoding="utf-8-sig")

# 월별 감성 차트(선택)
if not trend.empty:
    plt.figure(figsize=(7,4))
    plt.plot(trend["ym"], trend["avg_sentiment"], marker="o")
    plt.xticks(rotation=60)
    plt.title("Monthly Sentiment (Target)")
    plt.tight_layout()
    plt.savefig(OUT_DIR_NLP / "monthly_sentiment.png", dpi=150)

# 요약 카드
meta_series = df.loc[idx]
if not trend.empty:
    tmp = trend.copy()
    tmp["ym"] = pd.PeriodIndex(tmp["ym"], freq="M").to_timestamp()
    recent_3m = tmp.sort_values("ym").tail(3)["avg_sentiment"].mean()
else:
    recent_3m = np.nan

worst_aspect = ""
worst_aspect_sent = np.nan
if not aspect_df.empty and aspect_df["avg_sentiment"].notna().any():
    wr = aspect_df.sort_values("avg_sentiment").iloc[0]
    worst_aspect = wr["aspect"]
    worst_aspect_sent = float(wr["avg_sentiment"]) if pd.notna(wr["avg_sentiment"]) else np.nan

# 불만 Top 키워드 (bi/tri 혼합 상위 5)
_top = (pd.concat([
            bi_lo.sort_values("log_odds", ascending=False).head(5).assign(n=2),
            tri_lo.sort_values("log_odds", ascending=False).head(5).assign(n=3),
        ], ignore_index=True)
        .sort_values(["log_odds"], ascending=False)
        .head(5))
complaints_str = "; ".join(_top["ngram"].tolist())

card = {
    "name": meta_series.get("name", ""),
    "state": meta_series.get("state", ""),
    "city": meta_series.get("city", ""),
    "stars": meta_series.get("stars", ""),
    "review_count": meta_series.get("review_count", ""),
    "stability_score": meta_series.get("stability_score", ""),
    "loyalty_score": meta_series.get("loyalty_score", ""),
    "reliability_score": meta_series.get("reliability_score", ""),
    "recent_3m_sentiment": recent_3m,
    "worst_aspect": worst_aspect,
    "worst_aspect_sent": worst_aspect_sent,
    "top_complaints": complaints_str,
    "pred_proba": proba,
    "score_simple": score_dict["score_simple"],
    "score_log": score_dict["score_log"],
}
if rank_pct is not None:
    card["risk_percentile"] = rank_pct
    card["risk_top_percent"] = 100.0 - rank_pct

pd.DataFrame([card]).to_csv(OUT_DIR_NLP / "nlp_summary_card.csv", index=False, encoding="utf-8-sig")

print("\n✅ NLP 완료: 출력 경로 =", OUT_DIR_NLP.resolve())
print("- target_reviews.csv / peer_reviews.csv")
print("- monthly_trend.csv / monthly_sentiment.png")
print("- logodds_bigram_neg.csv / logodds_trigram_neg.csv")
print("- aspect_scores.csv")
print("- examples_negative.csv / examples_positive.csv")
print("- nlp_summary_card.csv")

try:
    r3 = f"{recent_3m:.3f}"
except Exception:
    r3 = "nan"

print(f"\n요약: {card['name']}({card['state']}, {card['city']}) | 최근3개월 감성={r3} | "
      f"최악측면={card['worst_aspect']}({card['worst_aspect_sent']}) | "
      f"불만Top: {card['top_complaints']}")