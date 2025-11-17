from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# =========================
# 파라미터 설정
# =========================
K_PRE = 20
M_EVENT = 3
TAU_DROP = 0.8
ROLL_POST = 5
DELTA_RECOV = 0.2
SUSTAIN_N = 5
MAX_DAYS = 180

W_TIME = 0.6
W_EXTENT = 0.4
ALPHA = 0.5
MIN_REVIEWS = 25
LOW_STAR_THRESHOLD = 2.0

# =========================
# 파일 경로
# =========================
BUSINESS_FN = "business.parquet"
REVIEW_FN   = "review.parquet"
base_dir = Path(".")

# =========================
# 데이터 로드
# =========================
business = pd.read_parquet(base_dir / BUSINESS_FN)
review = pd.read_parquet(base_dir / REVIEW_FN)

# 날짜 처리
if "date" not in review.columns:
    for c in ["review_date", "created_at", "timestamp"]:
        if c in review.columns:
            review["date"] = pd.to_datetime(review[c], errors="coerce")
            break
if "date" not in review.columns:
    raise ValueError("review 데이터에 날짜(date) 컬럼이 필요합니다.")

review["date"] = pd.to_datetime(review["date"], errors="coerce")
review = review.dropna(subset=["date"]).sort_values(["business_id", "date"])

# =========================
# 기본 집계 + 리뷰 25개 이상 필터
# =========================
agg = (
    review.groupby("business_id")
    .agg(
        review_count=("stars", "size"),
        stars_mean=("stars", "mean"),
        stars_std=("stars", "std"),
        low_star_count=("stars", lambda s: (s <= LOW_STAR_THRESHOLD).sum())
    )
    .reset_index()
)
agg["stars_std"] = agg["stars_std"].fillna(0.0)
agg["low_star_ratio"] = agg["low_star_count"] / agg["review_count"]

# 🔹 25개 이상 필터 적용
agg = agg[agg["review_count"] >= MIN_REVIEWS]
valid_ids = set(agg["business_id"])
review = review[review["business_id"].isin(valid_ids)]

# =========================
# 쇼크 탐지 및 회복력 계산
# =========================
def shock_recovery_metrics(g: pd.DataFrame) -> pd.Series:
    s = g["stars"].astype(float).to_numpy()
    d = g["date"].to_numpy()
    n = len(s)
    pre_mean = pd.Series(s).rolling(K_PRE, min_periods=K_PRE).mean().to_numpy()
    post_meanM = pd.Series(s).rolling(M_EVENT, min_periods=M_EVENT).mean().to_numpy()
    roll_post = pd.Series(s).rolling(ROLL_POST, min_periods=ROLL_POST).mean().to_numpy()

    shock_idx = None
    for i in range(1, n):
        if np.isfinite(pre_mean[i-1]) and np.isfinite(post_meanM[i]):
            if (pre_mean[i-1] - post_meanM[i]) >= TAU_DROP:
                shock_idx = i
                break

    if shock_idx is None:
        return pd.Series({"S_time": 0.7, "S_elas": 0.7})

    mu_pre = float(pre_mean[shock_idx-1])
    mu_event = float(post_meanM[shock_idx])
    t0 = pd.to_datetime(d[shock_idx])
    max_h = pd.Timedelta(days=MAX_DAYS)

    # 회복 탐색
    post_series = pd.Series(s[shock_idx:])
    post_vals = post_series.rolling(ROLL_POST, min_periods=ROLL_POST).mean().to_numpy()
    post_idx_offset = shock_idx + (ROLL_POST - 1)
    t_rec = None

    for j in range(len(post_vals)):
        if np.isnan(post_vals[j]): continue
        idx_global = post_idx_offset + j
        if idx_global >= n: break
        mu_post = post_vals[j]
        if abs(mu_post - mu_pre) <= DELTA_RECOV:
            ok, cnt, k = True, 1, j + 1
            while cnt < SUSTAIN_N:
                if k >= len(post_vals) or np.isnan(post_vals[k]): ok = False; break
                idx_gk = post_idx_offset + k
                if idx_gk >= n or abs(post_vals[k] - mu_pre) > DELTA_RECOV:
                    ok = False; break
                cnt += 1; k += 1
            if ok:
                t_rec = pd.to_datetime(d[idx_global])
                break
        idx_for_time = min(shock_idx + j, n-1)
        if (pd.to_datetime(d[idx_for_time]) - t0) > max_h: break

    # 속도 성분
    if t_rec is not None:
        days = (t_rec - t0) / np.timedelta64(1, "D")
    else:
        days = MAX_DAYS
    S_time = 1.0 / (1.0 + days / 60.0)

    # 탄성 성분
    delta_drop = mu_pre - mu_event
    horizon_end = t0 + max_h
    mask = (d >= t0) & (d <= horizon_end)
    post_h = s[mask]
    if post_h.size >= ROLL_POST:
        post_h_roll = pd.Series(post_h).rolling(ROLL_POST, min_periods=ROLL_POST).mean().to_numpy()
        valid = post_h_roll[~np.isnan(post_h_roll)]
        if valid.size > 0:
            mu_max = float(valid.max())
            S_elas = np.clip((mu_max - mu_event) / max(delta_drop, 1e-6), 0, 1)
        else:
            S_elas = 0.0
    else:
        S_elas = 0.0

    return pd.Series({"S_time": float(S_time), "S_elas": float(S_elas)})

metrics = review.groupby("business_id", as_index=False).apply(shock_recovery_metrics, include_groups=False)

# =========================
# BaseScore (5~95 분위수 클리핑)
# =========================
feat = agg.merge(metrics, on="business_id", how="left")
feat[["S_time","S_elas"]] = feat[["S_time","S_elas"]].fillna({"S_time":0.7,"S_elas":0.7})

feat["BaseScore"] = (
    feat["stars_mean"] * np.log1p(feat["review_count"])
) * (1.0 / (1.0 + feat["stars_std"])) * (1.0 - feat["low_star_ratio"])

p5, p95 = np.nanpercentile(feat["BaseScore"], [5, 95])
clipped = feat["BaseScore"].clip(lower=p5, upper=p95)
feat["BaseScore_norm"] = ((clipped - p5) / max(p95 - p5, 1e-12)).clip(0, 1)

# =========================
# RecoveryScore (MinMaxScaler)
# =========================
feat["RecoveryRaw"] = W_TIME * feat["S_time"] + W_EXTENT * feat["S_elas"]
scaler = MinMaxScaler()
feat["RecoveryScore"] = scaler.fit_transform(feat[["RecoveryRaw"]])

# =========================
# StableIndex (100점 척도)
# =========================
feat["Base_100"] = 100 * ALPHA * feat["BaseScore_norm"]
feat["Recovery_100"] = 100 * (1 - ALPHA) * feat["RecoveryScore"]
feat["StableIndex"] = feat["Base_100"] + feat["Recovery_100"]

# =========================
# df1 저장 및 출력
# =========================
keep_cols = ["business_id","name","state","is_open","categories"]
df1 = (
    business[keep_cols]
    .merge(feat, on="business_id", how="inner")
    .sort_values("StableIndex", ascending=False)
)

df1.to_csv("df1.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.max_columns", None)
print(df1.head(30))
print(f"\n✅ df1 저장 완료: rows={len(df1):,}")
# =========================
# 📊 기본 점수 & 회복력 분포 시각화
# =========================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 🔹 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 🔹 기본 점수 분포
plt.figure(figsize=(10,5))
plt.hist(feat["BaseScore_norm"], bins=50, color="skyblue", edgecolor="gray")
plt.title("기본 신뢰도(BaseScore_norm) 분포", fontsize=14)
plt.xlabel("BaseScore_norm (0~1 정규화)", fontsize=12)
plt.ylabel("매장 수", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 🔹 회복력 분포
plt.figure(figsize=(10,5))
plt.hist(feat["RecoveryScore"], bins=50, color="lightcoral", edgecolor="gray")
plt.title("회복력(RecoveryScore) 분포", fontsize=14)
plt.xlabel("RecoveryScore (0~1 정규화)", fontsize=12)
plt.ylabel("매장 수", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

print("\n📈 분포 시각화 완료 — BaseScore_norm / RecoveryScore 모두 확인 가능.")
