# -*- coding: utf-8 -*-
"""
run_nlp_one_store.py
- 단일 식당 NLP 파이프라인 (추출 → 감성/로그오즈/측면 → 요약카드) 일괄 실행
- peer = 같은 주(state) + 같은 카테고리(c_* ≥1 겹침)의 다른 식당
- 입력: df11.csv, review.csv (CSV만 사용)
- 대상 business_id 하드코딩: "1k6gLCvblOMzFLYeb6CPAQ"
- 출력: nlp_out/ 폴더에 결과 CSV/PNG 저장
"""

import re
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# 고정 설정 (CSV 경로 + 대상 식당)
# =========================
BUSINESS_ID = "5Md0YaxD5HiOoBmsnmIu7A"
DF11_PATH   = "df11.csv"      # 메타/카테고리/지수 포함
REVIEWS = "review_filtered.parquet"    # 컬럼: business_id,date,text,(선택)user_id,stars 등
OUT_DIR     = Path("nlp_out")

# =========================
# Util
# =========================
def ensure_cols(df: pd.DataFrame, need: set, name: str):
    miss = need - set(df.columns.astype(str))
    if miss:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(miss)}")

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
    df = pd.DataFrame(rows, columns=["ngram", "count_target", "count_peer", "log_odds"])
    return df.sort_values("log_odds", ascending=False)

# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # 0) 데이터 로드 (CSV 고정)
    df11 = pd.read_csv(DF11_PATH)
    ensure_cols(df11, {"business_id", "state"}, "df11")
    c_cols = [c for c in df11.columns if str(c).startswith("c_")]

    reviews = pd.read_parquet(REVIEWS)
    ensure_cols(reviews, {"business_id", "date", "text"}, "review.csv")
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date", "text"]).copy()
    reviews["text"] = reviews["text"].astype(str)

    # 1) 타깃/피어 선택
    row = df11.loc[df11["business_id"] == BUSINESS_ID]
    if row.empty:
        raise ValueError("df11에 해당 business_id가 없습니다.")
    row = row.iloc[0]  # Series
    state = row.get("state", None)
    target_cats = [c for c in c_cols if row.get(c, 0) == 1]

    meta_simple = df11[["business_id", "state"] + c_cols].copy()
    peers = meta_simple[
        (meta_simple["state"] == state) &
        (meta_simple[c_cols].mul([1 if c in target_cats else 0 for c in c_cols], axis=1).sum(axis=1) >= 1) &
        (meta_simple["business_id"] != BUSINESS_ID)
    ]["business_id"].tolist()

    target_df = reviews[reviews["business_id"] == BUSINESS_ID].copy()
    peer_df   = reviews[reviews["business_id"].isin(peers)].copy()

    if target_df.empty:
        raise ValueError("해당 business_id의 리뷰가 없습니다 (review.csv 확인).")
    if peer_df.empty:
        # 피어 없으면 같은 주(state) 전체로 보정 → 그래도 없으면 전체 샘플에서 일부
        if state and "state" in df11.columns:
            other_ids = df11[(df11["state"] == state) & (df11["business_id"] != BUSINESS_ID)]["business_id"].tolist()
            peer_df = reviews[reviews["business_id"].isin(other_ids)].copy()
        if peer_df.empty:
            peer_df = reviews.sample(min(len(reviews), 5000), random_state=42).copy()

    # 저장
    target_df.to_csv(OUT_DIR / "target_reviews.csv", index=False, encoding="utf-8-sig")
    peer_df.to_csv(OUT_DIR / "peer_reviews.csv", index=False, encoding="utf-8-sig")
    row.to_frame().T.to_csv(OUT_DIR / "target_meta.csv", index=False, encoding="utf-8-sig")

    # 2) 감성 + 길이 + 월별 트렌드
    analyzer = SentimentIntensityAnalyzer()
    def vsent(s):
        d = analyzer.polarity_scores(str(s))
        return d["compound"]

    for df in (target_df, peer_df):
        df["sentiment"] = df["text"].apply(vsent)
        df["review_len"] = df["text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split()))
        df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    trend = (target_df.groupby("ym")
             .agg(avg_sentiment=("sentiment", "mean"),
                  avg_length=("review_len", "mean"),
                  n=("text", "count"))
             .reset_index())
    trend.to_csv(OUT_DIR / "monthly_trend.csv", index=False, encoding="utf-8-sig")

    # 3) 로그오즈 (부정 리뷰만)
    target_neg = target_df[target_df["sentiment"] <= -0.05]
    peer_neg   = peer_df[peer_df["sentiment"] <= -0.05]

    bi_t  = ngram_counts(target_neg["text"], n=2)
    bi_p  = ngram_counts(peer_neg["text"], n=2)
    tri_t = ngram_counts(target_neg["text"], n=3)
    tri_p = ngram_counts(peer_neg["text"], n=3)

    bi_lo  = log_odds(bi_t, bi_p, k=1.0)
    tri_lo = log_odds(tri_t, tri_p, k=1.0)

    bi_lo.to_csv(OUT_DIR / "logodds_bigram_neg.csv",   index=False, encoding="utf-8-sig")
    tri_lo.to_csv(OUT_DIR / "logodds_trigram_neg.csv", index=False, encoding="utf-8-sig")

    # 4) 측면(Aspect) 점수
    aspects = {
        "service": ["service","server","staff","wait","rude","attitude","slow","attentive","friendly"],
        "taste":   ["taste","flavor","bland","salty","sweet","soggy","fresh","overcooked","undercooked","seasoning"],
        "price":   ["price","expensive","overpriced","cheap","worth","value","portion"],
        "clean":   ["clean","dirty","smell","sticky","sanitary","restroom","hair"],
        "speed":   ["slow","fast","wait","line","delay","quick","time"]
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
    aspect_df.to_csv(OUT_DIR / "aspect_scores.csv", index=False, encoding="utf-8-sig")

    # 5) 예시 문장 (부정/긍정 각 10개)
    ex_neg = target_df.sort_values("sentiment").head(10)[["date", "sentiment", "text"]]
    ex_pos = target_df.sort_values("sentiment").tail(10)[["date", "sentiment", "text"]]
    ex_neg.to_csv(OUT_DIR / "examples_negative.csv", index=False, encoding="utf-8-sig")
    ex_pos.to_csv(OUT_DIR / "examples_positive.csv", index=False, encoding="utf-8-sig")

    # 6) 월별 감성 차트
    if not trend.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(trend["ym"], trend["avg_sentiment"], marker="o")
        plt.xticks(rotation=60)
        plt.title("Monthly Sentiment (Target)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "monthly_sentiment.png", dpi=150)

    # 7) 요약 카드
    meta = row.to_dict()
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
        worst_aspect_sent = float(wr["avg_sentiment"])

    top_complaints = (pd.concat([
        bi_lo.sort_values("log_odds", ascending=False).head(5).assign(n=2),
        tri_lo.sort_values("log_odds", ascending=False).head(5).assign(n=3),
    ], ignore_index=True)
     .sort_values(["log_odds"], ascending=False)
     .head(5))
    complaints_str = "; ".join(top_complaints["ngram"].tolist())

    card = {
        "name": meta.get("name", ""),
        "state": meta.get("state", ""),
        "city": meta.get("city", ""),
        "stars": meta.get("stars", ""),
        "review_count": meta.get("review_count", ""),
        "stability_score": meta.get("stability_score", ""),
        "loyalty_score": meta.get("loyalty_score", ""),
        "reliability_score": meta.get("reliability_score", ""),
        "recent_3m_sentiment": recent_3m,
        "worst_aspect": worst_aspect,
        "worst_aspect_sent": worst_aspect_sent,
        "top_complaints": complaints_str
    }
    pd.DataFrame([card]).to_csv(OUT_DIR / "nlp_summary_card.csv", index=False, encoding="utf-8-sig")

    # 콘솔 요약
    print("\n✅ 완료: 출력 경로 =", OUT_DIR.resolve())
    print(f"- target_reviews.csv / peer_reviews.csv")
    print(f"- monthly_trend.csv / monthly_sentiment.png")
    print(f"- logodds_bigram_neg.csv / logodds_trigram_neg.csv")
    print(f"- aspect_scores.csv")
    print(f"- examples_negative.csv / examples_positive.csv")
    print(f"- nlp_summary_card.csv")
    # 깔끔 출력
    try:
        r3 = f"{card['recent_3m_sentiment']:.3f}"
    except Exception:
        r3 = "nan"
    print(f"\n요약: {card['name']}({card['state']}, {card['city']}) | 최근3개월 감성={r3} | "
          f"최악측면={card['worst_aspect']}({card['worst_aspect_sent']}) | "
          f"불만Top: {card['top_complaints']}")

if __name__ == "__main__":
    main()
