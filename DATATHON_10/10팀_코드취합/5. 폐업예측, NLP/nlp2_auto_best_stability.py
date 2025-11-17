# -*- coding: utf-8 -*-
"""
nlp1_auto_best_stability.py

1) model3_logit_v2.csv에서
   - reliability_flag == "ok"
   - logit_w_stability 최댓값인 (state, category) 조합 1개 선택

2) df11에서 해당 state & category 더미=1인 식당만 사용
   - 세 지수(stability/reliability/loyalty) 합 상위 10% vs 나머지

3) 두 그룹 리뷰에서 긍정 trigram 추출 후 log-odds 계산

4) 전체 리뷰로 Word2Vec 학습 → trigram을 브랜드/메뉴/공간으로 분류

출력:
- {state}_{category}_trigram_logodds_top50.csv
- {state}_{category}_trigram_logodds_top50.png
- {state}_{category}_trigram_top50_labeled.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 0. 기본 설정
# =========================
DF11_FN   = "df11.csv"
REVIEW_FN = "review.parquet"   # review.csv면 아래에서 분기 처리 있음

STAB_COL = "stability_score"
REL_COL  = "reliability_score"
LOY_COL  = "loyalty_score"

POS_THRESHOLD = 0.3     # 긍정 리뷰 기준 (VADER compound)
TOP_Q         = 0.9     # 세 지수 합 상위 10%
MIN_TOP_COUNT = 5       # top 그룹에서 최소 등장 횟수
TOP_K         = 50      # log-odds 상위 몇 개 trigram까지 볼지

MODEL3_FN = "model3_logit_v2.csv"


# =========================
# 1. model3_logit_v2에서 최고 안정성 조합 선택
# =========================
print(f"📂 model3_logit_v2 로드 중: {MODEL3_FN}")
m3 = pd.read_csv(MODEL3_FN)
print(f"   총 행 수: {len(m3):,}")

req_cols_m3 = ["state", "category", "logit_w_stability", "reliability_flag"]
missing_m3 = [c for c in req_cols_m3 if c not in m3.columns]
if missing_m3:
    raise ValueError(f"❌ model3_logit_v2에 다음 컬럼이 없습니다: {missing_m3}")

ok_m3 = m3[m3["reliability_flag"] == "ok"].copy()
if ok_m3.empty:
    raise ValueError("❌ reliability_flag == 'ok' 인 행이 없습니다.")

best_idx = ok_m3["logit_w_stability"].idxmax()
best_row = ok_m3.loc[best_idx]

STATE_TARGET   = str(best_row["state"]).upper()
CATEGORY_NAME  = str(best_row["category"])
CAT_COL        = f"c_{CATEGORY_NAME}"

tag = f"{STATE_TARGET.lower()}_{CATEGORY_NAME.lower()}"

OUT_LOGODDS_CSV  = f"{tag}_trigram_logodds_top50.csv"
OUT_PNG          = f"{tag}_trigram_logodds_top50.png"
OUT_LABELED_CSV  = f"{tag}_trigram_top50_labeled.csv"

print("\n🏆 선택된 조합 (최고 logit_w_stability & reliability_flag='ok'):")
print(best_row[["state", "category", "logit_w_stability", "logit_w_reliability",
                "logit_w_loyalty", "reliability_flag"]])
print(f"\n🎯 대상 state: {STATE_TARGET}, category: {CATEGORY_NAME} (더미 컬럼: {CAT_COL})")


# =========================
# 2. NLTK 리소스 체크 & 기본 객체
# =========================
def ensure_nltk_resources():
    needed = [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("vader_lexicon", "sentiment/vader_lexicon"),
    ]
    for pkg, path in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

ensure_nltk_resources()

# trigram용 stopwords: 기능어 위주 제거, 감정 단어는 살림
stop_words = set(stopwords.words("english"))
emotion_keep = {"good", "great", "love", "amazing", "best", "really", "so", "very", "favorite"}
stop_words = stop_words - emotion_keep

sia = SentimentIntensityAnalyzer()


# =========================
# 3. 긍정 trigram 추출 함수
# =========================
def get_positive_trigrams(df_reviews, text_col="text",
                          pos_threshold=0.3, stop_words=None):
    """
    - 리뷰 텍스트 중 VADER compound > pos_threshold 인 긍정 리뷰만 사용
    - 토큰화 + stopword 제거 후 trigram 생성
    """
    if stop_words is None:
        stop_words = set()

    trigram_list = []
    n_pos = 0

    for text in df_reviews[text_col]:
        if not isinstance(text, str):
            continue

        score = sia.polarity_scores(text)["compound"]
        if score <= pos_threshold:
            continue

        n_pos += 1

        tokens = [
            w.lower()
            for w in word_tokenize(text)
            if w.isalpha() and w.lower() not in stop_words
        ]

        trigram_list.extend(list(ngrams(tokens, 3)))

    print(f"   긍정 리뷰 수: {n_pos:,} / trigram 수: {len(trigram_list):,}")
    return trigram_list


# =========================
# 4. df11: 선택된 state & category 더미=1 상위10% / 나머지 분리
# =========================
print(f"\n📂 df11 로드 중: {DF11_FN}")
df = pd.read_csv(DF11_FN)
print(f"   총 행 수: {len(df):,}")

required_cols_df11 = ["business_id", "state", STAB_COL, REL_COL, LOY_COL, CAT_COL]
missing_df11 = [c for c in required_cols_df11 if c not in df.columns]
if missing_df11:
    raise ValueError(f"❌ df11에 다음 컬럼이 없습니다: {missing_df11}")

mask_state = df["state"].astype(str).str.upper() == STATE_TARGET
mask_cat   = df[CAT_COL] == 1
df_target = df[mask_state & mask_cat].copy()

print(f"\n🎯 대상: {STATE_TARGET} & {CAT_COL}=1 식당 수: {len(df_target):,}")
if df_target.empty:
    raise ValueError("❌ 해당 state & category 더미=1인 식당이 없습니다.")

df_target["sum_3idx"] = (
    df_target[STAB_COL] +
    df_target[REL_COL] +
    df_target[LOY_COL]
)

cutoff = df_target["sum_3idx"].quantile(TOP_Q)
df_top  = df_target[df_target["sum_3idx"] >= cutoff].copy()
df_rest = df_target[df_target["sum_3idx"] < cutoff].copy()

print(f"   세 지수 합 상위 {int(TOP_Q*100)}% cutoff: {cutoff:.4f}")
print(f"   상위 그룹 매장 수: {len(df_top):,}")
print(f"   나머지 그룹 매장 수: {len(df_rest):,}")

top_ids  = df_top["business_id"].unique().tolist()
rest_ids = df_rest["business_id"].unique().tolist()


# =========================
# 5. 리뷰 로드 & 그룹 분리
# =========================
print(f"\n📂 리뷰 로드 중: {REVIEW_FN}")
if REVIEW_FN.lower().endswith(".parquet"):
    reviews = pd.read_parquet(REVIEW_FN)
else:
    reviews = pd.read_csv(REVIEW_FN)

req_rev_cols = ["business_id", "text"]
missing_rev = [c for c in req_rev_cols if c not in reviews.columns]
if missing_rev:
    raise ValueError(f"❌ 리뷰 파일에 다음 컬럼이 없습니다: {missing_rev}")

reviews_target = reviews[reviews["business_id"].isin(df_target["business_id"])].copy()
print(f"   대상 매장 리뷰 수: {len(reviews_target):,}")

reviews_top  = reviews_target[reviews_target["business_id"].isin(top_ids)].copy()
reviews_rest = reviews_target[reviews_target["business_id"].isin(rest_ids)].copy()

print(f"   ▶ 상위 그룹 리뷰 수: {len(reviews_top):,}")
print(f"   ▶ 나머지 그룹 리뷰 수: {len(reviews_rest):,}")

if reviews_top.empty or reviews_rest.empty:
    raise ValueError("❌ 상위/나머지 그룹 리뷰가 부족합니다.")


# =========================
# 6. trigram 추출 (top / rest)
# =========================
print("\n🧠 상위 그룹 trigram 추출 중...")
trigrams_top = get_positive_trigrams(
    reviews_top,
    text_col="text",
    pos_threshold=POS_THRESHOLD,
    stop_words=stop_words,
)

print("\n🧠 나머지 그룹 trigram 추출 중...")
trigrams_rest = get_positive_trigrams(
    reviews_rest,
    text_col="text",
    pos_threshold=POS_THRESHOLD,
    stop_words=stop_words,
)

if not trigrams_top or not trigrams_rest:
    raise ValueError("❌ trigram이 한쪽 그룹에서 비어 있습니다.")


# =========================
# 7. log-odds 계산
# =========================
print("\n📊 log-odds 계산 중...")

fdist_top  = Counter(trigrams_top)
fdist_rest = Counter(trigrams_rest)

all_trigrams = set(fdist_top.keys()) | set(fdist_rest.keys())

rows = []
for tg in all_trigrams:
    top_freq  = fdist_top[tg]
    rest_freq = fdist_rest[tg]

    if top_freq < MIN_TOP_COUNT:
        continue

    # 라플라스 스무딩
    top_s  = top_freq + 1
    rest_s = rest_freq + 1

    log_odds = np.log(top_s / rest_s)
    phrase = " ".join(tg)
    rows.append((tg, phrase, top_freq, rest_freq, log_odds))

df_logodds = pd.DataFrame(
    rows,
    columns=["trigram", "phrase", "count_top", "count_rest", "log_odds"]
).sort_values("log_odds", ascending=False).reset_index(drop=True)

print("\n=== 📊 log-odds 상위 10개 예시 ===")
print(df_logodds.head(10))

# 상위 50개만 저장 (분석용)
df_topk = df_logodds.head(TOP_K).copy()
df_topk.to_csv(OUT_LOGODDS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ log-odds 상위 {TOP_K}개 저장 완료: {OUT_LOGODDS_CSV}")

# 간단 시각화
plt.figure(figsize=(10, 6))
df_plot = df_topk.sort_values("log_odds", ascending=True)
plt.barh(df_plot["phrase"], df_plot["log_odds"])
plt.xlabel("log-odds (Top vs Rest)")
plt.ylabel("Trigram")
plt.title(f"{STATE_TARGET} - {CATEGORY_NAME} (더미: {CAT_COL}=1) - Top {TOP_K} trigram log-odds")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.show()
print(f"📁 그래프 저장 완료: {OUT_PNG}")


# =========================
# 8. Word2Vec 학습 (해당 state+category 전체 리뷰 기반)
# =========================
print("\n🧬 Word2Vec 학습용 코퍼스 준비 중...")

sentences = []
for text in reviews_target["text"]:
    if not isinstance(text, str):
        continue
    tokens = [
        w.lower()
        for w in word_tokenize(text)
        if w.isalpha()
    ]
    if tokens:
        sentences.append(tokens)

print(f"   문장(리뷰) 수: {len(sentences):,}")

print("🧬 Word2Vec 학습 중...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,          # skip-gram
    epochs=10
)
print("✅ Word2Vec 학습 완료")


# =========================
# 9. trigram 임베딩 + 브랜드/메뉴/공간 분류
# =========================

# === 1) seed words (brand 제거, 4축만 사용) ===
menu_seeds  = ["ramen","soba","noodles","tikka","curry","dumpling","burger","wings",
               "pancakes","omelette","sandwich","salad","brunch","pho","bibimbap","taco"]
space_seeds = ["patio","view","inside","outside","rooftop","terrace","seating",
               "atmosphere","ambience","vibe","counter","bar","booth"]
ops_seeds   = ["seasonal","rotating","rotate","special","chef","daily","today",
               "weekly","tasting","limited","pop-up","prefix","prix","course"]
regi_seeds  = ["korean","japanese","thai","vietnamese","mexican","sicilian","tuscan",
               "philly","chicago","nashville","texas","cajun","bavarian","peruvian",
               "savoy","neapolitan","szechuan","hunan"]

def get_seed_vector(seed_words, model):
    vecs = [model.wv[w] for w in seed_words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else None

menu_vec  = get_seed_vector(menu_seeds,  w2v_model)
space_vec = get_seed_vector(space_seeds, w2v_model)
ops_vec   = get_seed_vector(ops_seeds,   w2v_model)
regi_vec  = get_seed_vector(regi_seeds,  w2v_model)

# === 2) 룰/키워드 패턴 (operation & regional) ===
OPS_PATTERNS = [
    "menu changes", "rotating menu", "seasonal menu", "chef special", "chef's choice",
    "tasting menu", "daily special", "today special", "limited time", "limited edition",
    "weekly special", "prix fixe", "prefix menu", "course menu", "pop-up", "only on", "weekend only"
]
REGI_PATTERNS = [
    # 국적/지역/도시/스타일 키워드(필요시 추가)
    "korean", "japanese", "thai", "vietnamese", "mexican", "sicilian", "tuscan",
    "philly", "philadelphia", "chicago", "nashville", "texas", "tex-mex", "cajun",
    "neapolitan", "szechuan", "hunan", "bavarian", "peruvian", "hawaiian"
]

def has_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)

from sklearn.metrics.pairwise import cosine_similarity

# === 3) 4축 분류기 (brand 제거) ===
def classify_trigram_four(phrase: str, model, menu_vec, space_vec, ops_vec, regi_vec):
    toks = phrase.split()
    vecs = [model.wv[t] for t in toks if t in model.wv]
    # 임베딩이 전혀 없으면 룰로만 판단
    if not vecs:
        if has_any(phrase, OPS_PATTERNS):  return "operation", None, None, None, None
        if has_any(phrase, REGI_PATTERNS): return "regional",  None, None, None, None
        # 메뉴/공간은 룰 정의가 애매하면 unknown 처리
        return "unknown", None, None, None, None

    v = np.mean(vecs, axis=0)
    sim = lambda a,b: float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]) if a is not None and b is not None else -999

    sims = {
        "menu":      sim(v, menu_vec),
        "space":     sim(v, space_vec),
        "operation": sim(v, ops_vec),
        "regional":  sim(v, regi_vec),
    }

    # 룰 가산(휴리스틱)
    if has_any(phrase, OPS_PATTERNS):
        sims["operation"] += 0.15
    if has_any(phrase, REGI_PATTERNS):
        sims["regional"]  += 0.15

    best_type = max(sims, key=sims.get)
    return best_type, sims["menu"], sims["space"], sims["operation"], sims["regional"]

# === 4) 라벨링 실행 ===
labels, s_menu, s_space, s_ops, s_regi = [], [], [], [], []
for phrase in df_topk["phrase"]:
    t, smn, ssp, sop, srg = classify_trigram_four(
        phrase, w2v_model, menu_vec, space_vec, ops_vec, regi_vec
    )
    labels.append(t); s_menu.append(smn); s_space.append(ssp); s_ops.append(sop); s_regi.append(srg)

df_topk["type"]         = labels
df_topk["sim_menu"]     = s_menu
df_topk["sim_space"]    = s_space
df_topk["sim_operation"]= s_ops
df_topk["sim_regional"] = s_regi

# === 5) 요약 지표(비중/강도) ===
def share_and_intensity(df, label):
    m = df["type"] == label
    return m.mean(), df.loc[m, "log_odds"].mean()

for lab in ["menu","space","regional","operation"]:
    sh, inten = share_and_intensity(df_topk, lab)
    print(f"[{lab}] share={sh:.2%}, intensity={inten if pd.notna(inten) else float('nan'):.3f}")


print("\n=== 🏷️ 상위 20개 trigram + 타입 예시 ===")
print(df_topk.head(20)[["phrase", "type", "sim_menu", "sim_space", "sim_operation", "sim_regional"]])

df_topk.to_csv(OUT_LABELED_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 라벨링 포함 결과 저장 완료: {OUT_LABELED_CSV}")

print("\n🎉 모든 단계 완료!")
