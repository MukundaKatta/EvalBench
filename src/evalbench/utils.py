"""Utility functions for text processing and scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return normalize_text(text).split()


def extract_ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    """Extract n-grams from a token list."""
    if n < 1 or n > len(tokens):
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    """Return a Counter of n-grams."""
    return Counter(extract_ngrams(tokens, n))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simple BLEU-like score (precision of n-grams, 1..max_n).

    Uses brevity penalty and geometric mean of n-gram precisions.
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ng = ngram_counts(ref_tokens, n)
        hyp_ng = ngram_counts(hyp_tokens, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # Avoid log(0)
    if any(p == 0.0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    geo_mean = math.exp(log_avg)

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * geo_mean


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L-like recall score based on longest common subsequence."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)
    # Recall-based: LCS / reference length
    return lcs_len / len(ref_tokens)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def exact_match(reference: str, hypothesis: str) -> float:
    """Return 1.0 if normalized texts match exactly, else 0.0."""
    return 1.0 if normalize_text(reference) == normalize_text(hypothesis) else 0.0


def length_ratio(reference: str, hypothesis: str) -> float:
    """Ratio of hypothesis length to reference length (clamped to [0, 2])."""
    ref_len = len(tokenize(reference))
    hyp_len = len(tokenize(hypothesis))
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 2.0
    return min(hyp_len / ref_len, 2.0)


def build_tfidf_vectors(
    doc_a: str, doc_b: str
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Build simple TF-IDF vectors for two documents (2-doc corpus)."""
    tokens_a = tokenize(doc_a)
    tokens_b = tokenize(doc_b)
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    vocab = set(tf_a.keys()) | set(tf_b.keys())
    # IDF with 2-document corpus
    idf: Dict[str, float] = {}
    for term in vocab:
        df = int(term in tf_a) + int(term in tf_b)
        idf[term] = math.log((2 + 1) / (df + 1)) + 1  # smoothed IDF

    vec_a = {t: tf_a.get(t, 0) * idf[t] for t in vocab}
    vec_b = {t: tf_b.get(t, 0) * idf[t] for t in vocab}
    return vec_a, vec_b


def compute_cosine_similarity(reference: str, hypothesis: str) -> float:
    """Compute TF-IDF cosine similarity between two texts."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    vec_a, vec_b = build_tfidf_vectors(reference, hypothesis)
    dot = sum(vec_a[t] * vec_b[t] for t in vec_a)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def aggregate_scores(scores: List[float]) -> Dict[str, float]:
    """Compute mean, min, max, and std deviation for a list of scores."""
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    return {
        "mean": round(mean, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(math.sqrt(variance), 4),
    }
