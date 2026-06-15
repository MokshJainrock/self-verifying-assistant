"""
metrics.py — text-comparison metrics for the evaluation harness (Layer 3).

WHY this file exists: to score an answer against SQuAD's gold answers we need to
compare free text fairly. Our Responder writes a full cited sentence; the gold answer
is a short span ("1985", "Barack Obama"). Naive exact-match would score ~0 even when
the answer is right. So we use the SAME normalization SQuAD's official scorer uses,
plus a token-F1 and a simple containment check.

These are deliberately simple and standard — in an interview you can say "I used the
official SQuAD normalization (lowercase, strip punctuation, drop articles) so my
scoring matches the benchmark's conventions."
"""

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """
    SQuAD-style normalization: lowercase, remove punctuation, drop the articles
    a/an/the, and collapse whitespace. This makes "The Beatles" and "beatles" match.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)   # drop articles
    text = " ".join(text.split())                  # collapse whitespace
    return text


def f1(prediction: str, gold: str) -> float:
    """
    Token-overlap F1 between prediction and gold (the standard SQuAD soft metric).
    Captures partial credit: getting "Barack Obama" when gold is "Obama" scores > 0.
    """
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        # If either is empty, F1 is 1.0 only if both are empty, else 0.0.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> bool:
    """
    Did the (normalized) gold answer appear verbatim inside the (normalized) prediction?
    A strict, very explainable proxy for "the right fact is present in the answer."
    """
    return normalize(gold) in normalize(prediction)


def best_over_golds(prediction: str, golds: list, fn) -> float:
    """
    SQuAD gives several acceptable gold answers per question; credit the best match.
    `fn` is f1 or contains_gold; returns the max score across all golds (0.0 if none).
    """
    if not golds:
        return 0.0
    return max(float(fn(prediction, g)) for g in golds)
