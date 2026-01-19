"""Lightweight embedding helpers for similarity search."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict

Vector = Dict[str, float]
_TOKEN_PATTERN = re.compile(r"[\w']+")


def embed(text: str) -> Vector:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return dict(counts)
    return {token: value / norm for token, value in counts.items()}


def cosine_similarity(left: Vector, right: Vector) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    score = 0.0
    for token, weight in left.items():
        score += weight * right.get(token, 0.0)
    return score
