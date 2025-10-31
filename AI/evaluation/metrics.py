"""Ranking and efficiency metrics for retrieval evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    k = min(k, len(relevance))
    if k == 0:
        return 0.0
    return sum(relevance[:k]) / float(k)


def recall_at_k(relevance: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    k = min(k, len(relevance))
    return sum(relevance[:k]) / float(total_relevant)


def mean_reciprocal_rank(relevance: Sequence[int]) -> float:
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(gains: Sequence[float], k: int) -> float:
    k = min(k, len(gains))
    if k == 0:
        return 0.0
    dcg = _discounted_cumulative_gain(gains, k)
    ideal = _discounted_cumulative_gain(sorted(gains, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def _discounted_cumulative_gain(gains: Sequence[float], k: int) -> float:
    score = 0.0
    for idx in range(k):
        gain = gains[idx]
        score += (2**gain - 1) / math.log2(idx + 2)
    return score


@dataclass
class EfficiencyStats:
    median_latency_ms: float
    p95_latency_ms: float
    index_build_ms: float
    throughput_qps: float


def summarise_latency(samples_ms: Sequence[float]) -> tuple[float, float]:
    if not samples_ms:
        return 0.0, 0.0
    sorted_samples = sorted(samples_ms)
    median = _percentile(sorted_samples, 0.5)
    p95 = _percentile(sorted_samples, 0.95)
    return median, p95


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    idx = percentile * (len(values) - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return values[int(idx)]
    fraction = idx - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
