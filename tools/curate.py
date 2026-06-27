"""Data-curation scaffold (paper-1 hook) — NOT used for v1 training.

PP-OCRv5's core lesson (arXiv 2603.24373) is data-centric: recognition quality is
driven by curating the training set along difficulty / accuracy / diversity, not
by changing the architecture. The "sweet spot" is medium-difficulty samples whose
model confidence sits ~0.95-0.97 — informative gradients while still reliably
labeled.

This module is a stand-alone hook for a FUTURE fine-tuning pass to adapt the
recognizer to our own document domain. It does NOT train anything in v1; it only
buckets a corpus by confidence so the sweet-spot band can be sampled.

Usage (offline, later):
    from tools.curate import confidence_buckets
    buckets = confidence_buckets(samples)   # samples: [{"image":PIL, "text":str}]
    sweet = buckets["sweet"]                # 0.95 <= conf <= 0.97
"""

from __future__ import annotations

SWEET_LO, SWEET_HI = 0.95, 0.97


def bucket_for(conf: float) -> str:
    if conf < SWEET_LO:
        return "hard"        # noisy / low-confidence — informative but risky labels
    if conf <= SWEET_HI:
        return "sweet"       # the target band per paper-1
    return "easy"            # trivial — low marginal value


def confidence_buckets(samples: list[dict]) -> dict[str, list[dict]]:
    """Bucket samples by recognizer confidence into hard/sweet/easy.

    Each sample must already carry a "confidence" float (run the recognizer first).
    """
    out: dict[str, list[dict]] = {"hard": [], "sweet": [], "easy": []}
    for s in samples:
        out[bucket_for(float(s.get("confidence", 0.0)))].append(s)
    return out
