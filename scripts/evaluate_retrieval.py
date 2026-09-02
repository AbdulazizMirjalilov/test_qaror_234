"""Retrieval quality evaluation against data/eval_questions.json.

Reports, per question, whether the threshold decision (found / not found)
matches the expectation, and whether the expected source appears in the
top-k results. Also prints the score distribution split by expectation,
which is exactly what you need to (re)tune QAROR_SCORE_THRESHOLD.

Run from the project root (requires the full requirements installed and
the index built):

    python -m scripts.evaluate_retrieval
"""

from __future__ import annotations

import json
import sys

from app.core.config import settings
from app.services.retriever import Retriever

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# Uzbek characters like U+02BB -- force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_PATH = settings.DATA_DIR / "eval_questions.json"


def matches_expected(metadata: dict, case: dict) -> bool:
    if "expected_punkt_num" in case and str(metadata.get("punkt_num")) != str(
        case["expected_punkt_num"]
    ):
        return False
    if "expected_ilova_num" in case and metadata.get("ilova_num") != case["expected_ilova_num"]:
        return False
    return True


def main():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()

    threshold_correct = 0
    hits_at_1 = 0
    hits_at_k = 0
    with_source = 0
    in_scope_scores: list[float] = []
    out_scope_scores: list[float] = []

    print(
        f"threshold={settings.SCORE_THRESHOLD}  top_k={settings.TOP_K}  "
        f"model={settings.EMBEDDING_MODEL}"
    )
    print("=" * 90)

    for case in cases:
        q = case["question"]
        results = retriever.search(q)
        top_score = results[0].score if results else 0.0
        predicted_found = top_score >= settings.SCORE_THRESHOLD

        (in_scope_scores if case["expect_found"] else out_scope_scores).append(top_score)

        ok_threshold = predicted_found == case["expect_found"]
        threshold_correct += ok_threshold

        line = (
            f"[{'OK ' if ok_threshold else 'ERR'}] top={top_score:.3f} found={predicted_found!s:5}"
        )

        has_source = case["expect_found"] and (
            "expected_punkt_num" in case or "expected_ilova_num" in case
        )
        if has_source:
            with_source += 1
            hit1 = bool(results) and matches_expected(results[0].metadata, case)
            hitk = any(matches_expected(r.metadata, case) for r in results)
            hits_at_1 += hit1
            hits_at_k += hitk
            line += f"  hit@1={hit1!s:5} hit@k={hitk!s:5}"

        print(f"{line}  | {q}")

    print("=" * 90)
    print(f"Threshold decision accuracy: {threshold_correct}/{len(cases)}")
    if with_source:
        print(
            f"Source hit@1: {hits_at_1}/{with_source}   "
            f"hit@{settings.TOP_K}: {hits_at_k}/{with_source}"
        )
    if in_scope_scores:
        print(
            f"In-scope top-1 scores:  min={min(in_scope_scores):.3f}  "
            f"max={max(in_scope_scores):.3f}"
        )
    if out_scope_scores:
        print(
            f"Out-of-scope top-1 scores: min={min(out_scope_scores):.3f}  "
            f"max={max(out_scope_scores):.3f}"
        )
        if in_scope_scores:
            lo, hi = max(out_scope_scores), min(in_scope_scores)
            if lo < hi:
                print(f"Any threshold in ({lo:.3f}, {hi:.3f}) separates the two groups cleanly.")
            else:
                print("Warning: score ranges overlap -- no single threshold separates them.")


if __name__ == "__main__":
    main()
