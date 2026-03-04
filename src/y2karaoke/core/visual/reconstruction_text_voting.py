"""Word-level text voting helpers for reconstruction tracks."""

from __future__ import annotations

import collections
import statistics
from typing import Any

from ..text_utils import LYRIC_FUNCTION_WORDS


def compute_voted_text_and_evidence(
    *,
    entries: list[dict[str, Any]],
    text_counts: collections.Counter[str],
    best_text: str,
) -> tuple[str, dict[str, Any]]:
    """Compute voted line text with evidence from per-frame OCR observations."""
    if not entries:
        return "", {}

    all_line_tokens = [e["words"] for e in entries]
    if not all_line_tokens:
        return best_text, {
            "observations": 0,
            "target_word_count": 0,
            "token_count_variants": 0,
            "valid_entry_fraction": 0.0,
            "text_variant_count": 0,
            "selected_text_support_ratio": 0.0,
            "position_support_mean": 0.0,
            "position_support_min": 0.0,
            "weak_vote_positions": 0,
            "used_observed_fallback": False,
            "uncertainty_score": 1.0,
        }

    word_counts = [len(tokens) for tokens in all_line_tokens]
    target_wc = int(statistics.median(word_counts))
    valid_entries = [tokens for tokens in all_line_tokens if len(tokens) == target_wc]
    if not valid_entries:
        total_entries = len(entries)
        support = text_counts.get(best_text, 0)
        return best_text, {
            "observations": total_entries,
            "target_word_count": target_wc,
            "token_count_variants": len(set(word_counts)),
            "valid_entry_fraction": 0.0,
            "text_variant_count": len(text_counts),
            "selected_text_support_ratio": round(
                support / float(max(total_entries, 1)), 3
            ),
            "position_support_mean": 0.0,
            "position_support_min": 0.0,
            "weak_vote_positions": target_wc,
            "used_observed_fallback": True,
            "uncertainty_score": 1.0,
        }

    valid_entry_texts = [
        str(e.get("text", "")) for e in entries if len(e.get("words", [])) == target_wc
    ]
    text_freq = collections.Counter(valid_entry_texts)

    final_words: list[str] = []
    position_vote_counts: list[collections.Counter[str]] = []
    for i in range(target_wc):
        word_votes = collections.Counter(tokens[i] for tokens in valid_entries)
        position_vote_counts.append(word_votes)
        weighted_votes: dict[str, float] = {}
        for word, count in word_votes.items():
            score = float(count)
            if word.lower() in LYRIC_FUNCTION_WORDS:
                score += 0.5
            weighted_votes[word] = score
        best_word = max(weighted_votes.items(), key=lambda x: x[1])[0]
        final_words.append(best_word)

    voted_text = " ".join(final_words)
    total_entries = len(entries)
    valid_fraction = len(valid_entries) / float(max(total_entries, 1))
    supports = [
        position_vote_counts[i].get(final_words[i], 0)
        / float(max(1, len(valid_entries)))
        for i in range(target_wc)
    ]
    avg_support = sum(supports) / len(supports) if supports else 1.0
    min_support = min(supports) if supports else 1.0
    weak_positions = sum(1 for s in supports if s < 0.5)
    used_observed_fallback = False

    if voted_text not in text_freq and (avg_support < 0.72 or weak_positions >= 2):
        best_tokens = max(
            valid_entries,
            key=lambda toks: (
                sum(position_vote_counts[i].get(tok, 0) for i, tok in enumerate(toks)),
                text_freq.get(" ".join(toks), 0),
                sum(len(tok) for tok in toks),
                " ".join(toks),
            ),
        )
        voted_text = " ".join(best_tokens)
        used_observed_fallback = True

    selected_support_ratio = text_freq.get(voted_text, 0) / float(max(total_entries, 1))
    top_text_variants: list[dict[str, object]] = []
    if text_counts:
        ranked_variants = sorted(
            text_counts.items(),
            key=lambda kv: (-int(kv[1]), -len(kv[0].split()), -len(kv[0]), kv[0]),
        )
        for txt, count in ranked_variants[:6]:
            top_text_variants.append(
                {
                    "text": txt,
                    "count": int(count),
                    "support_ratio": round(
                        int(count) / float(max(total_entries, 1)), 3
                    ),
                    "word_count": len(txt.split()),
                }
            )

    uncertainty_score = (
        (1.0 - min(max(valid_fraction, 0.0), 1.0)) * 0.35
        + (1.0 - min(max(avg_support, 0.0), 1.0)) * 0.35
        + (1.0 - min(max(selected_support_ratio, 0.0), 1.0)) * 0.2
        + (min(weak_positions, max(target_wc, 1)) / float(max(target_wc, 1))) * 0.1
    )
    evidence = {
        "observations": total_entries,
        "target_word_count": target_wc,
        "token_count_variants": len(set(word_counts)),
        "valid_entry_fraction": round(valid_fraction, 3),
        "text_variant_count": len(text_counts),
        "selected_text_support_ratio": round(selected_support_ratio, 3),
        "position_support_mean": round(avg_support, 3),
        "position_support_min": round(min_support, 3),
        "weak_vote_positions": weak_positions,
        "used_observed_fallback": used_observed_fallback,
        "top_text_variants": top_text_variants,
        "uncertainty_score": round(max(0.0, min(1.0, uncertainty_score)), 3),
    }
    return voted_text, evidence
