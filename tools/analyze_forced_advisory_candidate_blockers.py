"""Explain why aggressive advisory support does not become a live candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _collect_blockers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    live_candidate_indexes = {
        int(candidate["index"]) for candidate in payload.get("candidates", [])
    }
    blockers: list[dict[str, Any]] = []
    for line in payload.get("lines", []):
        index = int(line["index"])
        if index in live_candidate_indexes:
            continue
        aggressive_overlap = float(line.get("aggressive_best_overlap", 0.0))
        if aggressive_overlap < 0.99:
            continue
        default_overlap = float(line.get("default_best_overlap", 0.0))
        default_window_word_count = int(line.get("default_window_word_count", 0))
        current_window_word_count = int(line.get("current_window_word_count", 0))
        reasons: list[str] = []
        if default_overlap > 0.0:
            reasons.append(f"default_overlap={default_overlap:.3f}")
        if default_window_word_count > 0:
            reasons.append(f"default_window_word_count={default_window_word_count}")
        if current_window_word_count > 3:
            reasons.append(f"current_window_word_count={current_window_word_count}")
        if not reasons:
            reasons.append("not_promoted_by_runtime_bucket")
        blockers.append(
            {
                "index": index,
                "text": str(line["text"]),
                "aggressive_best_segment_text": str(
                    line.get("aggressive_best_segment_text", "")
                ),
                "aggressive_best_overlap": aggressive_overlap,
                "default_best_overlap": default_overlap,
                "default_window_word_count": default_window_word_count,
                "current_window_word_count": current_window_word_count,
                "blockers": reasons,
            }
        )
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", help="Path to forced advisory trace JSON")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    blockers = _collect_blockers(_load_payload(Path(args.trace_json)))
    if args.json:
        print(json.dumps({"blockers": blockers}, indent=2))
        return 0

    for blocker in blockers:
        print(
            f"line {blocker['index']}: {blocker['text']} "
            f"(aggressive='{blocker['aggressive_best_segment_text']}')"
        )
        print("  " + ", ".join(blocker["blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
