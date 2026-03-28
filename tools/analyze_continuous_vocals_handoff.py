#!/usr/bin/env python3
"""Analyze line changes between tail extension and continuous-vocals stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _snapshot_lines(trace: dict[str, Any], stage: str) -> dict[int, dict[str, Any]]:
    for snapshot in trace.get("snapshots", []):
        if snapshot.get("stage") == stage:
            return {int(line["line_index"]): line for line in snapshot.get("lines", [])}
    return {}


def analyze(
    *,
    stage_trace: dict[str, Any],
    from_stage: str = "postpass_extend_trailing",
    to_stage: str = "postpass_pull_continuous_vocals",
) -> dict[str, Any]:
    before = _snapshot_lines(stage_trace, from_stage)
    after = _snapshot_lines(stage_trace, to_stage)
    rows: list[dict[str, Any]] = []
    for line_index in sorted(set(before) | set(after)):
        prev = before.get(line_index)
        cur = after.get(line_index)
        if prev is None or cur is None:
            continue
        start_shift = round(float(cur["start"]) - float(prev["start"]), 3)
        end_shift = round(float(cur["end"]) - float(prev["end"]), 3)
        rows.append(
            {
                "line_index": line_index,
                "text": cur["text"],
                "from_start": prev["start"],
                "from_end": prev["end"],
                "to_start": cur["start"],
                "to_end": cur["end"],
                "start_shift": start_shift,
                "end_shift": end_shift,
                "grew_left": start_shift < 0,
                "grew_right": end_shift > 0,
            }
        )
    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "lines": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_trace_json", help="Whisper stage trace JSON")
    parser.add_argument("--from-stage", default="postpass_extend_trailing")
    parser.add_argument("--to-stage", default="postpass_pull_continuous_vocals")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        stage_trace=_load_json(Path(args.stage_trace_json)),
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{payload['from_stage']} -> {payload['to_stage']}")
    for row in payload["lines"]:
        print(
            f"{row['line_index']}: {row['text']} "
            f"start_shift={row['start_shift']:+.3f} end_shift={row['end_shift']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
