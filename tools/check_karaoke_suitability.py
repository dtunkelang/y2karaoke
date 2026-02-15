#!/usr/bin/env python3
"""Check suitability of a karaoke instrumental track compared to the original."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from y2karaoke.core.audio_analysis import calculate_harmonic_suitability


def main():
    parser = argparse.ArgumentParser(
        description="Check karaoke instrumental suitability."
    )
    parser.add_argument(
        "original", help="Path to original instrumental track (separated)"
    )
    parser.add_argument("karaoke", help="Path to karaoke instrumental track")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    orig_path = Path(args.original)
    kara_path = Path(args.karaoke)

    if not orig_path.exists():
        print(f"Error: Original track not found: {orig_path}")
        return 1
    if not kara_path.exists():
        print(f"Error: Karaoke track not found: {kara_path}")
        return 1

    print(f"Analyzing harmonic suitability...")
    print(f"  Original: {orig_path.name}")
    print(f"  Karaoke:  {kara_path.name}")

    metrics = calculate_harmonic_suitability(str(orig_path), str(kara_path))

    if "error" in metrics:
        print(f"Error during analysis: {metrics['error']}")
        return 1

    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0

    cost = metrics["similarity_cost"]

    print("\nResults:")
    print(f"  Similarity Cost:    {cost:.4f}")
    print(f"  Best Key Shift:     {metrics['best_key_shift']} half-steps")
    print(f"  Tempo Variance:     {metrics['tempo_variance']:.4f}")
    print(f"  Structure Jumps:    {metrics['structure_jump_count']}")
    print(f"  Estimated Offset:   {metrics['offset_seconds']:+.3f}s")

    print("\nInterpretation:")
    if cost < 0.15:
        print("  SUITABILITY: EXCELLENT - Very accurate backing track.")
    elif cost < 0.35:
        print("  SUITABILITY: ACCEPTABLE - Minor arrangement differences.")
    elif cost < 0.6:
        print("  SUITABILITY: POOR - Significant arrangement differences.")
    else:
        print("  SUITABILITY: UNUSABLE - Likely a different song or structure.")

    if metrics["structure_jump_count"] > 0:
        print(
            f"  WARNING: {metrics['structure_jump_count']} structural discontinuities detected (missing/extra bars)."
        )
    if metrics["tempo_variance"] > 0.1:
        print(
            f"  WARNING: High tempo variance ({metrics['tempo_variance']:.4f}) detected. Timing may drift."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
