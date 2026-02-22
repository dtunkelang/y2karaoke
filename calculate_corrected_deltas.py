import json
import yaml
from pathlib import Path
import statistics
import sys
from difflib import SequenceMatcher

# Set up path to use y2karaoke.core.text_utils
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from y2karaoke.core.text_utils import normalize_text_basic  # noqa: E402


def load_words(path):
    with open(path, "r") as f:
        doc = json.load(f)
    words = []
    # Support both "lines" -> "words" and flat structure if it exists
    for line in doc.get("lines", []):
        for word in line.get("words", []):
            text = normalize_text_basic(word["text"])
            if text:
                words.append({"text": text, "start": float(word["start"])})
    return words


def calculate_offset_corrected_metrics(ref_words, ext_words):
    # Align by text content using SequenceMatcher
    ref_texts = [w["text"] for w in ref_words]
    ext_texts = [w["text"] for w in ext_words]

    matcher = SequenceMatcher(a=ref_texts, b=ext_texts, autojunk=False)
    deltas = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Words match exactly, use these for delta calculation
            for offset in range(i2 - i1):
                ref_idx = i1 + offset
                ext_idx = j1 + offset
                deltas.append(ext_words[ext_idx]["start"] - ref_words[ref_idx]["start"])

    if not deltas:
        return None

    n = len(deltas)

    # Use Median delta as offset (robust)
    offset_median = statistics.median(deltas)
    corrected_median = [abs(d - offset_median) for d in deltas]

    return {
        "matched_count": n,
        "ref_count": len(ref_words),
        "ext_count": len(ext_words),
        "offset_median": offset_median,
        "corrected_median_avg_abs": sum(corrected_median) / n,
        "corrected_median_p95_abs": (
            sorted(corrected_median)[int(0.95 * n)] if n > 0 else 0
        ),
    }


with open("benchmarks/benchmark_songs.yaml", "r") as f:
    config = yaml.safe_load(f)

# Manually fix config to include both 07 songs if one is missing from YAML but present in files
# Actually, the YAML is the source of truth for the 13 songs.
# Let's check if 07 is Coldplay or Weeknd in YAML.

print("# Offset-Corrected Timing Accuracy (Visual Gold vs Reference Gold)")
print(
    "| Index | Song | Matched Words | Offset (sec) | Corrected Avg Abs Delta | Corrected P95 Delta |"
)
print("|---|---|---|---|---|---|")

gold_set_dir = Path("benchmarks/gold_set/")
visual_gold_dir = Path("benchmarks/gold_set_karaoke_seed/")

for idx, song in enumerate(config["songs"], 1):
    artist = song["artist"]
    title = song["title"]

    ref_path = None
    ext_path = None

    if "Viva la Vida" in title:
        ref_path = gold_set_dir / "07_coldplay-viva-la-vida.gold.json"
        ext_path = visual_gold_dir / "07_coldplay-viva-la-vida.visual.gold.json"
    elif "Blinding Lights" in title:
        ref_path = gold_set_dir / "07_the-weeknd-blinding-lights.gold.json"
        ext_path = visual_gold_dir / "07_the-weeknd-blinding-lights.visual.gold.json"
    else:
        # Standard index match
        rp = list(gold_set_dir.glob(f"{idx:02d}_*.gold.json"))
        ep = list(visual_gold_dir.glob(f"{idx:02d}_*.visual.gold.json"))
        if rp:
            ref_path = rp[0]
        if ep:
            ext_path = ep[0]

    if ref_path and ext_path and ref_path.exists() and ext_path.exists():
        ref_words = load_words(ref_path)
        ext_words = load_words(ext_path)

        metrics = calculate_offset_corrected_metrics(ref_words, ext_words)
        if metrics:
            print(
                f"| {idx:02d} | {artist} - {title} | {metrics['matched_count']}/{metrics['ref_count']} | "
                f"{metrics['offset_median']:+.3f} | {metrics['corrected_median_avg_abs']:.3f} | "
                f"{metrics['corrected_median_p95_abs']:.3f} |"
            )
        else:
            print(
                f"| {idx:02d} | {artist} - {title} | 0/{len(ref_words)} | ERROR | ERROR | ERROR |"
            )
    else:
        print(f"| {idx:02d} | {artist} - {title} | - | MISSING | MISSING | MISSING |")
