import yaml
import subprocess
from pathlib import Path

with open("benchmarks/benchmark_songs.yaml", "r") as f:
    config = yaml.safe_load(f)

results = []
summary = []
for idx, song in enumerate(config["songs"], 1):
    artist = song["artist"]
    title = song["title"]

    # Search for visual gold file starting with idx
    pattern = f"{idx:02d}_*.visual.gold.json"
    candidates = list(Path("benchmarks/gold_set_karaoke_seed/").glob(pattern))

    if candidates:
        gold_file = candidates[0]
        cmd = [
            "./.venv/bin/python3",
            "tools/evaluate_visual_lyrics_quality.py",
            "--gold-json",
            str(gold_file),
            "--title",
            title,
            "--artist",
            artist,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            lines = [ln.strip() for ln in res.stdout.strip().split("\n") if ln.strip()]
            strict_line = lines[0]
            repeat_line = lines[1]

            # Extract f1
            strict_f1 = float(strict_line.split("f1=")[1].split()[0])
            repeat_f1 = float(repeat_line.split("f1=")[1].split()[0])

            results.append(f"## {artist} - {title}\n{strict_line}\n{repeat_line}\n")
            summary.append(
                f"| {idx:02d} | {artist} - {title} | {strict_f1:.4f} | {repeat_f1:.4f} |"
            )
        else:
            results.append(f"## {artist} - {title}\nERROR: {res.stderr}\n")
            summary.append(f"| {idx:02d} | {artist} - {title} | ERROR | ERROR |")
    else:
        results.append(
            f"## {artist} - {title}\nERROR: No visual gold file found for index {idx:02d}\n"
        )
        summary.append(f"| {idx:02d} | {artist} - {title} | NOT FOUND | NOT FOUND |")

with open("visual_metrics_summary.md", "w") as f:
    f.write("# Visual Extraction Quality Summary\n\n")
    f.write("| Index | Song | Strict F1 | Repeat-Capped F1 |\n")
    f.write("|---|---|---|---|\n")
    f.write("\n".join(summary))
    f.write("\n\n" + "\n".join(results))

print("Results written to visual_metrics_summary.md")
