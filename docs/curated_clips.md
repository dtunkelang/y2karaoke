# Curated Clip Workflow

Last updated: 2026-03-20

Use curated clips to isolate specific timing or alignment behaviors without paying whole-song iteration costs on every pass.

## When To Add A Clip

Add a clip when a full-song metric is hiding a distinct failure mode or when you need a stable short control.

High-value clip categories:
- repeated-hook comparability
- duet or backing-vocal overlap
- weak-onset or clipped-onset starts
- tail and sparse-support endings
- source-text or source-timing mismatch stress cases
- clean control clips for sanity checks

## Manifest Rules

Curated clips live in `benchmarks/curated_clip_songs.yaml`.

Each clip entry should include:
- `clip_id`
- `clip_tags`
- `audio_start_sec`
- `clip_duration_sec`
- `notes`

Recommended tag vocabulary:
- `control`
- `stress`
- `chorus`
- `verse`
- `verse-hook`
- `repeated-hook`
- `duet`
- `tail`
- `source-text`
- `source-timing`
- `comparability`
- `clean`

Use a small set of stable tags rather than inventing one-off labels.

## Recommended Loop

1. Vet the source URL if it is new.
2. Add or update the clip entry in `benchmarks/curated_clip_songs.yaml`.
3. Validate the manifest:
   `./venv/bin/python tools/validate_benchmark_manifest.py benchmarks/curated_clip_songs.yaml`
4. Open the saved gold file, not a timing-report seed:
   `make curated-open MATCH="Song Or Clip Id"`
   Direct helper form:
   `PYTHONPATH=src ./.venv/bin/python tools/curated_clip_helper.py --match "Song Or Clip Id" --open-editor`
5. If the song is cold, use a quick first probe:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag stress --fast-clip-probe --max-songs 1`
6. For apples-to-apples measurement, rerun on the normal path and let the runner reuse full-song results where possible:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag duet --offline`

## Selection

Use regex matching for specific songs or clip ids:
- `--match "Blinding Lights|hook-repeat"`

Use tag selection for clip packs:
- `--clip-tag control`
- `--clip-tag duet --clip-tag tail`

Tag filters are additive at the CLI level: a song is selected if it matches any requested tag.

## Cost Model

- Prefer normal offline reruns once a song has cached audio and stems.
- The runner can now score many clip entries directly from a compatible cached full-song result, which avoids rerunning generation for those clips.
- `--fast-clip-probe` is still useful for a cold first pass, but it is a triage path, not the final benchmark path.
- Broad offline tag runs are still noisy if the selected pack includes uncached clips; use known cached match-based subsets when you need a clean quality signal.

## Curation Discipline

- Always open the editor from the saved gold JSON and canonical clip audio path.
- Prefer the stable shortcut:
  `make curated-open MATCH="Song Or Clip Id"`
- Use `tools/curated_clip_helper.py` instead of hand-building editor URLs or filenames.
- After any manual curation change:
  - verify the saved gold JSON on disk
  - verify the actual clip audio duration/path on disk
  - reopen the saved gold file once to confirm the editor is loading the right artifact
  - commit and push immediately before any further code or benchmark work
- If source and lyrics disagree structurally, fix the source before curating around it.
- Avoid changing the clip window and lyric span in the same pass unless you are deliberately reseeding the clip.
- If a hard clip remains the only outlier after other clips improve, add one or two companion clips in the same failure family before tuning further.

## Current Learnings

- Short curated clips are strong quality drivers when the pipeline stays clip-scoped:
  - bounded clip audio
  - clip-scoped lyric text
  - clip-scoped scoring against gold
- Repeated-hook clips should be optimized as families, not single songs. The current useful family is:
  - `Houdini`
  - `Without Me`
  - `I Gotta Feeling`
- For longer repeated-hook clips where the dominant repeated block starts after one or two unique setup lines, the prefix gap before the repeated block needs to be wider than the simpler compact-hook layout gives it. That improved `Houdini` without disturbing `Without Me` or `I Gotta Feeling`.
- Dense non-repeated short rap verses like `Rap God` need a different seed from both the repeated-hook path and the generic dense spread:
  - keep the opening anchor early
  - weight the first two dense lines heavily
  - preserve enough tail span for the final long line instead of letting the generic spread collapse it
- Plain-text clip lyrics need an audio-window-aware timing seed. Starting every plain-text clip at `0.0s` hides useful structure and biases repeated hooks toward early collapse.
- Short-title chorus clips like `Sweet Caroline` need a different seed from the generic compact spread:
  - give the title line less span
  - widen the setup gap into line 2
  - leave more room for the tail line
- Mixed-density chorus clips like `Con Calma` still need a bit more span on the repeated long lines than the generic chorus weighting gives them. A small increase there, plus a slightly looser long-line to short-response gap, improved the clip without hurting the focused lyrics tests.
- `Con Calma` then exposed a second failure mode after the seed improved:
  - some late lines had real earlier Whisper support, but rollback/correction behavior still left them pinned to later baseline starts
  - a narrow earlier-Whisper reanchor for lines with in-order prefix support improved `Con Calma` again without regressing the `Houdini|Without Me|I Gotta Feeling` canary
  - once a clip reaches that stage, inspect post-map and correction-pass traces before tuning seed helpers again
- Two-line falsetto/refrain clips exposed a different failure mode from longer repeated-hook clips:
  - WhisperX forced alignment previously could not help 2-line clips at all
  - weak onset detection could incorrectly fall back to a generic spread seed
  - subset-refrain clips need their own plain-text seed layout when line 2 is a shorter tail of line 1
- `Take On Me` revealed a different sparse/falsetto issue:
  - the accepted forced alignment can have correct line boundaries but poor within-line word distribution
  - the worst case is a short-function-word lead-in followed by a held final word
  - compare raw forced word timings against gold before changing line-level seeding again
  - a narrow within-line redistribution fix can improve this family without disturbing `Stayin' Alive`, `Time After Time`, or `Total Eclipse`
- When a live clip still looks wrong after a plausible fix, compare:
  - the helper-generated seed on the real cached clip audio
  - the accepted forced-alignment output
  - the final timing report
  This is faster than guessing which postpass is to blame.
- If a focused canary improves cleanly, commit and push before widening the benchmark set. That keeps the next step recoverable when iteration budget is tight.
- Do not drop difficult clips just because they are difficult. Keep them if they reflect real production failures, but add companion clips when a single clip is too underdetermined to tune against safely.
