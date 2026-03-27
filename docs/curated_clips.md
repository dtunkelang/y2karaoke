# Curated Clip Workflow

Last updated: 2026-03-27

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
   The helper now starts the local gold editor server automatically when needed before opening the browser.
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
- `Con Calma` later exposed a third failure mode after the obvious start fixes landed:
  - gold start/end metrics became decent while DTW / agreement coverage stayed poor
  - the practical signal was `dtw_line_coverage = 0.667` with `gold_start_mean_abs_sec = 0.2242`
  - implication: stop assuming the next gain is another boundary retime; the remaining issue is likely in mapping / coverage policy
- A deeper agreement diagnostic sharpened that read:
  - the remaining `Con Calma` agreement failures split into `anchor_outside_window` and `low_text_similarity`
  - but several `low_text_similarity` cases are not true text mismatches
  - they are exact local lyric phrases being compared against over-merged anchor text that spans multiple lines
  - implication: before relaxing similarity thresholds on bilingual / chorus-merging clips, inspect anchor granularity and anchor-selection policy
- A follow-up clipping simulation made that more concrete:
  - on the kept `Con Calma` baseline, clipping merged anchor text down to the best contiguous phrase would recover most of the `low_text_similarity` lines
  - implication: anchor-text clipping is now a more credible next strategy than another boundary retime or broad threshold relaxation
  - the stronger quantified read is:
    - baseline agreement coverage `0/5`
    - clipped-anchor simulated coverage `4/5`
  - that is strong enough to justify an explicit clipped-anchor comparability experiment before more song-specific retiming work
- A pack-level simulation showed the same idea is not automatically isolated:
  - on a mixed kept pack, clipped-anchor simulated agreement improves `Con Calma` strongly, but also recovers `Take On Me` and one `Without Me` line
  - implication: if this becomes a real comparability policy, it will need guards rather than a blanket enable
- The first simple guard that looked clean was:
  - `line_words >= 6`
  - `anchor_words - line_words >= 15`
  - on the mixed kept pack, that still recovers `Con Calma` while dropping the `Take On Me` / `Without Me` spillover
- On `Con Calma` itself, that guarded policy cleanly separates the remaining issues:
  - clipped-anchor recovery handles most of the `low_text_similarity` cases
  - the rest are still `anchor_outside_window`
  - implication: if you adopt clipped-anchor comparability, treat it as one layer, not the whole fix
- A follow-up window-phrase diagnostic tightened the second layer too:
  - most of the remaining `anchor_outside_window` lines in `Con Calma` already have strong local phrases inside the Whisper window
  - implication: the remaining blocker is largely anchor-start selection, not lack of local lexical evidence, except for the final `De guayarte, ma...` line
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

## Process Learnings

- See also `docs/development.md` for the broader local workflow and documentation-maintenance rules around this loop.
- Use the narrow iteration loop when:
  - one clip is clearly the top outlier
  - a clip has split into separate line-level failure modes
  - a plausible fix already improved the seed, and the remaining miss looks downstream
  - a broad canary is clean enough that you can afford to localize the next read
- Do not use the narrow loop as the default when:
  - the broad canary has not been reranked recently
  - the current top clip may still be curation drift rather than pipeline behavior
  - multiple clips are moving at once and you do not yet know the shared failure family
- The recent time-pressure loop was useful, and not only because of the deadline.
- The parts worth keeping even when time pressure is lower are:
  - narrow the next step to one concrete code path plus one concrete artifact before editing
  - state explicit success and failure criteria before broadening a fix
  - save negative results, not just wins, when they materially eliminate a suspect path
  - keep the current target clip split into separate failure modes when the lines are clearly failing for different reasons
  - pin the exact rerun command, trace env, unit check, and lint check next to the current hypothesis
- The part not worth keeping at full intensity is constant commit/push churn after every tiny note.
- Preferred normal mode:
  - commit/push after a clean behavioral win
  - commit/push after a meaningful diagnostic artifact that would be expensive to rediscover
  - batch small handoff-note updates together unless there is a real risk of losing context
- In practice, the good default loop is:
  1. identify one clip, one line, one likely code path
  2. name the exact artifact and rerun command
  3. run one focused probe
  4. either keep the win or record the elimination
  5. only then widen to the broader canary
