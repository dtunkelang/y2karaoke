# Curated Clip Workflow

Last updated: 2026-03-18

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
4. If the song is cold, use a quick first probe:
   `./venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag stress --fast-clip-probe --max-songs 1`
5. For apples-to-apples measurement, rerun on the normal path and let the runner reuse full-song results where possible:
   `./venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag duet --offline`

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

## Curation Discipline

- Commit curated clip gold files immediately.
- If source and lyrics disagree structurally, fix the source before curating around it.
- Avoid changing the clip window and lyric span in the same pass unless you are deliberately reseeding the clip.
