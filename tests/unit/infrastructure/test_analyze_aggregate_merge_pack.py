from __future__ import annotations

from pathlib import Path

from tools import analyze_aggregate_merge_pack as module


def test_analyze_aggregate_merge_pack_filters_and_collects(monkeypatch) -> None:
    songs = [
        {
            "artist": "Drake",
            "title": "Hotline Bling",
            "clip_id": "cell-phone-variant",
            "audio_start_sec": 12.0,
            "clip_duration_sec": 11.0,
            "youtube_id": "zt6aRKpf9T4",
        },
        {
            "artist": "Destiny's Child",
            "title": "Say My Name",
            "clip_id": "opening-hook-variant",
            "audio_start_sec": 1.0,
            "clip_duration_sec": 8.0,
            "youtube_id": "Ajez9PaCgek",
        },
    ]

    monkeypatch.setattr(module, "_load_manifest", lambda _p: songs)
    monkeypatch.setattr(
        module,
        "_gold_path",
        lambda index, song: Path(f"/tmp/{index}_{song['title']}.gold.json"),
    )

    def fake_cache(song: dict[str, object], *, aggressive: bool) -> Path | None:
        title = str(song["title"])
        if "Say My Name" in title and aggressive:
            return None
        suffix = "aggr" if aggressive else "vocals"
        return Path(f"/tmp/{title}_{suffix}.json")

    monkeypatch.setattr(module, "_cache_path_for", fake_cache)

    def fake_analyze(*, aggregate_path: Path, vocals_path: Path, gold_path: Path):
        if "Hotline Bling" in aggregate_path.name:
            return {"merge_count": 1, "rows": [{"left_line_index": 1}]}
        return {"merge_count": 0, "rows": []}

    monkeypatch.setattr(module.merge_tool, "analyze", fake_analyze)

    result = module.analyze(
        manifest_path=Path("benchmarks/curated_clip_songs.yaml"),
        match="Hotline|Say",
    )

    assert result["songs_analyzed"] == 2
    assert result["songs_with_merges"] == 1
    assert result["rows"] == [
        {
            "song": "Drake - Hotline Bling",
            "clip_id": "cell-phone-variant",
            "gold_path": "/tmp/1_Hotline Bling.gold.json",
            "aggregate_path": "/tmp/Hotline Bling_aggr.json",
            "vocals_path": "/tmp/Hotline Bling_vocals.json",
            "merge_count": 1,
            "rows": [{"left_line_index": 1}],
        }
    ]
    assert result["skipped"] == [
        {"song": "Destiny's Child - Say My Name", "reason": "missing_cache"}
    ]
