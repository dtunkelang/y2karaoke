from y2karaoke.core.karaoke_timing_report import _build_base_report
from y2karaoke.core.models import Line, Word


def test_build_base_report_includes_lyrics_source_routing_fields():
    lines = [
        Line(
            words=[
                Word(text="hello", start_time=1.0, end_time=1.5),
                Word(text="world", start_time=1.5, end_time=2.0),
            ]
        )
    ]
    lyrics_result = {
        "quality": {
            "source": "lyriq (LRCLib)",
            "lyrics_source_audio_scoring_used": True,
            "lyrics_source_disagreement_flagged": True,
            "lyrics_source_disagreement_reasons": ["duration spread 15.0s"],
            "lyrics_source_candidate_count": 3,
            "lyrics_source_comparable_candidate_count": 2,
            "lyrics_source_selection_mode": "audio_scored_disagreement",
            "alignment_method": "lrc_only",
            "whisper_requested": False,
            "whisper_force_dtw": False,
            "whisper_used": False,
            "whisper_corrections": 0,
            "issues": [],
            "dtw_metrics": {},
        }
    }

    report = _build_base_report(lines, "Song", "Artist", lyrics_result)

    assert report["lyrics_source"] == "lyriq (LRCLib)"
    assert report["lyrics_source_audio_scoring_used"] is True
    assert report["lyrics_source_disagreement_flagged"] is True
    assert report["lyrics_source_disagreement_reasons"] == ["duration spread 15.0s"]
    assert report["lyrics_source_candidate_count"] == 3
    assert report["lyrics_source_comparable_candidate_count"] == 2
    assert report["lyrics_source_selection_mode"] == "audio_scored_disagreement"
