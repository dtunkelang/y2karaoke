from tools import analyze_line_error_patterns as tool


def test_classify_line_marks_contaminated_sparse_followup() -> None:
    row = tool._classify_line(
        report_line={
            "index": 2,
            "text": "Take me on",
            "start": 6.84,
            "end": 10.42,
            "pre_whisper_start": 6.45,
            "whisper_window_word_count": 2,
            "whisper_window_words": [
                {"text": "on,"},
                {"text": "I'll"},
            ],
        },
        gold_line={"start": 6.85, "end": 9.95},
        contaminated_successors={2},
    )

    assert "sparse_window_support" in row["tags"]
    assert "low_lexical_overlap" in row["tags"]
    assert "contaminated_gap_predecessor" in row["tags"]
    assert "current_closer_than_pre_whisper" in row["tags"]


def test_classify_line_marks_late_exact_window_alignment() -> None:
    row = tool._classify_line(
        report_line={
            "index": 3,
            "text": "I've been inclined",
            "start": 12.0,
            "end": 14.12,
            "pre_whisper_start": 10.82,
            "whisper_window_word_count": 4,
            "whisper_window_words": [
                {"text": "I've"},
                {"text": "been"},
                {"text": "inclined"},
                {"text": "To"},
            ],
        },
        gold_line={"start": 11.95, "end": 14.1},
        contaminated_successors=set(),
    )

    assert "high_lexical_overlap" in row["tags"]
    assert "late_exact_window_alignment" in row["tags"]
    assert "current_closer_than_pre_whisper" in row["tags"]


def test_analyze_report_counts_tags_only_for_meaningful_errors() -> None:
    result = tool._analyze_report(
        report={
            "title": "Song",
            "artist": "Artist",
            "alignment_method": "whisperx",
            "lines": [
                {
                    "index": 1,
                    "text": "a",
                    "start": 0.0,
                    "end": 1.0,
                    "pre_whisper_start": 0.0,
                    "whisper_window_word_count": 0,
                    "whisper_window_words": [],
                },
                {
                    "index": 2,
                    "text": "Take me on",
                    "start": 6.84,
                    "end": 10.42,
                    "pre_whisper_start": 6.45,
                    "whisper_window_word_count": 2,
                    "whisper_window_words": [
                        {"text": "on,"},
                        {"text": "I'll"},
                    ],
                },
            ],
        },
        gold={
            "lines": [
                {"start": 0.0, "end": 1.0},
                {"start": 6.85, "end": 9.95},
            ]
        },
        contamination={"gaps": [{"gap_index": 1, "classification": "echo_fragment"}]},
    )

    assert result["tag_counts"]["contaminated_gap_predecessor"] == 1
    assert result["rows"][0]["index"] == 2
