from tools import simulate_phonetic_span_nudges as tool


def test_simulate_line_uses_best_span_word_timing() -> None:
    line = {
        "index": 12,
        "text": "De guayarte, ma...",
        "start": 28.89,
        "end": 29.94,
        "whisper_window_word_count": 4,
        "whisper_window_words": [
            {"text": "dam,", "start": 28.22, "end": 28.56},
            {"text": "degollarte", "start": 28.64, "end": 29.30},
            {"text": "mami.", "start": 29.30, "end": 29.98},
            {"text": "foo", "start": 29.98, "end": 30.10},
        ],
    }

    result = tool._simulate_line(line, language="es", min_joint_score=0.6)

    assert result["eligible"] is True
    assert result["candidate_start"] == 28.64
    assert result["candidate_end"] == 29.98
    assert result["start_shift"] == -0.25
    assert result["end_shift"] == 0.04


def test_simulate_line_rejects_low_score_span() -> None:
    line = {
        "index": 2,
        "text": "Take me on",
        "start": 6.84,
        "end": 10.42,
        "whisper_window_word_count": 0,
        "whisper_window_words": [],
    }

    result = tool._simulate_line(line, language="en", min_joint_score=0.6)

    assert result["eligible"] is False
    assert result["candidate_start"] is None
