from tools import analyze_phonetic_line_support as tool


def test_analyze_line_scores_noisy_spanish_tail_tokens() -> None:
    line = {
        "index": 12,
        "text": "De guayarte, ma...",
        "start": 28.89,
        "end": 29.94,
        "whisper_window_word_count": 4,
        "whisper_window_words": [
            {"text": "dam,"},
            {"text": "degollarte"},
            {"text": "mami"},
            {"text": "foo"},
        ],
    }

    result = tool._analyze_line(line, language="es")

    assert result["index"] == 12
    assert result["token_scores"][1]["best_match"] == "degollarte"
    assert result["token_scores"][1]["best_joint_similarity"] >= 0.55
    assert result["token_scores"][2]["best_match"] == "mami"
    assert result["token_scores"][2]["best_joint_similarity"] >= 0.66
    assert result["best_span"]["span_text"] == "degollarte mami"
    assert result["best_span"]["joint_score"] >= 0.55
