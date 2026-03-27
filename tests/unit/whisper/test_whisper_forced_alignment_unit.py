from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.whisper import whisper_forced_alignment as wfa


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None


def _line(*tokens: str) -> Line:
    words = [
        Word(text=token, start_time=10.0 + idx * 0.2, end_time=10.1 + idx * 0.2)
        for idx, token in enumerate(tokens)
    ]
    return Line(words=words)


def test_index_aligned_segments_falls_back_to_input_order_when_id_missing():
    aligned_segments = [{"text": "a"}, {"id": 9, "text": "b"}, {"text": "c"}]
    seg_by_idx = wfa._index_aligned_segments(aligned_segments, [3, 4, 5])
    assert seg_by_idx[3]["text"] == "a"
    assert seg_by_idx[9]["text"] == "b"
    assert seg_by_idx[5]["text"] == "c"


def test_align_lines_with_whisperx_accepts_segments_without_ids(monkeypatch):
    lines = [
        _line("Oh", "ma"),
        _line("douce", "souffrance"),
        _line("Pourquoi", "recommence"),
        _line("Je", "danse"),
    ]

    class _WhisperX:
        @staticmethod
        def load_audio(_path):
            return [0.0] * 32000

        @staticmethod
        def load_align_model(*, language_code, device):
            assert language_code == "fr"
            assert device == "cpu"
            return object(), {}

        @staticmethod
        def align(
            _segs, _align_model, _metadata, _audio, *, device, return_char_alignments
        ):
            assert device == "cpu"
            assert return_char_alignments is False
            return {
                "segments": [
                    {
                        "start": 10.0,
                        "end": 10.5,
                        "text": "Oh ma",
                        "words": [
                            {"word": "Oh", "start": 10.0, "end": 10.1},
                            {"word": "ma", "start": 10.2, "end": 10.3},
                        ],
                    },
                    {
                        "start": 11.0,
                        "end": 11.8,
                        "text": "douce souffrance",
                        "words": [
                            {"word": "douce", "start": 11.0, "end": 11.3},
                            {"word": "souffrance", "start": 11.35, "end": 11.8},
                        ],
                    },
                    {
                        "start": 12.0,
                        "end": 12.6,
                        "text": "Pourquoi recommence",
                        "words": [
                            {"word": "Pourquoi", "start": 12.0, "end": 12.3},
                            {"word": "recommence", "start": 12.35, "end": 12.6},
                        ],
                    },
                    {
                        "start": 13.0,
                        "end": 13.3,
                        "text": "Je danse",
                        "words": [
                            {"word": "Je", "start": 13.0, "end": 13.1},
                            {"word": "danse", "start": 13.15, "end": 13.3},
                        ],
                    },
                ]
            }

    monkeypatch.setattr(wfa, "patch_torchaudio_for_whisperx", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "whisperx", _WhisperX)

    forced = wfa.align_lines_with_whisperx(
        lines,
        "vocals.wav",
        "fr",
        _Logger(),
    )

    assert forced is not None
    forced_lines, metrics = forced
    assert len(forced_lines) == 4
    assert forced_lines[0].words[0].text == "Oh"
    assert forced_lines[1].words[0].start_time == 11.0
    assert metrics["forced_line_coverage"] == 1.0


def test_map_segment_words_to_line_tightens_leading_article_before_content_anchor():
    line = _line("The", "needle", "tears", "a", "hole")
    seg_words = [
        (19.84, 20.16, "real"),
        (20.16, 22.48, "the"),
        (22.48, 23.08, "needle"),
        (23.08, 24.18, "tears"),
        (24.18, 24.74, "the"),
        (24.74, 25.18, "hole"),
    ]

    mapped = wfa._map_segment_words_to_line(line, seg_words, 19.84, 25.18)

    assert mapped[0].text == "The"
    assert mapped[1].text == "needle"
    assert mapped[0].start_time >= 22.0
    assert mapped[0].end_time <= mapped[1].start_time
    assert abs(mapped[1].start_time - 22.48) < 0.05


def test_align_lines_with_whisperx_accepts_two_line_repeated_hook(monkeypatch):
    lines = [
        _line("Ah", "ha", "ha", "ha", "stayin'", "alive", "stayin'", "alive"),
        _line("Ah", "ha", "ha", "ha", "stayin'", "alive"),
    ]

    class _WhisperX:
        @staticmethod
        def load_audio(_path):
            return [0.0] * 32000

        @staticmethod
        def load_align_model(*, language_code, device):
            assert language_code == "en"
            assert device == "cpu"
            return object(), {}

        @staticmethod
        def align(
            _segs, _align_model, _metadata, _audio, *, device, return_char_alignments
        ):
            assert device == "cpu"
            assert return_char_alignments is False
            return {
                "segments": [
                    {
                        "start": 1.3,
                        "end": 5.7,
                        "text": "Ah ha ha ha stayin' alive stayin' alive",
                        "words": [
                            {"word": "Ah", "start": 1.3, "end": 1.55},
                            {"word": "ha", "start": 1.6, "end": 1.85},
                            {"word": "ha", "start": 1.9, "end": 2.15},
                            {"word": "ha", "start": 2.2, "end": 2.45},
                            {"word": "stayin'", "start": 2.7, "end": 3.4},
                            {"word": "alive", "start": 3.45, "end": 4.2},
                            {"word": "stayin'", "start": 4.25, "end": 4.95},
                            {"word": "alive", "start": 5.0, "end": 5.7},
                        ],
                    },
                    {
                        "start": 5.85,
                        "end": 8.95,
                        "text": "Ah ha ha ha stayin' alive",
                        "words": [
                            {"word": "Ah", "start": 5.85, "end": 6.1},
                            {"word": "ha", "start": 6.15, "end": 6.4},
                            {"word": "ha", "start": 6.45, "end": 6.7},
                            {"word": "ha", "start": 6.75, "end": 7.0},
                            {"word": "stayin'", "start": 7.25, "end": 8.0},
                            {"word": "alive", "start": 8.05, "end": 8.95},
                        ],
                    },
                ]
            }

    monkeypatch.setattr(wfa, "patch_torchaudio_for_whisperx", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "whisperx", _WhisperX)

    forced = wfa.align_lines_with_whisperx(
        lines,
        "vocals.wav",
        "en",
        _Logger(),
    )

    assert forced is not None
    forced_lines, metrics = forced
    assert len(forced_lines) == 2
    assert forced_lines[0].start_time == 1.3
    assert forced_lines[1].start_time == 5.85
    assert metrics["forced_line_coverage"] == 1.0
    assert metrics["forced_word_coverage"] == 1.0


def test_align_lines_with_whisperx_trace_includes_line_mapping_details(
    monkeypatch, tmp_path
):
    lines = [
        _line("I've", "been", "inclined"),
        _line("To", "believe", "they", "never", "would"),
    ]

    class _WhisperX:
        observed_return_char_alignments = None

        @staticmethod
        def load_audio(_path):
            return [0.0] * 32000

        @staticmethod
        def load_align_model(*, language_code, device):
            assert language_code == "en"
            assert device == "cpu"
            return object(), {}

        @staticmethod
        def align(
            _segs, _align_model, _metadata, _audio, *, device, return_char_alignments
        ):
            assert device == "cpu"
            _WhisperX.observed_return_char_alignments = return_char_alignments
            return {
                "segments": [
                    {
                        "start": 12.7,
                        "end": 14.1,
                        "text": "I've been inclined",
                        "words": [
                            {"word": "I've", "start": 12.741, "end": 13.01},
                            {"word": "been", "start": 13.02, "end": 13.31},
                            {"word": "inclined", "start": 13.33, "end": 14.125},
                        ],
                    },
                    {
                        "start": 16.6,
                        "end": 19.2,
                        "text": "To believe they never would",
                        "words": [
                            {"word": "To", "start": 16.656, "end": 17.01},
                            {"word": "believe", "start": 17.02, "end": 17.61},
                            {"word": "they", "start": 17.62, "end": 17.95},
                            {"word": "never", "start": 17.96, "end": 18.4},
                            {"word": "would", "start": 18.41, "end": 19.183},
                        ],
                    },
                ]
            }

    trace_path = tmp_path / "forced.json"
    monkeypatch.setattr(wfa, "patch_torchaudio_for_whisperx", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "whisperx", _WhisperX)
    monkeypatch.setenv("Y2K_TRACE_WHISPERX_FORCED_JSON", str(trace_path))
    monkeypatch.setenv("Y2K_TRACE_WHISPERX_FORCED_CHAR_ALIGN", "1")

    forced = wfa.align_lines_with_whisperx(
        lines,
        "vocals.wav",
        "en",
        _Logger(),
    )

    assert forced is not None
    trace = __import__("json").loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "accepted"
    assert trace["return_char_alignments"] is True
    assert _WhisperX.observed_return_char_alignments is True
    assert len(trace["requested_segments"]) == 2
    assert trace["requested_segments"][0]["text"] == "I've been inclined"
    assert len(trace["aligned_segments"]) == 2
    assert len(trace["line_mappings"]) == 2
    assert trace["line_mappings"][0]["line_index"] == 0
    assert trace["line_mappings"][0]["line_text"] == "I've been inclined"
    assert trace["line_mappings"][0]["match_count"] == 3
    assert trace["line_mappings"][0]["mapped_words"][0]["text"] == "I've"
    assert trace["line_mappings"][1]["segment_text"] == "To believe they never would"
