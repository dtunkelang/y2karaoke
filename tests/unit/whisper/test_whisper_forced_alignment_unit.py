from types import SimpleNamespace

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
