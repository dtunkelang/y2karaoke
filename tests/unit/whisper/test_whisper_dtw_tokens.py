import sys
from types import SimpleNamespace

from y2karaoke.core.components.whisper import whisper_dtw_tokens as tokens


def test_build_phoneme_dtw_path_reuses_cached_numeric_cost(monkeypatch):
    monkeypatch.setattr(tokens, "_phoneme_similarity_from_ipa", lambda *_: 0.3)

    observed = {}

    def fake_fastdtw(lrc_seq, whisper_seq, dist):
        first = dist(lrc_seq[0], whisper_seq[0])
        second = dist(lrc_seq[0], whisper_seq[0])
        observed["first"] = first
        observed["second"] = second
        return 0.0, [(0, 0)]

    monkeypatch.setitem(sys.modules, "fastdtw", SimpleNamespace(fastdtw=fake_fastdtw))

    path = tokens._build_phoneme_dtw_path(
        lrc_phonemes=[{"ipa": "a", "start": 0.0, "end": 0.1}],
        whisper_phonemes=[{"ipa": "b", "start": 0.0, "end": 0.1}],
        language="eng-Latn",
    )

    assert path == [(0, 0)]
    assert observed["first"] == observed["second"]
    assert observed["second"] == 0.595
