import json

from y2karaoke.core import timing_evaluator as te


def test_get_whisper_cache_path_none_for_missing(tmp_path):
    missing = tmp_path / "missing.wav"
    assert te._get_whisper_cache_path(str(missing), "base", None) is None


def test_save_and_load_whisper_cache(tmp_path):
    cache_path = tmp_path / "cache.json"
    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]
    words = [word]

    te._save_whisper_cache(str(cache_path), segments, words, "en")
    loaded = te._load_whisper_cache(str(cache_path))

    assert loaded is not None
    loaded_segments, loaded_words, lang = loaded
    assert lang == "en"
    assert loaded_segments[0].text == "hello"
    assert loaded_words[0].text == "hello"


def test_load_whisper_cache_handles_bad_json(tmp_path):
    cache_path = tmp_path / "bad.json"
    cache_path.write_text("{not-json")
    assert te._load_whisper_cache(str(cache_path)) is None


def test_transcribe_vocals_uses_cache(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    cache_path = te._get_whisper_cache_path(str(audio_path), "base", None)
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="cached", words=[])]
    words = [te.TranscriptionWord(start=0.1, end=0.2, text="cached")]
    te._save_whisper_cache(cache_path, segments, words, "en")

    monkeypatch.setattr(te, "_load_whisper_cache", lambda *_: (segments, words, "en"))

    got_segments, got_words, lang = te.transcribe_vocals(str(audio_path))
    assert got_segments[0].text == "cached"
    assert got_words[0].text == "cached"
    assert lang == "en"


def test_transcribe_vocals_handles_missing_whisper(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    import builtins

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no whisper")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(te, "_load_whisper_cache", lambda *_: None)

    segments, words, lang = te.transcribe_vocals(str(audio_path))
    assert segments == []
    assert words == []
    assert lang == ""


def test_align_dtw_whisper_falls_back_without_fastdtw(monkeypatch):
    from y2karaoke.core.models import Line, Word

    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    whisper_words = [te.TranscriptionWord(start=0.1, end=0.2, text="hello")]

    import builtins

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastdtw":
            raise ImportError("no fastdtw")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(te, "_get_ipa", lambda *a, **k: None)

    aligned, corrections, metrics = te.align_dtw_whisper(lines, whisper_words)
    assert aligned == lines
    assert corrections == []
    assert metrics["matched_ratio"] == 0.0
