from y2karaoke.core import timing_evaluator as te
from y2karaoke.core.components.whisper import whisper_dtw as wdtw
from y2karaoke.core.components.whisper import whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_cache as wc


def test_get_whisper_cache_path_none_for_missing(tmp_path):
    missing = tmp_path / "missing.wav"
    assert wc._get_whisper_cache_path(str(missing), "base", None) is None


def test_save_and_load_whisper_cache(tmp_path):
    cache_path = tmp_path / "cache.json"
    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]
    words = [word]

    wc._save_whisper_cache(str(cache_path), segments, words, "en", "base", False)
    loaded = wc._load_whisper_cache(str(cache_path))

    assert loaded is not None
    loaded_segments, loaded_words, lang = loaded
    assert lang == "en"
    assert loaded_segments[0].text == "hello"
    assert loaded_words[0].text == "hello"


def test_load_whisper_cache_handles_bad_json(tmp_path):
    cache_path = tmp_path / "bad.json"
    cache_path.write_text("{not-json")
    assert wc._load_whisper_cache(str(cache_path)) is None


def test_transcribe_vocals_uses_cache(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    cache_path = wc._get_whisper_cache_path(str(audio_path), "base", None)
    assert cache_path is not None
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="cached", words=[])]
    words = [te.TranscriptionWord(start=0.1, end=0.2, text="cached")]
    wc._save_whisper_cache(cache_path, segments, words, "en", "base", False)

    with wi.use_whisper_integration_hooks(
        load_whisper_cache_fn=lambda *_: (segments, words, "en")
    ):
        got_segments, got_words, lang, model = te.transcribe_vocals(str(audio_path))
    assert got_segments[0].text == "cached"
    assert got_words[0].text == "cached"
    assert lang == "en"
    assert model == "base"


def test_transcribe_vocals_handles_missing_whisper(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    def missing_whisper():
        raise ImportError("no whisper")

    with wi.use_whisper_integration_hooks(
        load_whisper_model_class_fn=missing_whisper,
        load_whisper_cache_fn=lambda *_: None,
    ):
        segments, words, lang, model = te.transcribe_vocals(str(audio_path))
    assert segments == []
    assert words == []
    assert lang == ""
    assert model == "base"


def test_find_best_cached_auto_accepts_explicit_language(tmp_path):
    """When language=None (auto), should find caches saved with explicit language."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    # Save a cache with explicit language "en"
    cache_path = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=True)
    assert cache_path is not None
    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]
    wc._save_whisper_cache(cache_path, segments, [word], "en", "large", True)

    # Search with language=None (auto) should find it
    result = wc._find_best_cached_whisper_model(str(audio), None, True, "large")
    assert result is not None
    found_path, found_model = result
    assert found_model == "large"
    assert "_en_" in found_path


def test_find_best_cached_explicit_accepts_auto_cache(tmp_path):
    """When language='en', should find caches saved with auto-detect."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    # Save a cache with auto-detect
    cache_path = wc._get_whisper_cache_path(str(audio), "large", None, aggressive=False)
    assert cache_path is not None
    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]
    wc._save_whisper_cache(cache_path, segments, [word], "en", "large", False)

    # Search with explicit language="en" should find the auto cache
    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is not None
    found_path, found_model = result
    assert found_model == "large"
    assert "_auto" in found_path


def test_find_best_cached_prefers_exact_language_match(tmp_path):
    """When both exact and cross-language caches exist, prefer exact match."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    # Save auto cache
    auto_path = wc._get_whisper_cache_path(str(audio), "large", None, aggressive=False)
    assert auto_path is not None
    wc._save_whisper_cache(auto_path, segments, [word], "en", "large", False)

    # Save explicit "en" cache
    en_path = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=False)
    assert en_path is not None
    wc._save_whisper_cache(en_path, segments, [word], "en", "large", False)

    # Search with language="en" should prefer the exact "en" cache
    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is not None
    found_path, found_model = result
    assert "_en" in found_path
    assert "_auto" not in found_path


def test_find_best_cached_rejects_wrong_explicit_language(tmp_path):
    """When language='en', should NOT match a cache saved with language='fr'."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    # Save a cache with explicit language "fr"
    cache_path = wc._get_whisper_cache_path(str(audio), "large", "fr", aggressive=False)
    assert cache_path is not None
    word = te.TranscriptionWord(start=0.1, end=0.2, text="bonjour")
    segments = [
        te.TranscriptionSegment(start=0.0, end=1.0, text="bonjour", words=[word])
    ]
    wc._save_whisper_cache(cache_path, segments, [word], "fr", "large", False)

    # Search with language="en" should NOT find the "fr" cache
    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is None


def test_find_best_cached_prefers_non_aggressive_when_requested(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    non_aggr = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=False)
    aggr = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=True)
    assert non_aggr is not None
    assert aggr is not None
    wc._save_whisper_cache(non_aggr, segments, [word], "en", "large", False)
    wc._save_whisper_cache(aggr, segments, [word], "en", "large", True)

    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is not None
    found_path, _ = result
    assert "_aggr" not in found_path


def test_find_best_cached_prefers_aggressive_when_requested(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    non_aggr = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=False)
    aggr = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=True)
    assert non_aggr is not None
    assert aggr is not None
    wc._save_whisper_cache(non_aggr, segments, [word], "en", "large", False)
    wc._save_whisper_cache(aggr, segments, [word], "en", "large", True)

    result = wc._find_best_cached_whisper_model(str(audio), "en", True, "large")
    assert result is not None
    found_path, _ = result
    assert "_aggr" in found_path


def test_find_best_cached_aggressive_requires_aggressive_cache(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    non_aggr = wc._get_whisper_cache_path(str(audio), "large", "en", aggressive=False)
    assert non_aggr is not None
    wc._save_whisper_cache(non_aggr, segments, [word], "en", "large", False)

    result = wc._find_best_cached_whisper_model(str(audio), "en", True, "large")
    assert result is None


def test_find_best_cached_prefers_higher_model_over_mode_match(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    # Non-aggressive lower model cache
    medium_path = wc._get_whisper_cache_path(
        str(audio), "medium", None, aggressive=False
    )
    assert medium_path is not None
    wc._save_whisper_cache(medium_path, segments, [word], "en", "medium", False)

    # Aggressive requested-model cache
    large_aggr_path = wc._get_whisper_cache_path(
        str(audio), "large", "en", aggressive=True
    )
    assert large_aggr_path is not None
    wc._save_whisper_cache(large_aggr_path, segments, [word], "en", "large", True)

    # Requesting non-aggressive large should still prefer large over medium.
    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is not None
    found_path, found_model = result
    assert found_model == "large"
    assert "_aggr" in found_path


def test_find_best_cached_requires_model_at_least_target(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"fake")

    word = te.TranscriptionWord(start=0.1, end=0.2, text="hello")
    segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])]

    # Only a lower-than-target cache exists.
    small_path = wc._get_whisper_cache_path(str(audio), "small", "en", aggressive=False)
    assert small_path is not None
    wc._save_whisper_cache(small_path, segments, [word], "en", "small", False)

    result = wc._find_best_cached_whisper_model(str(audio), "en", False, "large")
    assert result is None


def test_align_dtw_whisper_falls_back_without_fastdtw():
    from y2karaoke.core.models import Line, Word

    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    whisper_words = [te.TranscriptionWord(start=0.1, end=0.2, text="hello")]

    with wi.use_whisper_integration_hooks(get_ipa_fn=lambda *a, **k: None):
        with wdtw.use_whisper_dtw_hooks(
            load_fastdtw_fn=lambda: (_ for _ in ()).throw(ImportError("no fastdtw"))
        ):
            aligned, corrections, metrics = te.align_dtw_whisper(lines, whisper_words)
    assert aligned == lines
    assert corrections == []
    assert metrics["matched_ratio"] == 0.0
