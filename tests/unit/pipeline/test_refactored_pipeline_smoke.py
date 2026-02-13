import sys
import types

from y2karaoke.core.karaoke_generate import generate_karaoke
from y2karaoke.core.components.lyrics.sync_pipeline import (
    fetch_lyrics_multi_source_impl,
    _build_provider_search_terms,
)
from y2karaoke.core.components.lyrics.lyrics_whisper_pipeline import (
    get_lyrics_simple_impl,
)


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _CollectingLogger(_DummyLogger):
    def __init__(self):
        self.warning_calls = []

    def warning(self, message, *args, **_kwargs):
        self.warning_calls.append(message % args if args else message)


def test_generate_karaoke_smoke(monkeypatch, tmp_path):
    fake_audio_mod = types.ModuleType("y2karaoke.pipeline.audio")
    fake_audio_mod.extract_video_id = lambda _url: "vid123"
    monkeypatch.setitem(sys.modules, "y2karaoke.pipeline.audio", fake_audio_mod)

    class CacheManager:
        def auto_cleanup(self):
            return None

    class DummyGen:
        cache_manager = CacheManager()

        def __init__(self):
            self._original_prompt = None

        def _prepare_media(self, *_a, **_k):
            return (
                {"title": "Song", "artist": "Artist", "audio_path": "song.wav"},
                None,
                {"vocals_path": "vocals.wav", "instrumental_path": "inst.wav"},
            )

        def _resolve_final_metadata(self, audio_result, lyrics_title, lyrics_artist):
            return (
                lyrics_title or audio_result["title"],
                lyrics_artist or audio_result["artist"],
            )

        def _get_lyrics(self, *_a, **_k):
            return {"lines": [], "metadata": None, "quality": {"source": "test"}}

        def _process_audio_track(self, *_a, **_k):
            return "processed.wav", []

        def _scale_lyrics_timing(self, lines, _tempo):
            return lines

        def _apply_break_edits(self, lines, _edits):
            return lines

        def _apply_splash_offset(self, lines, **_kwargs):
            return lines

        def _build_output_path(self, _title):
            return tmp_path / "out.mp4"

        def _build_background_segments(self, *_a, **_k):
            return None

        def _summarize_quality(self, _lyrics_result):
            return 80.0, [], "high", "ok"

        def _render_video(self, **_kwargs):
            return None

    result = generate_karaoke(
        DummyGen(),
        url="https://youtube.com/watch?v=vid123",
        skip_render=True,
    )

    assert result["video_id"] == "vid123"
    assert result["rendered"] is False
    assert result["quality_score"] == 80.0


def test_fetch_lyrics_multi_source_impl_uses_cache():
    logger = _DummyLogger()

    class State:
        lrc_cache = {("artist", "song"): ("[00:00]Hi", True, "cached", 100)}
        disk_cache = {}

    result = fetch_lyrics_multi_source_impl(
        "Song",
        "Artist",
        synced_only=True,
        enhanced=False,
        target_duration=None,
        duration_tolerance=20,
        offline=False,
        runtime_state=State(),
        disk_cache_enabled_fn=lambda _s: False,
        load_disk_cache_fn=lambda _s: None,
        is_lyriq_available_fn=lambda _s: False,
        fetch_from_lyriq_fn=lambda *_a, **_k: None,
        has_timestamps_fn=lambda *_a, **_k: False,
        get_lrc_duration_fn=lambda *_a, **_k: None,
        set_lrc_cache_fn=lambda *_a, **_k: None,
        is_syncedlyrics_available_fn=lambda _s: False,
        search_with_state_fallback_fn=lambda *_a, **_k: (None, ""),
        logger=logger,
    )
    assert result == ("[00:00]Hi", True, "cached")


def test_fetch_lyrics_multi_source_impl_reuses_cached_on_refetch_failure():
    logger = _DummyLogger()

    class State:
        lrc_cache = {
            ("billie eilish", "bad guy"): (
                "[00:00]Hi",
                True,
                "cached",
                195,
            )
        }
        disk_cache = {}

    set_calls = []

    def _set_lrc_cache(cache_key, value, state=None):
        _ = state
        set_calls.append((cache_key, value))

    result = fetch_lyrics_multi_source_impl(
        "bad guy",
        "Billie Eilish",
        synced_only=True,
        enhanced=False,
        target_duration=220,
        duration_tolerance=8,
        offline=False,
        runtime_state=State(),
        disk_cache_enabled_fn=lambda _s: False,
        load_disk_cache_fn=lambda _s: None,
        is_lyriq_available_fn=lambda _s: False,
        fetch_from_lyriq_fn=lambda *_a, **_k: None,
        has_timestamps_fn=lambda *_a, **_k: False,
        get_lrc_duration_fn=lambda *_a, **_k: None,
        set_lrc_cache_fn=_set_lrc_cache,
        is_syncedlyrics_available_fn=lambda _s: True,
        search_with_state_fallback_fn=lambda *_a, **_k: (None, ""),
        logger=logger,
    )
    assert result == ("[00:00]Hi", True, "cached")
    assert set_calls


def test_fetch_lyrics_multi_source_impl_offline_reuses_cached_mismatch():
    logger = _DummyLogger()

    class State:
        lrc_cache = {
            ("billie eilish", "bad guy"): (
                "[00:00]Hi",
                True,
                "cached",
                195,
            )
        }
        disk_cache = {}
        warning_once_keys = set()

    result = fetch_lyrics_multi_source_impl(
        "bad guy",
        "Billie Eilish",
        synced_only=True,
        enhanced=False,
        target_duration=220,
        duration_tolerance=8,
        offline=True,
        runtime_state=State(),
        disk_cache_enabled_fn=lambda _s: False,
        load_disk_cache_fn=lambda _s: None,
        is_lyriq_available_fn=lambda _s: False,
        fetch_from_lyriq_fn=lambda *_a, **_k: None,
        has_timestamps_fn=lambda *_a, **_k: False,
        get_lrc_duration_fn=lambda *_a, **_k: None,
        set_lrc_cache_fn=lambda *_a, **_k: None,
        is_syncedlyrics_available_fn=lambda _s: True,
        search_with_state_fallback_fn=lambda *_a, **_k: (None, ""),
        logger=logger,
    )
    assert result == ("[00:00]Hi", True, "cached")


def test_fetch_lyrics_multi_source_impl_logs_not_found_once_per_state():
    logger = _CollectingLogger()

    class State:
        lrc_cache = {}
        disk_cache = {}
        warning_once_keys = set()

    state = State()

    kwargs = dict(
        synced_only=True,
        enhanced=False,
        target_duration=197,
        duration_tolerance=20,
        runtime_state=state,
        disk_cache_enabled_fn=lambda _s: False,
        load_disk_cache_fn=lambda _s: None,
        is_lyriq_available_fn=lambda _s: False,
        fetch_from_lyriq_fn=lambda *_a, **_k: None,
        has_timestamps_fn=lambda *_a, **_k: False,
        get_lrc_duration_fn=lambda *_a, **_k: None,
        set_lrc_cache_fn=lambda *_a, **_k: None,
        is_syncedlyrics_available_fn=lambda _s: True,
        search_with_state_fallback_fn=lambda *_a, **_k: (None, ""),
        logger=logger,
    )
    fetch_lyrics_multi_source_impl("Song", "Artist", offline=False, **kwargs)
    fetch_lyrics_multi_source_impl("Song", "Artist", offline=False, **kwargs)

    assert (
        sum(
            "No synced lyrics found from any provider" in line
            for line in logger.warning_calls
        )
        == 1
    )


def test_build_provider_search_terms_strips_noisy_suffixes():
    terms = _build_provider_search_terms(
        "Shape Of You (Penisland produced 2015 version)",
        "Ed Sheeran",
    )
    assert terms
    assert terms[0] == "Ed Sheeran Shape Of You"
    assert "Penisland produced 2015 version" not in terms[0]


def test_get_lyrics_simple_impl_whisper_only_no_vocals():
    lines, metadata = get_lyrics_simple_impl(
        title="Song",
        artist="Artist",
        vocals_path=None,
        lyrics_offset=None,
        romanize=False,
        filter_promos=True,
        target_duration=None,
        evaluate_sources=False,
        use_whisper=False,
        whisper_only=True,
        whisper_map_lrc=False,
        whisper_map_lrc_dtw=False,
        lyrics_file=None,
        whisper_language=None,
        whisper_model=None,
        whisper_force_dtw=False,
        whisper_aggressive=False,
        whisper_temperature=0.0,
        lenient_vocal_activity_threshold=0.3,
        lenient_activity_bonus=0.4,
        low_word_confidence_threshold=0.5,
        offline=False,
        create_no_lyrics_placeholder_fn=lambda *_a, **_k: (["placeholder"], None),
        transcribe_vocals_for_state_fn=lambda *_a, **_k: ([], [], "", "base"),
        create_lines_from_whisper_fn=lambda *_a, **_k: [],
        romanize_lines_fn=lambda *_a, **_k: None,
        load_lyrics_file_fn=lambda *_a, **_k: (None, None, []),
        fetch_lrc_text_and_timings_for_state_fn=lambda *_a, **_k: (None, None, ""),
        get_lrc_duration_fn=lambda *_a, **_k: None,
        fetch_genius_lyrics_with_singers_for_state_fn=lambda *_a, **_k: (None, None),
        detect_and_apply_offset_for_state_fn=lambda *_a, **_k: ([], 0.0),
        create_lines_from_lrc_timings_fn=lambda *_a, **_k: [],
        create_lines_from_lrc_fn=lambda *_a, **_k: [],
        apply_timing_to_lines_fn=lambda *_a, **_k: None,
        extract_text_lines_from_lrc_fn=lambda *_a, **_k: [],
        create_lines_from_plain_text_fn=lambda *_a, **_k: [],
        refine_timing_with_audio_for_state_fn=lambda *_a, **_k: [],
        apply_whisper_alignment_for_state_fn=lambda *_a, **_k: ([], [], {}),
        align_lrc_text_to_whisper_timings_for_state_fn=lambda *_a, **_k: ([], [], {}),
        whisper_lang_to_epitran_for_state_fn=lambda *_a, **_k: "eng-Latn",
        map_lrc_lines_to_whisper_segments_fn=lambda *_a, **_k: ([], 0, []),
        apply_singer_info_fn=lambda *_a, **_k: None,
        logger=_DummyLogger(),
    )
    assert lines == ["placeholder"]
    assert metadata is None
