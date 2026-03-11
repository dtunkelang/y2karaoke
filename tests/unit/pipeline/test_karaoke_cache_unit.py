import json
from pathlib import Path  # noqa: F401

import pytest
from unittest.mock import MagicMock

from y2karaoke.exceptions import Y2KaraokeError
from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.utils.cache import CacheManager


def test_download_audio_uses_cached_original(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    cache.save_metadata(video_id, {"title": "Song", "artist": "Artist"})

    video_dir = cache.get_video_cache_dir(video_id)
    original = video_dir / "Artist - Song.wav"
    original.write_bytes(b"data")
    (video_dir / "audio_vocals.wav").write_bytes(b"vocals")

    result = generator._download_audio(
        video_id,
        "https://youtu.be/abc123",
        False,
        expected_title="Song",
        expected_artist="Artist",
    )

    assert result["audio_path"] == str(original)
    generator.downloader.download_audio.assert_not_called()


def test_download_audio_rejects_cached_karaoke_asset(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock(
        return_value={
            "audio_path": "downloaded.wav",
            "title": "Song",
            "artist": "Artist",
        }
    )
    generator.downloader.download_audio.return_value = {
        "audio_path": "downloaded.wav",
        "title": "Song",
        "artist": "Artist",
    }

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    cache.save_metadata(
        video_id,
        {"title": "The Weeknd | Karaoke Version | KaraFun", "artist": "Song"},
    )

    video_dir = cache.get_video_cache_dir(video_id)
    bad_audio = video_dir / "Song - Artist Karaoke Version KaraFun.wav"
    bad_audio.write_bytes(b"data")

    result = generator._download_audio(
        video_id,
        "https://youtu.be/abc123",
        False,
        expected_title="Song",
        expected_artist="Artist",
    )

    assert result["audio_path"] == "downloaded.wav"
    generator.downloader.download_audio.assert_called_once()


def test_download_audio_offline_rejects_cached_karaoke_asset(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    cache.save_metadata(
        video_id,
        {"title": "The Weeknd | Karaoke Version | KaraFun", "artist": "Song"},
    )

    video_dir = cache.get_video_cache_dir(video_id)
    bad_audio = video_dir / "Song - Artist Karaoke Version KaraFun.wav"
    bad_audio.write_bytes(b"data")

    with pytest.raises(Y2KaraokeError):
        generator._download_audio(
            video_id,
            "https://youtu.be/abc123",
            False,
            offline=True,
            expected_title="Song",
            expected_artist="Artist",
        )


def test_download_audio_skips_cached_stem_files(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    cache.save_metadata(video_id, {"title": "Song", "artist": "Artist"})

    video_dir = cache.get_video_cache_dir(video_id)
    (video_dir / "Artist - Song_(Vocals)_htdemucs_ft.wav").write_bytes(b"vocals")
    (video_dir / "Artist - Song_instrumental.wav").write_bytes(b"inst")

    with pytest.raises(Y2KaraokeError):
        generator._download_audio(
            video_id,
            "https://youtu.be/abc123",
            False,
            offline=True,
            expected_title="Song",
            expected_artist="Artist",
        )


def test_download_video_uses_cached(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    generator.downloader = MagicMock()

    cache = CacheManager(tmp_path)
    video_id = "abc123"
    video_dir = cache.get_video_cache_dir(video_id)
    cached_video = video_dir / "abc123_video.mp4"
    cached_video.write_bytes(b"video")

    result = generator._download_video(video_id, "https://youtu.be/abc123", False)

    assert result["video_path"] == str(cached_video)
    generator.downloader.download_video.assert_not_called()


def test_get_lyrics_passes_arguments(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)

    captured = {}

    def fake_get_lyrics_with_quality(**kwargs):
        captured.update(kwargs)
        return ([], None, {"overall_score": 80.0, "issues": [], "source": "test"})

    monkeypatch.setattr(
        "y2karaoke.pipeline.lyrics.get_lyrics_with_quality",
        fake_get_lyrics_with_quality,
    )

    result = generator._get_lyrics(
        "Title",
        "Artist",
        "vocals.wav",
        "vid",
        False,
        lyrics_offset=1.0,
        target_duration=123,
        evaluate_sources=True,
        use_whisper=True,
        whisper_language="en",
        whisper_model="base",
        filter_promos=False,
    )

    assert result["quality"]["overall_score"] == 80.0
    assert captured["title"] == "Title"
    assert captured["artist"] == "Artist"
    assert captured["vocals_path"] == "vocals.wav"
    assert captured["lyrics_offset"] == 1.0
    assert captured["target_duration"] == 123
    assert captured["evaluate_sources"] is True
    assert captured["use_whisper"] is True
    assert captured["whisper_language"] == "en"
    assert captured["whisper_model"] == "base"
    assert captured["filter_promos"] is False


def test_shorten_breaks_uses_cached_edits(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    video_id = "vid"

    cache = generator.cache_manager
    shortened_name = "shortened_breaks_30s.wav"
    edits_name = "shortened_breaks_30s_edits.json"

    shortened_path = cache.get_file_path(video_id, shortened_name)
    shortened_path.write_bytes(b"audio")

    edits_path = cache.get_file_path(video_id, edits_name)
    edits_data = [
        {
            "original_start": 10.0,
            "original_end": 50.0,
            "new_end": 30.0,
            "time_removed": 20.0,
            "cut_start": 12.0,
        }
    ]
    edits_path.write_text(json.dumps(edits_data))

    shortened, edits = generator._shorten_breaks(
        "audio.wav",
        "vocals.wav",
        "inst.wav",
        video_id,
        max_break_duration=30.0,
        force=False,
    )

    assert shortened == str(shortened_path)
    assert len(edits) == 1
    assert edits[0].original_start == 10.0
