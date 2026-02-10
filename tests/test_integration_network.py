import pytest

from y2karaoke.core import sync
from y2karaoke.core.components.identify import youtube_metadata


@pytest.mark.network
@pytest.mark.integration
def test_youtube_metadata_real():
    info = youtube_metadata.extract_metadata_from_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    assert info["video_id"]
    assert len(info["video_id"]) == 11
    assert info["title"]


@pytest.mark.network
@pytest.mark.integration
def test_fetch_lyrics_multi_source_real():
    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source(
        "Imagine", "John Lennon", synced_only=True
    )
    if not lrc_text:
        pytest.skip("No synced lyrics found from providers")
    assert is_synced is True
    assert source
    assert sync._has_timestamps(lrc_text)
