import y2karaoke.core.track_identifier as ti


def test_search_matching_youtube_video_finds_close_match(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search(query, lrc_duration, artist, title):
        return {"url": "http://youtube", "duration": lrc_duration}

    monkeypatch.setattr(identifier, "_search_youtube_verified", fake_search)

    result = identifier._search_matching_youtube_video(
        artist="Artist",
        title="Title",
        lrc_duration=200,
        yt_duration=180,
    )

    assert result is not None
    assert result.youtube_url == "http://youtube"
    assert result.lrc_validated is True
    assert result.duration == 200


def test_search_matching_youtube_video_uses_radio_edit_when_needed(monkeypatch):
    identifier = ti.TrackIdentifier()

    calls = []

    def fake_search(query, lrc_duration, artist, title):
        calls.append(query)
        if "radio edit" in query:
            return {"url": "http://radio", "duration": lrc_duration - 1}
        return None

    monkeypatch.setattr(identifier, "_search_youtube_verified", fake_search)

    result = identifier._search_matching_youtube_video(
        artist="Artist",
        title="Title",
        lrc_duration=200,
        yt_duration=160,
    )

    assert result is not None
    assert result.youtube_url == "http://radio"
    assert any("radio edit" in q for q in calls)


def test_search_matching_youtube_video_skips_missing_duration(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search(query, lrc_duration, artist, title):
        return {"url": "http://youtube", "duration": None}

    monkeypatch.setattr(identifier, "_search_youtube_verified", fake_search)

    result = identifier._search_matching_youtube_video(
        artist="Artist",
        title="Title",
        lrc_duration=200,
        yt_duration=190,
    )

    assert result is None
