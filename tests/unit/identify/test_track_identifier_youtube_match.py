import y2karaoke.core.components.identify.implementation as ti


def test_search_matching_youtube_video_finds_close_match():
    def fake_search(query, lrc_duration, artist, title):
        return {"url": "http://youtube", "duration": lrc_duration}

    identifier = ti.TrackIdentifier(search_youtube_verified_fn=fake_search)

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


def test_search_matching_youtube_video_uses_radio_edit_when_needed():
    calls = []

    def fake_search(query, lrc_duration, artist, title):
        calls.append(query)
        if "radio edit" in query:
            return {"url": "http://radio", "duration": lrc_duration - 1}
        return None

    identifier = ti.TrackIdentifier(search_youtube_verified_fn=fake_search)

    result = identifier._search_matching_youtube_video(
        artist="Artist",
        title="Title",
        lrc_duration=200,
        yt_duration=160,
    )

    assert result is not None
    assert result.youtube_url == "http://radio"
    assert any("radio edit" in q for q in calls)


def test_search_matching_youtube_video_skips_missing_duration():
    def fake_search(query, lrc_duration, artist, title):
        return {"url": "http://youtube", "duration": None}

    identifier = ti.TrackIdentifier(search_youtube_verified_fn=fake_search)

    result = identifier._search_matching_youtube_video(
        artist="Artist",
        title="Title",
        lrc_duration=200,
        yt_duration=190,
    )

    assert result is None
