from y2karaoke.core.components.identify import youtube_rules as rules


def test_is_likely_non_studio_handles_live_and_radio_edit():
    assert rules.is_likely_non_studio("Artist - Song (Live at Wembley)")
    assert not rules.is_likely_non_studio("Artist - Song (Radio Edit)")


def test_is_preferred_audio_title():
    assert rules.is_preferred_audio_title("Artist - Song (Official Audio)")
    assert not rules.is_preferred_audio_title("Artist - Song (Official Video)")


def test_query_wants_non_studio_and_tolerance():
    assert rules.query_wants_non_studio("artist song live")
    assert rules.youtube_duration_tolerance(200) == 30
    assert rules.youtube_duration_tolerance(0) == 30


def test_extract_youtube_candidates_parses_duration():
    text = (
        '"videoRenderer":{"videoId":"abc123def45","title":{"runs":[{"text":"Song"}]},'
        '"lengthText":{"simpleText":"1:02:03"}}'
    )
    candidates = rules.extract_youtube_candidates(text)
    assert candidates == [
        {"video_id": "abc123def45", "title": "Song", "duration": 3723}
    ]
