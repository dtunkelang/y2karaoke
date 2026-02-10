import y2karaoke.core.components.identify.implementation as ti


def test_score_split_candidate_rejects_long_duration(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (721, "Artist", "Title")
    score = identifier._score_split_candidate(
        candidate, "Artist", "Title", "Artist Title"
    )

    assert score is None


def test_score_split_candidate_rejects_no_title_overlap(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (False, None)
    )

    candidate = (200, "Artist", "Alpha")
    score = identifier._score_split_candidate(
        candidate, "Artist", "Beta", "Artist Beta"
    )

    assert score is None


def test_score_split_candidate_rewards_matches(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (200, "Artist", "Hello")
    score = identifier._score_split_candidate(
        candidate, "Artist", "Hello", "Artist Hello"
    )

    assert score is not None
    assert score > 50


def test_score_split_candidate_penalizes_song_title_like_artist(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (200, "Symphony Orchestra", "Hello")
    score = identifier._score_split_candidate(
        candidate,
        "Symphony Orchestra",
        "Hello",
        "Symphony Orchestra Hello",
    )

    assert score is not None
    assert score < 100


def test_score_split_candidate_applies_query_penalty_and_overlap(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (200, "Artist Name", "Hello There")
    split_artist = "Artist Name"
    split_title = "Hello World"

    score_without_artist = identifier._score_split_candidate(
        candidate, split_artist, split_title, "Unrelated Query"
    )
    score_with_artist = identifier._score_split_candidate(
        candidate, split_artist, split_title, "Artist Name Hello"
    )

    assert score_without_artist is not None
    assert score_with_artist is not None
    assert score_with_artist > score_without_artist


def test_score_split_candidate_artist_match_bonus(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (200, "Artist Name", "Hello There")
    split_title = "Hello There"

    score_with_match = identifier._score_split_candidate(
        candidate, "Artist Name", split_title, "Artist Name Hello"
    )
    score_without_match = identifier._score_split_candidate(
        candidate, "Other Artist", split_title, "Artist Name Hello"
    )

    assert score_with_match is not None
    assert score_without_match is not None
    assert score_with_match > score_without_match


def test_score_split_candidate_partial_title_match(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    candidate = (200, "Artist", "Hello There")
    score = identifier._score_split_candidate(
        candidate, "Artist", "Hello", "Artist Hello"
    )

    assert score is not None
