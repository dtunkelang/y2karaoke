import y2karaoke.core.track_identifier as ti


def test_normalize_title_removes_stopwords():
    identifier = ti.TrackIdentifier()
    title = "The Sound of Silence"
    assert identifier._normalize_title(title, remove_stopwords=True) == "sound silence"


def test_normalize_title_basic_punctuation():
    identifier = ti.TrackIdentifier()
    title = "Hello, World!"
    assert identifier._normalize_title(title) == "hello world"


def test_score_and_select_best_prefers_clean_shorter():
    identifier = ti.TrackIdentifier()
    matches = [
        (200, "Artist", "Song"),
        (220, "Artist", "Song (Live)"),
        (205, "Artist", "Song [Remix]"),
        (200, "Artist", "Song"),
    ]
    best = identifier._score_and_select_best(matches)
    assert best[0] == 200
    assert best[2] == "Song"


def test_score_and_select_best_penalizes_non_keyword_parens():
    identifier = ti.TrackIdentifier()
    matches = [
        (200, "Artist", "Song"),
        (200, "Artist", "Song (Bonus Track)"),
    ]
    best = identifier._score_and_select_best(matches)
    assert best == (200, "Artist", "Song")


def test_score_and_select_best_all_penalized_prefers_target_duration():
    identifier = ti.TrackIdentifier()
    matches = [
        (210, "Artist", "Song (Live)"),
        (200, "Artist", "Song (Remix)"),
    ]
    best = identifier._score_and_select_best(matches)
    assert best == (210, "Artist", "Song (Live)")


def test_score_and_select_best_empty():
    identifier = ti.TrackIdentifier()
    assert identifier._score_and_select_best([]) is None
