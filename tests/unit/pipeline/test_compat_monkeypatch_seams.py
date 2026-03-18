from y2karaoke.core.components.lyrics import lyrics_whisper as lw
from y2karaoke.core.components.whisper import whisper_integration as wi


def test_whisper_integration_hooks_restore_transcribe():
    original = wi.transcribe_vocals

    def fake_transcribe(*_args, **_kwargs):
        return [], [], "", "base"

    with wi.use_whisper_integration_hooks(transcribe_vocals_fn=fake_transcribe):
        assert wi.transcribe_vocals is fake_transcribe

    assert wi.transcribe_vocals is original


def test_lyrics_whisper_hooks_restore_fetch():
    original = lw._fetch_lrc_text_and_timings_for_state

    def fake_fetch(*_args, **_kwargs):
        return None, None, ""

    with lw.use_lyrics_whisper_hooks(fetch_lrc_text_and_timings_fn=fake_fetch):
        assert lw._fetch_lrc_text_and_timings_for_state(
            title="Song", artist="Artist"
        ) == (None, None, "")

    assert lw._fetch_lrc_text_and_timings_for_state is original


def test_get_lyrics_simple_accepts_explicit_hooks():
    lines, metadata = lw.get_lyrics_simple(
        title="Song",
        artist="Artist",
        vocals_path=None,
        hooks=lw.LyricsWhisperHooks(
            fetch_lrc_text_and_timings_fn=lambda *_a, **_k: (None, None, ""),
            fetch_genius_lyrics_with_singers_fn=lambda *_a, **_k: (None, None),
        ),
    )

    assert metadata is not None
    assert lines
