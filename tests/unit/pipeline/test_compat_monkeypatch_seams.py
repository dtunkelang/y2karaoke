from y2karaoke.core.components.lyrics import lyrics_whisper as lw
from y2karaoke.core.components.whisper import whisper_integration as wi
import y2karaoke.core.karaoke as karaoke_mod


def test_karaoke_extract_video_id_alias_is_preserved():
    assert hasattr(karaoke_mod, "extract_video_id")
    assert callable(karaoke_mod.extract_video_id)


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
