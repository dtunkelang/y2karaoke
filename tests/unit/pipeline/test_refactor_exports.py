import y2karaoke.core.components.lyrics.sync as sync
import y2karaoke.core.components.lyrics.sync_quality as sync_quality
import y2karaoke.core.components.identify.implementation as ti
import y2karaoke.core.components.identify.implementation as ti_impl


def test_sync_reexports_quality_helpers():
    assert sync.get_lrc_duration is sync_quality.get_lrc_duration
    assert sync.validate_lrc_quality is sync_quality.validate_lrc_quality
    assert sync.get_lyrics_quality_report is sync_quality.get_lyrics_quality_report


def test_track_identifier_facade_reexports_musicbrainzngs():
    assert ti.TrackIdentifier is ti_impl.TrackIdentifier
    assert ti.musicbrainzngs is ti_impl.musicbrainzngs
