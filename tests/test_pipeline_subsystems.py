from y2karaoke.pipeline import alignment, audio, identify, lyrics


def test_lyrics_subsystem_exposes_core_entrypoints():
    from y2karaoke.core.lyrics_whisper import get_lyrics_with_quality

    assert lyrics.get_lyrics_with_quality is get_lyrics_with_quality


def test_alignment_subsystem_exposes_timing_report_entrypoint():
    from y2karaoke.core.timing_evaluator import print_comparison_report

    assert alignment.print_comparison_report is print_comparison_report


def test_identify_subsystem_exposes_track_identifier():
    from y2karaoke.core.track_identifier import TrackIdentifier

    assert identify.TrackIdentifier is TrackIdentifier


def test_audio_subsystem_exposes_downloader():
    from y2karaoke.core.downloader import YouTubeDownloader

    assert audio.YouTubeDownloader is YouTubeDownloader
