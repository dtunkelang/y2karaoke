from y2karaoke.pipeline import alignment, audio, identify, lyrics, render


def test_lyrics_subsystem_exposes_core_entrypoints():
    from y2karaoke.core.components.lyrics.lyrics_whisper import get_lyrics_with_quality

    assert lyrics.get_lyrics_with_quality is get_lyrics_with_quality


def test_alignment_subsystem_exposes_timing_report_entrypoint():
    from y2karaoke.core.components.alignment.timing_evaluator import (
        print_comparison_report,
    )

    assert alignment.print_comparison_report is print_comparison_report


def test_identify_subsystem_exposes_track_identifier():
    from y2karaoke.core.components.identify.implementation import TrackIdentifier

    assert identify.TrackIdentifier is TrackIdentifier


def test_audio_subsystem_exposes_downloader():
    from y2karaoke.core.components.audio.downloader import YouTubeDownloader

    assert audio.YouTubeDownloader is YouTubeDownloader


def test_render_subsystem_exposes_video_renderer():
    from y2karaoke.core.components.render.video_writer import render_karaoke_video

    assert render.render_karaoke_video is render_karaoke_video
