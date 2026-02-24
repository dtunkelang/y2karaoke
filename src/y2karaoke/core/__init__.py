"""Core functionality modules.

This package used to eagerly import audio/render stacks, which pulled optional
dependencies (for example `librosa`) into unrelated visual/lyrics utilities.
Keep imports resilient so submodules can be imported in lighter environments.
"""

from .models import Line, SingerID, SongMetadata, Word

__all__ = [
    "SingerID",
    "Word",
    "Line",
    "SongMetadata",
]

try:
    from .components.alignment import timing_evaluator  # noqa: F401

    __all__.append("timing_evaluator")
except ImportError:
    pass

try:
    from .components.lyrics import genius, lrc, lyrics_whisper, sync

    __all__ += ["sync", "genius", "lrc", "lyrics_whisper"]
except ImportError:
    pass

try:
    from .components.lyrics.api import LyricsProcessor, get_lyrics

    __all__ += ["LyricsProcessor", "get_lyrics"]
except ImportError:
    pass

try:
    from .components.audio.downloader import (
        YouTubeDownloader,
        download_audio,
        download_video,
    )
    from .components.audio.separator import AudioSeparator, separate_vocals
    from .components.audio.audio_effects import AudioProcessor, process_audio

    __all__ += [
        "YouTubeDownloader",
        "download_audio",
        "download_video",
        "AudioSeparator",
        "separate_vocals",
        "AudioProcessor",
        "process_audio",
    ]
except ImportError:
    pass

try:
    from .components.render.video_writer import render_karaoke_video
    from .components.render.backgrounds import (
        BackgroundProcessor,
        BackgroundSegment,
        create_background_segments,
    )

    __all__ += [
        "render_karaoke_video",
        "BackgroundProcessor",
        "BackgroundSegment",
        "create_background_segments",
    ]
except ImportError:
    pass

try:
    from .karaoke import KaraokeGenerator  # noqa: F401

    __all__.append("KaraokeGenerator")
except ImportError:
    pass
