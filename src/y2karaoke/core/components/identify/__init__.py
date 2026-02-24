"""Identify component facade."""

from ...models import TrackInfo

__all__ = [
    "TrackInfo",
]

try:
    from .implementation import TrackIdentifier

    __all__.append("TrackIdentifier")
except ImportError:
    pass

try:
    from .musicbrainz import MusicBrainzClient

    __all__.append("MusicBrainzClient")
except ImportError:
    pass

try:
    from .parser import QueryParser

    __all__.append("QueryParser")
except ImportError:
    pass

try:
    from .youtube import YouTubeSearcher

    __all__.append("YouTubeSearcher")
except ImportError:
    pass
