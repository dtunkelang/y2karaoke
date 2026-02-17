"""Visual timing refinement logic (Facade)."""

from .visual.reconstruction import reconstruct_lyrics_from_visuals
from .visual.refinement import refine_word_timings_at_high_fps, _detect_highlight_times

__all__ = [
    "reconstruct_lyrics_from_visuals",
    "refine_word_timings_at_high_fps",
    "_detect_highlight_times",
]
