"""Shared constants for visual bootstrap postprocessing modules."""

from __future__ import annotations

FUSED_FALLBACK_PREFIX_ANCHORS = (
    "i",
    "my",
)
FUSED_FALLBACK_SHORT_FUNCTIONS = ("i", "a", "in", "on", "to", "of", "my")
FUSED_FALLBACK_SUFFIX_ANCHORS = ("in",)

VOCALIZATION_NOISE_TOKENS = {
    "oh",
    "ooh",
    "oooh",
    "woah",
    "whoa",
    "woo",
    "ah",
    "aah",
    "la",
    "na",
    "mm",
    "mmm",
    "hmm",
}
HUM_NOISE_TOKENS = {"mm", "mmm", "hmm"}
COMMON_LYRIC_CHANT_TOKENS = {
    "hey",
    "yeah",
    "yea",
    "yo",
    "no",
    "na",
    "la",
    "oh",
    "ah",
    "woo",
    "ooh",
}
ADLIB_TAIL_TOKENS = {
    "uh",
    "ah",
    "aww",
    "oh",
    "hey",
    "come",
    "on",
}

COLLOQUIAL_EXPANSIONS = {
    "wanna": ("want", "to"),
    "gonna": ("going", "to"),
    "gotta": ("got", "to"),
    "lemme": ("let", "me"),
    "gimme": ("give", "me"),
}
CONTRACTION_RESTORES = {
    "wont": "won't",
    "cant": "can't",
    "dont": "don't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
}

OVERLAY_PLATFORM_TOKENS = {
    "youtube",
    "youtu",
    "tube",
    "facebook",
    "twitter",
    "instagram",
    "tiktok",
}
OVERLAY_CTA_TOKENS = {
    "subscribe",
    "subscribers",
    "subscriber",
    "follow",
    "like",
    "click",
    "watch",
    "channel",
}
OVERLAY_LEGAL_TOKENS = {
    "rights",
    "reserved",
    "association",
    "ltd",
    "limited",
    "copyright",
    "produced",
}
OVERLAY_BRAND_TOKENS = {
    "karaoke",
    "instrumental",
    "collection",
}
