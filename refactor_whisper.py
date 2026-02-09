import re

file_path = "src/y2karaoke/core/whisper_integration.py"
with open(file_path, "r") as f:
    content = f.read()

# Avoid double prefixing
replacements = [
    (r"(?<!models\.)\bLine\b", "models.Line"),
    (r"(?<!models\.)\bWord\b", "models.Word"),
    (r"(?<!timing_models\.)\bTranscriptionWord\b", "timing_models.TranscriptionWord"),
    (
        r"(?<!timing_models\.)\bTranscriptionSegment\b",
        "timing_models.TranscriptionSegment",
    ),
    (r"(?<!timing_models\.)\bAudioFeatures\b", "timing_models.AudioFeatures"),
    (r"(?<!phonetic_utils\.)\b_get_ipa\b", "phonetic_utils._get_ipa"),
    (
        r"(?<!phonetic_utils\.)\b_phonetic_similarity\b",
        "phonetic_utils._phonetic_similarity",
    ),
    (
        r"(?<!phonetic_utils\.)\b_whisper_lang_to_epitran\b",
        "phonetic_utils._whisper_lang_to_epitran",
    ),
    (
        r"(?<!whisper_cache\.)\b_get_whisper_cache_path\b",
        "whisper_cache._get_whisper_cache_path",
    ),
    (
        r"(?<!whisper_cache\.)\b_load_whisper_cache\b",
        "whisper_cache._load_whisper_cache",
    ),
    (
        r"(?<!whisper_cache\.)\b_save_whisper_cache\b",
        "whisper_cache._save_whisper_cache",
    ),
    (
        r"(?<!whisper_cache\.)\b_find_best_cached_whisper_model\b",
        "whisper_cache._find_best_cached_whisper_model",
    ),
    (
        r"(?<!whisper_alignment\.)\balign_hybrid_lrc_whisper\b",
        "whisper_alignment.align_hybrid_lrc_whisper",
    ),
    (
        r"(?<!whisper_alignment\.)\b_fix_ordering_violations\b",
        "whisper_alignment._fix_ordering_violations",
    ),
    (
        r"(?<!whisper_phonetic_dtw\.)\b_assess_lrc_quality\b",
        "whisper_phonetic_dtw._assess_lrc_quality",
    ),
    (
        r"(?<!whisper_phonetic_dtw\.)\b_extract_lrc_words\b",
        "whisper_phonetic_dtw._extract_lrc_words",
    ),
    (
        r"(?<!whisper_phonetic_dtw\.)\b_compute_phonetic_costs\b",
        "whisper_phonetic_dtw._compute_phonetic_costs",
    ),
    (
        r"(?<!whisper_utils\.)\b_compute_speech_blocks\b",
        "whisper_utils._compute_speech_blocks",
    ),
    (r"(?<!whisper_utils\.)\b_word_idx_to_block\b", "whisper_utils._word_idx_to_block"),
    (r"(?<!whisper_utils\.)\b_block_time_range\b", "whisper_utils._block_time_range"),
    (r"(?<!whisper_utils\.)\b_segment_start\b", "whisper_utils._segment_start"),
    (r"(?<!whisper_utils\.)\b_segment_end\b", "whisper_utils._segment_end"),
    (r"(?<!whisper_utils\.)\b_get_segment_text\b", "whisper_utils._get_segment_text"),
    (
        r"(?<!whisper_blocks\.)\b_build_segment_text_overlap_assignments\b",
        "whisper_blocks._build_segment_text_overlap_assignments",
    ),
    (
        r"(?<!whisper_blocks\.)\b_build_block_segmented_syllable_assignments\b",
        "whisper_blocks._build_block_segmented_syllable_assignments",
    ),
]

# Specifically exclude imports and __all__ from these replacements if possible,
# but since they already have prefixes or quotes, the regex might handle it.
# Actually, the regex above will prefix them in __all__ too, which is BAD.
# __all__ should have the original names if they are meant to be exported.

# Let's do a more careful replacement.
# Skip the header (imports and __all__)
header_end = content.find("def transcribe_vocals")
header = content[:header_end]
body = content[header_end:]

for pattern, replacement in replacements:
    body = re.sub(pattern, replacement, body)

new_content = header + body
with open(file_path, "w") as f:
    f.write(new_content)
print("Refactored whisper_integration.py")
