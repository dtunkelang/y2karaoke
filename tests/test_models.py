"""Tests for data models and quality reporting."""

import pytest
from y2karaoke.core.models import (
    Word,
    Line,
    SongMetadata,
    SingerID,
    StepQuality,
    TrackIdentificationQuality,
    LyricsQuality,
    TimingAlignmentQuality,
    PipelineQualityReport,
)

# ------------------------------
# Word Tests
# ------------------------------


class TestWord:
    def test_word_creation(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        assert word.text == "hello"
        assert word.start_time == 0.0
        assert word.end_time == 0.5
        assert word.singer == ""

    def test_word_with_singer(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5, singer="singer1")
        assert word.singer == "singer1"

    def test_word_validate_success(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        word.validate()  # Should not raise

    def test_word_validate_negative_start(self):
        word = Word(text="hello", start_time=-1.0, end_time=0.5)
        with pytest.raises(ValueError, match="non-negative"):
            word.validate()

    def test_word_validate_end_before_start(self):
        word = Word(text="hello", start_time=1.0, end_time=0.5)
        with pytest.raises(ValueError, match="end_time must be"):
            word.validate()


# ------------------------------
# Line Tests
# ------------------------------


class TestLine:
    def test_line_creation(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words)
        assert len(line.words) == 2

    def test_line_start_time(self):
        words = [
            Word(text="hello", start_time=1.0, end_time=1.3),
            Word(text="world", start_time=1.3, end_time=1.6),
        ]
        line = Line(words=words)
        assert line.start_time == 1.0

    def test_line_end_time(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words)
        assert line.end_time == 0.6

    def test_line_text(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words)
        assert line.text == "hello world"

    def test_empty_line_times(self):
        line = Line(words=[])
        assert line.start_time == 0.0
        assert line.end_time == 0.0
        assert line.text == ""

    def test_line_validate_empty(self):
        line = Line(words=[])
        with pytest.raises(ValueError, match="at least one word"):
            line.validate()

    def test_line_with_singer(self):
        words = [Word(text="hello", start_time=0.0, end_time=0.5)]
        line = Line(words=words, singer=SingerID.SINGER1)
        assert line.singer == SingerID.SINGER1


# ------------------------------
# SongMetadata Tests
# ------------------------------


class TestSongMetadata:
    def test_creation(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.singers == ["Alice", "Bob"]
        assert metadata.is_duet is True

    def test_get_singer_id_first_singer(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice") == SingerID.SINGER1

    def test_get_singer_id_second_singer(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Bob") == SingerID.SINGER2

    def test_get_singer_id_empty_string(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("") == SingerID.UNKNOWN

    def test_get_singer_id_both_with_ampersand(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice & Bob") == SingerID.BOTH

    def test_get_singer_id_both_with_and(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice and Bob") == SingerID.BOTH

    def test_get_singer_id_both_with_feat(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice feat Bob") == SingerID.BOTH

    def test_get_singer_id_unknown_defaults_to_first(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Unknown Person") == SingerID.SINGER1

    def test_get_singer_id_partial_match(self):
        metadata = SongMetadata(singers=["Alice Johnson", "Bob Smith"], is_duet=True)
        assert metadata.get_singer_id("Alice") == SingerID.SINGER1
        assert metadata.get_singer_id("Bob") == SingerID.SINGER2

    def test_get_singer_id_case_insensitive(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("ALICE") == SingerID.SINGER1
        assert metadata.get_singer_id("bob") == SingerID.SINGER2


# ------------------------------
# StepQuality Tests
# ------------------------------


class TestStepQuality:
    def test_is_good(self):
        step = StepQuality(step_name="test", quality_score=80)
        assert step.is_good is True

    def test_is_good_threshold(self):
        step = StepQuality(step_name="test", quality_score=70)
        assert step.is_good is True

    def test_is_good_below_threshold(self):
        step = StepQuality(step_name="test", quality_score=69)
        assert step.is_good is False

    def test_needs_review(self):
        step = StepQuality(step_name="test", quality_score=50)
        assert step.needs_review is True

    def test_needs_review_boundaries(self):
        step_low = StepQuality(step_name="test", quality_score=40)
        step_high = StepQuality(step_name="test", quality_score=69)
        assert step_low.needs_review is True
        assert step_high.needs_review is True

    def test_is_poor(self):
        step = StepQuality(step_name="test", quality_score=30)
        assert step.is_poor is True

    def test_is_poor_threshold(self):
        step = StepQuality(step_name="test", quality_score=40)
        assert step.is_poor is False

    def test_default_values(self):
        step = StepQuality(step_name="test", quality_score=50)
        assert step.status == "success"
        assert step.cached is False
        assert step.issues == []
        assert step.details == {}


# ------------------------------
# TrackIdentificationQuality Tests
# ------------------------------


class TestTrackIdentificationQuality:
    def test_step_name_set(self):
        quality = TrackIdentificationQuality(
            step_name="track_identification", quality_score=80
        )
        assert quality.step_name == "track_identification"

    def test_default_values(self):
        quality = TrackIdentificationQuality(
            step_name="track_identification", quality_score=80
        )
        assert quality.match_confidence == 0.0
        assert quality.source == ""
        assert quality.fallback_used is False


# ------------------------------
# LyricsQuality Tests
# ------------------------------


class TestLyricsQuality:
    def test_step_name_set(self):
        quality = LyricsQuality(step_name="lyrics_fetch", quality_score=80)
        assert quality.step_name == "lyrics_fetch"

    def test_default_values(self):
        quality = LyricsQuality(step_name="lyrics_fetch", quality_score=80)
        assert quality.coverage == 0.0
        assert quality.timestamp_density == 0.0
        assert quality.duration_match is True


# ------------------------------
# TimingAlignmentQuality Tests
# ------------------------------


class TestTimingAlignmentQuality:
    def test_step_name_set(self):
        quality = TimingAlignmentQuality(step_name="timing_alignment", quality_score=80)
        assert quality.step_name == "timing_alignment"

    def test_alignment_rate(self):
        quality = TimingAlignmentQuality(
            step_name="timing_alignment",
            quality_score=80,
            lines_aligned=80,
            total_lines=100,
        )
        assert quality.alignment_rate == 80.0

    def test_alignment_rate_zero_total(self):
        quality = TimingAlignmentQuality(
            step_name="timing_alignment",
            quality_score=80,
            lines_aligned=0,
            total_lines=0,
        )
        assert quality.alignment_rate == 0.0


# ------------------------------
# PipelineQualityReport Tests
# ------------------------------


class TestPipelineQualityReport:
    def test_from_steps_empty(self):
        report = PipelineQualityReport.from_steps([])
        assert report.overall_score == 0
        assert report.confidence_level == "low"

    def test_from_steps_high_quality(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=90
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=85),
            TimingAlignmentQuality(step_name="timing_alignment", quality_score=80),
        ]
        report = PipelineQualityReport.from_steps(steps)
        assert report.overall_score >= 80
        assert report.confidence_level == "high"

    def test_from_steps_medium_quality(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=60
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=55),
            TimingAlignmentQuality(step_name="timing_alignment", quality_score=50),
        ]
        report = PipelineQualityReport.from_steps(steps)
        assert 50 <= report.overall_score < 80
        assert report.confidence_level == "medium"

    def test_from_steps_low_quality(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=30
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=25),
            TimingAlignmentQuality(step_name="timing_alignment", quality_score=20),
        ]
        report = PipelineQualityReport.from_steps(steps)
        assert report.overall_score < 50
        assert report.confidence_level == "low"

    def test_from_steps_collects_warnings(self):
        step = TrackIdentificationQuality(
            step_name="track_identification",
            quality_score=50,
            status="degraded",
            issues=["Issue 1", "Issue 2"],
        )
        report = PipelineQualityReport.from_steps([step])
        assert len(report.warnings) > 0
        assert any("degraded" in w.lower() for w in report.warnings)

    def test_from_steps_recommendations_for_low_quality(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=30
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=25),
        ]
        report = PipelineQualityReport.from_steps(steps)
        assert len(report.recommendations) > 0
        assert any("whisper" in r.lower() for r in report.recommendations)

    def test_from_steps_weighted_average(self):
        # Timing alignment has weight 2.5, higher than track ID (1.5)
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=100
            ),  # weight 1.5
            TimingAlignmentQuality(
                step_name="timing_alignment", quality_score=0
            ),  # weight 2.5
        ]
        report = PipelineQualityReport.from_steps(steps)
        # Weighted: (100*1.5 + 0*2.5) / (1.5+2.5) = 150/4 = 37.5
        assert report.overall_score == pytest.approx(37.5, abs=0.1)

    def test_summary(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=80
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=75),
        ]
        report = PipelineQualityReport.from_steps(steps)
        summary = report.summary()
        assert "Quality:" in summary
        assert "/100" in summary

    def test_steps_dict(self):
        steps = [
            TrackIdentificationQuality(
                step_name="track_identification", quality_score=80
            ),
            LyricsQuality(step_name="lyrics_fetch", quality_score=75),
        ]
        report = PipelineQualityReport.from_steps(steps)
        assert "track_identification" in report.steps
        assert "lyrics_fetch" in report.steps
