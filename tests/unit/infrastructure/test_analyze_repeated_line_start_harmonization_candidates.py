from tools import analyze_repeated_line_start_harmonization_candidates as tool


def test_analyze_song_reports_improvement_and_changed_lines(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    report_path.write_text(
        """
        {
          "lines": [
            {
              "index": 1,
              "text": "Foo",
              "start": 0.78,
              "end": 1.8,
              "pre_whisper_start": 0.78
            },
            {
              "index": 2,
              "text": "Foo",
              "start": 11.55,
              "end": 12.5,
              "pre_whisper_start": 11.55
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    gold_path.write_text(
        """
        {
          "lines": [
            {"start": 0.8, "end": 1.9},
            {"start": 11.05, "end": 12.6}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    row = tool._analyze_song(
        {
            "artist": "Artist",
            "title": "Song",
            "report_path": str(report_path),
            "gold_path": str(gold_path),
        },
        min_start_error_span=0.3,
    )

    assert row["song"] == "Artist - Song"
    assert row["improvement"] > 0
    assert len(row["changed_lines"]) == 1
