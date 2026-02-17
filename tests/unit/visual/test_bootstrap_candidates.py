from pathlib import Path

from y2karaoke.core.visual.bootstrap_candidates import (
    rank_candidates_by_suitability,
    search_karaoke_candidates,
)


def test_search_karaoke_candidates_with_fake_ytdlp():
    class FakeYDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def extract_info(self, _term, download=False):
            assert download is False
            return {
                "entries": [
                    {"id": "abc123", "title": "t", "uploader": "u", "duration": 10}
                ]
            }

    class FakeYTDLP:
        YoutubeDL = FakeYDL

    out = search_karaoke_candidates(
        "Artist",
        "Song",
        3,
        yt_dlp_module=FakeYTDLP(),
    )
    assert len(out) == 1
    assert out[0]["url"].endswith("abc123")


def test_rank_candidates_by_suitability_orders_by_score(tmp_path):
    class FakeDownloader:
        def download_video(self, url, output_dir):
            path = output_dir / f"{url.split('=')[-1]}.mp4"
            path.write_bytes(b"v")
            return {"video_path": str(path)}

    def fake_analyze(video_path, fps, work_dir):
        score = 0.9 if "a.mp4" in str(video_path) else 0.5
        return (
            {
                "detectability_score": score,
                "word_level_score": score,
                "avg_ocr_confidence": score,
            },
            (0, 0, 1, 1),
        )

    ranked = rank_candidates_by_suitability(
        [
            {"url": "https://youtube.com/watch?v=a"},
            {"url": "https://youtube.com/watch?v=b"},
        ],
        downloader=FakeDownloader(),
        song_dir=tmp_path,
        suitability_fps=1.0,
        analyze_fn=fake_analyze,
    )
    assert ranked[0]["url"].endswith("=a")
    assert ranked[1]["url"].endswith("=b")
