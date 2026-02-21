from y2karaoke.core.visual.bootstrap_candidates import (
    _metadata_prefilter_score,
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
        def __init__(self):
            self.output_dirs = []

        def download_video(self, url, output_dir):
            self.output_dirs.append(output_dir)
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

    downloader = FakeDownloader()
    ranked = rank_candidates_by_suitability(
        [
            {"url": "https://youtube.com/watch?v=a"},
            {"url": "https://youtube.com/watch?v=b"},
        ],
        downloader=downloader,
        song_dir=tmp_path,
        suitability_fps=1.0,
        analyze_fn=fake_analyze,
    )
    assert ranked[0]["url"].endswith("=a")
    assert ranked[1]["url"].endswith("=b")
    assert downloader.output_dirs[0].name == "candidate_a"
    assert downloader.output_dirs[1].name == "candidate_b"


def test_metadata_prefilter_score_prefers_karaoke_signal():
    strong = {
        "title": "Artist Song Karaoke Instrumental",
        "uploader": "karaoke channel",
        "duration": 210,
    }
    weak = {
        "title": "Artist Song Lyrics Video",
        "uploader": "Artist - Topic",
        "duration": 210,
    }
    assert _metadata_prefilter_score(strong) > _metadata_prefilter_score(weak)


def test_rank_candidates_prefilters_before_download(tmp_path):
    class FakeDownloader:
        def __init__(self):
            self.downloaded_urls = []

        def download_video(self, url, output_dir):
            self.downloaded_urls.append(url)
            path = output_dir / f"{url.split('=')[-1]}.mp4"
            path.write_bytes(b"v")
            return {"video_path": str(path)}

    def fake_analyze(video_path, fps, work_dir):
        _ = fps
        _ = work_dir
        return (
            {
                "detectability_score": 0.5,
                "word_level_score": 0.5,
                "avg_ocr_confidence": 0.5,
            },
            (0, 0, 1, 1),
        )

    candidates = [
        {
            "url": "https://youtube.com/watch?v=top1",
            "title": "Song karaoke instrumental",
            "uploader": "chan",
            "duration": 200,
        },
        {
            "url": "https://youtube.com/watch?v=top2",
            "title": "Song karaoke",
            "uploader": "chan",
            "duration": 190,
        },
        {
            "url": "https://youtube.com/watch?v=top3",
            "title": "Song karaoke track",
            "uploader": "chan",
            "duration": 220,
        },
        {
            "url": "https://youtube.com/watch?v=drop1",
            "title": "Song lyrics video",
            "uploader": "Artist - Topic",
            "duration": 210,
        },
    ]

    downloader = FakeDownloader()
    rank_candidates_by_suitability(
        candidates,
        downloader=downloader,
        song_dir=tmp_path,
        suitability_fps=1.0,
        analyze_fn=fake_analyze,
        prefilter_limit=3,
    )
    assert len(downloader.downloaded_urls) == 3
    assert "https://youtube.com/watch?v=drop1" not in downloader.downloaded_urls
