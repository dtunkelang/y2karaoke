from pathlib import Path

from y2karaoke.core.visual.bootstrap_media import (
    extract_audio_from_video,
    resolve_media_paths,
)


def test_extract_audio_from_video_uses_run_once(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    output_dir = tmp_path / "out"
    calls = {"n": 0}

    def fake_run(cmd, check, stdout, stderr):
        calls["n"] += 1
        Path(cmd[-1]).write_bytes(b"wav")
        return None

    first = extract_audio_from_video(video_path, output_dir, run_fn=fake_run)
    second = extract_audio_from_video(video_path, output_dir, run_fn=fake_run)
    assert first == second
    assert calls["n"] == 1


def test_resolve_media_paths_falls_back_to_download(tmp_path):
    class FakeDownloader:
        def download_video(self, url, output_dir):
            path = output_dir / "video.mp4"
            path.write_bytes(b"v")
            return {"video_path": str(path)}

        def download_audio(self, url, output_dir):
            path = output_dir / "audio.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"a")
            return {"audio_path": str(path)}

    def fail_extract(_v_path, _output_dir):
        raise RuntimeError("ffmpeg failed")

    v_path, a_path = resolve_media_paths(
        downloader=FakeDownloader(),
        candidate_url="https://youtube.com/watch?v=abc",
        cached_video_path=tmp_path / "cached.mp4",
        song_dir=tmp_path,
        extract_audio_fn=fail_extract,
    )
    assert v_path.name == "cached.mp4"
    assert a_path.name == "audio.wav"
