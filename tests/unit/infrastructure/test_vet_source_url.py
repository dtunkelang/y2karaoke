from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "vet_source_url.py"
    spec = importlib.util.spec_from_file_location("vet_source_url", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_audio_hint_detects_audio_title() -> None:
    module = _load_module()
    assert module._official_audio_hint("Sia - Chandelier (Official Audio)")
    assert not module._official_audio_hint("Sia - Chandelier (Official Video)")


def test_duration_delta_handles_missing_values() -> None:
    module = _load_module()
    assert module._duration_delta(10, 7) == 3
    assert module._duration_delta(None, 7) is None
    assert module._duration_delta(10, None) is None


def test_title_matches_hint_rejects_unrelated_titles() -> None:
    module = _load_module()
    assert module._title_matches_hint("Radioactive", "Radioactive")
    assert not module._title_matches_hint("Levitating", "Whatcha Doing")


def test_feature_version_mismatch_detected() -> None:
    module = _load_module()
    assert module._has_feature_version_mismatch(
        "Levitating", "Dua Lipa - Levitating Featuring DaBaby (Official Music Video)"
    )
    assert not module._has_feature_version_mismatch(
        "Levitating feat. DaBaby",
        "Dua Lipa - Levitating Featuring DaBaby (Official Music Video)",
    )
