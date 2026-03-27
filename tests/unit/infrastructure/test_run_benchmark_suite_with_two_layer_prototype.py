import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, cast

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "run_benchmark_suite_with_two_layer_prototype.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_suite_with_two_layer_prototype_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MODULE = cast(Any, _MODULE)


def test_main_writes_two_layer_sidecar_for_resume_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps({"aggregate": {}, "songs": []}),
        encoding="utf-8",
    )

    called: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> SimpleNamespace:
        assert check is False
        called.append(cmd)
        return SimpleNamespace(returncode=0)

    tool = SimpleNamespace(
        analyze=lambda report_doc: {
            "baseline": {
                "agreement_coverage_ratio_total": 0.35,
                "agreement_bad_ratio_total": 0.1,
            },
            "prototype": {
                "agreement_coverage_ratio_total": 0.65,
                "agreement_bad_ratio_total": 0.1,
            },
            "prototype_hotspots": [{"song": "a-ha - Take On Me"}],
        },
        _write_markdown=lambda path, payload: path.write_text(
            "# prototype\n", encoding="utf-8"
        ),
    )

    old_run = _MODULE.subprocess.run
    old_loader = MODULE._load_two_layer_benchmark_prototype_tool
    MODULE.subprocess.run = fake_run
    MODULE._load_two_layer_benchmark_prototype_tool = lambda: tool
    try:
        code = MODULE.main(["--resume-run-dir", str(run_dir), "--offline"])
    finally:
        MODULE.subprocess.run = old_run
        MODULE._load_two_layer_benchmark_prototype_tool = old_loader

    assert code == 0
    assert called
    assert (run_dir / "two_layer_benchmark_prototype.json").exists()
    assert (run_dir / "two_layer_benchmark_prototype.md").exists()
