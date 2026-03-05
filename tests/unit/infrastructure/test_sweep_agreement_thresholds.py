import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "sweep_agreement_thresholds.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sweep_agreement_thresholds_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_parse_float_list():
    assert _MODULE._parse_float_list("0.60, 0.58") == [0.6, 0.58]


def test_candidate_label():
    assert _MODULE._candidate_label(0.6, 0.5) == "ts60_to50"
