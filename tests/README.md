# Test Suite Layout

This repository organizes tests by scope and subsystem.

## Directories

- `tests/unit/alignment`: timing evaluation, pause/onset logic, timing refinement.
- `tests/unit/whisper`: Whisper alignment/integration helpers.
- `tests/unit/lyrics`: lyrics providers, parsing, sync quality, Genius/LRC flows.
- `tests/unit/identify`: track identification, YouTube/MusicBrainz matching.
- `tests/unit/audio`: downloader, separator, audio effects/utilities.
- `tests/unit/render`: frame/background rendering and video writing.
- `tests/unit/core_shared`: shared core utilities (`models`, `text_utils`, romanization, validation).
- `tests/unit/pipeline`: karaoke orchestration and subsystem boundary checks.
- `tests/unit/cli`: CLI command behavior.
- `tests/unit/infrastructure`: test helpers and infrastructure-level utilities.
- `tests/integration`: network and integration coverage.
- `tests/e2e`: end-to-end behavior.
- `tests/conftest.py`: global fixtures and test options.

## Running Tests

Run all tests:

```bash
PYTHONPATH=src pytest tests -v
```

Run the fast local gate (format/lint/type/unit+perf smoke):

```bash
make check
```

Run fast unit tests only:

```bash
PYTHONPATH=src pytest tests/unit -v
```

Or via marker:

```bash
PYTHONPATH=src pytest -m unit -v
```

Run integration tests (normally skipped unless opted in):

```bash
PYTHONPATH=src pytest tests/integration -v --run-network
```

Run end-to-end tests only:

```bash
PYTHONPATH=src pytest -m e2e -v
```
