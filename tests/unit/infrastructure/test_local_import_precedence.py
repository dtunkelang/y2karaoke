from pathlib import Path

import y2karaoke


def test_y2karaoke_import_resolves_to_local_src():
    module_path = Path(y2karaoke.__file__).resolve()
    assert "/src/y2karaoke/" in str(module_path).replace("\\", "/")
