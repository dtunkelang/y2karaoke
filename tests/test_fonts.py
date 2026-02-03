import y2karaoke.utils.fonts as fonts


def test_get_font_uses_first_available(monkeypatch):
    seen = {}

    def fake_exists(self):
        return str(self).endswith("Helvetica.ttc")

    def fake_truetype(path, size):
        seen["path"] = path
        seen["size"] = size
        return "font"

    monkeypatch.setattr(fonts.Path, "exists", fake_exists)
    monkeypatch.setattr(fonts.ImageFont, "truetype", fake_truetype)

    result = fonts.get_font(size=42)
    assert result == "font"
    assert seen["size"] == 42
    assert seen["path"].endswith("Helvetica.ttc")


def test_get_font_falls_back_to_default(monkeypatch):
    def fake_exists(self):
        return False

    monkeypatch.setattr(fonts.Path, "exists", fake_exists)
    monkeypatch.setattr(fonts.ImageFont, "load_default", lambda: "default-font")

    assert fonts.get_font() == "default-font"


def test_get_font_fallback_logs_when_load_default_fails(monkeypatch, caplog):
    def fake_exists(self):
        return False

    calls = {"count": 0}

    def fake_load_default():
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("boom")
        return "default-font"

    monkeypatch.setattr(fonts.Path, "exists", fake_exists)
    monkeypatch.setattr(fonts.ImageFont, "load_default", fake_load_default)

    with caplog.at_level("WARNING"):
        result = fonts.get_font()

    assert result == "default-font"
    assert "Could not load any font" in caplog.text
