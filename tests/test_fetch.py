import pytest

from y2karaoke.core import fetch


class FakeResponse:
    def __init__(self, text="", json_data=None, raise_error=False):
        self.text = text
        self._json_data = json_data
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error:
            raise RuntimeError("bad status")

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        if not self._responses:
            raise RuntimeError("no response")
        return self._responses.pop(0)


def test_fetch_html_success():
    session = FakeSession([FakeResponse(text="ok")])
    assert (
        fetch.fetch_html(
            "https://example.com", session=session, max_retries=1, retry_sleep=0
        )
        == "ok"
    )


def test_fetch_html_retries_and_returns_none(monkeypatch):
    session = FakeSession(
        [FakeResponse(raise_error=True), FakeResponse(raise_error=True)]
    )
    monkeypatch.setattr(fetch.time, "sleep", lambda *_: None)
    monkeypatch.setattr(fetch.random, "random", lambda: 0.0)

    assert (
        fetch.fetch_html(
            "https://example.com", session=session, max_retries=2, retry_sleep=0
        )
        is None
    )


def test_fetch_json_success():
    session = FakeSession([FakeResponse(json_data={"ok": True})])
    assert fetch.fetch_json(
        "https://example.com", session=session, max_retries=1, retry_sleep=0
    ) == {"ok": True}


def test_fetch_json_retries_and_returns_none(monkeypatch):
    session = FakeSession([FakeResponse(json_data=RuntimeError("bad"))])
    monkeypatch.setattr(fetch.time, "sleep", lambda *_: None)
    monkeypatch.setattr(fetch.random, "random", lambda: 0.0)

    assert (
        fetch.fetch_json(
            "https://example.com", session=session, max_retries=1, retry_sleep=0
        )
        is None
    )
