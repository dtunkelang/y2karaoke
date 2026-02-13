import logging
from pathlib import Path  # noqa: F401

from y2karaoke.utils.logging import setup_logging, get_logger


def test_setup_logging_adds_file_handler(tmp_path):
    log_path = tmp_path / "logs" / "app.log"

    logger = setup_logging(level="INFO", log_file=log_path, verbose=False)

    handlers = [type(h) for h in logger.handlers]
    assert logging.FileHandler in handlers
    assert logging.StreamHandler in handlers
    assert log_path.parent.exists()


def test_setup_logging_verbose_formatter():
    logger = setup_logging(level="DEBUG", verbose=True)

    # Ensure formatter includes timestamp in verbose mode
    formatters = [h.formatter for h in logger.handlers if h.formatter]
    assert any("%(asctime)s" in f._fmt for f in formatters)


def test_get_logger_returns_named_logger():
    logger = get_logger("custom")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "custom"
