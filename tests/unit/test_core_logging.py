import logging

import pytest

from quadguide.core.logging import setup_logging


@pytest.fixture
def restore_root():
    """setup_logging() takes over the root logger — put it back afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield
    root.handlers.clear()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _cfg(tmp_path, level="INFO"):
    return {"logging": {"level": level, "dir": str(tmp_path)}}


def _read(tmp_path, name):
    for h in logging.getLogger().handlers:
        h.flush()
    return (tmp_path / f"{name}.log").read_text()


def test_process_logger_writes_to_file(tmp_path, restore_root):
    log = setup_logging("proc_a", _cfg(tmp_path))
    log.info("hello from the worker")
    assert "hello from the worker" in _read(tmp_path, "proc_a")


def test_third_party_logger_is_captured(tmp_path, restore_root):
    # The reason handlers live on the root logger: uvicorn/pymavlink/GStreamer
    # log through the stdlib and would otherwise be dropped entirely.
    setup_logging("proc_b", _cfg(tmp_path))
    logging.getLogger("uvicorn.error").warning("third-party line")
    assert "third-party line" in _read(tmp_path, "proc_b")


def test_debug_level_captures_debug_records(tmp_path, restore_root):
    log = setup_logging("proc_c", _cfg(tmp_path, level="DEBUG"))
    log.debug("verbose detail")
    logging.getLogger("some.library").debug("library detail")
    text = _read(tmp_path, "proc_c")
    assert "verbose detail" in text
    assert "library detail" in text


def test_info_level_drops_debug_records(tmp_path, restore_root):
    log = setup_logging("proc_d", _cfg(tmp_path, level="INFO"))
    log.debug("should not appear")
    log.info("should appear")
    text = _read(tmp_path, "proc_d")
    assert "should not appear" not in text
    assert "should appear" in text


def test_process_logger_does_not_double_log(tmp_path, restore_root):
    # propagate=True + root handlers must still yield exactly one line, not two.
    log = setup_logging("proc_e", _cfg(tmp_path))
    log.info("once only")
    assert _read(tmp_path, "proc_e").count("once only") == 1


def test_repeated_setup_does_not_accumulate_handlers(tmp_path, restore_root):
    setup_logging("proc_f", _cfg(tmp_path))
    first = len(logging.getLogger().handlers)
    setup_logging("proc_f", _cfg(tmp_path))
    assert len(logging.getLogger().handlers) == first
