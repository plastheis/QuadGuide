from __future__ import annotations
import logging
import os
import time
from logging.handlers import RotatingFileHandler

__all__ = ["setup_logging"]


class _MonotonicFormatter(logging.Formatter):
    """Formatter that injects timestamp_ns (monotonic) into every log record."""

    def format(self, record: logging.LogRecord) -> str:
        record.timestamp_ns = time.monotonic_ns()
        return super().format(record)


_FMT = "%(timestamp_ns)d %(name)s %(levelname)s %(message)s"


def setup_logging(process_name: str, config: dict) -> logging.Logger:
    """Create a rotating file logger for a worker process.

    Also attaches a StreamHandler so output is visible during development.
    Calling this multiple times for the same process_name is safe — handlers
    are cleared before adding new ones.
    """
    lg_cfg    = config.get("logging", {})
    level_str = lg_cfg.get("level", "INFO")
    log_dir   = lg_cfg.get("dir", "/var/log/quadguide")
    max_bytes = lg_cfg.get("max_bytes", 10 * 1024 * 1024)
    backup    = lg_cfg.get("backup_count", 3)

    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(process_name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = _MonotonicFormatter(fmt=_FMT)

    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, f"{process_name}.log"),
            maxBytes=max_bytes,
            backupCount=backup,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except PermissionError:
        pass  # dev machine without /var/log/quadguide — fall through to stderr

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
