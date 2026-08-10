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

    # /var/log/quadguide is created root-owned by the systemd unit, so a bench
    # run started as a normal user cannot open the file and every log line —
    # including the FC's own STATUSTEXT reasoning, which exists nowhere else —
    # would survive only in terminal scrollback. Fall back to a writable dir
    # next to the trace output rather than degrading to stderr-only.
    for candidate in (log_dir, os.path.join(os.getcwd(), "quadguide-logs")):
        try:
            os.makedirs(candidate, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(candidate, f"{process_name}.log"),
                maxBytes=max_bytes,
                backupCount=backup,
            )
        except (PermissionError, OSError):
            continue
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        break

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
