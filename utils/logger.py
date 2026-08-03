"""Centralised logging setup for the project.

get_logger configures the ROOT logger's handlers (once), not just the
named logger it returns. This matters: every other module in this
codebase calls plain `logging.getLogger(__name__)` (the standard,
lightweight pattern -- see e.g. training/trainer_vqvae.py,
training/callbacks.py, data/generate_dataset.py) rather than importing
this function themselves. Those loggers have no handlers of their own,
so they rely on propagation to a configured root logger to be visible at
all -- attaching handlers only to the caller's own named logger (e.g.
"__main__") leaves every other module's INFO-level messages silently
dropped (Python's logging falls back to a WARNING-and-above "handler of
last resort" when the root has no handlers). Call get_logger(__name__)
once near the top of each CLI entry point (scripts/*.py, run.py,
smoke_test.py already do this) and every module's logging becomes
visible through the same formatted, UTF-8-safe handler.

Also forces UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError
when log messages contain characters outside the cp1252 range).
"""

import io
import logging
import sys
from pathlib import Path

_ROOT_CONFIGURED = False


def _utf8_stream(stream=None):
    """Return a UTF-8 wrapped version of a stream if needed.

    On Windows the default console encoding is cp1252. Wrapping the
    stream in a UTF-8 TextIOWrapper prevents UnicodeEncodeError when log
    messages contain characters outside the cp1252 range.
    """
    if stream is None:
        stream = sys.stdout
    enc = getattr(stream, "encoding", "utf-8") or "utf-8"
    if enc.lower().replace("-", "") == "utf8":
        return stream
    if hasattr(stream, "buffer"):
        return io.TextIOWrapper(
            stream.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    return stream


def _configure_root(level: int, log_file: str | None) -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=_utf8_stream(sys.stdout))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _ROOT_CONFIGURED = True


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root logger's handlers (once) and return a named logger.

    Args:
        name: Logger name (use __name__ in calling modules).
        log_file: Optional path to also write log output to disk.
        level: Logging level (default INFO), applied to the root logger.

    Returns:
        A logging.Logger for `name`. Every other logger in the process
        (including plain `logging.getLogger(__name__)` calls in modules
        that never import this function) becomes visible through the
        same root handler once this has been called anywhere in the process.
    """
    _configure_root(level, log_file)
    return logging.getLogger(name)
