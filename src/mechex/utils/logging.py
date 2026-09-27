"""Process logger used by every entry point.

Library modules never configure handlers themselves; they take a child logger from
``get_logger`` so that a caller decides where records go.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import IO

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_ROOT = "mechex"
_configured = False


def configure_logging(level: str | int | None = None, stream: IO[str] | None = None) -> None:
    """Attach a single stream handler to the package root logger.

    Repeated calls are no-ops so that an entry point invoked from a test does not
    stack duplicate handlers onto the same records.
    """
    global _configured
    root = logging.getLogger(_ROOT)
    if _configured:
        return
    resolved = level if level is not None else os.environ.get("MECHEX_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(resolved)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package root."""
    configure_logging()
    return logging.getLogger(f"{_ROOT}.{name}" if not name.startswith(_ROOT) else name)
