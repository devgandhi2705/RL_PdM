"""
Centralized logging configuration for the PdM pipeline.

Usage in any module:
    from src.logging_config import setup_logger
    logger = setup_logger(__name__)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

_RUN_ID = time.strftime("%Y%m%d_%H%M%S")

_FMT = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s"


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID  # type: ignore[attr-defined]
        return True


def setup_logger(name: str) -> logging.Logger:
    """Return a logger wired to console (INFO) and rotating file (DEBUG).

    Idempotent — calling twice with the same name returns the existing logger.
    All handlers share the same run_id stamp so a single training run can be
    grepped with:  grep <run_id> logs/pdm_pipeline.log
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT)
    filt = _RunIdFilter()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch.addFilter(filt)

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(
        logs_dir / "pdm_pipeline.log",
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(filt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False  # prevent double-printing via root's lastResort handler
    return logger
