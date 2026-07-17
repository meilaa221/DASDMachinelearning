"""Centralized logging setup for the Rumahku FastAPI app.

Writes to both the console and a rotating log file (logs/app.log) so request
activity, predictions, and errors are traceable after the fact instead of
only showing up in whatever terminal happened to be running uvicorn.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rumahku")
    if logger.handlers:
        return logger  # avoid duplicate handlers on module reload (--reload)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = _build_logger()
