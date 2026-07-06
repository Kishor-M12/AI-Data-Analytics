"""
logger.py — Structured logging setup for the entire application.
Writes JSON-structured logs to file and readable logs to console.
"""
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with console and file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console)

    # File handler — JSON structured
    Path(settings.log_dir).mkdir(exist_ok=True)
    log_file = os.path.join(settings.log_dir, "app.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger
