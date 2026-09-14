"""Centralized logging configuration for TickVault."""

import logging
import sys

from .config import CONFIG


def setup_logger() -> logging.Logger:
    """
    Configure and return a logger instance.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("tick_vault")

    # Only configure if not already configured
    if logger.handlers:
        return logger

    logger.setLevel(CONFIG.base_log_level)

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONFIG.console_log_level)

    # File handler
    log_file = CONFIG.log_file_path
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # File gets everything

    # Formatter
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Root logger for the package
logger = setup_logger()
