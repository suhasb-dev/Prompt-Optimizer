"""
Logging setup for GEPA Optimizer
"""

import logging

def setup_logging(level="INFO"):
    """
    Configure logging for GEPA Optimizer

    Args:
        level (str): Logging level (e.g. "DEBUG", "INFO", "WARNING")
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Logging configured at {level} level")

def get_logger(name):
    """Get a logger for a specific module"""
    return logging.getLogger(name)
