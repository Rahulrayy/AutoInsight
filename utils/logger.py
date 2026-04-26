import logging
import os


def get_logger(name):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid adding duplicate handlers if called multiple times

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger