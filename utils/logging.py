import logging
import os


def configure_logging(default_level: str | int = "INFO") -> None:
    level = os.getenv("LOG_LEVEL", default_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
