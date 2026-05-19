"""
config

Loads environment variables from .env and exposes them as a typed Config
dataclass. Call get_config() at startup; it validates required keys and raises
EnvironmentError with a clear message if anything is missing.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    PAPER: bool = True
    LOG_DIR: str = "logs/"

    def __post_init__(self) -> None:
        load_dotenv()
        self.ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
        self.ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
        raw_paper = os.environ.get("ALPACA_PAPER", "true").lower()
        self.PAPER = raw_paper not in ("false", "0", "no")
        self.LOG_DIR = os.environ.get("LOG_DIR", "logs/")

    def validate(self) -> list[str]:
        """Return a list of error strings. Empty list means config is valid."""
        errors: list[str] = []
        if not self.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is missing or empty")
        if not self.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is missing or empty")
        return errors


def get_config() -> Config:
    """Create and validate Config. Raises EnvironmentError if invalid."""
    cfg = Config()
    errors = cfg.validate()
    if errors:
        raise EnvironmentError(
            "Synk config validation failed:\n  " + "\n  ".join(errors)
        )
    return cfg
