"""
telegram_util

Single shared Telegram notification helper for the Synk bot.
All modules import from here — zero local implementations elsewhere.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env at module load.
Safe to import from any entry point (main.py, watchdog.py, kill_switch.py, etc.).
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_TIMEOUT = 5  # seconds


def send_telegram(message: str) -> bool:
    """Send a Telegram message. Returns True on 200 OK, False on any failure."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("Telegram not configured — alert suppressed")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        # Log type only — never log exc itself, which would str() the URL + token
        log.error("Telegram send failed: %s", type(exc).__name__)
        return False
