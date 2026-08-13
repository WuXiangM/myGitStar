"""
Daily API call counter for OpenRouter rate limit tracking.

Tracks API calls per UTC day to enforce RPD (Requests Per Day) limits.
Persists to a local JSON file so summarize and classify stages share the same counter.
"""
import json
import os
import time
from datetime import datetime, timezone


def _get_counter_path() -> str:
    """Return path to the daily counter JSON file."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, ".openrouter_daily_counter.json")


def _today_utc() -> str:
    """Return today's date string in UTC (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_counter() -> dict:
    """Load the daily counter from disk. Returns {date, count}."""
    path = _get_counter_path()
    if not os.path.exists(path):
        return {"date": _today_utc(), "count": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"date": _today_utc(), "count": 0}
        # Reset if date changed
        if data.get("date") != _today_utc():
            return {"date": _today_utc(), "count": 0}
        return data
    except Exception:
        return {"date": _today_utc(), "count": 0}


def save_counter(data: dict) -> None:
    """Save the daily counter to disk."""
    path = _get_counter_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def increment_and_check(rpd_limit: int) -> tuple[bool, int]:
    """
    Increment the daily counter and check if RPD limit is exceeded.

    Returns (allowed, current_count).
    If allowed is False, the call should be skipped.
    """
    if rpd_limit <= 0:
        # No limit configured
        data = load_counter()
        data["count"] = data.get("count", 0) + 1
        save_counter(data)
        return True, data["count"]

    data = load_counter()
    current = data.get("count", 0)

    if current >= rpd_limit:
        return False, current

    data["count"] = current + 1
    save_counter(data)
    return True, data["count"]


def get_remaining(rpd_limit: int) -> int:
    """Return remaining calls for today. Returns -1 if no limit."""
    if rpd_limit <= 0:
        return -1
    data = load_counter()
    return max(0, rpd_limit - data.get("count", 0))


def get_count() -> int:
    """Return current day's call count."""
    data = load_counter()
    return data.get("count", 0)
