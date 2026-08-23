"""
Daily API call counter for OpenRouter rate limit tracking.

Tracks API calls per UTC day to enforce RPD (Requests Per Day) limits.
Persists to a local JSON file so summarize and classify stages share the same counter.
"""
import json
import os
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


def check_and_reserve(rpd_limit: int, reserve: int = 0) -> tuple[bool, int]:
    """
    Check RPD limit and reserve a slot (increment counter) before making the call.

    `reserve` holds back that many calls for a later stage (e.g. classify),
    so the summarize stage can never consume the whole daily quota.

    Returns (allowed, current_count).
    If allowed is False, the counter is NOT incremented.

    Note: requests rejected with 429 never reached the model, so the caller
    (api_clients.openrouter_summarize) releases the reserved slot via
    rollback_increment() to keep the counter tracking accepted requests.
    """
    if rpd_limit > 0 and reserve > 0:
        rpd_limit = max(1, rpd_limit - reserve)

    if rpd_limit <= 0:
        data = load_counter()
        data["count"] = data.get("count", 0) + 1
        save_counter(data)
        return True, data["count"]

    data = load_counter()
    current = data.get("count", 0)

    if current >= rpd_limit:
        return False, current

    # Reserve the slot by incrementing
    data["count"] = current + 1
    save_counter(data)
    return True, data["count"]


def check_limit(rpd_limit: int, reserve: int = 0) -> tuple[bool, int]:
    """
    Only check RPD limit without incrementing.

    `reserve` holds back that many calls for a later stage, mirroring
    check_and_reserve().

    Used for pre-check before API call to avoid exceeding limit.

    Returns (allowed, current_count).
    """
    if rpd_limit > 0 and reserve > 0:
        rpd_limit = max(1, rpd_limit - reserve)

    if rpd_limit <= 0:
        data = load_counter()
        return True, data.get("count", 0)

    data = load_counter()
    current = data.get("count", 0)

    if current >= rpd_limit:
        return False, current

    return True, current


def increment() -> int:
    """
    Only increment counter without checking limit.

    Used after successful API call.

    Returns new count.
    """
    data = load_counter()
    data["count"] = data.get("count", 0) + 1
    save_counter(data)
    return data["count"]


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


def rollback_increment() -> int:
    """
    Rollback the counter by 1 (decrement).

    Used when a request is rejected with 429 — the reserved slot is
    released because the request never reached the model and consumed
    no daily quota.

    Returns new count after rollback.
    """
    data = load_counter()
    current = data.get("count", 0)
    if current > 0:
        data["count"] = current - 1
        save_counter(data)
    return data.get("count", 0)
