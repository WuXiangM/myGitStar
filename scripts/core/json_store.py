import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

# json_store.py lives at <repo>/scripts/core/, so THREE dirnames are needed
# to reach the repo root. (Two only reached <repo>/scripts/, which made
# get_summary_json_path() point at a non-existent scripts/repo_summaries.json
# and silently discarded all incremental progress in CI.)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Schema constants
# ============================================================================
# Each entry in repo_summaries.json now carries metadata alongside the
# content fields, so subsequent runs can do incremental updates without
# re-calling the LLM for repos whose upstream description has not changed.
SUMMARY_CONTENT_FIELDS = (
    "Repository Name",
    "Repository URL",
    "Brief Introduction",
    "Innovations",
    "Basic Usage",
    "Summary",
)
SUMMARY_META_FIELDS = (
    "last_summarized_at",       # ISO8601 UTC timestamp of last successful summary
    "description_hash",         # sha256 of (full_name + description) at summarize time
    "summary_model",            # model identifier used to produce the summary
    "summary_source",           # "ai" | "reused" | "backfill" | "manual"
    "summary_attempts",         # number of LLM attempts that produced this entry
)

# Error/backoff metadata (written when an LLM attempt fails, consumed by the
# priority queue so failed repos back off instead of being retried every run).
ERROR_META_FIELDS = (
    "last_attempt_at",          # ISO8601 UTC timestamp of the last (failed) attempt
    "last_error",               # short machine-readable failure reason
    "next_retry_at",            # earliest ISO8601 UTC time this repo may be retried
    "source_absent",            # True = upstream has no README/description; terminal state
)

QUEUE_STATE_FILE = os.path.join(REPO_ROOT, "queue_state.json")


def compute_description_hash(full_name: str, description: str) -> str:
    """Stable hash of (full_name + description). Used to detect upstream changes.

    We include full_name so a repo that is forked/renamed still counts as
    'changed' even if its description happens to collide.
    """
    payload = f"{str(full_name or '').strip()}\x00{str(description or '').strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def get_summary_json_path(language: str) -> str:
    if language == "zh":
        return os.path.join(REPO_ROOT, "repo_summaries_zh.json")
    return os.path.join(REPO_ROOT, "repo_summaries.json")


def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json_atomic(data: dict, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def make_metadata(
    full_name: str,
    description: str,
    model: str,
    source: str,
    attempts: int = 1,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the metadata dict to embed inside a summary entry.

    This is the single place that knows how a summary is timestamped / hashed
    so the format stays consistent across summarize + classify paths.
    """
    from datetime import datetime, timezone
    return {
        "last_summarized_at": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description_hash": compute_description_hash(full_name, description),
        "summary_model": model or "unknown",
        "summary_source": source,
        "summary_attempts": int(attempts) if attempts else 1,
    }


def make_error_meta(
    full_name: str,
    description: str,
    model: str,
    reason: str,
    attempts: int = 1,
    backoff_base_days: float = 1.0,
    backoff_max_days: float = 8.0,
) -> Dict[str, Any]:
    """Build metadata for a FAILED summarization attempt.

    Unlike make_metadata(), this does NOT set last_summarized_at (the entry
    was not successfully summarized) and instead records the failure reason,
    the attempt count and an exponential backoff window (2^attempts days,
    capped) so the priority queue can defer the next retry.
    """
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    attempts = max(1, int(attempts or 1))
    delay_days = min(backoff_base_days * (2 ** (attempts - 1)), backoff_max_days)
    next_retry = now + timedelta(days=delay_days)
    return {
        "description_hash": compute_description_hash(full_name, description),
        "summary_model": model or "unknown",
        "summary_source": "error",
        "summary_attempts": attempts,
        "last_attempt_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_error": str(reason or "unknown_error")[:200],
        "next_retry_at": next_retry.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# Fields considered when scoring how much useful content an entry has.
QUALITY_FIELDS = ("Brief Introduction", "Innovations", "Basic Usage", "Summary")


def count_invalid_fields(entry: Optional[Dict[str, Any]]) -> int:
    """Return how many of the 4 content fields lack useful text (0~4).

    A field is 'invalid' when it is missing, empty, or the 'Not specified'
    placeholder. Repos with a HIGHER count are summarised first.
    """
    if not isinstance(entry, dict):
        return len(QUALITY_FIELDS)
    n = 0
    for f in QUALITY_FIELDS:
        v = str(entry.get(f) or "").strip()
        if not v or v.strip().lower() in ("not specified.", "not specified"):
            n += 1
    return n


def _entry_meta(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(entry, dict) and isinstance(entry.get("__meta__"), dict):
        return entry["__meta__"]
    return {}


def entry_attempts(entry: Optional[Dict[str, Any]]) -> int:
    """Number of recorded LLM attempts for this entry (0 = never attempted)."""
    try:
        return int(_entry_meta(entry).get("summary_attempts") or 0)
    except Exception:
        return 0


def _parse_iso(ts: Optional[str]):
    from datetime import datetime, timezone
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def is_retry_due(entry: Optional[Dict[str, Any]], now=None) -> bool:
    """True when the entry may be retried now (backoff expired / never failed)."""
    from datetime import datetime, timezone
    meta = _entry_meta(entry)
    if not meta.get("next_retry_at"):
        return True
    nxt = _parse_iso(meta.get("next_retry_at"))
    if nxt is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    return now >= nxt


def is_source_absent(entry: Optional[Dict[str, Any]]) -> bool:
    """True when upstream has no README and no description (terminal state)."""
    return bool(_entry_meta(entry).get("source_absent"))


def load_queue_state() -> Dict[str, Any]:
    """Load the round-robin cursor state (survives across runs via git)."""
    state = load_json(QUEUE_STATE_FILE)
    if not isinstance(state, dict):
        state = {}
    try:
        state["cursor"] = max(0, int(state.get("cursor") or 0))
    except Exception:
        state["cursor"] = 0
    return state


def save_queue_state(state: Dict[str, Any]) -> None:
    from datetime import datetime, timezone
    if not isinstance(state, dict):
        state = {}
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_json_atomic(state, QUEUE_STATE_FILE)


def _is_not_specified(value: Any) -> bool:
    """Check if a field value is the 'Not specified' placeholder.

    This is a LEGITIMATE placeholder when the repo genuinely has no content
    (e.g. no README, no usage examples, no innovations).  It should only be
    treated as INVALID when *all* key fields are "Not specified" (indicating
    a total LLM failure).
    """
    if isinstance(value, str):
        return value.strip().lower() in ("not specified.", "not specified")
    return False


def _is_all_not_specified(entry: Dict[str, Any], fields: Tuple[str, ...]) -> bool:
    """Return True when *all* given fields are empty or 'Not specified'.

    This distinguishes between:
      - LEGITIMATE partial emptiness: some fields are "Not specified" because
        the repo genuinely lacks that content (e.g. no Basic Usage for a lib
        without examples)
      - INVALID total failure: ALL key fields are "Not specified", which
        indicates the LLM call completely failed to produce useful content.
    """
    return all(
        _is_not_specified(entry.get(f)) or not str(entry.get(f) or "").strip()
        for f in fields
    )


def is_entry_fresh(
    entry: Optional[Dict[str, Any]],
    full_name: str,
    description: str,
    refresh_after_days: int = 0,
) -> bool:
    """Return True if the entry's summary is still considered fresh.

    "Fresh" means:
      - has the 4 user-visible content fields populated (Repository URL is
        always re-derivable from full_name, so we don't require it), AND
      - if *all* key fields are "Not specified" -> LLM totally failed,
        mark as stale (needs re-summarization);
      - if *some* fields are "Not specified" -> LEGITIMATE placeholder
        (repo genuinely has no such content), do NOT re-summarize;
      - description_hash matches the current upstream description, AND
      - if refresh_after_days > 0, last_summarized_at is within that window.

    refresh_after_days=0 means "never auto-refresh even if old" - we still
    invalidate when the description changes. This is the default for the
    GitHub Actions workflow since we want to minimise API calls.
    """
    if not isinstance(entry, dict):
        return False

    # Legacy poison string written by old budget-exhaustion code. It must be
    # re-summarised, never reused as a valid summary.
    summary_text = str(entry.get("Summary") or "").strip()
    if summary_text.startswith("(deferred"):
        return False

    required_fields = (
        "Repository Name",
        "Brief Introduction",
        "Innovations",
        "Summary",
    )

    # Total LLM failure: ALL key fields are "Not specified" or empty.
    # This must be re-summarized.
    if _is_all_not_specified(entry, required_fields):
        return False

    # Individual fields: "Not specified" is a LEGITIMATE placeholder.
    # Only truly missing (None / empty string) fields invalidate the entry.
    for f in required_fields:
        v = entry.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
        # "Not specified" is valid per-field — do NOT invalidate here.

    # `Basic Usage` is allowed to be empty or "Not specified".
    # Some repos genuinely have no usage snippet (e.g. libraries without examples).
    # We no longer treat "Not specified" on Basic Usage as invalid.
    basic = entry.get("Basic Usage")
    if basic is not None:
        if isinstance(basic, str) and not basic.strip():
            pass  # ok - empty is acceptable
        # "Not specified." on Basic Usage is now LEGAL — the repo has no usage.
        # If the LLM totally failed, the check above (_is_all_not_specified)
        # would have caught it already.

    current_hash = compute_description_hash(full_name, description)
    meta = entry.get("__meta__") if isinstance(entry.get("__meta__"), dict) else {}
    stored_hash = meta.get("description_hash") or entry.get("description_hash")
    if stored_hash and stored_hash != current_hash:
        return False

    if refresh_after_days and refresh_after_days > 0:
        ts = entry.get("__meta__", {}).get("last_summarized_at") if isinstance(entry.get("__meta__"), dict) else None
        if not ts:
            return False
        try:
            from datetime import datetime, timezone, timedelta
            last = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last > timedelta(days=refresh_after_days):
                return False
        except Exception:
            return False

    return True


def normalize_json_store(data: Any) -> Dict[str, Dict]:
    """Normalize JSON store to {full_name: entry_dict}.

    Supports:
    1) full_name -> entry dict (current format)
    2) category -> [entry dict] (extracted from README)
    3) list of entry dicts
    """
    normalized: Dict[str, Dict] = {}
    if not data:
        return normalized

    def _add_entry(full_name: str, entry: Dict) -> None:
        if not full_name:
            return
        normalized[full_name] = entry

    if isinstance(data, dict):
        # category -> [entry dict]
        if all(isinstance(v, list) for v in data.values()):
            for _, items in data.items():
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    full_name = item.get("full_name") or item.get("Repository Name") or item.get("repo")
                    if not full_name:
                        continue
                    summary = item.get("summary") or item.get("Summary") or ""
                    normalized_entry = dict(item)
                    normalized_entry["summary"] = summary
                    normalized_entry["full_name"] = full_name
                    _add_entry(full_name, normalized_entry)
            return normalized

        # full_name -> entry dict or summary string
        for key, value in data.items():
            if isinstance(value, dict):
                full_name = value.get("full_name") or value.get("Repository Name") or key
                summary = value.get("summary") or value.get("Summary") or ""
                normalized_entry = dict(value)
                normalized_entry["summary"] = summary
                normalized_entry["full_name"] = full_name
                _add_entry(full_name, normalized_entry)
            elif isinstance(value, str):
                normalized_entry = {"full_name": key, "summary": value}
                _add_entry(key, normalized_entry)
        return normalized

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            full_name = item.get("full_name") or item.get("Repository Name") or item.get("repo")
            if not full_name:
                continue
            summary = item.get("summary") or item.get("Summary") or ""
            normalized_entry = dict(item)
            normalized_entry["summary"] = summary
            normalized_entry["full_name"] = full_name
            _add_entry(full_name, normalized_entry)
    return normalized


def build_summary_index(json_store: Dict[str, Dict]) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
    for full_name, entry in (json_store or {}).items():
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary") or entry.get("Summary") or ""
        if summary:
            summaries[full_name] = str(summary)
    return summaries


def get_summary_from_json(json_store: Dict[str, Dict], full_name: str) -> str:
    if not json_store or not full_name:
        return ""
    entry = json_store.get(full_name)
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("summary") or entry.get("Summary") or "")


def merge_summary_store(existing_store: Dict[str, Dict], updates: Dict[str, Dict]) -> Dict[str, Dict]:
    merged = dict(existing_store or {})
    for key, value in (updates or {}).items():
        if not key:
            continue
        merged[key] = value
    return merged


def load_summary_store(json_path: str) -> Dict[str, Dict]:
    raw = load_json(json_path)
    return normalize_json_store(raw)
