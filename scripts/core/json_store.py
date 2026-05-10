import json
import os
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_summary_json_path(language: str) -> str:
    filename = "repo_summaries_zh.json" if language == "zh" else "repo_summaries_en.json"
    return os.path.join(REPO_ROOT, filename)


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
