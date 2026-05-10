import os
import re
from typing import Dict

from scripts.core.json_store import build_summary_index, load_json, normalize_json_store


def load_old_summaries(json_path: str, readme_sum_path: str, language: str) -> Dict[str, str]:
    """Prefer JSON summaries; fallback to README summary file."""
    json_data = load_json(json_path)
    json_store = normalize_json_store(json_data)
    summaries = build_summary_index(json_store)
    if summaries:
        return summaries

    if not readme_sum_path or not isinstance(readme_sum_path, str) or not os.path.exists(readme_sum_path):
        return {}

    summaries = {}
    current_repo = None
    current_lines = []
    with open(readme_sum_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("### 📌 ["):
                if current_repo and current_lines:
                    summary = _extract_summary_block(current_lines, language)
                    if summary:
                        summaries[current_repo] = summary
                left = line.find("[") + 1
                right = line.find("]")
                current_repo = line[left:right]
                current_lines = []
            elif current_repo:
                current_lines.append(line)
        if current_repo and current_lines:
            summary = _extract_summary_block(current_lines, language)
            if summary:
                summaries[current_repo] = summary
    return summaries


def _extract_summary_block(lines, language: str) -> str:
    summary_block = "".join(lines)
    summary = summary_block.split("---")[0].strip()
    summary = re.sub(r"\*\*⭐ Stars:.*更新:.*\n", "", summary)
    summary = re.sub(r"\*\*⭐ Stars:.*Updated:.*\n", "", summary)
    if language == "en":
        if re.search(r"[\u4e00-\u9fa5]", summary):
            return ""
    else:
        if re.search(r"[A-Za-z]", summary) and not re.search(r"[\u4e00-\u9fa5]", summary):
            return ""
    return summary
