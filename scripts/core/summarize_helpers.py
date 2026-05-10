import re
from typing import Dict, List


def repo_key(repo: Dict) -> str:
    return str(repo.get("full_name") or repo.get("Repository Name") or "").strip()


def is_valid_summary(summary: str, language: str) -> bool:
    """Check if summary is valid for the given language."""
    if not summary or not summary.strip():
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached"]
    for phrase in invalid_phrases:
        if phrase in summary:
            return False

    lang = (language or "zh").strip().lower()

    common_english_templates = [
        r"Here'?s the summary",
        r"Here is the summary",
        r"Repository Name",
        r"Brief Introduction",
        r"Innovations",
        r"Basic Usage",
        r"Summary\s*:",
        r"Please summarize",
    ]

    common_chinese_templates = [
        r"仓库名称",
        r"简要介绍",
        r"创新点",
        r"简单用法",
        r"总结\s*[:：]",
        r"请对以下 GitHub 仓库进行内容总结",
    ]

    s_head = summary.strip()[:200]
    if lang != "en":
        for p in common_english_templates:
            if re.search(p, s_head, flags=re.IGNORECASE):
                return False
    if lang == "en":
        for p in common_chinese_templates:
            if re.search(p, s_head):
                return False

    full_text = summary.strip()
    if lang == "en":
        patterns = [r"Summary\s*[:：]", r"Repository Name", r"Brief Introduction", r"Innovations"]
    else:
        patterns = [r"总结\s*[:：]", r"仓库名称", r"简要介绍", r"创新点"]

    missing = []
    for p in patterns:
        if not re.search(p, full_text, flags=re.IGNORECASE):
            missing.append(p)
    if missing:
        return False

    try:
        s = summary
        if lang == "en":
            m = re.search(r"Brief Introduction\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", s, flags=re.IGNORECASE | re.S)
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                if len(intro_text) < 20:
                    return False
        else:
            m = re.search(r"简要介绍\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", s, flags=re.S)
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                if len(intro_text) < 10:
                    return False
    except Exception:
        pass

    return True


def should_update_repo(repo: Dict, summary_store: Dict[str, Dict], fallback_summary: str, language: str) -> bool:
    key = repo_key(repo)
    if not key:
        return True
    summary = (summary_store.get(key, {}) or {}).get("summary") or fallback_summary
    return not is_valid_summary(str(summary or ""), language)


def build_repo_entry(repo: Dict, summary: str) -> Dict:
    return {
        "full_name": repo.get("full_name"),
        "Repository Name": repo.get("full_name"),
        "Repository URL": repo.get("html_url"),
        "description": repo.get("description"),
        "language": repo.get("language"),
        "stargazers_count": repo.get("stargazers_count"),
        "forks_count": repo.get("forks_count"),
        "updated_at": repo.get("updated_at"),
        "summary": summary or "",
    }


def select_repos_for_update(
    classified: Dict[str, List[Dict]],
    summary_store: Dict[str, Dict],
    old_summaries: Dict[str, str],
    mode: str,
    language: str,
) -> Dict[str, List[Dict]]:
    if mode != "missing_only":
        return classified
    filtered: Dict[str, List[Dict]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = repo_key(repo)
            fallback = old_summaries.get(key, "")
            if should_update_repo(repo, summary_store, fallback, language):
                needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def classify_by_language(repos: List[Dict]) -> Dict[str, List[Dict]]:
    classified: Dict[str, List[Dict]] = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified


def update_existing_summaries(lines: List[str], old_summaries: Dict[str, str]) -> List[str]:
    updated_lines: List[str] = []
    current_repo = None
    for line in lines:
        if line.startswith("### ["):
            left = line.find("[") + 1
            right = line.find("]")
            current_repo = line[left:right]
            updated_lines.append(line)
        elif current_repo and current_repo in old_summaries:
            updated_lines.append(old_summaries[current_repo] + "\n")
            current_repo = None
        else:
            updated_lines.append(line)
    return updated_lines


def github_anchor(text: str) -> str:
    anchor = text.strip().lower()
    anchor = re.sub(r"[\s]+", "-", anchor)
    anchor = re.sub(r"[^\w\u4e00-\u9fa5-]", "", anchor)
    return anchor
