import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


LANG_ICONS = {
    "Python": "🐍",
    "JavaScript": "🟨",
    "TypeScript": "🔷",
    "Java": "☕",
    "Go": "🐹",
    "Rust": "🦀",
    "C++": "⚡",
    "C": "🔧",
    "C#": "💜",
    "PHP": "🐘",
    "Ruby": "💎",
    "Swift": "🐦",
    "Kotlin": "🅺",
    "Dart": "🎯",
    "Shell": "🐚",
    "HTML": "🌐",
    "CSS": "🎨",
    "Vue": "💚",
    "React": "⚛️",
    "Other": "📦",
}


def classify_by_language(repos: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    classified: Dict[str, List[Dict[str, Any]]] = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified


def github_anchor(text: str) -> str:
    anchor = text.strip().lower()
    anchor = re.sub(r"[\s]+", "-", anchor)
    anchor = re.sub(r"[^\w\u4e00-\u9fa5-]", "", anchor)
    return anchor


def _get_lang_icon(lang: str) -> str:
    return LANG_ICONS.get(lang, "📝")


def _format_updated_at(updated_at: str) -> str:
    if not updated_at:
        return ""
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return updated_at[:10]


def _select_repos_for_update(
    classified: Dict[str, List[Dict[str, Any]]],
    summary_store: Dict[str, Dict[str, Any]],
    old_summaries: Dict[str, str],
    mode: str,
    is_valid_func: callable,
) -> Dict[str, List[Dict[str, Any]]]:
    if mode != "missing_only":
        return classified
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = str(repo.get("full_name") or repo.get("Repository Name") or "").strip()
            fallback = old_summaries.get(key, "")
            summary = (summary_store.get(key, {}) or {}).get("summary") or fallback
            if not is_valid_func(str(summary or "")):
                needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def build_readme_header(
    language: str,
    github_username: str,
    api_name: str,
    total_repos: int,
    current_time: str,
) -> List[str]:
    current_account = (github_username or "").strip()
    if current_account:
        current_account_html = f'<a href="https://github.com/{current_account}">{current_account}</a>'
    else:
        current_account_html = "未知" if language != "en" else "Unknown"

    readme_links = (
        '<a href="README.md">README（内容分类）</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a> | '
        '<a href="README_lang.md">README classified by language</a>'
    ) if language != "en" else (
        '<a href="README.md">README (content classified)</a> | '
        '<a href="README_lang.md">README classified by language</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a>'
    )

    guide_links = (
        '<a href="GUIDE_zh.md">中文教程</a> | <a href="GUIDE_en.md">English GUIDE</a>'
    ) if language != "en" else (
        '<a href="GUIDE_en.md">English GUIDE</a> | <a href="GUIDE_zh.md">中文教程</a>'
    )

    lines: List[str] = []

    if language == "en":
        lines.append(
            "<div align=\"center\">\n\n"
            "<h1>My GitHub Star Project AI Summary</h1>\n\n"
            "<p><b>Reference Repository:</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n\n"
            f"<p>{readme_links}</p>\n"
            f"<p>{guide_links}</p>\n\n"
            "<hr/>\n\n"
            f"<p><b>Current account:</b> {current_account_html}</p>\n"
            f"<p><b>Generated on:</b> {current_time}</p>\n"
            f"<p><b>AI Model:</b> {api_name}</p>\n"
            f"<p><b>Total repositories:</b> {total_repos}</p>\n\n"
            "</div>\n\n"
        )
    else:
        lines.append(
            "<div align=\"center\">\n\n"
            "<h1>我的 GitHub Star 项目AI总结</h1>\n\n"
            "<p><b>参考仓库：</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n\n"
            f"<p>{readme_links}</p>\n"
            f"<p>{guide_links}</p>\n\n"
            "<hr/>\n\n"
            f"<p><b>当前账号：</b> {current_account_html}</p>\n"
            f"<p><b>生成时间：</b> {current_time}</p>\n"
            f"<p><b>AI模型：</b> {api_name}</p>\n"
            f"<p><b>总仓库数：</b> {total_repos} 个</p>\n\n"
            "</div>\n\n"
        )

    return lines


def build_table_of_contents(
    classified: Dict[str, List[Dict[str, Any]]],
    language: str,
) -> List[str]:
    lines: List[str] = []
    lines.append("## 📖 目录\n\n" if language != "en" else "## 📖 Table of Contents\n\n")

    lang_counts: Dict[str, int] = {lang: len(repos) for lang, repos in classified.items()}
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        anchor = github_anchor(lang)
        count_str = f"（{count}个）" if language != "en" else f"(Total {count})"
        lines.append(f"- [{lang}](#{anchor}) {count_str}\n")
    lines.append("\n---\n\n")
    return lines


def build_repo_section(
    lang: str,
    repos: List[Dict[str, Any]],
    language: str,
    summary_store: Dict[str, Dict[str, Any]],
    old_summaries: Dict[str, Any],
    rate_limit_delay: float,
    printed_repos: set,
    printed_langs: set,
    processed_repos: int,
) -> Tuple[List[str], set, set, int]:
    if lang in printed_langs:
        return [], printed_repos, printed_langs, processed_repos

    printed_langs.add(lang)
    lines: List[str] = []
    icon = _get_lang_icon(lang)
    count_str = f"（共{len(repos)}个）" if language != "en" else f"(Total {len(repos)})"
    lines.append(f"## {icon} {lang}{count_str}\n\n")

    for repo in repos:
        if repo["full_name"] in printed_repos:
            continue
        printed_repos.add(repo["full_name"])

        summary_entry = summary_store.get(repo["full_name"], {}) or old_summaries.get(repo["full_name"], {})

        if isinstance(summary_entry, dict):
            repo_name = summary_entry.get("Repository Name", repo["full_name"])
            brief = summary_entry.get("Brief Introduction", "")
            innovations = summary_entry.get("Innovations", "")
            basic = summary_entry.get("Basic Usage", "")
            summary_text = summary_entry.get("Summary", "")

            if language == "en":
                summary_parts = []
                summary_parts.append(f"1. **Repository Name:** {repo_name}")
                summary_parts.append(f"2. **Brief Introduction:** {(brief or 'Not specified.')}")
                summary_parts.append(f"3. **Innovations:** {(innovations or 'Not specified.')}")
                summary_parts.append(f"4. **Basic Usage:** {(basic or 'Not specified.')}")
                summary_parts.append(f"5. **Summary:** {(summary_text or 'Not specified.')}")
                summary = "\n".join(summary_parts)
            else:
                summary_parts = []
                summary_parts.append(f"1. **仓库名称：** {repo_name}")
                summary_parts.append(f"2. **简要介绍：** {(brief or '未指定。')}")
                summary_parts.append(f"3. **创新点：** {(innovations or '未指定。')}")
                summary_parts.append(f"4. **基本用法：** {(basic or '未指定。')}")
                summary_parts.append(f"5. **总结：** {(summary_text or '未指定。')}")
                summary = "\n".join(summary_parts)
        elif isinstance(summary_entry, str):
            if language == "en":
                summary = f"1. **Repository Name:** {repo['full_name']}\n2. **Brief Introduction:** {summary_entry}\n3. **Innovations:** Not specified.\n4. **Basic Usage:** Not specified.\n5. **Summary:** Not specified."
            else:
                summary = f"1. **仓库名称：** {repo['full_name']}\n2. **简要介绍：** {summary_entry}\n3. **创新点：** 未指定。\n4. **基本用法：** 未指定。\n5. **总结：** 未指定。"
        else:
            if language == "en":
                summary = f"1. **Repository Name:** {repo['full_name']}\n2. **Brief Introduction:** Not specified.\n3. **Innovations:** Not specified.\n4. **Basic Usage:** Not specified.\n5. **Summary:** Not specified."
            else:
                summary = f"1. **仓库名称：** {repo['full_name']}\n2. **简要介绍：** 未指定。\n3. **创新点：** 未指定。\n4. **基本用法：** 未指定。\n5. **总结：** 未指定。"

        url = repo["html_url"]
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        updated_at = _format_updated_at(repo.get("updated_at", ""))

        lines.append(f"### 📌 [{repo['full_name']}]({url})\n\n")

        if language == "en":
            lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **📅 Updated:** {updated_at}\n\n")
        else:
            lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **📅 更新:** {updated_at}\n\n")

        if summary and summary.strip():
            lines.append(f"{summary}\n\n")
        else:
            if language == "en":
                lines.append("*No AI summary available*\n\n")
            else:
                lines.append("*暂无AI总结*\n\n")

        lines.append("---\n\n")
        processed_repos += 1
        time.sleep(rate_limit_delay)

    return lines, printed_repos, printed_langs, processed_repos


def build_readme_footer(
    processed_repos: int,
    num_languages: int,
    current_time: str,
    api_name: str,
    api_call_counts: Tuple[int, int, int],
    language: str,
) -> List[str]:
    lines: List[str] = []
    copilot_count, openrouter_count, gemini_count = api_call_counts

    if language == "en":
        lines.append("\n## 📊 Statistics\n\n")
        lines.append(f"- **Total repositories:** {processed_repos}\n")
        lines.append(f"- **Languages:** {num_languages}\n")
        lines.append(f"- **Generated on:** {current_time}\n")
        lines.append(f"- **AI Model:** {api_name}\n\n")
        lines.append(f"- **API Calls:** Copilot={copilot_count}, OpenRouter={openrouter_count}, Gemini={gemini_count}\n")
        lines.append("---\n\n")
        lines.append("*This document is generated by AI. For any errors, please refer to the original repository information.*\n")
    else:
        lines.append(f"\n## 📊 统计信息\n\n")
        lines.append(f"- **总仓库数：** {processed_repos} 个\n")
        lines.append(f"- **编程语言数：** {num_languages} 种\n")
        lines.append(f"- **生成时间：** {current_time}\n")
        lines.append(f"- **AI模型：** {api_name}\n\n")
        lines.append(f"- **API 调用次数：** Copilot={copilot_count}，OpenRouter={openrouter_count}，Gemini={gemini_count}\n")
        lines.append("---\n\n")
        lines.append("*本文档由AI自动生成，如有错误请以原仓库信息为准。*\n")

    return lines
