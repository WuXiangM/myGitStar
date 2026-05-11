import concurrent.futures
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _repo_key(repo: Dict) -> str:
    return str(repo.get("full_name") or repo.get("Repository Name") or "").strip()


def generate_summarize_prompt(repo: Dict[str, Any], language: str = "zh") -> str:
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo["html_url"]

    if language == "zh":
        return (
            f"请对以下 GitHub 仓库进行内容总结，按如下格式输出：\n"
            f"1. **仓库名称：** {repo_name}\n"
            f"2. **简要介绍：** （50字以内）\n"
            f"3. **创新点：** （简述本仓库最有特色的地方）\n"
            f"4. **简单用法：** （给出最简关键用法或调用示例，如无则略）\n"
            f"5. **总结：** （一句话总结它的用途/价值）\n"
            f"**仓库描述：** {desc}\n"
            f"**仓库地址：** {url}\n"
        )
    else:
        return (
            f"Please summarize the following GitHub repository in the specified format:\n"
            f"1. **Repository Name:** {repo_name}\n"
            f"2. **Brief Introduction:** (within 50 words)\n"
            f"3. **Innovations:** (Briefly describe the most distinctive features)\n"
            f"4. **Basic Usage:** (Provide the simplest key usage or example, omit if none)\n"
            f"5. **Summary:** (One sentence summarizing its purpose/value)\n"
            f"**Repository Description:** {desc}\n"
            f"**Repository URL:** {url}\n"
        )


def is_valid_summary(summary: str, language: str = "zh") -> bool:
    if not summary or not summary.strip():
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached"]
    for phrase in invalid_phrases:
        if phrase in summary:
            return False

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
    if language != "en":
        for p in common_english_templates:
            if re.search(p, s_head, flags=re.IGNORECASE):
                return False
    if language == "en":
        for p in common_chinese_templates:
            if re.search(p, s_head):
                return False

    full_text = summary.strip()
    if language == "en":
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
        if language == "en":
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
    old_summaries: Dict[str, str],
    mode: str,
    language: str = "zh",
) -> Dict[str, List[Dict]]:
    if mode != "missing_only":
        return classified
    filtered: Dict[str, List[Dict]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = _repo_key(repo)
            fallback = old_summaries.get(key, "")
            if not is_valid_summary(fallback, language):
                needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def summarize_batch(
    repos: List[Dict],
    old_summaries: Dict[str, str],
    summarize_func: Callable[[Dict], Optional[str]],
    update_mode: str,
    language: str,
    max_workers: int = 5,
) -> List[str]:
    results: List[str] = ["" for _ in repos]

    repos_with_prompts = []
    for repo in repos:
        repo_copy = dict(repo)
        repo_copy["prompt"] = generate_summarize_prompt(repo_copy, language)
        repos_with_prompts.append(repo_copy)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(summarize_func, repo): idx for idx, repo in enumerate(repos_with_prompts)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            repo = repos[idx]
            try:
                existing_summary = old_summaries.get(repo["full_name"], "")
                reuse_existing = (update_mode == "missing_only") and is_valid_summary(existing_summary, language)
                if reuse_existing:
                    summary = existing_summary
                else:
                    summary = future.result()
                    if summary is None:
                        api_name = summarize_func.__name__.replace("_summarize", "").upper()
                        summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败或429")
                print(f"[DEBUG] repo: {repo['full_name']} | summary: {repr(summary)}")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                api_name = summarize_func.__name__.replace("_summarize", "").upper()
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary if summary is not None else "*暂无AI总结*"
    return results


def sort_repos_by_validity(
    repos: List[Dict],
    old_summaries: Dict[str, str],
    language: str = "zh",
) -> List[Dict]:
    try:
        return sorted(repos, key=lambda r: is_valid_summary(old_summaries.get(r.get("full_name", ""), ""), language))
    except Exception:
        return repos


def get_summarize_func(
    model_choice: str,
    github_token: str,
    openrouter_api_key: str,
    gemini_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    language: str,
    config: Dict[str, Any],
    throttle: Any,
    request_timeout: float,
    request_retry_delay: float,
    retry_attempts: int,
    api_call_counter: Callable,
):
    from scripts.ai.api_clients import create_summarize_func

    return create_summarize_func(
        model_choice=model_choice,
        github_token=github_token,
        openrouter_api_key=openrouter_api_key,
        gemini_api_key=gemini_api_key,
        default_copilot_model=default_copilot_model,
        default_openrouter_model=default_openrouter_model,
        default_gemini_model=default_gemini_model,
        language=language,
        config=config,
        throttle=throttle,
        request_timeout=request_timeout,
        request_retry_delay=request_retry_delay,
        retry_attempts=retry_attempts,
        api_call_counter=api_call_counter,
    )
