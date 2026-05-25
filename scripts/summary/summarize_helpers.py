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


def generate_combined_summarize_prompt(repos: List[Dict[str, Any]], language: str = "zh") -> str:
    if language == "zh":
        repo_list = []
        for i, repo in enumerate(repos, 1):
            repo_name = repo["full_name"]
            desc = repo.get("description") or ""
            url = repo["html_url"]
            repo_list.append(
                f"## 仓库 {i}\n"
                f"- 仓库名称: {repo_name}\n"
                f"- 描述: {desc}\n"
                f"- 地址: {url}"
            )

        return (
            f"你是一个GitHub仓库总结助手。请对以下 {len(repos)} 个仓库分别进行总结，\n"
            f"每个仓库按以下固定格式输出（注意：仓库名称必须与输入完全一致）：\n\n"
            "输出格式（必须是JSON数组）：\n"
            "```json\n"
            "[\n"
            '  {"repo": "owner/repo", "Repository Name": "...", "Repository URL": "...", "Brief Introduction": "...", "Innovations": "...", "Basic Usage": "...", "Summary": "..."},\n'
            "]\n"
            "```\n\n"
            "要求：\n"
            "- Repository Name: 仓库全名（必须与输入完全一致）\n"
            "- Repository URL: 仓库地址\n"
            "- Brief Introduction: 简要介绍（50字以内）\n"
            "- Innovations: 创新点\n"
            "- Basic Usage: 简单用法\n"
            "- Summary: 一句话总结\n"
            "- 只输出JSON数组，不要输出其他内容\n\n"
            "## 待总结的仓库：\n"
            + "\n\n".join(repo_list)
        )
    else:
        repo_list = []
        for i, repo in enumerate(repos, 1):
            repo_name = repo["full_name"]
            desc = repo.get("description") or ""
            url = repo["html_url"]
            repo_list.append(
                f"## Repository {i}\n"
                f"- Name: {repo_name}\n"
                f"- Description: {desc}\n"
                f"- URL: {url}"
            )

        return (
            f"You are a GitHub repository summarization assistant. Please summarize the following {len(repos)} repositories.\n"
            f"Each repository must follow this exact format (note: repository name must match exactly):\n\n"
            "Output format (must be JSON array):\n"
            "```json\n"
            "[\n"
            '  {"repo": "owner/repo", "Repository Name": "...", "Repository URL": "...", "Brief Introduction": "...", "Innovations": "...", "Basic Usage": "...", "Summary": "..."},\n'
            "]\n"
            "```\n\n"
            "Requirements:\n"
            "- Repository Name: full repository name (must match exactly)\n"
            "- Repository URL: repository URL\n"
            "- Brief Introduction: brief intro (within 50 words)\n"
            "- Innovations: key innovations\n"
            "- Basic Usage: basic usage\n"
            "- Summary: one sentence summary\n"
            "- Output only JSON array, nothing else\n\n"
            "## Repositories to summarize:\n"
            + "\n\n".join(repo_list)
        )


def parse_combined_summaries(response_text: str, repos: List[Dict[str, Any]]) -> Dict[str, Dict]:
    import json

    results: Dict[str, Dict] = {}
    for repo in repos:
        results[repo["full_name"]] = {}

    if not response_text:
        return results

    text = response_text.strip()

    json_match = None
    for pattern in [r"```json\s*(\[[\s\S]*?)\s*```", r"(\[[\s\S]*?\])"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    json_match = data
                    break
            except json.JSONDecodeError:
                continue

    if not json_match:
        try:
            arr_start = text.find("[")
            arr_end = text.rfind("]")
            if arr_start != -1 and arr_end != -1:
                json_str = text[arr_start:arr_end+1]
                json_match = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            print(f"[WARN] JSON parse failed for batch, repos: {[r['full_name'] for r in repos]}")
            pass

    if json_match and isinstance(json_match, list):
        for item in json_match:
            if isinstance(item, dict):
                repo_name = item.get("repo", "") or item.get("Repository Name", "")
                if not repo_name:
                    continue

                brief_intro = item.get("Brief Introduction", "") or item.get("简要介绍", "")
                innovations = item.get("Innovations", "") or item.get("创新点", "")
                basic_usage = item.get("Basic Usage", "") or item.get("简单用法", "")
                summary = item.get("Summary", "") or item.get("总结", "")

                full_entry = {
                    "Repository Name": repo_name,
                    "Repository URL": item.get("Repository URL", "") or item.get("仓库地址", ""),
                    "Brief Introduction": brief_intro,
                    "Innovations": innovations,
                    "Basic Usage": basic_usage,
                    "Summary": summary,
                }

                for existing_repo in repos:
                    if existing_repo["full_name"] == repo_name:
                        full_entry["Repository URL"] = existing_repo.get("html_url", full_entry["Repository URL"])
                        break

                results[repo_name] = full_entry

    return results


def is_valid_summary(summary: str, language: str = "zh") -> bool:
    if not summary or not summary.strip():
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached", "Not specified"]
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


def build_repo_entry(repo: Dict, summary: Any) -> Dict:
    if isinstance(summary, dict):
        entry = {
            "Repository Name": summary.get("Repository Name", repo.get("full_name")),
            "Repository URL": summary.get("Repository URL", repo.get("html_url")),
            "Brief Introduction": summary.get("Brief Introduction", ""),
            "Innovations": summary.get("Innovations", ""),
            "Basic Usage": summary.get("Basic Usage", ""),
            "Summary": summary.get("Summary", ""),
        }
        entry["Repository URL"] = repo.get("html_url") or entry.get("Repository URL", "")
        return entry

    return {
        "Repository Name": repo.get("full_name"),
        "Repository URL": repo.get("html_url"),
        "Brief Introduction": "",
        "Innovations": "",
        "Basic Usage": "",
        "Summary": summary or "",
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
                print(f"[DEBUG] [repo]: {repo['full_name']} | [AI summary]: {repr(summary.strip())}")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                api_name = summarize_func.__name__.replace("_summarize", "").upper()
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary if summary is not None else "*暂无AI总结*"
    return results


def summarize_batch_combined(
    repos: List[Dict],
    old_summaries: Dict[str, Any],
    summarize_func: Callable[[Dict], Optional[str]],
    update_mode: str,
    language: str,
    batch_size: int = 5,
    batch_num: int = 1,
) -> List[Any]:
    results: List[Any] = [{} for _ in repos]

    repos_need_call = []
    repos_indices = []

    for idx, repo in enumerate(repos):
        existing_summary = old_summaries.get(repo["full_name"], {})
        if isinstance(existing_summary, dict) and existing_summary.get("Summary"):
            if update_mode == "missing_only":
                results[idx] = existing_summary
                print(f"[REUSE] repo: {repo['full_name']} | existing summary")
            else:
                repos_need_call.append(repo)
                repos_indices.append(idx)
        elif isinstance(existing_summary, str) and existing_summary:
            if update_mode == "missing_only":
                results[idx] = {"Repository Name": repo["full_name"], "Repository URL": repo.get("html_url", ""), "Brief Introduction": "", "Innovations": "", "Basic Usage": "", "Summary": existing_summary}
                print(f"[REUSE] repo: {repo['full_name']} | existing summary (legacy format)")
            else:
                repos_need_call.append(repo)
                repos_indices.append(idx)
        else:
            repos_need_call.append(repo)
            repos_indices.append(idx)

    for i in range(0, len(repos_need_call), batch_size):
        batch = repos_need_call[i : i + batch_size]
        indices = repos_indices[i : i + batch_size]
        print(f"[COMBINED] Processing batch {batch_num}, {len(batch)} repos...")

        combined_prompt = generate_combined_summarize_prompt(batch, language)
        repo_with_prompt = {"prompt": combined_prompt, "repos": [r["full_name"] for r in batch]}
        print(f"[DEBUG] Batch {batch_num} prompt length: {len(combined_prompt)} chars, repos: {[r['full_name'] for r in batch]}", flush=True)
        print(f"[DEBUG] Batch {batch_num} calling summarize_func...", flush=True)

        try:
            response_text = summarize_func(repo_with_prompt)
            print(f"[DEBUG] Batch {batch_num} summarize_func returned, response_text type={type(response_text)}, len={len(response_text) if response_text else 0}", flush=True)
            if response_text:
                parsed = parse_combined_summaries(response_text, batch)
                for full_name, summary_dict in parsed.items():
                    if summary_dict and summary_dict.get("Summary"):
                        for idx, repo in zip(indices, batch):
                            if repo["full_name"] == full_name:
                                results[idx] = summary_dict
                                print(f"[DEBUG] [repo]: {full_name} | [AI summary]: {repr(summary_dict.get('Summary', '').strip()[:80])}...")
                                break
                    else:
                        api_name = summarize_func.__name__.replace("_summarize", "").upper()
                        results[idx] = {"Repository Name": full_name, "Repository URL": "", "Brief Introduction": "", "Innovations": "", "Basic Usage": "", "Summary": old_summaries.get(full_name, f"{api_name} 解析失败或为空")}
                        print(f"[WARN] repo: {full_name} | empty summary from LLM")
            else:
                print(f"[ERROR] Batch {batch_num} response is None or empty, repos: {[r['full_name'] for r in batch]}")
                for idx, repo in zip(indices, batch):
                    api_name = summarize_func.__name__.replace("_summarize", "").upper()
                    results[idx] = {"Repository Name": repo["full_name"], "Repository URL": repo.get("html_url", ""), "Brief Introduction": "", "Innovations": "", "Basic Usage": "", "Summary": old_summaries.get(repo["full_name"], f"{api_name} API返回空")}
                    print(f"[ERROR] repo: {repo['full_name']} | empty response")
        except Exception as exc:
            import traceback
            print(f"[ERROR] Batch {batch_num} exception: {exc}")
            print(f"[ERROR] Exception details: {traceback.format_exc()}")
            for idx, repo in zip(indices, batch):
                api_name = summarize_func.__name__.replace("_summarize", "").upper()
                results[idx] = {"Repository Name": repo["full_name"], "Repository URL": repo.get("html_url", ""), "Brief Introduction": "", "Innovations": "", "Basic Usage": "", "Summary": old_summaries.get(repo["full_name"], f"{api_name} API调用失败")}
                print(f"[ERROR] repo: {repo['full_name']} | {exc}")

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
    modelscope_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    default_modelscope_model: str,
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
        modelscope_api_key=modelscope_api_key,
        default_copilot_model=default_copilot_model,
        default_openrouter_model=default_openrouter_model,
        default_gemini_model=default_gemini_model,
        default_modelscope_model=default_modelscope_model,
        language=language,
        config=config,
        throttle=throttle,
        request_timeout=request_timeout,
        request_retry_delay=request_retry_delay,
        retry_attempts=retry_attempts,
        api_call_counter=api_call_counter,
    )
