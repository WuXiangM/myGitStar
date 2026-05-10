import concurrent.futures
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from scripts.core.config import (
    load_config,
    get_int_config,
    get_float_config,
    env_truthy,
    resolve_update_mode,
    normalize_update_mode,
)
from scripts.core.secrets import load_api_keys
from scripts.core.throttle import SimpleThrottle
from scripts.core.json_store import (
    get_summary_json_path,
    load_json,
    normalize_json_store,
    build_summary_index,
    get_summary_from_json,
    save_json_atomic,
    merge_summary_store,
    load_summary_store,
)
from scripts.api_clients import make_api_request
from scripts.github_api import get_starred_repos
from scripts.prompts import generate_summarize_prompt
from scripts.readme_builder import (
    classify_by_language,
    build_readme_header,
    build_table_of_contents,
    build_repo_section,
    build_readme_footer,
)


config = load_config()

DEBUG_API = env_truthy("DEBUG_API") or bool(config.get("test_first_repo", False))

GITHUB_TOKEN, OPENROUTER_API_KEY, GEMINI_API_KEY = load_api_keys(config)

update_mode = resolve_update_mode(config)

github_username = config.get("github_username")
model_choice = config.get("model_choice", "copilot")

default_copilot_model = config.get("default_copilot_model")
default_openrouter_model = config.get("default_openrouter_model")
default_gemini_model = config.get("default_gemini_model", "gemini-pro")

max_workers = get_int_config(config, "max_workers", 5)
batch_size = get_int_config(config, "batch_size", 1)
request_timeout = get_float_config(config, "request_timeout", 10.0)
rate_limit_delay = get_float_config(config, "rate_limit_delay", 1.0)
request_retry_delay = get_int_config(config, "request_retry_delay", 5)
retry_attempts = get_int_config(config, "retry_attempts", 3)
readme_sum_path = config.get("readme_sum_path")

if github_username == "0" or github_username == 0:
    GITHUB_USERNAME = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
    if not GITHUB_USERNAME:
        print("未检测到 workflow 账号环境变量 GITHUB_ACTOR/GITHUB_USERNAME，请检查 workflow 配置！")
else:
    GITHUB_USERNAME = github_username

MAX_REPOS: Optional[int] = None
max_repos_env = os.environ.get("MAX_REPOS")
if max_repos_env:
    try:
        mr = int(max_repos_env)
        if mr > 0:
            MAX_REPOS = mr
    except Exception:
        MAX_REPOS = None

if MAX_REPOS is None:
    try:
        cfg_mr = config.get("max_repos") if isinstance(config, dict) else None
        if cfg_mr is not None:
            mr = int(cfg_mr)
            if mr > 0:
                MAX_REPOS = mr
    except Exception:
        pass

GLOBAL_QPS = get_float_config(config, "global_qps", 0.5)
THROTTLE = SimpleThrottle(GLOBAL_QPS)

copilot_api_call_count = 0
openrouter_api_call_count = 0
gemini_api_call_count = 0


def _repo_key(repo: Dict) -> str:
    return str(repo.get("full_name") or repo.get("Repository Name") or "").strip()


def _make_request_with_throttle(url: str, headers: Dict, data: Dict, retries: int, retry_delay: float, timeout: float) -> Optional[Dict]:
    return make_api_request(url, headers, data, retries, retry_delay, timeout, THROTTLE)


def copilot_summarize(repo: Dict) -> Optional[str]:
    global copilot_api_call_count
    copilot_api_call_count += 1
    remaining = 150 - copilot_api_call_count
    print(f"[Copilot API调用] 第 {copilot_api_call_count} 次调用，仓库: {repo['full_name']}，剩余可用: {remaining}")
    if not GITHUB_TOKEN:
        print("缺少 STARRED_GITHUB_TOKEN，无法调用 GitHub Copilot API")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", default_copilot_model) or "openai/gpt-4o-mini"
        prompt = generate_summarize_prompt(repo, LANGUAGE)
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.4,
        }
        response = _make_request_with_throttle(
            "https://models.github.ai/inference/chat/completions",
            headers,
            data,
            retry_attempts,
            float(request_retry_delay),
            request_timeout,
        )
        if response and isinstance(response, dict) and response.get("error"):
            err = response["error"]
            if err.get("code") == "RateLimitReached":
                msg = err.get("message", "Copilot API限额已用尽，请明天再试。")
                print(f"[Copilot限额] {msg}")
                return f"Copilot API限额已用尽：{msg}"
        content = None
        if response:
            choices = response.get("choices", [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if message and isinstance(message, dict):
                    content = message.get("content", "")
                elif "content" in choices[0]:
                    content = choices[0]["content"]
            if content is not None:
                content = str(content).strip()
        print(f"Copilot内容: {content!r}")
        return content if content else None
    except Exception as e:
        print(f"Copilot总结异常: {e}")
        return None


def openrouter_summarize(repo: Dict) -> Optional[str]:
    global openrouter_api_call_count
    openrouter_api_call_count += 1
    if not OPENROUTER_API_KEY:
        print("缺少 OPENROUTER_API_KEY，无法调用 OpenRouter API")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = generate_summarize_prompt(repo, LANGUAGE)
        data = {
            "model": default_openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = _make_request_with_throttle(
            "https://openrouter.ai/api/v1/chat/completions",
            headers,
            data,
            retry_attempts,
            float(request_retry_delay),
            request_timeout,
        )
        content = None
        if response:
            choices = response.get("choices", [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if message and isinstance(message, dict):
                    content = message.get("content", "")
                elif "content" in choices[0]:
                    content = choices[0]["content"]
            if content is not None:
                content = str(content).strip()
        print(f"OpenRouter内容: {content!r}")
        return content if content else None
    except Exception as e:
        print(f"OpenRouter总结异常: {e}")
        return None


def gemini_summarize(repo: Dict) -> Optional[str]:
    global gemini_api_call_count
    gemini_api_call_count += 1
    if not GEMINI_API_KEY:
        print("缺少 GEMINI_API_KEY，无法调用 Gemini API")
        return None
    prompt = generate_summarize_prompt(repo, LANGUAGE)
    if not prompt.strip():
        print(f"[Gemini] 仓库 {repo.get('full_name')} 的生成提示为空，跳过请求")
        return None
    try:
        model_name = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-pro"
        model_path = str(model_name).strip()
        if model_path.startswith("models/"):
            model_path = model_path[len("models/"):]
        model_path = model_path.strip()
        if not model_path.startswith("gemini-"):
            print(f"[Gemini] 模型名称 {model_name} 非标准格式，建议使用 gemini-pro/gemini-1.5-pro 等")

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        request_url = f"{api_url}?key={GEMINI_API_KEY}"

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GitHub Star Summary Bot/1.0",
            "X-Goog-Api-Key": GEMINI_API_KEY,
        }

        temperature = config.get("gemini_temperature", 0.4)
        max_output_tokens = config.get("gemini_max_output_tokens", 800)
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.8,
                "topK": 40,
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

        gen_retries = int(config.get("gemini_generation_retries", 3))
        gen_backoff = int(config.get("gemini_retry_backoff", 5))
        base_max_tokens = int(config.get("gemini_max_output_tokens", max_output_tokens))

        final_content = None
        for attempt in range(1, gen_retries + 1):
            attempt_max_tokens = min(base_max_tokens + (attempt - 1) * 200, 2048)
            payload["generationConfig"]["maxOutputTokens"] = attempt_max_tokens

            response = _make_request_with_throttle(
                url=request_url,
                headers=headers,
                data=payload,
                retries=get_int_config(config, "gemini_retry_attempts", retry_attempts),
                retry_delay=float(get_float_config(config, "gemini_retry_delay", float(request_retry_delay))),
                timeout=request_timeout,
            )

            if not response or not isinstance(response, dict):
                if attempt < gen_retries:
                    wait = gen_backoff * attempt
                    print(f"[Gemini] 响应异常或为空，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[Gemini] 仓库 {repo.get('full_name')} 响应为空或格式异常，已放弃")
                    return None

            if "error" in response:
                error = response["error"]
                error_code = error.get("code")
                error_msg = error.get("message", "未知错误")
                print(f"[Gemini] API 错误 - 代码: {error_code}, 信息: {error_msg}")
                if error_code in (429, 503):
                    if attempt < gen_retries:
                        wait = gen_backoff * attempt
                        print(f"[Gemini] 遇到 {error_code}，等待 {wait} 秒后重试...")
                        time.sleep(wait)
                        continue
                if error_code == 401:
                    return "Gemini API Key 无效或无权限"
                elif error_code == 404:
                    return f"Gemini 模型 {model_path} 不存在"
                elif error_code == 400:
                    return "Gemini 请求参数格式错误"
                else:
                    return f"Gemini API 错误: {error_msg}"

            content = ""
            truncated = False
            try:
                candidates = response.get("candidates", [])
                for candidate in candidates:
                    finish = candidate.get("finishReason")
                    content_obj = candidate.get("content", {})
                    parts = content_obj.get("parts", [])
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            content += part["text"].strip() + "\n"
                    if finish == "MAX_TOKENS":
                        truncated = True
                    break

                if not content:
                    choices = response.get("choices", [])
                    for choice in choices:
                        if isinstance(choice, dict):
                            content = choice.get("message", {}).get("content", "") or choice.get("text", "")
                            if content:
                                break

                content = content.strip()
            except Exception as e:
                print(f"[Gemini] 解析响应异常: {e}")
                content = ""

            if content and not truncated:
                final_content = content
                break

            if attempt < gen_retries:
                wait = gen_backoff * attempt
                print(f"[Gemini] 生成内容为空或被截断 (attempt={attempt})，等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            else:
                if content:
                    final_content = content
                else:
                    print(f"[Gemini] 仓库 {repo.get('full_name')} 无有效总结内容，已放弃")
                    return None

        if final_content:
            print(f"[Gemini] 仓库 {repo.get('full_name')} 总结内容: {final_content[:50]}...")
            return final_content
        return None

    except Exception as e:
        print(f"[Gemini] 总结仓库 {repo.get('full_name')} 异常: {e}")
        return None


def get_summarize_func():
    if model_choice == "copilot":
        return copilot_summarize
    elif model_choice == "openrouter":
        return openrouter_summarize
    elif model_choice == "gemini":
        return gemini_summarize
    else:
        raise ValueError(f"不支持的模型选择: {model_choice}")


summarize_func = get_summarize_func()

API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

README_SUM_PATH = readme_sum_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "README-sum.md")
LANGUAGE = config.get("language", "zh")

LOG_FILE = config.get("log_file", os.path.join(os.path.dirname(__file__), "summarize_stars.log"))
LOG_MAX_BYTES = get_int_config(config, "log_max_bytes", 5 * 1024 * 1024)
LOG_BACKUP_COUNT = get_int_config(config, "log_backup_count", 3)

logger = logging.getLogger("summarize_stars")
logger.setLevel(logging.DEBUG if DEBUG_API else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

orig_stdout = sys.stdout
orig_stderr = sys.stderr


class TeeStream:
    def __init__(self, orig, lg, level):
        self.orig = orig
        self.lg = lg
        self.level = level

    def write(self, msg):
        try:
            self.orig.write(msg)
        except Exception:
            pass
        if msg and msg.strip():
            try:
                self.lg.log(self.level, msg.rstrip())
            except Exception:
                pass

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


sys.stdout = TeeStream(orig_stdout, logger, logging.INFO)
sys.stderr = TeeStream(orig_stderr, logger, logging.ERROR)

if OPENROUTER_API_KEY:
    print(f"OpenRouter API Key 前缀: {OPENROUTER_API_KEY[:6]}...")
if GITHUB_TOKEN:
    print(f"GitHub Token 前缀: {GITHUB_TOKEN[:6]}...")
if GEMINI_API_KEY:
    try:
        print(f"Gemini API Key 前缀: {GEMINI_API_KEY[:4]}...")
    except Exception:
        print("Gemini API Key 前缀: (已设置)")


def is_valid_summary(summary: str) -> bool:
    if not summary or not summary.strip():
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached"]
    for phrase in invalid_phrases:
        if phrase in summary:
            return False

    import re

    try:
        lang = config.get("language", "zh")
    except Exception:
        lang = "zh"

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


def summarize_batch(
    repos: List[Dict],
    old_summaries: Dict[str, str],
    use_copilot: bool = False,
    use_gemini: bool = False,
) -> List[str]:
    results: List[str] = ["" for _ in repos]
    if use_gemini:
        func = gemini_summarize
        api_name = "Gemini"
    elif use_copilot:
        func = copilot_summarize
        api_name = "Copilot"
    else:
        func = openrouter_summarize
        api_name = "OpenRouter"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, repo): idx for idx, repo in enumerate(repos)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            repo = repos[idx]
            try:
                existing_summary = old_summaries.get(repo["full_name"], "")
                reuse_existing = (update_mode == "missing_only") and is_valid_summary(existing_summary)
                if reuse_existing:
                    summary = existing_summary
                else:
                    summary = future.result()
                    if summary is None:
                        summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败或429")
                print(f"[DEBUG] repo: {repo['full_name']} | summary: {repr(summary)}")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary if summary is not None else "*暂无AI总结*"
    return results


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
) -> Dict[str, List[Dict]]:
    if mode != "missing_only":
        return classified
    filtered: Dict[str, List[Dict]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = _repo_key(repo)
            fallback = old_summaries.get(key, "")
            if not is_valid_summary(fallback):
                needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def main():
    if model_choice:
        api_choice = model_choice.lower()
    else:
        api_choice = "copilot" if os.environ.get("USE_COPILOT_API", "true").lower() == "true" else "openrouter"

    if api_choice == "gemini":
        api_name = "Gemini"
    elif api_choice == "openrouter":
        api_name = "OpenRouter (DeepSeek)"
    else:
        api_name = "GitHub Copilot"

    print(f"开始使用 {api_name} 生成 GitHub Star 项目总结...")
    print(f"[mode] update_mode={update_mode} (missing_only=仅补缺失/新增；all=全量重汇总)")

    try:
        starred = get_starred_repos(GITHUB_TOKEN, GITHUB_USERNAME, THROTTLE, request_timeout, MAX_REPOS)

        try:
            test_first_repo = bool(config.get("test_first_repo", False))
        except Exception:
            test_first_repo = False
        if test_first_repo and isinstance(starred, list) and len(starred) > 0:
            print("[TEST MODE] test_first_repo 已启用：仅处理第一个仓库进行调试")
            starred = [starred[0]]

        if MAX_REPOS and isinstance(starred, list):
            try:
                limit = int(MAX_REPOS)
                if limit > 0 and len(starred) > limit:
                    print(f"[LIMIT] 因环境变量 MAX_REPOS={limit}，仅处理前 {limit} 个仓库以避免超时")
                    starred = starred[:limit]
            except Exception:
                pass

        classified = classify_by_language(starred)

        json_path = get_summary_json_path(LANGUAGE)
        summary_store = load_summary_store(json_path)
        old_summaries = build_summary_index(summary_store)

        from scripts.core.summary_reader import load_old_summaries
        if not old_summaries:
            old_summaries = load_old_summaries(json_path, README_SUM_PATH, LANGUAGE)
        if not summary_store and old_summaries:
            for full_name, summary in old_summaries.items():
                summary_store[full_name] = {"full_name": full_name, "summary": summary}

        repos_to_update = select_repos_for_update(classified, summary_store, old_summaries, update_mode)

        classified_to_process: Dict[str, List[Dict]] = {}
        for lang, repos in classified.items():
            try:
                sorted_repos = sorted(repos, key=lambda r: is_valid_summary(old_summaries.get(r.get("full_name", ""), "")))
            except Exception:
                sorted_repos = repos
            if sorted_repos:
                classified_to_process[lang] = sorted_repos

        current_time = time.strftime("%Y-%m-%d", time.localtime())

        lines: List[str] = []
        lines.extend(build_readme_header(LANGUAGE, GITHUB_USERNAME, api_name, len(starred), current_time))
        lines.extend(build_table_of_contents(classified_to_process, LANGUAGE))

        printed_repos: set = set()
        printed_langs: set = set()
        total_repos = sum(len(repos) for repos in classified_to_process.values())
        processed_repos = 0
        repo_summary_map: Dict[str, Dict] = {}

        for lang, repos in sorted(classified_to_process.items(), key=lambda x: -len(x[1])):
            if lang in printed_langs:
                continue

            repos_to_call = repos_to_update.get(lang, []) if update_mode == "missing_only" else repos

            for i in range(0, len(repos_to_call), batch_size):
                this_batch = repos_to_call[i : i + batch_size]
                print(f"处理批次 {i // batch_size + 1}，包含 {len(this_batch)} 个仓库...")

                if api_choice == "gemini":
                    summaries = summarize_batch(this_batch, old_summaries, use_gemini=True)
                elif api_choice == "copilot":
                    summaries = summarize_batch(this_batch, old_summaries, use_copilot=True)
                else:
                    summaries = summarize_batch(this_batch, old_summaries)

                for repo, summary in zip(this_batch, summaries):
                    key = _repo_key(repo)
                    entry = build_repo_entry(repo, summary)
                    if key:
                        repo_summary_map[key] = entry

                summary_store = merge_summary_store(summary_store, repo_summary_map)
                save_json_atomic(summary_store, json_path)

            section_lines, printed_repos, printed_langs, processed_repos = build_repo_section(
                lang,
                repos,
                LANGUAGE,
                summary_store,
                old_summaries,
                rate_limit_delay,
                printed_repos,
                printed_langs,
                processed_repos,
            )
            lines.extend(section_lines)

        lines.extend(
            build_readme_footer(
                processed_repos,
                len(classified_to_process),
                current_time,
                api_name,
                (copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count),
                LANGUAGE,
            )
        )

        with open(README_SUM_PATH, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        print(f"\n✅ {README_SUM_PATH} 已生成，共处理了 {processed_repos} 个仓库。")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate language-classified README summaries.")
    parser.add_argument("--language", type=str, default=None, help="Override language: en or zh.")
    parser.add_argument("--out", type=str, default=None, help="Override output markdown path.")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=None,
        help="Override update mode: missing_only or all (also supports env MYGITSTAR_UPDATE_MODE).",
    )
    parser.add_argument("--copilot-count", action="store_true", help="Print Copilot API call count (for this run) and exit.")
    args = parser.parse_args()

    if args.copilot_count or (len(sys.argv) > 1 and sys.argv[1] == "--copilot-count"):
        print(copilot_api_call_count)
        raise SystemExit(0)

    if args.language:
        lang = str(args.language).strip().lower()
        if lang in {"cn", "zh-cn", "zh_cn", "zh"}:
            LANGUAGE = "zh"
        elif lang in {"en", "eng", "english"}:
            LANGUAGE = "en"
        else:
            print(f"Unsupported --language: {args.language}")
            raise SystemExit(2)

    if args.out:
        out_path = str(args.out).strip()
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), out_path)
        README_SUM_PATH = out_path

    if args.update_mode is not None:
        update_mode = normalize_update_mode(args.update_mode)

    main()
    print(f"Copilot API 总调用次数: {copilot_api_call_count}")
