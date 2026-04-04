import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

# NOTE:
# - This script reuses the same config + env secret strategy as scripts/summarize_stars.py
# - It classifies repos by CONTENT (based mainly on repository description), not by language.

CONFIG_PATH_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
CONFIG_PATH_YAML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    try:
        with open(CONFIG_PATH_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return {}


config = load_config()


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _env_truthy(name: str) -> bool:
    try:
        return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return False


def _get_int_config(key: str, default: int) -> int:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def _get_float_config(key: str, default: float) -> float:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _get_secret(config_env_key: str, default_env_names: List[str], config_plain_key: str = "") -> str:
    for name in default_env_names:
        val = os.environ.get(name)
        if val:
            return val

    if isinstance(config, dict):
        for name in default_env_names:
            cfg_val = config.get(name)
            if cfg_val:
                return cfg_val

    env_name_in_config = config.get(config_env_key, "") if isinstance(config, dict) else ""
    if env_name_in_config:
        val = os.environ.get(env_name_in_config)
        if val:
            return val

    if config_plain_key and isinstance(config, dict):
        val = config.get(config_plain_key, "")
        if val:
            return val

    return ""


GITHUB_TOKEN = _get_secret("github_token_env", ["STARRED_GITHUB_TOKEN", "GITHUB_TOKEN"], "github_token")
OPENROUTER_API_KEY = _get_secret("openrouter_api_key_env", ["OPENROUTER_API_KEY"], "openrouter_api_key")
GEMINI_API_KEY = _get_secret("gemini_api_key_env", ["GEMINI_API_KEY"], "gemini_api_key")

MODEL_CHOICE = (config.get("model_choice", "copilot") if isinstance(config, dict) else "copilot").strip().lower()

DEFAULT_COPILOT_MODEL = config.get("default_copilot_model", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_MODEL = config.get("default_openrouter_model")
DEFAULT_GEMINI_MODEL = config.get("default_gemini_model", "gemini-2.0-flash")

REQUEST_TIMEOUT = _get_float_config("request_timeout", 30.0)
REQUEST_RETRY_DELAY = _get_float_config("request_retry_delay", 2.0)
RETRY_ATTEMPTS = _get_int_config("retry_attempts", 1)
RATE_LIMIT_DELAY = _get_float_config("rate_limit_delay", 3.0)
GLOBAL_QPS = _get_float_config("global_qps", 0.5)

DEBUG_API = _env_truthy("DEBUG_API")


class SimpleThrottle:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.time()
        if now < self.next_allowed:
            to_sleep = self.next_allowed - now
            time.sleep(to_sleep + random.uniform(0, 0.1))
            now = time.time()
        self.next_allowed = now + self.interval


THROTTLE = SimpleThrottle(GLOBAL_QPS)


API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def _safe_print_debug(msg: str) -> None:
    if DEBUG_API:
        print(msg)


def make_api_request(url: str, headers: Dict[str, str], data: Dict[str, Any], retries: int, retry_delay: float) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            THROTTLE.wait()
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=REQUEST_TIMEOUT)
            _safe_print_debug(f"[API] POST {url} -> {resp.status_code}")
            if DEBUG_API:
                _safe_print_debug(f"[API] Response text: {resp.text[:2000]}")

            if resp.status_code == 429:
                retry_after = None
                try:
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = int(ra)
                except Exception:
                    retry_after = None

                if attempt < retries - 1:
                    wait = retry_after if retry_after else (retry_delay * (2 ** attempt) + random.uniform(0, 1))
                    print(f"[429] Rate limited, sleep {wait:.1f}s then retry ({attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue
                return {"error": {"code": 429, "message": "Too Many Requests"}, "status_code": 429}

            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"text": resp.text}
        except requests.HTTPError as e:
            if attempt < retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"[HTTPError] {e}; sleep {wait:.1f}s then retry")
                time.sleep(wait)
                continue
            print(f"[HTTPError] final: {e}")
            return None
        except Exception as e:
            if attempt < retries - 1:
                wait = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[Error] {e}; sleep {wait:.1f}s then retry")
                time.sleep(wait)
                continue
            print(f"[Error] final: {e}")
            return None


def _extract_json_from_text(text: str) -> Any:
    """Best-effort JSON extraction (models sometimes wrap JSON in markdown fences)."""
    if text is None:
        raise ValueError("empty response")
    s = str(text).strip()

    # Strip markdown fences
    s = re.sub(r"^```(json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)

    # Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Find first JSON object/array
    obj_start = s.find("{")
    arr_start = s.find("[")
    start = -1
    if obj_start != -1 and arr_start != -1:
        start = min(obj_start, arr_start)
    else:
        start = obj_start if obj_start != -1 else arr_start

    if start == -1:
        raise ValueError("no json found")

    tail = s[start:]
    # Greedy attempt: try to parse progressively shorter tails
    for end in range(len(tail), 1, -1):
        candidate = tail[:end]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("failed to parse json")


def _get_llm_text(response: Optional[Dict[str, Any]]) -> str:
    if not response:
        return ""
    if "choices" in response:
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and "content" in msg:
                return str(msg.get("content") or "").strip()
            if "content" in choices[0]:
                return str(choices[0].get("content") or "").strip()
    if "text" in response:
        return str(response.get("text") or "").strip()
    return ""


def call_llm(prompt: str) -> str:
    """Call the selected LLM backend with a user prompt. Returns raw text."""
    if MODEL_CHOICE == "copilot":
        if not GITHUB_TOKEN:
            raise RuntimeError("Missing STARRED_GITHUB_TOKEN (required for copilot backend).")
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", DEFAULT_COPILOT_MODEL) or "openai/gpt-4o-mini"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1800,
            "temperature": 0.2,
        }
        resp = make_api_request(API_ENDPOINTS["copilot"], headers, data, retries=RETRY_ATTEMPTS, retry_delay=REQUEST_RETRY_DELAY)
        return _get_llm_text(resp)

    if MODEL_CHOICE == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY (required for openrouter backend).")
        model_name = DEFAULT_OPENROUTER_MODEL or "openai/gpt-4o-mini"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        resp = make_api_request(API_ENDPOINTS["openrouter"], headers, data, retries=RETRY_ATTEMPTS, retry_delay=REQUEST_RETRY_DELAY)
        return _get_llm_text(resp)

    if MODEL_CHOICE == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY (required for gemini backend).")

        model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or "gemini-2.0-flash"
        model_path = str(model_name).strip()
        if model_path.startswith("models/"):
            model_path = model_path[len("models/"):]

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        request_url = f"{api_url}?key={GEMINI_API_KEY}"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GitHub Star Content Classifier/1.0",
            "X-Goog-Api-Key": GEMINI_API_KEY,
        }
        temperature = config.get("gemini_temperature", 0.2)
        max_output_tokens = int(config.get("gemini_max_output_tokens", 2000))
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.8,
                "topK": 40,
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

        resp = make_api_request(request_url, headers, payload, retries=_get_int_config("gemini_retry_attempts", RETRY_ATTEMPTS), retry_delay=_get_float_config("gemini_retry_delay", REQUEST_RETRY_DELAY))
        if not resp or not isinstance(resp, dict):
            return ""
        if "error" in resp:
            return json.dumps(resp, ensure_ascii=False)

        # Gemini: candidates[0].content.parts[].text
        text = ""
        try:
            candidates = resp.get("candidates") or []
            if candidates and isinstance(candidates[0], dict):
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        text += str(part.get("text") or "")
        except Exception:
            pass
        return text.strip()

    raise RuntimeError(f"Unsupported model_choice: {MODEL_CHOICE}")


def call_llm_json(prompt: str, attempts: int = 2) -> Any:
    """Call LLM and parse JSON, retrying once if the model returns non-JSON or truncated JSON."""
    last_error: Optional[Exception] = None
    for i in range(1, max(1, attempts) + 1):
        p = prompt
        if i > 1:
            p = (
                prompt
                + "\n\nIMPORTANT: Return STRICT JSON only. Do not include markdown fences, explanations, or extra text."
            )
        try:
            text = call_llm(p)
            return _extract_json_from_text(text)
        except Exception as e:
            last_error = e
            if i < attempts:
                time.sleep(1.0)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Failed to get JSON from LLM")


def get_github_username() -> str:
    gh = config.get("github_username") if isinstance(config, dict) else None
    if gh == "0" or gh == 0:
        actor = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
        if not actor:
            raise RuntimeError("github_username is 0 but GITHUB_ACTOR/GITHUB_USERNAME is not set.")
        return actor
    if not gh:
        raise RuntimeError("Missing config.github_username")
    return str(gh)


def get_starred_repos(max_repos: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GITHUB_TOKEN:
        raise RuntimeError("Missing STARRED_GITHUB_TOKEN")

    username = get_github_username()
    print(f"Fetching starred repos for: {username}")

    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"https://api.github.com/users/{username}/starred?per_page={per_page}&page={page}"
        THROTTLE.wait()
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
        if not data:
            break
        repos.extend(data)
        print(f"  got {len(repos)} repos... (page {page})")
        page += 1
        time.sleep(1)

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            repos = repos[:max_repos]
            break

    return repos


def _clean_inline_md(text: str) -> str:
    s = (text or "").strip()
    # remove trailing markdown double spaces used for line breaks
    s = re.sub(r"\s{2,}$", "", s)
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_repos_from_readme(readme_path: str, max_repos: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse repo list from an existing generated README.

    Supports headings like:
    - ### 📌 [owner/repo](https://github.com/owner/repo)

    Extracts a short content description from one of:
    - '2. **Brief Introduction:** ...'
    - '2. **简要介绍：** ...'
    - '**Repository Description:** ...'
    - '**仓库描述：** ...'
    """
    if not os.path.isabs(readme_path):
        readme_path = os.path.join(REPO_ROOT, readme_path)

    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README not found: {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by repo heading
    # Example: ### 📌 [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0)
    pattern = re.compile(
        r"^###\s+📌\s+\[(?P<full_name>[^\]]+)\]\((?P<url>https?://[^\)\s]+)\)\s*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))
    repos: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        heading_line = m.group(0)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block = content[start:end]
        block_md = (heading_line + "\n" + block).strip() + "\n"

        full_name = _clean_inline_md(m.group("full_name"))
        url = _clean_inline_md(m.group("url"))

        # Try to extract a brief description
        desc = ""
        extract_patterns = [
            r"^2\.\s*\*\*Brief Introduction:\*\*\s*(?P<d>.+?)\s*$",
            r"^2\.\s*\*\*简要介绍：\*\*\s*(?P<d>.+?)\s*$",
            r"^\*\*Repository Description:\*\*\s*(?P<d>.+?)\s*$",
            r"^\*\*仓库描述：\*\*\s*(?P<d>.+?)\s*$",
        ]
        for ep in extract_patterns:
            mm = re.search(ep, block, flags=re.MULTILINE)
            if mm:
                desc = _clean_inline_md(mm.group("d"))
                break

        # Fall back: use the first non-empty paragraph line (skip stars/forks/updated and separators)
        if not desc:
            for line in block.splitlines():
                line = _clean_inline_md(line)
                if not line:
                    continue
                if line.startswith("**⭐") or line.startswith("---"):
                    continue
                if line.startswith("**🍴") or line.startswith("**📅"):
                    continue
                # skip numbered items other than introduction if they exist
                if re.match(r"^\d+\.\s*\*\*", line):
                    continue
                desc = line
                break

        repos.append(
            {
                "id": idx + 1,
                "full_name": full_name,
                "description": desc,
                "html_url": url,
                "block_md": block_md,
            }
        )

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            break

    if not repos:
        raise ValueError(
            "No repositories found in README. Expected headings like: ### 📌 [owner/repo](https://github.com/owner/repo)"
        )

    return repos


def _slugify_heading(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "-")
    return s


@dataclass
class Taxonomy:
    categories: List[Dict[str, Any]]


def build_taxonomy_prompt(items: List[Dict[str, Any]], min_categories: int, max_categories: int) -> str:
    examples = []
    for r in items:
        examples.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "description": r.get("description") or "",
            }
        )

    return (
        "You are a taxonomy designer.\n"
        "Task: create a content-based taxonomy to classify GitHub repositories, mainly using the repository description.\n"
        f"Constraints: create BETWEEN {min_categories} and {max_categories} categories total (inclusive), and include an 'Other' category.\n"
        "Rules:\n"
        "- Categories must be based on CONTENT/domains (e.g., LLM tooling, CV, data engineering), NOT programming languages.\n"
        "- Category names should be short and clear.\n"
        "- Include an 'Other' category for anything that doesn't fit.\n"
        "Output strictly as JSON, with this shape:\n"
        "{\n"
        "  \"categories\": [\n"
        "    {\"id\": \"C1\", \"name\": \"...\", \"description\": \"...\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Here are example repositories (id, full_name, description):\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
    )


def build_classification_prompt(taxonomy: Taxonomy, repos: List[Dict[str, Any]]) -> str:
    items = []
    for r in repos:
        items.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "description": r.get("description") or "",
            }
        )

    return (
        "You are a classifier.\n"
        "Classify each GitHub repository into exactly ONE category from the provided taxonomy.\n"
        "Use mainly the repository description.\n"
        "Return STRICT JSON only.\n\n"
        "Taxonomy JSON:\n"
        + json.dumps({"categories": taxonomy.categories}, ensure_ascii=False, indent=2)
        + "\n\n"
        "Repositories to classify (id, full_name, description):\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
        + "\n\n"
        "Output JSON shape:\n"
        "{\n"
        "  \"assignments\": [\n"
        "    {\"id\": 123, \"category_id\": \"C1\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
    )


def _normalize_taxonomy(raw: Any, min_categories: int, max_categories: int) -> Taxonomy:
    if not isinstance(raw, dict) or "categories" not in raw or not isinstance(raw["categories"], list):
        raise ValueError("Invalid taxonomy JSON")

    cats = []
    for c in raw["categories"]:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        name = str(c.get("name") or "").strip()
        desc = str(c.get("description") or "").strip()
        if not cid or not name:
            continue
        cats.append({"id": cid, "name": name, "description": desc})

    # Enforce max categories
    cats = cats[:max_categories]

    # Ensure there is an Other category
    if not any(str(c.get("name", "")).strip().lower() == "other" for c in cats):
        # If at limit, replace the last category with Other
        if len(cats) >= max_categories:
            cats[-1] = {"id": cats[-1]["id"], "name": "Other", "description": "Everything that does not fit other categories."}
        else:
            cats.append({"id": f"C{len(cats) + 1}", "name": "Other", "description": "Everything that does not fit other categories."})

    # Ensure minimum category count (5~8 requirement). If model returns too few, pad with generic buckets.
    # We insert before Other to keep Other as the last bucket.
    other_idx = next((i for i, c in enumerate(cats) if str(c.get("name", "")).strip().lower() == "other"), len(cats) - 1)
    while len(cats) < min_categories:
        cats.insert(
            other_idx,
            {
                "id": f"C{len(cats) + 1}",
                "name": f"Misc {len(cats) + 1}",
                "description": "Additional bucket to satisfy the minimum category count.",
            },
        )
        other_idx += 1

    # Re-number ids if needed for stability
    normalized = []
    for idx, c in enumerate(cats, start=1):
        normalized.append({"id": f"C{idx}", "name": c["name"], "description": c.get("description", "")})

    return Taxonomy(categories=normalized)


def _model_display_name() -> str:
    if MODEL_CHOICE == "copilot":
        return "GitHub Copilot"
    if MODEL_CHOICE == "openrouter":
        return "OpenRouter"
    if MODEL_CHOICE == "gemini":
        return "Gemini"
    return MODEL_CHOICE


def render_classified_readme(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    *,
    reference_repo: str = "WuXiangM/myGitStar",
) -> str:
    categories = {c["id"]: c for c in taxonomy.categories}
    buckets: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in categories.keys()}
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])

    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid, other_id)
        if cid not in buckets:
            cid = other_id
        buckets[cid].append(r)

    generated_on = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    model_name = _model_display_name()
    total_repos = len(repos)

    lines: List[str] = []
    lines.append(
        "<div align=\"center\">\n\n"
        "<h1>My GitHub Star Project AI Summary (Classified)</h1>\n\n"
        f"<p><b>Generated on:</b> {generated_on}</p>\n"
        f"<p><b>AI Model:</b> {model_name}</p>\n"
        f"<p><b>Total repositories:</b> {total_repos}</p>\n\n"
        "<hr/>\n\n"
        f"<p><b>Reference Repository:</b> <a href=\"https://github.com/{reference_repo}\">{reference_repo}</a></p>\n\n"
        "<p>"
        "<a href=\"README.md\">English README</a> | "
        "<a href=\"README2.md\">中文 README</a> | "
        "<a href=\"README_classified.md\">📂 Content Classified</a>"
        "</p>\n"
        "<p>"
        "<a href=\"GUIDE_en.md\">English GUIDE</a> | "
        "<a href=\"GUIDE_zh.md\">中文教程</a>"
        "</p>\n\n"
        "</div>\n\n"
    )
    lines.append("## 📖 Table of Contents\n\n")
    for c in taxonomy.categories:
        anchor = _slugify_heading(c["name"])
        lines.append(f"- [{c['name']}](#{anchor}) ({len(buckets.get(c['id'], []))})\n")
    lines.append("\n---\n\n")

    for c in taxonomy.categories:
        bucket = buckets.get(c["id"], [])
        lines.append(f"## {c['name']} (Total {len(bucket)})\n\n")
        if c.get("description"):
            lines.append(f"> {c['description']}\n\n")

        # Keep the original per-repo blocks (best mimic of README.md)
        for r in bucket:
            block_md = r.get("block_md")
            if isinstance(block_md, str) and block_md.strip():
                lines.append(block_md.strip() + "\n")
            else:
                full_name = r.get("full_name") or ""
                url = r.get("html_url") or f"https://github.com/{full_name}"
                desc = (r.get("description") or "").strip()
                lines.append(f"### 📌 [{full_name}]({url})\n\n")
                if desc:
                    lines.append(desc + "\n\n")
                lines.append("---\n\n")

    return "".join(lines).rstrip() + "\n"


def _parse_assignments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict) or "assignments" not in raw or not isinstance(raw["assignments"], list):
        raise ValueError("Invalid assignments JSON")
    out: List[Dict[str, Any]] = []
    for a in raw["assignments"]:
        if not isinstance(a, dict):
            continue
        rid = a.get("id")
        cid = a.get("category_id")
        if rid is None or cid is None:
            continue
        out.append({"id": rid, "category_id": str(cid)})
    return out


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def render_markdown(taxonomy: Taxonomy, repos: List[Dict[str, Any]], assignment_map: Dict[Any, str]) -> str:
    # Build reverse index
    categories = {c["id"]: c for c in taxonomy.categories}
    buckets: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in categories.keys()}
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])

    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid, other_id)
        if cid not in buckets:
            cid = other_id
        buckets[cid].append(r)

    lines: List[str] = []
    lines.append("# Repo Content Categories (AI-generated)\n")
    lines.append(f"**Generated on:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}\n")
    lines.append(f"**Model choice:** {MODEL_CHOICE}\n")
    lines.append(f"**Total repositories:** {len(repos)}\n")
    lines.append("---\n")

    # Table of contents
    lines.append("## Table of Contents\n")
    for c in taxonomy.categories:
        anchor = re.sub(r"[^a-z0-9\- ]", "", c["name"].lower()).strip().replace(" ", "-")
        lines.append(f"- [{c['name']}](#{anchor}) ({len(buckets.get(c['id'], []))})")
    lines.append("\n---\n")

    for c in taxonomy.categories:
        anchor = re.sub(r"[^a-z0-9\- ]", "", c["name"].lower()).strip().replace(" ", "-")
        lines.append(f"## {c['name']}\n")
        if c.get("description"):
            lines.append(f"> {c['description']}\n")

        bucket = sorted(buckets.get(c["id"], []), key=lambda x: (x.get("full_name") or ""))
        for r in bucket:
            full_name = r.get("full_name") or ""
            desc = (r.get("description") or "").strip()
            url = r.get("html_url") or f"https://github.com/{full_name}"
            lines.append(f"- [{full_name}]({url})" + (f" — {desc}" if desc else ""))
        lines.append("\n")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify GitHub repositories into 5~8 content categories using the selected LLM.")
    parser.add_argument(
        "--from-readme",
        type=str,
        default=None,
        help="Parse repositories from an existing generated README (e.g. README.md) instead of calling GitHub API.",
    )
    parser.add_argument("--max-repos", type=int, default=None, help="Limit number of repos (overrides config/env MAX_REPOS).")
    parser.add_argument("--min-categories", type=int, default=5, help="Min number of categories (default: 5).")
    parser.add_argument("--max-categories", type=int, default=8, help="Max number of categories (default: 8).")
    parser.add_argument("--sample-size", type=int, default=60, help="How many repos to sample to design taxonomy (default: 60).")
    parser.add_argument("--batch-size", type=int, default=30, help="Repos per LLM classification call (default: 30).")
    parser.add_argument("--out-json", type=str, default="repo_categories.json", help="Output JSON filename.")
    parser.add_argument("--out-md", type=str, default="README_classified.md", help="Output Markdown filename (default: README_classified.md).")
    args = parser.parse_args()

    if args.min_categories < 1:
        print("--min-categories must be >= 1")
        return 2
    if args.max_categories < args.min_categories:
        print("--max-categories must be >= --min-categories")
        return 2

    # NOTE:
    # - For --from-readme mode, we want to classify ALL repos already present in README by default.
    #   So we only apply a limit if the user explicitly passes --max-repos.
    # - For GitHub API mode, we keep the old behavior: env MAX_REPOS / config.max_repos can limit the run.
    max_repos = args.max_repos
    if max_repos is None and not args.from_readme:
        env_mr = os.environ.get("MAX_REPOS")
        if env_mr:
            try:
                mr = int(env_mr)
                if mr > 0:
                    max_repos = mr
            except Exception:
                max_repos = None
        if max_repos is None:
            cfg_mr = config.get("max_repos") if isinstance(config, dict) else None
            try:
                if cfg_mr is not None and int(cfg_mr) > 0:
                    max_repos = int(cfg_mr)
            except Exception:
                pass

    if args.from_readme:
        try:
            repos = parse_repos_from_readme(args.from_readme, max_repos=max_repos)
        except Exception as e:
            print(f"Failed to parse repos from README: {e}")
            return 2
    else:
        try:
            repos = get_starred_repos(max_repos=max_repos)
        except Exception as e:
            print(f"Failed to fetch starred repos: {e}")
            print("\nChecklist:")
            print("- Ensure you have set STARRED_GITHUB_TOKEN in environment variables, or in GitHub Actions Secrets.")
            print("- If config.yaml sets github_username: 0, ensure GITHUB_ACTOR is available (Actions) or set github_username to your username locally.")
            return 2

    if not repos:
        print("No repos fetched.")
        return 2

    # Build taxonomy from a sample
    sample_size = max(5, min(args.sample_size, len(repos)))
    sample = repos[:sample_size]
    taxonomy_prompt = build_taxonomy_prompt(sample, min_categories=args.min_categories, max_categories=args.max_categories)
    print(
        f"Designing taxonomy from {len(sample)} sample repos (min_categories={args.min_categories}, max_categories={args.max_categories})..."
    )
    try:
        raw_tax = call_llm_json(taxonomy_prompt, attempts=2)
    except Exception as e:
        print(f"Failed to call LLM for taxonomy: {e}")
        print("\nChecklist:")
        print("- model_choice=copilot requires STARRED_GITHUB_TOKEN")
        print("- model_choice=openrouter requires OPENROUTER_API_KEY")
        print("- model_choice=gemini requires GEMINI_API_KEY")
        return 3

    try:
        taxonomy = _normalize_taxonomy(raw_tax, min_categories=args.min_categories, max_categories=args.max_categories)
    except Exception as e:
        print(f"Failed to normalize taxonomy JSON: {e}")
        return 3

    print("Taxonomy:")
    for c in taxonomy.categories:
        print(f"  {c['id']}: {c['name']}")

    # Classify repos in batches
    assignment_map: Dict[Any, str] = {}
    all_ids = {r.get("id") for r in repos}

    print(f"Classifying {len(repos)} repos in batches of {args.batch_size}...")
    for i, batch in enumerate(chunk_list(repos, args.batch_size), start=1):
        prompt = build_classification_prompt(taxonomy, batch)
        try:
            raw = call_llm_json(prompt, attempts=2)
            assignments = _parse_assignments(raw)
        except Exception as e:
            print(f"[Batch {i}] failed to parse assignments: {e}")
            assignments = []

        # validate and fill
        valid_category_ids = {c["id"] for c in taxonomy.categories}
        other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])
        for a in assignments:
            rid = a["id"]
            cid = a["category_id"]
            if cid not in valid_category_ids:
                cid = other_id
            assignment_map[rid] = cid

        # Fill missing in this batch as Other
        batch_ids = {r.get("id") for r in batch}
        for rid in batch_ids:
            if rid not in assignment_map:
                assignment_map[rid] = other_id

        print(f"  batch {i}/{(len(repos) + args.batch_size - 1) // args.batch_size} done")
        time.sleep(RATE_LIMIT_DELAY)

    # Ensure every repo is assigned
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])
    for rid in all_ids:
        if rid not in assignment_map:
            assignment_map[rid] = other_id

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_choice": MODEL_CHOICE,
        "min_categories": args.min_categories,
        "max_categories": args.max_categories,
        "taxonomy": {"categories": taxonomy.categories},
        "assignments": [{"id": r.get("id"), "full_name": r.get("full_name"), "category_id": assignment_map.get(r.get("id"), other_id)} for r in repos],
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = render_classified_readme(taxonomy, repos, assignment_map)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
