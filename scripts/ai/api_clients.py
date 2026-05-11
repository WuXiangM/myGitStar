import json
import os
import random
import time
from typing import Any, Dict, Optional

import requests

from scripts.core.throttle import SimpleThrottle


API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}


def make_api_request(
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    retries: int = 3,
    retry_delay: float = 5.0,
    timeout: float = 30.0,
    throttle: Optional[SimpleThrottle] = None,
) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            if throttle:
                try:
                    throttle.wait()
                except Exception:
                    pass
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            if resp.status_code == 429:
                retry_after = None
                try:
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = int(ra)
                except Exception:
                    retry_after = None

                if retry_after and attempt < retries - 1:
                    wait = retry_after
                    time.sleep(wait)
                    continue

                if attempt < retries - 1:
                    wait = int(retry_delay) * (2**attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                else:
                    return {
                        "error": {"code": 429, "message": "Too Many Requests"},
                        "status_code": 429,
                    }

            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"text": resp.text}
        except requests.HTTPError as e:
            if attempt < retries - 1:
                wait = int(retry_delay) * (2**attempt)
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            if attempt < retries - 1:
                wait = int(retry_delay) * (2**attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            return None
    return None


def copilot_summarize(
    repo: Dict[str, Any],
    github_token: str,
    default_copilot_model: str,
    api_request_func: callable,
) -> Optional[str]:
    if not github_token:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", default_copilot_model) or "openai/gpt-4o-mini"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": repo.get("prompt", "")}],
            "max_tokens": 600,
            "temperature": 0.4,
        }
        response = api_request_func(API_ENDPOINTS["copilot"], headers, data)
        if response and isinstance(response, dict) and response.get("error"):
            err = response["error"]
            if err.get("code") == "RateLimitReached":
                return f"Copilot API限额已用尽：{err.get('message', 'RateLimitReached')}"
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
        return content if content else None
    except Exception as e:
        return None


def openrouter_summarize(
    repo: Dict[str, Any],
    openrouter_api_key: str,
    default_openrouter_model: str,
    api_request_func: callable,
) -> Optional[str]:
    if not openrouter_api_key:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": default_openrouter_model,
            "messages": [{"role": "user", "content": repo.get("prompt", "")}],
        }
        response = api_request_func(API_ENDPOINTS["openrouter"], headers, data)
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
        return content if content else None
    except Exception as e:
        return None


def gemini_summarize(
    repo: Dict[str, Any],
    gemini_api_key: str,
    default_gemini_model: str,
    config: Dict[str, Any],
    api_request_func: callable,
) -> Optional[str]:
    if not gemini_api_key:
        return None

    prompt = repo.get("prompt", "")
    if not prompt.strip():
        return None

    model_name = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-pro"
    model_path = str(model_name).strip()
    if model_path.startswith("models/"):
        model_path = model_path[len("models/"):]
    model_path = model_path.strip()

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
    request_url = f"{api_url}?key={gemini_api_key}"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "GitHub Star Summary Bot/1.0",
        "X-Goog-Api-Key": gemini_api_key,
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

        response = api_request_func(
            url=request_url,
            headers=headers,
            data=payload,
            retries=int(config.get("gemini_retry_attempts", 3)),
            retry_delay=float(config.get("gemini_retry_delay", 5.0)),
        )

        if not response or not isinstance(response, dict):
            if attempt < gen_retries:
                wait = gen_backoff * attempt
                time.sleep(wait)
                continue
            else:
                return None

        if "error" in response:
            error = response["error"]
            error_code = error.get("code")
            error_msg = error.get("message", "未知错误")
            if error_code in (429, 503):
                if attempt < gen_retries:
                    wait = gen_backoff * attempt
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
        except Exception:
            content = ""

        if content and not truncated:
            final_content = content
            break

        if attempt < gen_retries:
            wait = gen_backoff * attempt
            time.sleep(wait)
            continue
        else:
            if content:
                final_content = content
            else:
                return None

    return final_content


def create_summarize_func(
    model_choice: str,
    github_token: str,
    openrouter_api_key: str,
    gemini_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    language: str,
    config: dict,
    throttle: Any,
    request_timeout: float,
    request_retry_delay: float,
    retry_attempts: int,
    api_call_counter: callable,
):
    def make_request(url, headers, data, retries, retry_delay, timeout):
        return make_api_request(url, headers, data, retries, retry_delay, timeout, throttle)

    if model_choice == "copilot":
        def summarize(repo: Dict) -> Optional[str]:
            return copilot_summarize(
                repo,
                github_token,
                default_copilot_model,
                make_request,
            )
    elif model_choice == "openrouter":
        def summarize(repo: Dict) -> Optional[str]:
            return openrouter_summarize(
                repo,
                openrouter_api_key,
                default_openrouter_model,
                make_request,
            )
    elif model_choice == "gemini":
        def summarize(repo: Dict) -> Optional[str]:
            return gemini_summarize(
                repo,
                gemini_api_key,
                default_gemini_model,
                config,
                make_request,
            )
    else:
        raise ValueError(f"不支持的模型选择: {model_choice}")

    return summarize
