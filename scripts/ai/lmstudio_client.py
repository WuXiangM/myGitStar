import json
import time
from typing import Any, Dict, Optional, Union

import requests

LM_STUDIO_API_ENDPOINT = "http://localhost:1234/v1/chat/completions"
MODEL_DEFAULT = "qwen/qwen3-4b-2507"


def _extract_prompt(repo: Union[Dict, Any]) -> str:
    if isinstance(repo, dict):
        if "prompt" in repo:
            return repo["prompt"]
    return repo.get("prompt", "") if isinstance(repo, dict) else ""


def lmstudio_make_request(
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    retries: int = 3,
    retry_delay: float = 5.0,
    timeout: float = 300.0,
) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            if response.status_code == 429:
                if attempt < retries - 1:
                    wait = retry_delay * (2**attempt)
                    time.sleep(wait)
                    continue
                return {"error": {"code": 429, "message": "Too Many Requests"}, "status_code": 429}

            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if attempt < retries - 1:
                wait = retry_delay * (2**attempt)
                time.sleep(wait)
                continue
            return {"error": {"code": "http_error", "message": str(e)}}
        except Exception as e:
            if attempt < retries - 1:
                wait = retry_delay * (2**attempt)
                time.sleep(wait)
                continue
            return {"error": {"code": "exception", "message": str(e)}}
    return {"error": {"code": "unknown", "message": "request failed"}}


def lmstudio_summarize(
    repo: Union[Dict[str, Any], Any],
    model: str = MODEL_DEFAULT,
    api_endpoint: str = LM_STUDIO_API_ENDPOINT,
    timeout: float = 300.0,
) -> Optional[str]:
    if not repo:
        return None

    prompt = _extract_prompt(repo)
    if not prompt.strip():
        return None

    is_combined = isinstance(repo, dict) and "repos" in repo
    max_tokens = 2500 if is_combined else 600

    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        content = None
        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")

        if content:
            return content.strip()
        return None

    except requests.exceptions.Timeout:
        return f"LM Studio 请求超时 (model={model})"
    except requests.exceptions.ConnectionError:
        return f"LM Studio 连接失败，请确保本地 LM Studio 服务正在运行 (endpoint={api_endpoint})"
    except Exception as e:
        return f"LM Studio 错误: {str(e)}"


def lmstudio_call_llm(
    prompt: str,
    model: str = MODEL_DEFAULT,
    max_tokens: int = 1800,
    temperature: float = 0.2,
) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = lmstudio_make_request(LM_STUDIO_API_ENDPOINT, headers, payload)
    if not resp or resp.get("error"):
        raise RuntimeError(f"LM Studio error: {resp.get('error', {}).get('message', 'unknown')}")

    content = None
    if resp.get("choices") and len(resp["choices"]) > 0:
        message = resp["choices"][0].get("message", {})
        content = message.get("content", "")

    if not content:
        raise RuntimeError("LM Studio returned empty content")
    return content


def lmstudio_call_llm_json(prompt: str, model: str = MODEL_DEFAULT) -> Any:
    text = lmstudio_call_llm(prompt, model)
    return extract_json_from_text(text)


def extract_json_from_text(text: str) -> Any:
    if text is None:
        raise ValueError("empty response")
    s = str(text).strip()

    s = s.replace("```json", "").replace("```", "")
    s = s.strip()

    obj_start = s.find("{")
    arr_start = s.find("[")
    start = -1
    if obj_start != -1 and arr_start != -1:
        start = min(obj_start, arr_start)
    else:
        start = obj_start if obj_start != -1 else arr_start

    if start == -1:
        raise ValueError(f"no json found in response: {s[:500]}")

    tail = s[start:]
    for end in range(len(tail), 1, -1):
        candidate = tail[:end]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError(f"failed to parse json. Response preview: {s[:1000]}")


def create_lmstudio_summarize_func(model: str = MODEL_DEFAULT):
    def summarize(repo: Dict[str, Any]) -> Optional[str]:
        return lmstudio_summarize(repo, model=model)
    return summarize
