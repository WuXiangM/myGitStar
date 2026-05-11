import json
import time
from typing import Any, Dict, Optional

import requests

OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_DEFAULT = "qwen3.5:4b"


def ollama_make_request(
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    retries: int = 3,
    retry_delay: float = 5.0,
    timeout: float = 300.0,
) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, timeout=timeout)
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


def ollama_call_llm(prompt: str, model: str = MODEL_DEFAULT) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1800,
        },
    }

    resp = ollama_make_request(OLLAMA_API_ENDPOINT, {}, payload)
    if not resp or resp.get("error"):
        raise RuntimeError(f"Ollama error: {resp.get('error', {}).get('message', 'unknown')}")

    content = resp.get("response", "")
    if not content:
        raise RuntimeError("Ollama returned empty content")
    return content


def ollama_call_llm_json(prompt: str, model: str = MODEL_DEFAULT) -> Any:
    text = ollama_call_llm(prompt, model)
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
        raise ValueError("no json found")

    tail = s[start:]
    for end in range(len(tail), 1, -1):
        candidate = tail[:end]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("failed to parse json")
