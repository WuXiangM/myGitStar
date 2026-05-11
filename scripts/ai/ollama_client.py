import json
import time
from typing import Any, Dict, Optional

import requests

OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate"

MODEL_DEFAULT = "qwen3.5:4b"


def ollama_summarize(
    repo: Dict[str, Any],
    model: str = MODEL_DEFAULT,
    api_endpoint: str = OLLAMA_API_ENDPOINT,
    timeout: float = 300.0,
) -> Optional[str]:
    if not repo:
        return None

    prompt = repo.get("prompt", "")
    if not prompt.strip():
        return None

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 600,
            },
        }

        response = requests.post(api_endpoint, json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        content = result.get("response", "")

        if content:
            return content.strip()
        return None

    except requests.exceptions.Timeout:
        return f"Ollama 请求超时 (model={model})"
    except requests.exceptions.ConnectionError:
        return f"Ollama 连接失败，请确保本地 Ollama 服务正在运行 (endpoint={api_endpoint})"
    except Exception as e:
        return f"Ollama 错误: {str(e)}"


def create_ollama_summarize_func(model: str = "qwen3.5:4b"):
    def summarize(repo: Dict[str, Any]) -> Optional[str]:
        return ollama_summarize(repo, model=model)
    return summarize
