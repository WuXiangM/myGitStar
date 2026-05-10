import json
import os
from typing import Any, Dict

import yaml

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

    return {"language": "zh"}


def get_int_config(config: Dict[str, Any], key: str, default: int) -> int:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def get_float_config(config: Dict[str, Any], key: str, default: float) -> float:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def env_truthy(name: str) -> bool:
    try:
        return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return False


def normalize_update_mode(mode: Any) -> str:
    try:
        s = str(mode or "").strip().lower()
    except Exception:
        s = ""
    s = s.replace("-", "_")
    if s in {"missing", "missingonly", "missing_only", "incremental"}:
        return "missing_only"
    if s in {"all", "full"}:
        return "all"
    return "all"


def resolve_update_mode(config: Dict[str, Any]) -> str:
    return normalize_update_mode(
        os.environ.get("MYGITSTAR_UPDATE_MODE")
        or (config.get("update_mode") if isinstance(config, dict) else None)
        or "all"
    )


def get_model_choice(config: Dict[str, Any]) -> str:
    try:
        val = os.environ.get("MYGITSTAR_MODEL_CHOICE")
        if val is not None and str(val).strip():
            return str(val).strip().lower()
        return (config.get("model_choice", "copilot") if isinstance(config, dict) else "copilot").strip().lower()
    except Exception:
        return "copilot"
