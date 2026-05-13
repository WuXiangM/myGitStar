import os
from typing import Dict, List, Tuple


def get_secret(config: Dict, config_env_key: str, default_env_names: List[str], config_plain_key: str = "") -> str:
    # 1) Environment variables
    for name in default_env_names:
        val = os.environ.get(name)
        if val:
            return val

    # 2) Config uppercase keys
    if isinstance(config, dict):
        for name in default_env_names:
            cfg_val = config.get(name)
            if cfg_val:
                return cfg_val

    # 3) Config-specified env name
    env_name_in_config = config.get(config_env_key, "") if isinstance(config, dict) else ""
    if env_name_in_config:
        val = os.environ.get(env_name_in_config)
        if val:
            return val

    # 4) Plaintext config key
    if config_plain_key and isinstance(config, dict):
        val = config.get(config_plain_key, "")
        if val:
            return val

    return ""


def load_api_keys(config: Dict) -> Tuple[str, str, str, str]:
    github_token = get_secret(config, "github_token_env", ["STARRED_GITHUB_TOKEN", "GITHUB_TOKEN"], "github_token")
    openrouter_api_key = get_secret(config, "openrouter_api_key_env", ["OPENROUTER_API_KEY"], "openrouter_api_key")
    gemini_api_key = get_secret(config, "gemini_api_key_env", ["GEMINI_API_KEY"], "gemini_api_key")
    modelscope_api_key = get_secret(config, "modelscope_api_key_env", ["MODELSCOPE_API_KEY"], "modelscope_api_key")
    return github_token, openrouter_api_key, gemini_api_key, modelscope_api_key
