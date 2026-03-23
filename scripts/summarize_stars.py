import os
import time
import requests
import json
from sympy import li
import yaml
import concurrent.futures
from typing import Dict, List, Optional
import re
import logging
from logging.handlers import RotatingFileHandler
import sys
import random
import threading
import time as _time

# 请勿直接把密钥写在代码中。下面使用 config + 环境变量优先策略读取密钥。

# 加载配置文件
CONFIG_PATH_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
CONFIG_PATH_YAML = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    # 优先读取 YAML 配置，其次回退到 JSON 配置
    try:
        with open(CONFIG_PATH_YAML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    try:
        with open(CONFIG_PATH_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'language': 'zh'}  # 默认中文
    except Exception:
        return {'language': 'zh'}

config = load_config()

# 安全读取数值型配置的辅助方法，避免 config.get 返回 None/str 导致类型错误
def _get_int_config(key: str, default: int) -> int:
    try:
        if isinstance(config, dict):
            val = config.get(key, default)
        else:
            val = default
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def _get_float_config(key: str, default: float) -> float:
    try:
        if isinstance(config, dict):
            val = config.get(key, default)
        else:
            val = default
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)

# 合并调试与测试模式：当环境变量 DEBUG_API=1 或 config.test_first_repo 为 true 时启用详细 API 调试日志
DEBUG_API = bool(os.environ.get("DEBUG_API")) or bool(config.get('test_first_repo', False))

# 通用获取密钥逻辑：优先环境变量（常见或大写名），其次 config 中的大写键，
# 然后 config 指定的 env 名称字段（如 github_token_env），最后 config 中的明文字段（不推荐）
def _get_secret(config_env_key: str, default_env_names: List[str], config_plain_key: str = "") -> str:
    # 1) 检查环境变量
    for name in default_env_names:
        val = os.environ.get(name)
        if val:
            return val

    # 2) 检查 config 中是否存在大写键（用户可能直接把 key 写在 config）
    if isinstance(config, dict):
        for name in default_env_names:
            cfg_val = config.get(name)
            if cfg_val:
                return cfg_val

    # 3) 兼容旧配置：config 中可能指定了一个 env 名称字段（如 github_token_env）
    env_name_in_config = config.get(config_env_key, "") if isinstance(config, dict) else ""
    if env_name_in_config:
        val = os.environ.get(env_name_in_config)
        if val:
            return val

    # 4) 最后尝试 config 中的明文字段（不推荐）
    if config_plain_key and isinstance(config, dict):
        val = config.get(config_plain_key, "")
        if val:
            return val

    return ""

# 读取各类 API Key（优先环境变量 / Secret）
GITHUB_TOKEN = _get_secret("github_token_env", ["STARRED_GITHUB_TOKEN", "GITHUB_TOKEN"], "github_token")
OPENROUTER_API_KEY = _get_secret("openrouter_api_key_env", ["OPENROUTER_API_KEY"], "openrouter_api_key")
GEMINI_API_KEY = _get_secret("gemini_api_key_env", ["GEMINI_API_KEY"], "gemini_api_key")

# 新增：读取 update_mode 配置
update_mode = config.get("update_mode", "all")  # 默认全部更新

# 从配置文件加载参数
github_username = config.get("github_username")
github_token_env = config.get("github_token_env")
openrouter_api_key_env = config.get("openrouter_api_key_env")
model_choice = config.get("model_choice", "copilot")

default_copilot_model = config.get("default_copilot_model")
default_openrouter_model = config.get("default_openrouter_model")
default_gemini_model = config.get("default_gemini_model", "gemini-pro")
# 使用安全读取，确保为正确类型
max_workers = _get_int_config("max_workers", 5)
batch_size = _get_int_config("batch_size", 1)
request_timeout = _get_float_config("request_timeout", 10.0)
rate_limit_delay = _get_float_config("rate_limit_delay", 1.0)
request_retry_delay = _get_int_config("request_retry_delay", 5)
retry_attempts = _get_int_config("retry_attempts", 3)
readme_sum_path = config.get("readme_sum_path")

# 环境变量加载
# 支持 config.json 配置为 0 时自动获取 workflow 账号
if github_username == "0" or github_username == 0:
    GITHUB_USERNAME = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
    if not GITHUB_USERNAME:
        print("未检测到 workflow 账号环境变量 GITHUB_ACTOR/GITHUB_USERNAME，请检查 workflow 配置！")
else:
    GITHUB_USERNAME = github_username

# 限制每次运行处理的最大仓库数：优先使用环境变量 `MAX_REPOS`，若不存在则使用 config 中的 `max_repos`。
MAX_REPOS = None
max_repos_env = os.environ.get('MAX_REPOS')
if max_repos_env:
    try:
        mr = int(max_repos_env)
        if mr > 0:
            MAX_REPOS = mr
    except Exception:
        MAX_REPOS = None

# 全局速率限制配置（请求每秒数），默认保守 0.5 req/s（即每 2s 一次）
GLOBAL_QPS = _get_float_config('global_qps', 0.5)


class SimpleThrottle:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = _time.time()
            if now < self.next_allowed:
                to_sleep = self.next_allowed - now
                # small jitter
                _time.sleep(to_sleep + random.uniform(0, 0.1))
                now = _time.time()
            # schedule next
            self.next_allowed = now + self.interval


# 全局节流器实例
THROTTLE = SimpleThrottle(GLOBAL_QPS)

# 将 copilot_summarize 和 openrouter_summarize 函数移动到 get_summarize_func 之前

# Copilot API调用计数器
copilot_api_call_count = 0
copilot_api_limit = 150  # 默认每日限额
openrouter_api_call_count = 0
gemini_api_call_count = 0

def copilot_summarize(repo: Dict) -> Optional[str]:
    """使用 GitHub Copilot API 进行总结"""
    global copilot_api_call_count
    copilot_api_call_count += 1
    remaining = copilot_api_limit - copilot_api_call_count
    print(f"[Copilot API调用] 第 {copilot_api_call_count} 次调用，仓库: {repo['full_name']}，剩余可用: {remaining}")
    if not GITHUB_TOKEN:
        print("缺少 STARRED_GITHUB_TOKEN，无法调用 GitHub Copilot API")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json"
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", DEFAULT_COPILOT_MODEL) or "openai/gpt-4o-mini"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": generate_prompt(repo)}],
            "max_tokens": 600,
            "temperature": 0.4
        }
        response = make_api_request(API_ENDPOINTS["copilot"], headers, data)
        # 限额提醒处理
        if response and isinstance(response, dict) and response.get("error"):
            err = response["error"]
            if err.get("code") == "RateLimitReached":
                msg = err.get("message", "Copilot API限额已用尽，请明天再试。")
                print(f"[Copilot限额] {msg}")
                return f"Copilot API限额已用尽：{msg}"
        content = None
        if response:
            choices = response.get('choices', [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get('message')
                if message and isinstance(message, dict):
                    content = message.get('content', '')
                elif 'content' in choices[0]:
                    content = choices[0]['content']
            if content is not None:
                content = str(content).strip()
        print(f"Copilot内容: {content!r}")
        if not content:
            print("大模型输出为空 (Copilot)")
        return content
    except Exception as e:
        print(f"Copilot总结异常: {e}")
        return None


def openrouter_summarize(repo: Dict) -> Optional[str]:
    """使用 OpenRouter API 进行总结"""
    global openrouter_api_call_count
    openrouter_api_call_count += 1
    if not OPENROUTER_API_KEY:
        print("缺少 OPENROUTER_API_KEY，无法调用 OpenRouter API")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": DEFAULT_OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": generate_prompt(repo)}]
        }
        response = make_api_request(API_ENDPOINTS["openrouter"], headers, data)
        content = None
        if response:
            # OpenRouter API返回结构兼容性处理
            choices = response.get('choices', [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get('message')
                if message and isinstance(message, dict):
                    content = message.get('content', '')
                elif 'content' in choices[0]:
                    content = choices[0]['content']
            if content is not None:
                content = str(content).strip()
        print(f"OpenRouter内容: {content!r}")
        if not content:
            print("大模型输出为空 (OpenRouter)")
        return content
    except Exception as e:
        print(f"OpenRouter总结异常: {e}")
        return None


def gemini_summarize(repo: Dict) -> Optional[str]:
    """使用 Gemini API 进行总结（优化版）"""
    global gemini_api_call_count
    gemini_api_call_count += 1
    # 1. 前置参数验证
    if not GEMINI_API_KEY:
        print("缺少 GEMINI_API_KEY，无法调用 Gemini API")
        return None
    
    prompt = generate_prompt(repo)
    if not prompt.strip():
        print(f"[Gemini] 仓库 {repo.get('full_name')} 的生成提示为空，跳过请求")
        return None

    try:
        # 2. 模型名称标准化（兼容 Gemini 1.0/1.5 全系列模型）
        model_name = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-pro"
        # 严格处理模型路径：移除多余的 models/ 前缀，兼容用户输入的不同格式
        model_path = model_name.lstrip("models/").strip()
        # 补充官方模型前缀（确保 URL 符合规范）
        if not model_path.startswith("gemini-"):
            print(f"[Gemini] 模型名称 {model_name} 非标准格式，建议使用 gemini-pro/gemini-1.5-pro 等")

        # 3. 构造符合官方最新规范的请求 URL
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        
        # 4. 优化请求头（添加 User-Agent、明确 Content-Type）
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GitHub Star Summary Bot/1.0",  # 标识请求来源
            "X-Goog-Api-Key": GEMINI_API_KEY  # 部分场景下该 Header 更兼容，同时 URL 仍保留 key 参数
        }

        # 5. 优化请求体（符合 Gemini 官方规范，增加可控参数）
        # 可配置生成参数，提升总结质量
        temperature = config.get("gemini_temperature", 0.4)
        max_output_tokens = config.get("gemini_max_output_tokens", 800)
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.8,
                "topK": 40
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # 6. 支持基于内容完整性的重试（例如被截断导致 finishReason == 'MAX_TOKENS'）
        request_url = f"{api_url}?key={GEMINI_API_KEY}"

        gen_retries = int(config.get("gemini_generation_retries", 3))
        gen_backoff = int(config.get("gemini_retry_backoff", 5))  # seconds, will multiply per attempt
        base_max_tokens = int(config.get("gemini_max_output_tokens", max_output_tokens))

        final_content = None
        for attempt in range(1, gen_retries + 1):
            # adjust max tokens slightly on retry to try to avoid truncation
            attempt_max_tokens = min(base_max_tokens + (attempt - 1) * 200, 2048)
            payload["generationConfig"]["maxOutputTokens"] = attempt_max_tokens

            if DEBUG_API:
                logger.info(f"[Gemini] 生成尝试 {attempt}/{gen_retries}, maxOutputTokens={attempt_max_tokens}")

            response = make_api_request(
                url=request_url,
                headers=headers,
                data=payload,
                retries=_get_int_config("gemini_retry_attempts", RETRY_ATTEMPTS),
                retry_delay=_get_int_config("gemini_retry_delay", int(REQUEST_RETRY_DELAY))
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

            # 处理错误响应（Gemini 官方错误格式）
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
                # 非重试错误或重试用尽
                if error_code == 401:
                    return "Gemini API Key 无效或无权限"
                elif error_code == 404:
                    return f"Gemini 模型 {model_path} 不存在"
                elif error_code == 400:
                    return "Gemini 请求参数格式错误"
                else:
                    return f"Gemini API 错误: {error_msg}"

            # 解析响应并判断是否被截断
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
                    # prefer first candidate
                    break

                # 兜底解析
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

            # 如果为空或被截断，决定是否重试
            if attempt < gen_retries:
                wait = gen_backoff * attempt
                print(f"[Gemini] 生成内容为空或被截断 (attempt={attempt})，等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            else:
                # 重试用尽，返回当前可能不完整的内容或 None
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
    
    
# 根据配置选择总结函数
def get_summarize_func():
    if model_choice == 'copilot':
        return copilot_summarize
    elif model_choice == 'openrouter':
        return openrouter_summarize
    elif model_choice == 'gemini':
        return gemini_summarize
    else:
        raise ValueError(f"不支持的模型选择: {model_choice}")

summarize_func = get_summarize_func()

# API 配置
DEFAULT_COPILOT_MODEL = default_copilot_model
DEFAULT_OPENROUTER_MODEL = default_openrouter_model
MAX_WORKERS = max_workers
BATCH_SIZE = batch_size
REQUEST_TIMEOUT = request_timeout
RATE_LIMIT_DELAY = rate_limit_delay
REQUEST_RETRY_DELAY = request_retry_delay
RETRY_ATTEMPTS = retry_attempts

# 输出配置
README_SUM_PATH = readme_sum_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README-sum.md')
LANGUAGE = config.get('language', 'zh')

# 日志配置：支持将输出写入文件（可通过 config.log_file 配置路径）
LOG_FILE = config.get('log_file', os.path.join(os.path.dirname(__file__), 'summarize_stars.log'))
LOG_MAX_BYTES = _get_int_config('log_max_bytes', 5 * 1024 * 1024)
LOG_BACKUP_COUNT = _get_int_config('log_backup_count', 3)

logger = logging.getLogger('summarize_stars')
logger.setLevel(logging.DEBUG if DEBUG_API else logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 保留控制台输出，同时将控制台输出也记录到日志文件
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 将 print()/stderr 输出同时写入日志（tee）
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

# 打印 API Key 前缀用于调试
if OPENROUTER_API_KEY:
    print(f"OpenRouter API Key 前缀: {OPENROUTER_API_KEY[:6]}...")
if GITHUB_TOKEN:
    print(f"GitHub Token 前缀: {GITHUB_TOKEN[:6]}...")
if GEMINI_API_KEY:
    try:
        print(f"Gemini API Key 前缀: {GEMINI_API_KEY[:4]}...")
    except Exception:
        # 防止非字符串或长度不足导致异常
        print("Gemini API Key 前缀: (已设置)")

# 常量定义
API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models"
}

# 通用函数

def make_api_request(url: str, headers: Dict, data: Dict, retries: int = RETRY_ATTEMPTS, retry_delay: int = REQUEST_RETRY_DELAY) -> Optional[Dict]:
    """通用的 API 请求函数，支持重试逻辑"""
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=REQUEST_TIMEOUT)
            if DEBUG_API:
                logger.info('[API调试]')
                logger.info(f"请求URL: {url}")
                logger.info(f"请求Headers: {headers}")
                logger.info(f"请求Data: {data}")
                logger.info(f"响应Status: {resp.status_code}")
                logger.info(f"响应Text: {resp.text}")

            if resp.status_code == 429:
                # 优先使用服务器返回的 Retry-After
                retry_after = None
                try:
                    ra = resp.headers.get('Retry-After')
                    if ra is not None:
                        retry_after = int(ra)
                except Exception:
                    retry_after = None

                if retry_after and attempt < retries - 1:
                    wait = retry_after
                    logger.warning(f"遇到 429, 使用 Retry-After 等待 {wait}s 后重试 (尝试 {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue

                # 指数回退并带抖动
                if attempt < retries - 1:
                    base = int(retry_delay)
                    wait = base * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"遇到 429, 等待 {wait:.1f}s 后重试 (尝试 {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
                else:
                    logger.error("API 429 Too Many Requests 并且重试用尽")
                    return {'error': {'code': 429, 'message': 'Too Many Requests'}, 'status_code': 429}

            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {'text': resp.text}
        except requests.HTTPError as e:
            logger.warning(f"API HTTP 错误: {e}")
            if attempt < retries - 1:
                wait = int(retry_delay) * (2 ** attempt)
                logger.info(f"HTTP 错误，等待 {wait}s 后重试")
                time.sleep(wait)
                continue
            else:
                logger.error(f"API 调用最终失败: {e}")
                return None
        except Exception as e:
            logger.warning(f"API 调用失败: {e}")
            if attempt < retries - 1:
                wait = int(retry_delay) * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"请求失败，等待 {wait:.1f}s 后重试: {e}")
                time.sleep(wait)
                continue
            else:
                logger.error(f"API 调用最终失败: {e}")
                return None


def generate_prompt(repo: Dict) -> str:
    """生成通用的总结提示"""
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo["html_url"]
    if LANGUAGE == 'zh':
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

def get_starred_repos() -> List[Dict]:
    """获取用户的 GitHub 星标仓库"""
    if not GITHUB_TOKEN:
        raise ValueError("缺少 GITHUB_TOKEN 环境变量")
    
    print("正在获取星标仓库...")
    repos = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    while True:
        try:
            url = f"https://api.github.com/users/{GITHUB_USERNAME}/starred?per_page={per_page}&page={page}"
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            repos.extend(data)
            print(f"已获取 {len(repos)} 个仓库... (第 {page} 页)")
            page += 1
            
            # 避免 GitHub API 限制
            time.sleep(1)
            
        except requests.RequestException as e:
            print(f"获取星标仓库失败: {e}")
            break
    
    print(f"总共获取到 {len(repos)} 个星标仓库")
    return repos


def load_old_summaries():
    """读取旧的README-sum.md，返回字典: {repo_full_name: summary}，只保留与 config.language 一致的内容"""
    if not README_SUM_PATH or not isinstance(README_SUM_PATH, str) or not os.path.exists(README_SUM_PATH):
        print(f"[DEBUG] {README_SUM_PATH} 不存在，跳过加载旧总结")
        return {}
    print(f"[DEBUG] 开始加载旧总结，文件路径: {README_SUM_PATH}")
    summaries = {}
    current_repo = None
    current_lines = []
    with open(README_SUM_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("### 📌 ["):
                if current_repo and current_lines:
                    summary_block = ''.join(current_lines)
                    summary = summary_block.split('---')[0].strip()
                    summary = re.sub(r"\*\*⭐ Stars:.*更新:.*\n", "", summary)
                    summary = re.sub(r"\*\*⭐ Stars:.*Updated:.*\n", "", summary)
                    # 语言一致性判断
                    if LANGUAGE == 'en':
                        if re.search(r'[\u4e00-\u9fa5]', summary):
                            summary = ''
                    else:
                        if re.search(r'[A-Za-z]', summary) and not re.search(r'[\u4e00-\u9fa5]', summary):
                            summary = ''
                    if summary:
                        summaries[current_repo] = summary
                left = line.find('[') + 1
                right = line.find(']')
                current_repo = line[left:right]
                current_lines = []
            elif current_repo:
                current_lines.append(line)
        if current_repo and current_lines:
            summary_block = ''.join(current_lines)
            summary = summary_block.split('---')[0].strip()
            summary = re.sub(r"\*\*⭐ Stars:.*更新:.*\n", "", summary)
            summary = re.sub(r"\*\*⭐ Stars:.*Updated:.*\n", "", summary)
            if LANGUAGE == 'en':
                if re.search(r'[\u4e00-\u9fa5]', summary):
                    summary = ''
            else:
                if re.search(r'[A-Za-z]', summary) and not re.search(r'[\u4e00-\u9fa5]', summary):
                    summary = ''
            if summary:
                summaries[current_repo] = summary
    print(f"[DEBUG] 加载旧总结完成，仓库名称列表: {list(summaries.keys())}")
    return summaries


# 新增：使用 GitHub Copilot / GitHub Models API 进行总结
# 需要 STARRED_GITHUB_TOKEN 具备访问 models:read & copilot 范围（一般 PAT 启用 copilot 即可）
# 可通过环境变量 GITHUB_COPILOT_MODEL 指定模型，默认 gpt-4o-mini（依据 GitHub Models 可用模型自行调整）


def is_valid_summary(summary: str) -> bool:
    """检查给定的总结是否有效（只要包含无效短语或仅为换行都判定为False）"""
    if not summary or not summary.strip():
        print(f"[DEBUG] is_valid_summary: False (空字符串或仅换行)")
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached"]
    for phrase in invalid_phrases:
        if phrase in summary:
            print(f"[DEBUG] is_valid_summary: False (包含无效短语: {phrase})")
            return False
    # 额外检测：若摘要使用了与当前语言不一致的模板关键词，或包含常见的不完整英文/模板开头，则视为无效，触发更新
    try:
        lang = config.get('language', 'zh')
    except Exception:
        lang = 'zh'

    # 常见的、不完整或占位性质的英文模板短语/关键词
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

    # 常见的、不完整或占位性质的中文模板短语
    common_chinese_templates = [
        r"仓库名称",
        r"简要介绍",
        r"创新点",
        r"简单用法",
        r"总结\s*[:：]",
        r"请对以下 GitHub 仓库进行内容总结",
    ]

    s_head = summary.strip()[:200]

    # 如果当前语言为中文，但摘要中出现明显英文模板关键词，则视为不完善（需要更新）
    if lang != 'en':
        for p in common_english_templates:
            if re.search(p, s_head, flags=re.IGNORECASE):
                print(f"[DEBUG] is_valid_summary: False (包含英文模板关键词，需更新: {p})")
                return False
    # 如果当前语言为英文，但摘要中出现明显中文模板关键词，则视为不完善（需要更新）
    if lang == 'en':
        for p in common_chinese_templates:
            if re.search(p, s_head):
                print(f"[DEBUG] is_valid_summary: False (包含中文模板关键词，需更新: {p})")
                return False
    # 检查是否仅为换行（如 '\n', '\r\n' 等）
    if summary.strip() == "":
        print(f"[DEBUG] is_valid_summary: False (仅换行)")
        return False
    # 内容完整性检查：根据语言确认摘要包含预期的段落标题或关键字段
    try:
        lang = config.get('language', 'zh')
    except Exception:
        lang = 'zh'

    head = summary.strip()[:200]  # 只检查开头部分
    if lang == 'en':
        patterns = [r'Summary\s*[:：]', r'Repository Name', r'Brief Introduction', r'Innovations']
    else:
        patterns = [r'总结\s*[:：]', r'仓库名称', r'简要介绍', r'创新点']

    # 要求所有关键词都存在；任一缺失都视为不完整，需要更新
    missing = []
    for p in patterns:
        if not re.search(p, head, flags=re.IGNORECASE):
            missing.append(p)

    if missing:
        print(f"[DEBUG] is_valid_summary: False (缺少关键词，需更新: {missing})")
        return False

    # 额外检查：若摘要包含明确的“简要介绍/Brief Introduction”字段，但该字段内容非常短或仅为占位字符（如单字'A'），则视为无效
    try:
        s = summary
        if lang == 'en':
            m = re.search(r'Brief Introduction\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)', s, flags=re.IGNORECASE | re.S)
            if m:
                intro = m.group(1).strip()
                # 去除markdown标记和多余空白
                intro_text = re.sub(r'\*|\*\*|`|\\n', '', intro).strip()
                if len(intro_text) < 20:
                    print(f"[DEBUG] is_valid_summary: False (英文简要介绍过短或占位: {intro_text!r})")
                    return False
        else:
            m = re.search(r'简要介绍\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)', s, flags=re.S)
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r'\*|\*\*|`|\\n', '', intro).strip()
                if len(intro_text) < 10:
                    print(f"[DEBUG] is_valid_summary: False (中文简要介绍过短或占位: {intro_text!r})")
                    return False
    except Exception:
        pass

    print(f"[DEBUG] is_valid_summary: True")
    return True


def summarize_batch(repos: List[Dict], old_summaries: Dict[str, str], use_copilot: bool = False, use_gemini: bool = False) -> List[str]:
    """批量总结仓库，支持选择使用 OpenRouter、GitHub Copilot 或 Gemini"""
    results: List[str] = [""] * len(repos)
    if use_gemini:
        summarize_func = gemini_summarize
        api_name = "Gemini"
    else:
        summarize_func = copilot_summarize if use_copilot else openrouter_summarize
        api_name = "Copilot" if use_copilot else "OpenRouter"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(summarize_func, repo): idx
            for idx, repo in enumerate(repos)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            repo = repos[idx]
            try:
                # 检查是否已有有效总结
                existing_summary = old_summaries.get(repo["full_name"], "")
                if is_valid_summary(existing_summary):
                    summary = existing_summary
                else:
                    summary = future.result()
                    if summary is None:
                        summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败或429")
                # Debug: 输出每个 summary 内容
                print(f"[DEBUG] repo: {repo['full_name']} | summary: {repr(summary)}")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary if summary is not None else "*暂无AI总结*"
    return results


def copilot_summarize_batch(repos: List[Dict], old_summaries: Dict[str, str]) -> List[str]:
    """使用 GitHub Copilot 批量总结仓库"""
    return summarize_batch(repos, old_summaries, use_copilot=True)


def gemini_summarize_batch(repos: List[Dict], old_summaries: Dict[str, str]) -> List[str]:
    """使用 Gemini 批量总结仓库"""
    return summarize_batch(repos, old_summaries, use_gemini=True)


def classify_by_language(repos):
    classified = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified


def update_existing_summaries(lines, old_summaries):
    """更新已有的 README-sum.md 文件中的总结内容"""
    updated_lines = []
    current_repo = None
    for line in lines:
        if line.startswith("### ["):
            # 解析仓库名
            left = line.find('[') + 1
            right = line.find(']')
            current_repo = line[left:right]
            updated_lines.append(line)
        elif current_repo and current_repo in old_summaries:
            # Debug: 输出替换内容
            print(f"[DEBUG] 替换 {current_repo} 的 summary 为: {repr(old_summaries[current_repo])}")
            updated_lines.append(old_summaries[current_repo] + "\n")
            current_repo = None
        else:
            updated_lines.append(line)
    return updated_lines

def github_anchor(text):
    # GitHub锚点规则：小写，空格转-，去除特殊字符，仅保留字母、数字、中文和'-'
    anchor = text.strip().lower()
    anchor = re.sub(r'[\s]+', '-', anchor)  # 空格转-
    anchor = re.sub(r'[^\w\u4e00-\u9fa5-]', '', anchor)  # 去除非字母数字中文和-
    return anchor

###########################################
def main():
    # 通过环境变量控制使用哪种 API，默认使用 Copilot
    # 通过 config 的 model_choice 优先选择模型；若未设置，则使用环境变量 USE_COPILOT_API
    if model_choice:
        api_choice = model_choice.lower()
    else:
        api_choice = 'copilot' if os.environ.get("USE_COPILOT_API", "true").lower() == "true" else 'openrouter'

    if api_choice == 'gemini':
        api_name = 'Gemini'
    elif api_choice == 'openrouter':
        api_name = 'OpenRouter (DeepSeek)'
    else:
        api_name = 'GitHub Copilot'

    print(f"开始使用 {api_name} 生成 GitHub Star 项目总结...")
    
    try:
        starred = get_starred_repos()
        # 测试模式：若 config 中启用了 test_first_repo，则只保留第一个仓库以便快速调试
        try:
            test_first_repo = bool(config.get('test_first_repo', False))
        except Exception:
            test_first_repo = False
        if test_first_repo and isinstance(starred, list) and len(starred) > 0:
            print('[TEST MODE] test_first_repo 已启用：仅处理第一个仓库进行调试')
            starred = [starred[0]]
        # 测试模式：若配置中启用了 test_first_repo，则只保留第一个仓库以便快速调试
        test_first_repo = config.get('test_first_repo', False)
        if test_first_repo and isinstance(starred, list) and len(starred) > 0:
            print("[TEST MODE] test_first_repo 已启用：仅处理第一个仓库进行调试")
            starred = [starred[0]]
        # 根据环境变量 MAX_REPOS 限制处理数量，方便在 CI 中避免超时
        if MAX_REPOS and isinstance(starred, list):
            try:
                limit = int(MAX_REPOS)
                if limit > 0 and len(starred) > limit:
                    print(f"[LIMIT] 因环境变量 MAX_REPOS={limit}，仅处理前 {limit} 个仓库以避免超时")
                    starred = starred[:limit]
            except Exception:
                pass
        classified = classify_by_language(starred)
        old_summaries = load_old_summaries()
        
        # 新增：根据 update_mode 过滤需要处理的仓库
        if update_mode == "missing_only":
            # 只对缺失或无效的 summary 做实际的 API 调用，但保留完整仓库列表用于最终输出。
            # 构建一个 process_set，包含需要调用 AI 的仓库 full_name
            process_set = set()
            for lang, repos in classified.items():
                for repo in repos:
                    if not is_valid_summary(old_summaries.get(repo.get("full_name", ""), "")):
                        process_set.add(repo.get('full_name'))
            # classified_to_process 保留全部仓库（用于生成最终文档），但实际只对 process_set 中的仓库发起调用
            classified_to_process = classified
        else:
            # 全部仓库都处理
            # 优先处理已有总结不完整或包含错误的仓库（使它们在每次更新时先被刷新）
            classified_to_process = {}
            for lang, repos in classified.items():
                try:
                    # 将 is_valid_summary 为 False 的仓库排在前面
                    sorted_repos = sorted(repos, key=lambda r: is_valid_summary(old_summaries.get(r.get('full_name', ''), '')))
                except Exception:
                    sorted_repos = repos
                if sorted_repos:
                    classified_to_process[lang] = sorted_repos

        # 更新标题以反映实际使用的 API
        current_time = time.strftime("%Y-%m-%d", time.localtime())
        if LANGUAGE == 'en':
            title = f"# My GitHub Star Project AI Summary\n\n"
            title += f"**Generated on:** {current_time}\n\n"
            title += f"**AI Model:** {api_name}\n\n"
            title += f"**Total repositories:** {len(starred)}\n\n"
            title += "---\n\n"
            lines = [title]
            # Add repository reference link to generated document
            lines.append("**Reference Repository:** [WuXiangM/myGitStar](https://github.com/WuXiangM/myGitStar)\n\n")
            # 根据配置决定中英文 README 链接的显示顺序（默认 English first）
            repo_display_language = bool(config.get('repo_display_language', True))
            if repo_display_language:
                lines.append("[English README](README.md) | [中文 README](README2.md)\n\n")            
            lines.append("[English GUIDE](GUIDE_en.md) | [中文 GUIDE](GUIDE_zh.md)\n\n")


            # 添加目录
            lines.append("## 📖 Table of Contents\n\n")
            lang_counts = {}
            for lang, repos in classified_to_process.items():
                lang_counts[lang] = len(repos)
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                anchor = github_anchor(lang)
                lines.append(f"- [{lang}](#{anchor}) ({count})\n")
            lines.append("\n---\n\n")
        else:
            title = f"# 我的 GitHub Star 项目AI总结\n\n"
            title += f"**生成时间：** {current_time}\n\n"
            title += f"**AI模型：** {api_name}\n\n"
            title += f"**总仓库数：** {len(starred)} 个\n\n"
            title += "---\n\n"
            lines = [title]
            # 在中文文档顶部加入仓库引用
            lines.append("**参考仓库：** [WuXiangM/myGitStar](https://github.com/WuXiangM/myGitStar)\n\n")
            # 根据配置决定中英文 README 链接的显示顺序（默认 中文 first）
            repo_display_language = bool(config.get('repo_display_language', True))
            if repo_display_language:
                lines.append("[中文 README](README.md) | [English README](README2.md)\n\n")            
            lines.append("[中文 GUIDE](GUIDE_zh.md) | [English GUIDE](GUIDE_en.md)\n\n")
            
            # 添加目录
            lines.append("## 📖 目录\n\n")
            lang_counts = {}
            for lang, repos in classified_to_process.items():
                lang_counts[lang] = len(repos)
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                anchor = github_anchor(lang)
                lines.append(f"- [{lang}](#{anchor})（{count}个）\n")
            lines.append("\n---\n\n")
        
        printed_repos = set()
        printed_langs = set()  # 记录已输出的语言
        
        total_repos = sum(len(repos) for repos in classified_to_process.values())
        processed_repos = 0
        
        repo_summary_map = {}  # 新增：全局仓库总结映射

        for lang, repos in sorted(classified_to_process.items(), key=lambda x: -len(x[1])):
            if lang in printed_langs:
                continue  # 跳过已输出的语言标题
            printed_langs.add(lang)
            print(f"正在处理 {lang} 类型的仓库（共{len(repos)}个）...")
            
            # 添加语言标题和图标
            if LANGUAGE == 'en':
                lang_icon = {
                    "Python": "🐍", "JavaScript": "🟨", "TypeScript": "🔷", 
                    "Java": "☕", "Go": "🐹", "Rust": "🦀", "C++": "⚡", 
                    "C": "🔧", "C#": "💜", "PHP": "🐘", "Ruby": "💎", 
                    "Swift": "🐦", "Kotlin": "🅺", "Dart": "🎯", 
                    "Shell": "🐚", "HTML": "🌐", "CSS": "🎨", 
                    "Vue": "💚", "React": "⚛️", "Other": "📦"
                }.get(lang, "📝")
                lines.append(f"## {lang_icon} {lang} (Total {len(repos)})\n\n")
            else:
                lang_icon = {
                    "Python": "🐍", "JavaScript": "🟨", "TypeScript": "🔷", 
                    "Java": "☕", "Go": "🐹", "Rust": "🦀", "C++": "⚡", 
                    "C": "🔧", "C#": "💜", "PHP": "🐘", "Ruby": "💎", 
                    "Swift": "🐦", "Kotlin": "🅺", "Dart": "🎯", 
                    "Shell": "🐚", "HTML": "🌐", "CSS": "🎨", 
                    "Vue": "💚", "React": "⚛️", "Other": "📦"
                }.get(lang, "📝")
                lines.append(f"## {lang_icon} {lang}（共{len(repos)}个）\n\n")
            
            # 为了避免在 missing_only 模式下把完整仓库列表裁剪掉，我们只对 process_list 发起请求
            if update_mode == "missing_only":
                # 只处理需要更新的仓库
                repos_to_call = [r for r in repos if r.get('full_name') in process_set]
            else:
                repos_to_call = repos

            for i in range(0, len(repos_to_call), BATCH_SIZE):
                this_batch = repos_to_call[i:i+BATCH_SIZE]
                print(f"处理批次 {i//BATCH_SIZE + 1}，包含 {len(this_batch)} 个仓库...")
                
                # 根据选择使用不同的总结函数（优先使用 config.model_choice）
                if api_choice == 'gemini':
                    summaries = gemini_summarize_batch(this_batch, old_summaries)
                elif api_choice == 'copilot':
                    summaries = copilot_summarize_batch(this_batch, old_summaries)
                else:
                    summaries = summarize_batch(this_batch, old_summaries, use_copilot=False)
                
                for repo, summary in zip(this_batch, summaries):
                    repo_summary_map[repo['full_name']] = summary  # 收集更新后的 summary

                # 当使用 missing_only 时，this_batch 仅包含需要更新的仓库；后续在写入阶段会合并老的 summary

            # 在写入阶段遍历原始 repos 列表，优先使用更新后的 summary（若存在），否则使用旧的 summary
            for repo in repos:
                if repo['full_name'] in printed_repos:
                    continue  # 跳过已输出的仓库
                printed_repos.add(repo['full_name'])

                summary = repo_summary_map.get(repo['full_name']) or old_summaries.get(repo['full_name'], "")

                # 获取仓库信息
                url = repo["html_url"]
                stars = repo.get("stargazers_count", 0)
                forks = repo.get("forks_count", 0)
                language = repo.get("language", "Unknown")
                updated_at = repo.get("updated_at", "")
                if updated_at:
                    try:
                        # 解析时间并格式化
                        from datetime import datetime
                        dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        updated_at = dt.strftime("%Y-%m-%d")
                    except:
                        updated_at = updated_at[:10]  # 取前10个字符作为日期
                
                # 构建仓库条目
                lines.append(f"### 📌 [{repo['full_name']}]({url})\n\n")

                # 添加仓库元信息
                if LANGUAGE == 'en':
                    lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **📅 Updated:** {updated_at}\n\n")
                else:
                    lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **📅 更新:** {updated_at}\n\n")

                # 添加AI总结内容
                if summary and summary.strip():
                    print(f"[DEBUG] 写入MD: {repo['full_name']} | 内容: {summary[:60]}...")
                    lines.append(f"{summary}\n\n")
                else:
                    print(f"[DEBUG] 写入MD: {repo['full_name']} | 内容: *暂无AI总结*")
                    if LANGUAGE == 'en':
                        lines.append("*No AI summary available*\n\n")
                    else:
                        lines.append("*暂无AI总结*\n\n")

                lines.append("---\n\n")
                processed_repos += 1
                
                print(f"已处理 {processed_repos}/{total_repos} 个仓库")
                time.sleep(RATE_LIMIT_DELAY)  # 避免 API 限流
        
        # 添加页脚
        if LANGUAGE == 'en':
            lines.append(f"\n## 📊 Statistics\n\n")
            lines.append(f"- **Total repositories:** {processed_repos}\n")
            lines.append(f"- **Languages:** {len(classified_to_process)}\n")
            lines.append(f"- **Generated on:** {current_time}\n")
            lines.append(f"- **AI Model:** {api_name}\n\n")
            lines.append(f"- **API Calls:** Copilot={copilot_api_call_count}, OpenRouter={openrouter_api_call_count}, Gemini={gemini_api_call_count}\n")
            lines.append("---\n\n")
            lines.append("*This document is generated by AI. For any errors, please refer to the original repository information.*\n")
        else:
            lines.append(f"\n## 📊 统计信息\n\n")
            lines.append(f"- **总仓库数：** {processed_repos} 个\n")
            lines.append(f"- **编程语言数：** {len(classified_to_process)} 种\n")
            lines.append(f"- **生成时间：** {current_time}\n")
            lines.append(f"- **AI模型：** {api_name}\n\n")
            lines.append(f"- **API 调用次数：** Copilot={copilot_api_call_count}，OpenRouter={openrouter_api_call_count}，Gemini={gemini_api_call_count}\n")
            lines.append("---\n\n")
            lines.append("*本文档由AI自动生成，如有错误请以原仓库信息为准。*\n")

        # 始终生成完整的新md内容，直接覆盖写入
        with open(README_SUM_PATH, "w", encoding="utf-8") as f:
            f.write(''.join(lines))
        print(f"\n✅ {README_SUM_PATH} 已生成，共处理了 {processed_repos} 个仓库。")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--copilot-count":
        print(copilot_api_call_count)
    else:
        main()
        print(f"Copilot API 总调用次数: {copilot_api_call_count}")