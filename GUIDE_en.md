# 🌟 GitHub Stars AI Auto-Summary System — English Guide

This project fetches a GitHub account’s starred repositories (Stars), uses AI to generate structured Markdown summaries, and outputs them as README documents. It is suitable for local runs or scheduled updates via GitHub Actions.

## 1) 🎁 What you get (outputs)

- 📗 README (content classified): `README.md`
- 📘 README classified by language (English): `README_lang.md`
- 📙 README 按语言分类 (Chinese): `README_lang_cn.md`

> Note: The language-classified output file is controlled by `readme_sum_path` in `config.yaml` or can be overridden via `--out`. The content-classified README is generated separately by `classify_stars_by_content.py` as `README.md`.

## 2) 📂 Project layout (key files)

```
myGitStar/
├── scripts/summarize_stars.py     # main script
├── config.yaml                   # configuration
├── requirements.txt              # Python dependencies
├── README.md                     # content-classified output (main README)
├── README_lang.md                # language-classified (English)
├── README_lang_cn.md             # language-classified (Chinese)
├── GUIDE_en.md                   # English guide (this file)
└── GUIDE_zh.md                   # Chinese guide
```

## 3) 🚀 Quick start

### 3.1 🤖 Run with GitHub Actions (recommended)

In your repository: Settings → Secrets and variables → Actions, add:

- ✅ `STARRED_GITHUB_TOKEN` (required)
- ➕ `OPENROUTER_API_KEY` (optional)
- ➕ `GEMINI_API_KEY` (optional)

Then inject them into the workflow as environment variables (there is already a workflow in this repo you can reference).

### 3.2 💻 Run locally (Windows PowerShell)

1. Install dependencies (using a virtual environment is recommended):

```powershell
pip install -r requirements.txt
```

2. Set the token (at minimum you need a GitHub token; it is used to fetch Stars and is also used by Copilot Models):

```powershell
$env:STARRED_GITHUB_TOKEN = "ghp_xxx"
```

3. Run the script:

```powershell
python scripts/summarize_stars.py --language en --out README_lang.md
python scripts/summarize_stars.py --language zh --out README_lang_cn.md

# Generate the content-classified README (parse repos from the language-classified README)
python scripts/classify_stars_by_content.py --from-readme README_lang.md --out-md README.md --out-json repo_categories.json
```

## 4) ⚙️ Configuration (config.yaml)

The script reads `config.yaml` first (comments are supported).

Common / easy-to-misconfigure fields:

```yaml
# Output language: zh or en
language: en

# AI backend: copilot / openrouter / gemini
model_choice: copilot

# Output path (language-classified): recommended en->README_lang.md, zh->README_lang_cn.md
readme_sum_path: README_lang.md

# Update mode: all (rewrite all) / missing_only (only fill missing/invalid summaries)
update_mode: missing_only

# Concurrency & throttling (if you hit 429, reduce concurrency and increase delays)
max_workers: 1
batch_size: 4
rate_limit_delay: 5
request_timeout: 30

# Workflow classify-only: when true, GitHub Actions skips summarize_stars.py and only runs classify_stars_by_content.py
# Note: when enabled, README_lang.md must already exist in the repo (used as --from-readme input)
workflow_classify_only: false

# README top language-switch link order
# true: put the language matching `language` first; false: reverse
repo_display_language: true
```

### 4.1 📌 Config field quick reference (with “Required”)

The table below summarizes supported `config.yaml` fields and marks whether each one is required.

| Field | Required | Type/Example | Purpose | Recommendation |
| --- | --- | --- | --- | --- |
| `github_username` | Yes (account) | `0` / `WuXiangM` | Which account’s Stars to fetch. `0` means use Actions `GITHUB_ACTOR`; locally it’s better to set a fixed username | CI: `0`; Local: set a fixed username |
| `STARRED_GITHUB_TOKEN` | Yes (runtime; prefer env) | `""` | GitHub token (used for fetching Stars and Copilot Models) | Keep empty in config; use Actions Secrets / environment variables |
| `OPENROUTER_API_KEY` | Conditional (openrouter) | `""` | OpenRouter key (needed only when `model_choice: openrouter`) | Use env |
| `GEMINI_API_KEY` | Conditional (gemini) | `""` | Gemini key (needed only when `model_choice: gemini`) | Use env |
| `language` | No | `zh` / `en` | Output language | Defaults to `zh` if omitted |
| `model_choice` | No | `copilot` / `openrouter` / `gemini` | Choose AI backend | Defaults to `copilot` |
| `readme_sum_path` | No | `README_lang.md` / `README_lang_cn.md` | Language-classified output path | en→`README_lang.md`; zh→`README_lang_cn.md` (if omitted: `README-sum.md`) |
| `update_mode` | No | `all` / `missing_only` | Update strategy: rewrite all / only fill missing or invalid summaries | Use `missing_only` for stable incremental updates |
| `repo_display_language` | No | `true` / `false` | Order of README top language-switch links | `true`: put the language matching `language` first |
| `default_copilot_model` | No | `openai/gpt-4o-mini` | Default Copilot Models model name | Can be overridden via env (see below) |
| `default_openrouter_model` | No | `deepseek/deepseek-prover-v2:free` | Default OpenRouter model name | Pick a model available to your account |
| `default_gemini_model` | No | `gemini-2.0-flash` | Default Gemini model name | Can be overridden via env (see below) |
| `max_repos` | No | `50` / `0` | Max repositories per run; `0`/empty means no limit | CI: 20–80; Local: can use `0` |
| `max_workers` | No | `1` | Thread pool concurrency | Start with `1` to avoid rate limits |
| `batch_size` | No | `4` | Repos per batch (batches trigger summarization requests) | Lower it if you hit limits/timeouts |
| `request_timeout` | No | `30` | HTTP timeout (seconds) | Increase on unstable networks |
| `rate_limit_delay` | No | `5` | Sleep between batches (seconds) | Increase if you hit 429 |
| `request_retry_delay` | No | `2` | Delay between network retries (seconds) | 2–10 |
| `retry_attempts` | No | `1` | Network retry attempts (generic request wrapper) | 1–3 |
| `global_qps` | No | `0.5` | Global throttling (QPS). Default 0.5 ≈ one request per ~2s | Reduce further if you hit 429 (e.g. 0.2) |
| `workflow_classify_only` | No | `true` / `false` | Actions: run content-classifier only (skip summarize step) | Before setting `true`, ensure `README_lang.md` exists; keep `false` for normal updates |
| `test_first_repo` | No | `false` | Debug switch: process only the first repo (also enables more verbose logs) | Use `true` for local debugging |
| `log_file` | No | `scripts/summarize_stars.log` | Log file path (default is in the script directory) | Default is fine for CI |
| `log_max_bytes` | No | `5242880` | Max single log file size (bytes), then rotate | Default is fine |
| `log_backup_count` | No | `3` | Number of rotated log backups to keep | Default is fine |

#### ✨ Gemini fields (only useful when `model_choice: gemini`)

| Field | Required | Type/Example | Purpose | Recommendation |
| --- | --- | --- | --- | --- |
| `gemini_temperature` | No | `0.4` | Generation temperature (higher = more diverse) | 0.2–0.6 |
| `gemini_max_output_tokens` | No | `2000` | Output token limit (too small can truncate) | Increase if truncated/missing fields |
| `gemini_generation_retries` | No | `1` | Generation-level retries (e.g. truncated or incomplete response) | 1–3 |
| `gemini_retry_backoff` | No | `2` | Backoff multiplier for generation retries | 2–5 |
| `gemini_retry_attempts` | No | `1` | Network retry attempts for Gemini requests | 1–3 |
| `gemini_retry_delay` | No | `2` | Delay between Gemini network retries (seconds) | 2–10 |

### 4.2 📌 Environment variable overrides (common in CI)

- `MAX_REPOS`: higher priority than `max_repos` in `config.yaml` (handy to prevent CI timeouts).
- `GITHUB_COPILOT_MODEL`: overrides `default_copilot_model`.
- `GEMINI_MODEL`: overrides `default_gemini_model`.

### 4.3 📌 Legacy compatibility fields (usually not needed)

These exist mainly for compatibility (not recommended for new setups):

- `github_token_env`: specify the env var name for GitHub token (recommended default is `STARRED_GITHUB_TOKEN`).
- `openrouter_api_key_env`: specify the env var name for OpenRouter key.
- `gemini_api_key_env`: specify the env var name for Gemini key.
- `github_token` / `openrouter_api_key` / `gemini_api_key`: plaintext keys (not recommended; avoid committing secrets).

### 4.4 📌 Secrets in config (not recommended; fallback only)

`config.yaml` may contain the following fields (this repo’s config includes them), and the script can read them as a fallback — but you should keep them empty and use Actions Secrets / environment variables instead:

- `STARRED_GITHUB_TOKEN`
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`

## 5) 🔒 Secrets / Environment Variables (Keys & Token)

Always prefer CI Secrets or local environment variables. Do not commit real keys into the repository.

### Recommended env var names

| Env var | Purpose |
| --- | --- |
| `STARRED_GITHUB_TOKEN` | GitHub token (used to fetch Stars and Copilot Models) |
| `OPENROUTER_API_KEY` | OpenRouter (optional) |
| `GEMINI_API_KEY` | Gemini (optional) |

### Read priority (high → low)

1. Environment variables (Actions Secrets / local env)
2. Same-name uppercase keys in `config.yaml` (fallback only; not recommended)
3. `*_env` fields in `config.yaml` (legacy compatibility)
4. Plaintext keys (not recommended)

## 6) 🧯 Troubleshooting checklist

- **429 / rate limits**: lower `batch_size`, keep `max_workers: 1`, increase `rate_limit_delay`.
- **Output file doesn’t change**: check `update_mode`; `missing_only` only updates missing/invalid summaries.
- **Generated into an unexpected file**: check `readme_sum_path`.
- **Missing dependencies locally**: run `pip install -r requirements.txt` and ensure you are in the right venv.

## 7) 🧱 Minimal Actions snippet (reference)

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with:
    python-version: '3.x'
- run: pip install -r requirements.txt
- name: Run summarizer
  env:
    STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  run: python scripts/summarize_stars.py
```
