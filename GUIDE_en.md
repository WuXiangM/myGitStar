# 🌟 GitHub Stars AI Auto-Summary System — Full Guide

## Overview

This project automates collection of a GitHub user's starred repositories and uses AI to generate structured Markdown summaries. It supports GitHub Copilot, OpenRouter, and Google Gemini, and includes automated deployment, caching, and robust error handling.

## Core Features

- Automatically fetch starred repositories
- Multiple AI backends: Copilot, OpenRouter, Gemini
- Categorize summaries by programming language
- Nicely formatted Markdown output with emoji icons
- GitHub Actions friendly for scheduled updates
- Smart caching to skip unchanged repositories
- Retry/backoff and rate-limit handling

## File Layout

```
myGitStar/
├── scripts/
│   └── summarize_stars.py          # main script
├── .github/
│   └── workflows/
│       └── update_myGitStar_sum.yml # Actions workflow
├── README-sum.md                    # generated summary
├── README.md                        # project README
└── GUIDE_en.md                       # this English guide
```

## New config key

- `repo_display_language: true|false` (optional, default: `true`)
  - Purpose: Controls whether the generated README includes quick language-switch links at the top and which language link appears first.
  - Example: with `language: en` and `repo_display_language: true`, the top will show `[English README](README.md) | [中文 README](README2.md)`. If `false`, the Chinese link will appear first. The same logic applies when `language: zh`.
  - Note: The script uses example filenames (`README.md` and `README2.md`) for link targets. If your repository uses different filenames (e.g., `README_cn.md`), adjust the files or the script accordingly.

## Environment & Secrets

The script reads API keys and tokens using the following priority (recommended to provide them via CI secrets):

1. Environment variables (CI Secrets or local env)
2. Uppercase keys in `config.yaml` (if you stored keys there)
3. Env-name fields in `config.yaml` (compatibility, e.g., `github_token_env`)
4. Plaintext keys in `config.yaml` (not recommended)

Recommended environment variable names (default):

- `STARRED_GITHUB_TOKEN` — GitHub Personal Access Token, used to fetch starred repos and (optionally) call Copilot/Models
- `OPENROUTER_API_KEY` — OpenRouter API key (optional)
- `GEMINI_API_KEY` — Google Gemini API key (optional)

The script prints key prefixes for debugging (first few characters) so you can verify secrets are injected in CI.

## Quick Start (local / CI)

1. Clone repository and enter:

```bash
git clone https://github.com/<your>/myGitStar.git
cd myGitStar
```

2. Install dependencies:

```bash
pip install requests pyyaml
```

3. Add Secrets in GitHub Actions (if using CI):

- `STARRED_GITHUB_TOKEN` (required)
- `OPENROUTER_API_KEY` (optional)
- `GEMINI_API_KEY` (optional)

4. Example local run:

```bash
export STARRED_GITHUB_TOKEN="ghp_xxx"
python scripts/summarize_stars.py
```

## Using Gemini

- To enable Gemini, set `model_choice: gemini` in `config.yaml` and provide `GEMINI_API_KEY` in the environment.
- You can override the model via `default_gemini_model` in config or `GEMINI_MODEL` env var.

## Troubleshooting & Tips

- If you get many `429` errors, increase `rate_limit_delay` or reduce `batch_size` / `max_workers` in `config.yaml`.
- Verify Secrets are correctly injected in Actions by checking the run logs (the script prints a small prefix of keys for debug purposes).
- If Gemini parsing fails, inspect the logs and consider adjusting `default_gemini_model`.

## Minimal GitHub Actions Example

```yaml
name: Update myGitStar Summaries
on:
  schedule:
    - cron: '0 4 * * *'
  workflow_dispatch:
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.x' }
      - run: pip install requests pyyaml
      - name: Run summarizer
        env:
          STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: python scripts/summarize_stars.py
      - name: Commit
        run: |
          git add README-sum.md
          git commit -m "update AI summarized stars [bot]" || echo "No changes"
          git push origin HEAD:main || true
```

## Conclusion

Configuration moved to `config.yaml` (supports comments). Prefer CI Secrets over storing plaintext keys. If you want, I can also update the README files or the Actions workflow to reflect these options.
