# 🌟 GitHub Stars AI 自动总结系统 - 完整说明文档

## 📋 项目概述

这是一个自动化系统，用于获取 GitHub 用户的星标仓库并使用 AI 进行智能总结，生成结构化的 Markdown 文档。支持 GitHub Copilot、OpenRouter 和 Gemini 三种 AI 服务，具备自动化部署、智能缓存、错误处理等功能。

## ✨ 核心功能

- 🔍 自动获取所有 GitHub 星标仓库
- 🤖 多 AI 引擎：Copilot API、OpenRouter API、Gemini（可选）
- 📊 按编程语言分类展示
- 🎨 美化 Markdown 输出，带 emoji 图标
- ⚡ GitHub Actions 自动定时更新
- 🔄 智能缓存，避免重复处理
- 🛡️ 健壮错误处理与重试机制

## 📂 文件结构

```
myGitStar/
├── scripts/
│   └── summarize_stars.py          # 主脚本
├── .github/
│   └── workflows/
│       └── update_myGitStar_sum.yml # Actions 工作流
├── README-sum.md                    # 生成总结文档
├── README.md                        # 项目说明
└── GUIDE.md                          # 本完整说明文档
```

## 🔧 环境配置（Secrets 与 Keys）

本项目在运行时会使用若干 API Key / Token，推荐在 CI（GitHub Actions）中通过 Secrets 提供。脚本在读取时遵循以下优先级：

1. 环境变量（来自 CI 的 Secrets，或本地环境导出，常见为全大写名称）
2. `config.json` 中的大写键（如果你不使用 Secrets 且将 key 写入 config）
3. `config.json` 中指定的 env 名称字段（兼容旧配置，例如 `github_token_env`）
4. `config.json` 中的明文字段（不推荐，在仓库中不要保存真实密钥）

### 推荐的 Secret 名称（默认配置）

| 名称 (env)              | 用途                        | 在 `config.json` 中的对应字段 |
| ----------------------- | --------------------------- | --------------------------- |
| `STARRED_GITHUB_TOKEN`  | GitHub Personal Access Token，用于获取星标仓库与 Copilot Models API | `STARRED_GITHUB_TOKEN` 或 `github_token_env` |
| `OPENROUTER_API_KEY`    | OpenRouter API Key（可选）  | `OPENROUTER_API_KEY` 或 `openrouter_api_key_env` |
| `GEMINI_API_KEY`        | Gemini（Google）API Key（可选） | `GEMINI_API_KEY` 或 `gemini_api_key_env` |

> 说明：脚本默认已把这些 env 名称作为首选（全大写）。如果你在 `config.json` 内使用了不同的 env 名称，可把该名称写入 `github_token_env`/`openrouter_api_key_env`/`gemini_api_key_env`，脚本会使用该 env 名称去读取。
# 🌟 GitHub Stars AI 自动总结系统 — 使用指南

## 📋 项目概述

本项目会获取指定 GitHub 账号的星标仓库并使用 AI（支持 Copilot / OpenRouter / Gemini）生成结构化的 Markdown 总结，适合在 GitHub Actions 中定时运行并自动提交结果。

## 📂 主要文件

```
myGitStar/
├── scripts/summarize_stars.py    # 主脚本（支持 copilot/openrouter/gemini）
├── config.yaml                   # 配置（优先读取 YAML）
├── README-sum.md                 # 自动生成的总结文档
├── README.md
├── .github/workflows/            # Actions 工作流
└── GUIDE.md                      # 本指南
```

## 🔧 配置与 Secrets（关键点）

1) 配置文件：当前脚本优先读取 `config.yaml`（支持注释），若不存在会回退到 `config.json`。建议使用 `config.yaml` 并不要在仓库中提交真实密钥。

2) 优先级（脚本读取顺序）：
- 环境变量（来自 CI 的 Secrets 或本地导出，通常为全大写名称）
- `config.yaml` 中的大写键（如 `STARRED_GITHUB_TOKEN`）
- `config.yaml` 中指定的 env 名称字段（如 `github_token_env`）
- `config.yaml` 中的明文字段（不推荐）

3) 推荐的 Secrets 名称（默认）
- `STARRED_GITHUB_TOKEN` — GitHub PAT（用于读取星标仓库与 Copilot Models）
- `OPENROUTER_API_KEY` — OpenRouter API Key（可选）
- `GEMINI_API_KEY` — Google Gemini API Key（可选）

4) `config.yaml` 中常用字段示例（位于仓库根目录）：

```yaml
language: zh
model_choice: copilot    # 可选: copilot | openrouter | gemini
github_username: 0      # 0 表示使用 workflow 账号(GITHUB_ACTOR)
default_gemini_model: models/gemini-pro
STARRED_GITHUB_TOKEN: ""   # 可做回退（不推荐）
OPENROUTER_API_KEY: ""
GEMINI_API_KEY: ""
max_workers: 1
batch_size: 5
request_timeout: 60
```

5) 注意：脚本里也支持通过如下 env 覆盖模型：
- `GITHUB_COPILOT_MODEL` 覆盖 Copilot 模型
- `GEMINI_MODEL` 覆盖 Gemini 模型

## 🚀 快速开始（本地 / CI）

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/<your>/myGitStar.git
cd myGitStar
```

2. 安装依赖（包含 PyYAML）：

```bash
pip install requests pyyaml
```

3. 在 GitHub 仓库 Settings → Secrets and variables → Actions 添加 Secrets：

- `STARRED_GITHUB_TOKEN`（必需）
- `OPENROUTER_API_KEY`（可选）
- `GEMINI_API_KEY`（可选）

4. 在 Actions workflow 中注入 env（示例）：

```yaml
env:
  STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

5. 本地运行：

```bash
export STARRED_GITHUB_TOKEN="ghp_xxx"
python scripts/summarize_stars.py
```

## 🤖 使用 Gemini

- 要启用 Gemini，请在 `config.yaml` 中将 `model_choice: gemini`，并在运行环境提供 `GEMINI_API_KEY`。
- 可通过 `default_gemini_model` 或环境变量 `GEMINI_MODEL` 指定模型。脚本对常见的 Gemini 返回结构做了兼容解析，但若 Google 更新了 API，可能需要调整解析逻辑。

## ✅ 运行与故障排查要点

- 若遇 429，增大 `rate_limit_delay` 或减小 `batch_size` / `max_workers`。
- 验证 Secrets 是否正确注入：在 Actions 中可以临时打印前缀（脚本会输出前6位前缀以便调试）。
- 遇到 Gemini 解析异常，请查看脚本日志（有请求/响应打印）并调整 `default_gemini_model`。

## ⚙️ Action 示例（精简）

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

## 📄 结语

已将配置格式迁移为 `config.yaml`（支持注释），脚本仍兼容 `config.json` 作为回退。建议在 CI 中使用 Secrets 而非在仓库保留明文密钥。

如需我同步 README 或 Actions workflow，我可以继续更新。
