# 🌟 GitHub Stars AI 自动总结系统 — 中文使用指南

本项目会拉取指定 GitHub 账号的星标仓库（Stars），调用 AI 生成结构化 Markdown 总结，并输出为 README 文档，适合本地手动运行或在 GitHub Actions 中定时自动更新。

## 1) 🎁 你会得到什么（产物）

- 📗 README（内容分类）：`README.md`
- 📘 README classified by language（英文）：`README_lang.md`
- 📙 README 按语言分类（中文）：`README_lang_cn.md`

> 说明：按语言分类的输出文件名可由 `config.yaml` 的 `readme_sum_path` 或命令行 `--out` 覆盖；内容分类版由 `classify_stars_by_content.py` 生成到 `README.md`。

## 2) 📂 目录结构（关键文件）

```
myGitStar/
├── scripts/summarize_stars.py     # 主脚本
├── config.yaml                   # 配置
├── requirements.txt              # Python 依赖
├── README.md                     # 内容分类输出（主 README）
├── README_lang.md                # 按语言分类（英文）
├── README_lang_cn.md             # 按语言分类（中文）
├── GUIDE_en.md                   # 英文指南
└── GUIDE_zh.md                   # 中文指南（本文件）
```

## 3) 🚀 快速开始

### 3.1 🤖 GitHub Actions 运行（推荐）

在仓库 Settings → Secrets and variables → Actions 中添加：

- ✅ `STARRED_GITHUB_TOKEN`（必需）
- ➕ `OPENROUTER_API_KEY`（可选）
- ➕ `GEMINI_API_KEY`（可选）

工作流里将 secrets 注入为环境变量即可运行（仓库里已有 workflow 可参考）。

### 3.2 💻 本地运行（Windows PowerShell）

1. 安装依赖（建议使用虚拟环境）：

```powershell
pip install -r requirements.txt
```

2. 设置 Token（至少需要 GitHub Token；用于拉取星标列表 + Copilot Models 时也会用到）：

```powershell
$env:STARRED_GITHUB_TOKEN = "ghp_xxx"
```

3. 运行脚本：

```powershell
python scripts/summarize_stars.py --language en --out README_lang.md
python scripts/summarize_stars.py --language zh --out README_lang_cn.md

# 生成内容分类版（从按语言分类的 README 解析仓库列表）
python scripts/classify_stars_by_content.py --from-readme README_lang.md --out-md README.md --out-json repo_categories.json
```

## 4) ⚙️ 配置说明（config.yaml）

脚本优先读取 `config.yaml`（支持注释）

下面是最常用、最容易踩坑的字段：

```yaml
# 输出语言：zh 或 en
language: en

# 选择 AI 引擎：copilot / openrouter / gemini
model_choice: copilot

# 生成文件路径：建议 en->README.md，zh->README2.md
# 生成文件路径（按语言分类版）：建议 en->README_lang.md，zh->README_lang_cn.md
readme_sum_path: README_lang.md

# 更新模式：all（全量重写）/ missing_only（仅补全缺失或无效的总结）
update_mode: missing_only

# 并发与节流（遇到 429 建议调小并发、增大 delay）
max_workers: 1
batch_size: 4
rate_limit_delay: 5
request_timeout: 30

# 是否在 README 顶部显示中英互跳链接，以及链接顺序
# true：优先显示与 language 一致的语言在前；false：反过来
repo_display_language: true
```

### 4.1 📌 配置字段速查表（含是否必需）

下表汇总 `config.yaml` 中可用的配置字段（含常用字段），并标注“是否必需”。

| 字段                         | 必需修改                 | 类型/示例                                 | 作用                                                                           | 建议                                                                  |
| ---------------------------- | ------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `github_username`          | 是（账号名）             | `0` / `WuXiangM`                      | 拉取谁的 Stars。`0` 表示用 Actions 的 `GITHUB_ACTOR`；本地建议直接填用户名 | CI 用 `0`；本地填固定用户名更稳定                                   |
| `STARRED_GITHUB_TOKEN`     | 是（必需，建议用 env）   | `""`                                    | GitHub Token（拉取 Stars + Copilot Models 会用到）                             | 不建议写入 config，改用 Actions Secrets / 环境变量                    |
| `OPENROUTER_API_KEY`       | 条件（ openrouter 必需） | `""`                                    | OpenRouter Key（仅 `model_choice: openrouter` 时需要）                       | 建议用 env                                                            |
| `GEMINI_API_KEY`           | 条件（ gemini 必需）     | `""`                                    | Gemini Key（仅 `model_choice: gemini` 时需要）                               | 建议用 env                                                            |
| `language`                 | 否                       | `zh` / `en`                           | 输出语言                                                                       | 不填默认 `zh`                                                       |
| `model_choice`             | 否                       | `copilot` / `openrouter` / `gemini` | 选择 AI 引擎                                                                   | 不填默认 `copilot`                                                  |
| `readme_sum_path`          | 否                       | `README_lang.md` / `README_lang_cn.md` | 按语言分类 README 输出路径                                                     | en→`README_lang.md`；zh→`README_lang_cn.md`（不填则默认 `README-sum.md`） |
| `update_mode`              | 否                       | `all` / `missing_only`                | 更新策略：全量重写 / 仅补全缺失或无效总结                                      | 想稳定增量更新就用 `missing_only`                                   |
| `repo_display_language`    | 否                       | `true` / `false`                      | README 顶部中英互跳链接的显示顺序                                              | `true`：与 `language` 一致的链接在前                              |
| `default_copilot_model`    | 否                       | `openai/gpt-4o-mini`                    | Copilot Models 默认模型名                                                      | 也可用环境变量覆盖（见下）                                            |
| `default_openrouter_model` | 否                       | `deepseek/deepseek-prover-v2:free`      | OpenRouter 默认模型名                                                          | 选你账号可用的模型                                                    |
| `default_gemini_model`     | 否                       | `gemini-2.0-flash`                      | Gemini 默认模型名                                                              | 也可用环境变量覆盖（见下）                                            |
| `max_repos`                | 否                       | `50` / `0`                            | 限制每次运行最多处理多少个仓库；`0` 或留空表示不限制                         | CI 建议 20~80；本地可 `0`                                           |
| `max_workers`              | 否                       | `1`                                     | 线程池并发数（影响并发请求）                                                   | 为避免限流，建议先从 `1` 开始                                       |
| `batch_size`               | 否                       | `4`                                     | 每批处理仓库数（批内会发起总结请求）                                           | 限流/超时就调小                                                       |
| `request_timeout`          | 否                       | `30`                                    | HTTP 超时时间（秒）                                                            | 网络不稳就调大                                                        |
| `rate_limit_delay`         | 否                       | `5`                                     | 批次之间 sleep（秒）                                                           | 429 就调大                                                            |
| `request_retry_delay`      | 否                       | `2`                                     | 网络层失败后的重试间隔（秒）                                                   | 2~10                                                                  |
| `retry_attempts`           | 否                       | `1`                                     | 网络层重试次数（适用于通用请求封装）                                           | 1~3                                                                   |
| `global_qps`               | 否                       | `0.5`                                   | 全局节流（QPS），默认 0.5 表示约每 2 秒 1 次请求                               | 429 就降低（如 0.2）                                                  |
| `test_first_repo`          | 否                       | `false`                                 | 调试开关：只处理第一个仓库（也会开启更详细日志）                               | 本地排错用 `true`                                                   |
| `log_file`                 | 否                       | `scripts/summarize_stars.log`           | 日志文件路径（默认写到脚本目录）                                               | CI 可保持默认                                                         |
| `log_max_bytes`            | 否                       | `5242880`                               | 单个日志文件最大大小（字节），到达后滚动                                       | 默认即可                                                              |
| `log_backup_count`         | 否                       | `3`                                     | 日志滚动保留的历史份数                                                         | 默认即可                                                              |

#### Gemini 相关（仅 `model_choice: gemini` 时有用）

| 字段                          | 必需 | 类型/示例 | 作用                                             | 建议                |
| ----------------------------- | ---- | --------- | ------------------------------------------------ | ------------------- |
| `gemini_temperature`        | 否   | `0.4`   | 生成温度（越大越发散）                           | 0.2~0.6             |
| `gemini_max_output_tokens`  | 否   | `2000`  | 输出 token 上限（过小可能被截断）                | 缺字段/被截断就调大 |
| `gemini_generation_retries` | 否   | `1`     | Gemini 生成层面的重试次数（如被截断/返回不完整） | 1~3                 |
| `gemini_retry_backoff`      | 否   | `2`     | Gemini 生成重试的退避倍数                        | 2~5                 |
| `gemini_retry_attempts`     | 否   | `1`     | Gemini 网络请求重试次数                          | 1~3                 |
| `gemini_retry_delay`        | 否   | `2`     | Gemini 网络请求重试间隔（秒）                    | 2~10                |

### 4.2 📌 环境变量可覆盖项（CI 常用）

- `MAX_REPOS`：优先级高于 `config.yaml` 的 `max_repos`，用于 CI 临时限量，避免超时。
- `GITHUB_COPILOT_MODEL`：覆盖 `default_copilot_model`。
- `GEMINI_MODEL`：覆盖 `default_gemini_model`。

### 4.3 📌 兼容旧配置的字段（一般不必写）

脚本兼容一些旧写法，用于“在 config 里指定环境变量名”或“直接写小写明文 key”（不推荐）：

- `github_token_env`：指定 GitHub Token 的环境变量名（默认推荐直接用 `STARRED_GITHUB_TOKEN`）。
- `openrouter_api_key_env`：指定 OpenRouter Key 的环境变量名。
- `gemini_api_key_env`：指定 Gemini Key 的环境变量名。
- `github_token` / `openrouter_api_key` / `gemini_api_key`：明文 key（不推荐，避免提交到仓库）。

### 4.4  📌不推荐写入配置文件的密钥字段（仅作回退）

`config.yaml` 里也允许出现以下字段（仓库当前配置文件里就有），脚本会读取它们作为回退，但强烈建议保持为空，改用 Actions Secrets / 本地环境变量：

- `STARRED_GITHUB_TOKEN`
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`

## 5) 🔒Secrets / 环境变量（Keys & Token）

推荐永远用 CI Secrets 或本地环境变量提供密钥，不要把真实密钥提交到仓库。

### 推荐的环境变量名

| 环境变量                 | 用途                                                   |
| ------------------------ | ------------------------------------------------------ |
| `STARRED_GITHUB_TOKEN` | GitHub Token（拉取星标仓库 + Copilot Models 时会用到） |
| `OPENROUTER_API_KEY`   | OpenRouter（可选）                                     |
| `GEMINI_API_KEY`       | Gemini（可选）                                         |

### 读取优先级（从高到低）

1. 环境变量（Actions Secrets / 本地 `$env:...`）
2. `config.yaml` 中的同名大写键（不推荐保存真实值，仅作回退）
3. `config.yaml` 中的 `*_env` 字段指定的 env 名称（兼容旧配置）
4. 明文字段（不推荐）

## 6) 🧯 常见问题（排查清单）

- **出现 429 / 限流**：降低 `batch_size`、保持 `max_workers: 1`、增大 `rate_limit_delay`。
- **输出文件不更新**：确认 `update_mode`；`missing_only` 模式只会更新“缺失/无效”的仓库总结。
- **生成到了意料之外的文件**：检查 `readme_sum_path` 是否指向了你想要的文件。
- **本地运行找不到依赖**：用 `pip install -r requirements.txt`，或确认你在正确的 venv 里运行。

## 7) 🧱 一个最小可用的 Actions 片段（参考）

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
