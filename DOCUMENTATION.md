# GitHub Stars AI 自动总结系统文档

## 📋 目录

- [系统概述](#-系统概述)
- [功能特性](#-功能特性)
- [文件结构](#-文件结构)
- [环境配置](#-环境配置)
- [使用方法](#-使用方法)
- [API 配置](#-api-配置)
- [GitHub Actions 工作流](#-github-actions-工作流)
- [输出格式](#-输出格式)
- [故障排除](#-故障排除)
- [自定义配置](#-自定义配置)

## 🎯 系统概述

这是一个自动化的 GitHub Stars 仓库总结系统，能够：

- 📊 **自动获取**您的 GitHub 星标仓库列表
- 🤖 **AI 智能总结**每个仓库的功能和特点
- 📝 **生成文档**格式化的 Markdown 总结文档
- ⏰ **定时执行**通过 GitHub Actions 每日自动更新
- 🔄 **增量更新**仅对新增或变更的仓库重新总结

### 支持的 AI 模型

1. **GitHub Copilot API**（默认）
   - 模型：`openai/gpt-4o-mini`
   - 需要：GitHub Token 具备 Copilot 访问权限

2. **OpenRouter API**（备选）
   - 模型：`deepseek/deepseek-prover-v2:free`
   - 需要：OpenRouter API Key

## ✨ 功能特性

### 🚀 核心功能
- **智能分类**：按编程语言自动分类仓库
- **元数据展示**：显示 Stars、Forks、更新时间等信息
- **语言图标**：为不同编程语言添加 Emoji 图标
- **增量处理**：智能检测已有总结，避免重复请求
- **容错机制**：网络异常时使用缓存的旧总结

### 🛡️ 稳定性保障
- **速率限制**：智能控制 API 请求频率，避免 429 错误
- **重试机制**：遇到网络问题自动重试（最多3次）
- **并发控制**：限制同时请求数量，保护 API 配额
- **错误处理**：完善的异常处理和日志记录

## 📂 文件结构

```
myGitStar/
├── scripts/
│   └── summarize_stars.py          # 主要 Python 脚本
├── .github/
│   └── workflows/
│       └── update_myGitStar_sum.yml # GitHub Actions 工作流
├── README-sum.md                    # 生成的总结文档
├── README.md                        # 项目说明
└── DOCUMENTATION.md                 # 本文档
```

## 🔧 环境配置

### 必需的环境变量

| 变量名 | 描述 | 获取方式 |
|--------|------|----------|
| `STARRED_GITHUB_TOKEN` | GitHub 个人访问令牌 | [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens) |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥（可选） | [OpenRouter 官网](https://openrouter.ai/) |

### GitHub Token 权限要求

创建 GitHub Token 时需要以下权限：
- ✅ `public_repo` - 访问公共仓库
- ✅ `read:user` - 读取用户信息
- ✅ `copilot` - 访问 GitHub Copilot API（如果使用 Copilot）

### 可选环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `USE_COPILOT_API` | 是否使用 GitHub Copilot API | `true` |
| `GITHUB_COPILOT_MODEL` | Copilot 模型名称 | `openai/gpt-4o-mini` |

## 🚀 使用方法

### 本地运行

1. **克隆仓库**
   ```bash
   git clone https://github.com/你的用户名/myGitStar.git
   cd myGitStar
   ```

2. **安装依赖**
   ```bash
   pip install requests openai
   ```

3. **配置环境变量**
   ```bash
   # Windows PowerShell
   $env:STARRED_GITHUB_TOKEN="你的GitHub令牌"
   $env:OPENROUTER_API_KEY="你的OpenRouter密钥"  # 可选
   
   # Linux/Mac
   export STARRED_GITHUB_TOKEN="你的GitHub令牌"
   export OPENROUTER_API_KEY="你的OpenRouter密钥"  # 可选
   ```

4. **运行脚本**
   ```bash
   python scripts/summarize_stars.py
   ```

### GitHub Actions 自动运行

系统会在以下情况下自动运行：
- 🕐 **每天 4:00 AM UTC**（北京时间 12:00 PM）
- 🔄 **手动触发**（通过 GitHub Actions 页面）

## 🤖 API 配置

### GitHub Copilot API（推荐）

**优势：**
- 🆓 免费使用（需要 GitHub Copilot 订阅）
- 🚀 响应速度快
- 🔒 无需额外的 API Key

**配置：**
```python
# 使用默认 Copilot 配置
USE_COPILOT_API=true
GITHUB_COPILOT_MODEL=openai/gpt-4o-mini
```

**API 端点：**
```
https://models.github.ai/inference/chat/completions
```

### OpenRouter API（备选）

**优势：**
- 🌟 支持多种开源模型
- 💰 有免费额度
- 🔄 作为备选方案

**配置：**
```python
# 切换到 OpenRouter
USE_COPILOT_API=false
```

**支持的模型：**
- `deepseek/deepseek-prover-v2:free`
- 其他 OpenRouter 支持的模型

## ⚙️ GitHub Actions 工作流

### 工作流程详解

```yaml
name: Update myGitStar Summaries

on:
  schedule:
    - cron: '0 4 * * *'        # 每日 4:00 UTC 执行
  workflow_dispatch:           # 支持手动触发

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4        # 检出代码
      
      - name: Set up Python             # 设置 Python 环境
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'
          
      - name: Install dependencies      # 安装依赖
        run: |
          pip install requests
          pip install openai
          
      - name: Summarize starred repos   # 执行总结脚本
        env:
          STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY}}
        run: python scripts/summarize_stars.py
        
      - name: Commit and push changes   # 提交更新
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add README-sum.md
          git commit -m "update AI summarized stars [bot]" || echo "No changes to commit"
          git push origin HEAD:main
```

### 配置 GitHub Secrets

在仓库设置中添加以下 Secrets：

1. 进入 `Settings > Secrets and variables > Actions`
2. 点击 `New repository secret`
3. 添加以下 Secrets：
   - `STARRED_GITHUB_TOKEN`：您的 GitHub 个人访问令牌
   - `OPENROUTER_API_KEY`：OpenRouter API 密钥（可选）

## 📄 输出格式

生成的 `README-sum.md` 文件包含以下部分：

### 1. 文档头部
```markdown
# 我的 GitHub Star 项目AI总结

**生成时间：** 2024年12月25日
**AI模型：** GitHub Copilot
**总仓库数：** 150 个
```

### 2. 目录导航
- 按语言分类的快速导航
- 显示每种语言的仓库数量

### 3. 按语言分组的仓库列表

每个仓库条目包含：
- 📌 **仓库标题**：包含链接
- ⭐ **元数据**：Stars、Forks、更新时间
- 🤖 **AI 总结**：结构化的内容分析

### 4. 统计信息
- 总仓库数、编程语言数
- 生成时间和使用的 AI 模型

## 🔧 故障排除

### 常见问题

#### 1. 429 错误（请求过于频繁）
**症状：** API 返回 429 状态码
**解决方案：**
- 系统已内置重试机制（3次重试，30秒间隔）
- 调整 `RATE_LIMIT_DELAY` 参数增加请求间隔
- 减少 `MAX_WORKERS` 并发数

#### 2. GitHub Token 权限不足
**症状：** 403 错误或无法访问仓库
**解决方案：**
- 确保 Token 具备必要权限
- 重新生成 Token 并更新 Secrets

#### 3. AI 总结生成失败
**症状：** 显示 "API生成失败" 或空白总结
**解决方案：**
- 检查 API Key 是否正确
- 确认网络连接正常
- 系统会自动使用缓存的旧总结

#### 4. 工作流执行失败
**症状：** GitHub Actions 显示红色失败状态
**解决方案：**
- 检查 Actions 日志查看具体错误
- 确认 Secrets 配置正确
- 手动触发工作流进行测试

### 日志和调试

脚本运行时会输出详细日志：
```
OpenRouter API Key 前缀: sk-or-v1...
GitHub Token 前缀: ghp_1234...
正在获取星标仓库...
已获取 100 个仓库... (第 1 页)
总共获取到 150 个星标仓库
正在处理 Python 类型的仓库（共45个）...
处理批次 1，包含 5 个仓库...
已处理 45/150 个仓库
✅ README-sum.md 已生成，共处理了 150 个仓库。
```

## ⚡ 自定义配置

### 修改配置参数

在 `summarize_stars.py` 中可以调整以下参数：

```python
# 性能配置
MAX_WORKERS = 3              # 并发线程数
BATCH_SIZE = 5               # 批处理大小
RATE_LIMIT_DELAY = 10        # 请求间隔（秒）
REQUEST_RETRY_DELAY = 30     # 重试延迟（秒）

# 用户配置
GITHUB_USERNAME = "你的用户名"  # 修改为您的 GitHub 用户名
README_SUM_PATH = "README-sum.md"  # 输出文件路径
```

### 自定义 AI 总结格式

修改 `prompt` 变量来自定义 AI 总结的格式：

```python
prompt = (
    f"请对以下 GitHub 仓库进行内容总结，按如下格式输出（用中文）：\n"
    f"**仓库名称：** {repo_name}\n\n"
    f"**简要介绍：** （50字以内）\n\n"
    f"**创新点：** （简述本仓库最有特色的地方）\n\n"
    f"**简单用法：** （给出最简关键用法或调用示例，如无则略）\n\n"
    f"**总结：** （一句话总结它的用途/价值）\n\n"
    # 可以在这里添加更多自定义字段
)
```

### 添加更多编程语言图标

在 `lang_icon` 字典中添加新的语言图标：

```python
lang_icon = {
    "Python": "🐍", "JavaScript": "🟨", "TypeScript": "🔷",
    "Scala": "🔴", "Clojure": "🟢", "Haskell": "🟣",
    # 添加更多语言图标
}
```

## 📞 技术支持

如果您遇到问题或有改进建议，请：

1. 📋 查看 [Issues 页面](https://github.com/你的用户名/myGitStar/issues)
2. 🐛 提交新的 Issue 描述问题
3. 💡 提出功能改进建议
4. 🤝 提交 Pull Request 贡献代码

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

**更新时间：** 2024年12月25日  
**版本：** v1.0.0  
**维护者：** [您的用户名](https://github.com/您的用户名)
