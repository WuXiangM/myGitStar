# 🌟 GitHub星标仓库AI总结系统 - 完整说明文档

## 📋 项目概述

这是一个自动化系统，用于获取GitHub用户的星标仓库并使用AI进行智能总结，生成结构化的Markdown文档。系统支持两种AI服务（GitHub Copilot和OpenRouter），具备完善的错误处理和自动化部署功能。

### ✨ 核心功能

- 🔍 **智能仓库获取**：自动获取用户的所有GitHub星标仓库
- 🤖 **双AI引擎支持**：GitHub Copilot API 和 OpenRouter API
- 📊 **智能分类展示**：按编程语言自动分类和组织
- 🎨 **美化文档输出**：带emoji图标和丰富格式的Markdown
- ⚡ **GitHub Actions自动化**：定时更新，无需手动干预
- 🔄 **智能缓存机制**：避免重复处理，提高效率
- 🛡️ **健壮错误处理**：429错误重试、网络异常恢复

## 🚀 快速开始指南

### 环境要求

- Python 3.7+
- GitHub账户
- GitHub个人访问令牌
- 可选：OpenRouter API密钥

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/WuXiangM/myGitStar.git
   cd myGitStar
   ```

2. **安装依赖**
   ```bash
   pip install requests openai
   ```

3. **配置环境变量**
   ```bash
   # 必需配置
   export STARRED_GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
   
   # 可选配置
   export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
   export USE_COPILOT_API="true"
   export GITHUB_COPILOT_MODEL="openai/gpt-4o-mini"
   ```

4. **运行脚本**
   ```bash
   python scripts/summarize_stars.py
   ```

## 🔧 详细配置说明

### GitHub Token配置

#### 创建Personal Access Token

1. 访问 [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. 点击 "Generate new token" > "Fine-grained personal access token"
3. 配置以下权限：

**必需权限：**
- `contents:read` - 读取仓库内容
- `metadata:read` - 读取仓库元数据
- `public_repo` - 访问公共仓库

**可选权限（使用Copilot API时）：**
- `copilot` - 访问GitHub Copilot API

#### Token验证
```bash
# 验证token有效性
curl -H "Authorization: Bearer $STARRED_GITHUB_TOKEN" \
     https://api.github.com/user
```

### API服务配置

#### 方案一：GitHub Copilot API（推荐）

**优势：**
- ✅ 与GitHub深度集成
- ✅ 响应速度快
- ✅ 对部分用户免费
- ✅ 无需额外API密钥

**配置：**
```bash
export USE_COPILOT_API="true"
export GITHUB_COPILOT_MODEL="openai/gpt-4o-mini"  # 可选
```

**支持的模型：**
- `openai/gpt-4o-mini`（默认，推荐）
- `openai/gpt-4o`
- `openai/gpt-3.5-turbo`

#### 方案二：OpenRouter API

**优势：**
- ✅ 支持多种AI模型
- ✅ 包含免费模型选项
- ✅ 灵活的定价方案

**配置：**
```bash
export USE_COPILOT_API="false"
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
```

**免费模型推荐：**
- `deepseek/deepseek-prover-v2:free`（默认）
- `google/gemma-2-9b-it:free`
- `meta-llama/llama-3.1-8b-instruct:free`

### 性能调优参数

脚本中的关键配置常量：

```python
# 并发控制
MAX_WORKERS = 3          # 并发线程数，避免API限流
BATCH_SIZE = 5           # 每批处理的仓库数量

# 延迟控制
RATE_LIMIT_DELAY = 10    # API调用间隔（秒）
REQUEST_RETRY_DELAY = 30 # 429错误重试延迟（秒）
REQUEST_TIMEOUT = 60     # 单次请求超时（秒）
```

**调优建议：**
- 遇到429错误频繁时，降低`MAX_WORKERS`到1-2
- 网络较慢时，增加`REQUEST_TIMEOUT`到120
- 仓库数量很多时，适当增加`BATCH_SIZE`到10

## 🤖 GitHub Actions自动化

### 工作流文件解析

`.github/workflows/update_myGitStar_sum.yml` 文件配置：

```yaml
name: Update myGitStar Summaries

on:
  schedule:
    - cron: '0 4 * * *'    # 每天凌晨4点运行
  workflow_dispatch:       # 支持手动触发

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'
      
      - name: Install dependencies
        run: |
          pip install requests
          pip install openai
      
      - name: Summarize starred repos
        env:
          STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: python scripts/summarize_stars.py
      
      - name: Commit and push changes
        env:
          STARRED_GITHUB_TOKEN: ${{ secrets.STARRED_GITHUB_TOKEN }}
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add README-sum.md
          git commit -m "update AI summarized stars [bot]" || echo "No changes to commit"
          git remote set-url origin https://x-access-token:${STARRED_GITHUB_TOKEN}@github.com/${{ github.repository }}.git
          git pull --rebase origin main || echo "No remote changes to pull"
          git push origin HEAD:main
```

### Secrets配置

在GitHub仓库的 `Settings > Secrets and variables > Actions` 中添加：

| Secret名称 | 必需性 | 说明 |
|-----------|--------|------|
| `STARRED_GITHUB_TOKEN` | 必需 | GitHub个人访问令牌 |
| `OPENROUTER_API_KEY` | 可选 | OpenRouter API密钥 |

### 自定义运行时间

修改cron表达式来自定义运行时间：

```yaml
schedule:
  - cron: '0 8 * * 1'    # 每周一上午8点
  - cron: '0 0 1 * *'    # 每月1号午夜
  - cron: '0 */6 * * *'  # 每6小时一次
```

## 📊 输出文档结构

生成的 `README-sum.md` 文档包含以下结构：

### 1. 文档头部
```markdown
# 我的 GitHub Star 项目AI总结

**生成时间：** 2024年01月01日
**AI模型：** GitHub Copilot
**总仓库数：** 150 个
```

### 2. 目录导航
```markdown
## 📖 目录

- [Python](#-python)（25个）
- [JavaScript](#-javascript)（20个）
- [TypeScript](#-typescript)（15个）
- [Go](#-go)（10个）
- [Other](#-other)（80个）
```

### 3. 分类仓库列表
```markdown
## 🐍 Python（共25个）

### 📌 [用户名/仓库名](https://github.com/用户名/仓库名)

**⭐ Stars:** 1,234 | **🍴 Forks:** 567 | **📅 更新:** 2024-01-01

**仓库名称：** 项目的完整名称

**简要介绍：** 50字以内的项目简介

**创新点：** 项目最有特色的功能或优势

**简单用法：** 基本使用方法或安装命令

**总结：** 一句话总结项目的核心价值

---
```

### 4. 统计信息
```markdown
## 📊 统计信息

- **总仓库数：** 150 个
- **编程语言数：** 12 种
- **生成时间：** 2024年01月01日
- **AI模型：** GitHub Copilot

---

*本文档由AI自动生成，如有错误请以原仓库信息为准。*
```

### 支持的编程语言图标

| 语言 | 图标 | 语言 | 图标 |
|------|------|------|------|
| Python | 🐍 | JavaScript | 🟨 |
| TypeScript | 🔷 | Java | ☕ |
| Go | 🐹 | Rust | 🦀 |
| C++ | ⚡ | C | 🔧 |
| C# | 💜 | PHP | 🐘 |
| Ruby | 💎 | Swift | 🐦 |
| Kotlin | 🅺 | Dart | 🎯 |
| Shell | 🐚 | HTML | 🌐 |
| CSS | 🎨 | Vue | 💚 |
| React | ⚛️ | Other | 📦 |

## 🛠️ 故障排除指南

### 常见问题及解决方案

#### 1. 429 Too Many Requests 错误

**问题描述：** API调用过于频繁，触发速率限制

**解决方案：**
```python
# 在 summarize_stars.py 中调整以下参数
MAX_WORKERS = 1          # 降低并发数
BATCH_SIZE = 3           # 减小批次大小
RATE_LIMIT_DELAY = 15    # 增加延迟时间
```

**额外措施：**
- 错开运行时间，避免高峰期
- 使用不同的API密钥轮换
- 考虑升级API计划

#### 2. GitHub Token权限不足

**错误信息：** `401 Unauthorized` 或 `403 Forbidden`

**检查清单：**
- ✅ Token是否正确设置且未过期
- ✅ 用户名 `GITHUB_USERNAME` 是否正确
- ✅ Token权限是否包含 `contents:read`、`metadata:read`
- ✅ 如使用Copilot API，是否包含 `copilot` 权限

**验证方法：**
```bash
# 测试token有效性
curl -H "Authorization: Bearer $STARRED_GITHUB_TOKEN" \
     https://api.github.com/user/starred?per_page=1
```

#### 3. Copilot API访问被拒绝

**错误信息：** `403 Forbidden` 或 `Access denied`

**可能原因：**
- 账户没有GitHub Copilot订阅
- Token缺少copilot权限
- 请求格式不正确

**解决方案：**
```bash
# 切换到OpenRouter API
export USE_COPILOT_API="false"
export OPENROUTER_API_KEY="your_openrouter_key"
```

#### 4. 网络连接问题

**症状：** 连接超时、DNS解析失败

**解决方案：**
```python
# 增加超时时间
REQUEST_TIMEOUT = 120

# 添加代理设置（如需要）
proxies = {
    'http': 'http://proxy.example.com:8080',
    'https': 'https://proxy.example.com:8080'
}
```

#### 5. 文件编码问题

**错误信息：** `UnicodeDecodeError` 或乱码

**解决方案：**
```python
# 确保使用UTF-8编码
with open(README_SUM_PATH, "w", encoding="utf-8") as f:
    f.write(''.join(lines))
```

**Windows用户额外配置：**
```bash
# 设置控制台编码
chcp 65001
set PYTHONIOENCODING=utf-8
```

### 调试技巧

#### 1. 启用详细日志
```bash
# 方法1：重定向输出
python scripts/summarize_stars.py > output.log 2>&1

# 方法2：实时查看
python scripts/summarize_stars.py | tee output.log
```

#### 2. 测试单个功能模块
```python
# 测试GitHub API连接
repos = get_starred_repos()
print(f"获取到 {len(repos)} 个仓库")

# 测试AI API调用
test_repo = {
    "full_name": "test/repo", 
    "description": "test description", 
    "html_url": "https://github.com/test/repo"
}
result = copilot_summarize(test_repo)
print(result)
```

#### 3. 环境变量验证
```bash
# 检查环境变量
echo "GitHub Token: ${STARRED_GITHUB_TOKEN:0:8}..."
echo "OpenRouter Key: ${OPENROUTER_API_KEY:0:8}..."
echo "Use Copilot: $USE_COPILOT_API"
```

#### 4. GitHub Actions调试
```yaml
# 在workflow中添加调试步骤
- name: Debug environment
  run: |
    echo "Python version: $(python --version)"
    echo "Current directory: $(pwd)"
    echo "Files: $(ls -la)"
    echo "Environment variables:"
    env | grep -E "(GITHUB|TOKEN|API)" | sed 's/=.*/=***/'
```

## 📈 性能优化建议

### 1. API调用优化
- **使用缓存**：脚本自动缓存已处理的仓库，避免重复调用
- **批量处理**：合理设置 `BATCH_SIZE` 平衡速度和稳定性
- **错峰运行**：避开API使用高峰期

### 2. 并发控制
```python
# 保守配置（推荐）
MAX_WORKERS = 3
BATCH_SIZE = 5
RATE_LIMIT_DELAY = 10

# 激进配置（仅在网络和API稳定时使用）
MAX_WORKERS = 5
BATCH_SIZE = 10
RATE_LIMIT_DELAY = 5
```

### 3. 错误恢复机制
- **重试机制**：自动重试失败的API调用
- **降级处理**：API失败时使用缓存的历史数据
- **部分成功**：即使部分仓库处理失败，也保存成功的结果

## 🔮 扩展开发指南

### 添加新的AI服务

1. **实现总结函数**
```python
def custom_ai_summarize(repo: Dict) -> Optional[str]:
    """自定义AI服务总结函数"""
    # 实现你的AI服务调用逻辑
    pass
```

2. **集成到主流程**
```python
# 在 summarize_batch 函数中添加选项
if use_custom_ai:
    summaries = custom_ai_summarize_batch(repos, old_summaries)
```

### 自定义输出格式

修改 `main()` 函数中的文档生成逻辑：

```python
# 自定义仓库条目格式
lines.append(f"### 🎯 [{repo['full_name']}]({url})\n\n")
lines.append(f"**描述：** {repo.get('description', '无描述')}\n\n")
# 添加更多自定义字段
```

### 添加新的分类维度

```python
def classify_by_topic(repos):
    """按主题分类仓库"""
    classified = {}
    for repo in repos:
        # 基于描述或标签进行主题分类
        topic = extract_topic(repo)
        classified.setdefault(topic, []).append(repo)
    return classified
```

## 📞 技术支持

### 社区资源
- **GitHub Issues**：[项目Issue页面](https://github.com/WuXiangM/myGitStar/issues)
- **讨论区**：[GitHub Discussions](https://github.com/WuXiangM/myGitStar/discussions)

### 相关文档
- [GitHub API文档](https://docs.github.com/en/rest)
- [GitHub Copilot API](https://docs.github.com/en/copilot)
- [OpenRouter API文档](https://openrouter.ai/docs)
- [GitHub Actions文档](https://docs.github.com/en/actions)

### 联系方式
- **开发者**：[@WuXiangM](https://github.com/WuXiangM)
- **邮箱**：通过GitHub联系

---

*最后更新：2024年1月 | 版本：2.0 | 作者：[@WuXiangM](https://github.com/WuXiangM)*
