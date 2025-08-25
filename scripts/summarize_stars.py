import os
import time
import requests
import json
import concurrent.futures
from typing import Dict, List, Optional

# 配置常量
GITHUB_USERNAME = "WuXiangM"
GITHUB_TOKEN = os.environ.get("STARRED_GITHUB_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# API 配置
DEFAULT_COPILOT_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-prover-v2:free"
MAX_WORKERS = 10
BATCH_SIZE = 10
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 5

# 输出配置
README_SUM_PATH = "README-sum.md"

# 打印 API Key 前缀用于调试
if OPENROUTER_API_KEY:
    print(f"OpenRouter API Key 前缀: {OPENROUTER_API_KEY[:8]}...")
if GITHUB_TOKEN:
    print(f"GitHub Token 前缀: {GITHUB_TOKEN[:8]}...")


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
    """读取旧的README-sum.md，返回字典: {repo_full_name: summary}"""
    if not os.path.exists(README_SUM_PATH):
        return {}
    summaries = {}
    current_repo = None
    current_lines = []
    with open(README_SUM_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("### ["):
                if current_repo and current_lines:
                    summaries[current_repo] = "".join(current_lines).strip()
                # 解析仓库名
                left = line.find('[') + 1
                right = line.find(']')
                current_repo = line[left:right]
                current_lines = []
            elif current_repo:
                current_lines.append(line)
        if current_repo and current_lines:
            summaries[current_repo] = "".join(current_lines).strip()
    return summaries


def openrouter_summarize(repo: Dict) -> Optional[str]:
    """使用 OpenRouter API 总结仓库"""
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo["html_url"]

    prompt = (
        f"请对以下 GitHub 仓库进行内容总结，按如下格式输出（用中文）：\n"
        f"1. 仓库名称：{repo_name}\n"
        f"2. 简要介绍：（50字以内）\n"
        f"3. 创新点：（简述本仓库最有特色的地方）\n"
        f"4. 简单用法：（给出最简关键用法或调用示例，如无则略）\n"
        f"5. 总结：（一句话总结它的用途/价值）\n"
        f"仓库描述：{desc}\n"
        f"仓库地址：{url}\n"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": DEFAULT_OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 429:
            raise requests.HTTPError("429 Too Many Requests")
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        return content
    except Exception as e:
        print(f"OpenRouter API 调用失败: {e}")
        return None

# 新增：使用 GitHub Copilot / GitHub Models API 进行总结
# 需要 STARRED_GITHUB_TOKEN 具备访问 models:read & copilot 范围（一般 PAT 启用 copilot 即可）
# 可通过环境变量 GITHUB_COPILOT_MODEL 指定模型，默认 gpt-4o-mini（依据 GitHub Models 可用模型自行调整）
def copilot_summarize(repo: Dict) -> Optional[str]:
    """使用 GitHub Copilot / GitHub Models API 进行总结"""
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo["html_url"]
    model = os.environ.get("GITHUB_COPILOT_MODEL", DEFAULT_COPILOT_MODEL)

    prompt = (
        f"请对以下 GitHub 仓库进行内容总结，按如下格式输出（用中文）：\n"
        f"1. 仓库名称：{repo_name}\n"
        f"2. 简要介绍：（50字以内）\n"
        f"3. 创新点：（简述本仓库最有特色的地方）\n"
        f"4. 简单用法：（给出最简关键用法或调用示例，如无则略）\n"
        f"5. 总结：（一句话总结它的用途/价值）\n"
        f"仓库描述：{desc}\n"
        f"仓库地址：{url}\n"
    )

    if not GITHUB_TOKEN:
        print("缺少 STARRED_GITHUB_TOKEN，无法调用 GitHub Copilot API")
        return None

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/json",
        "X-GitHub-Api-Version": "2023-07-01",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.4
    }

    try:
        resp = requests.post(
            "https://models.github.ai/inference/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 429:
            raise requests.HTTPError("429 Too Many Requests")
        resp.raise_for_status()
        j = resp.json()
        content = j.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip() if content else None
    except Exception as e:
        print(f"GitHub Models API 调用失败: {e}")
        return None


def summarize_batch(repos: List[Dict], old_summaries: Dict[str, str], use_copilot: bool = False) -> List[str]:
    """批量总结仓库，支持选择使用 OpenRouter 或 GitHub Copilot"""
    results = [None] * len(repos)
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
                summary = future.result()
                if summary is None:  # 429等失败
                    summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败或429")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary
    return results


def copilot_summarize_batch(repos: List[Dict], old_summaries: Dict[str, str]) -> List[str]:
    """使用 GitHub Copilot 批量总结仓库"""
    return summarize_batch(repos, old_summaries, use_copilot=True)


def classify_by_language(repos):
    classified = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified

###########################################
def main():
    # 通过环境变量控制使用哪种 API，默认使用 Copilot
    use_copilot_api = os.environ.get("USE_COPILOT_API", "true").lower() == "true"
    api_name = "GitHub Copilot" if use_copilot_api else "OpenRouter (DeepSeek)"
    
    print(f"开始使用 {api_name} 生成 GitHub Star 项目总结...")
    
    try:
        starred = get_starred_repos()
        classified = classify_by_language(starred)
        old_summaries = load_old_summaries()
        
        # 更新标题以反映实际使用的 API
        title = f"# 我的 GitHub Star 项目AI总结（由 {api_name} 自动生成）\n"
        lines = [title]
        
        printed_repos = set()
        printed_langs = set()  # 记录已输出的语言
        
        total_repos = sum(len(repos) for repos in classified.values())
        processed_repos = 0
        
        for lang, repos in sorted(classified.items(), key=lambda x: -len(x[1])):
            if lang in printed_langs:
                continue  # 跳过已输出的语言标题
            printed_langs.add(lang)
            print(f"正在处理 {lang} 类型的仓库（共{len(repos)}个）...")
            lines.append(f"\n## {lang}（共{len(repos)}个）\n")
            
            for i in range(0, len(repos), BATCH_SIZE):
                this_batch = repos[i:i+BATCH_SIZE]
                print(f"处理批次 {i//BATCH_SIZE + 1}，包含 {len(this_batch)} 个仓库...")
                
                # 根据选择使用不同的总结函数
                if use_copilot_api:
                    summaries = copilot_summarize_batch(this_batch, old_summaries)
                else:
                    summaries = summarize_batch(this_batch, old_summaries, use_copilot=False)
                
                for repo, summary in zip(this_batch, summaries):
                    if repo['full_name'] in printed_repos:
                        continue  # 跳过已输出的仓库
                    printed_repos.add(repo['full_name'])
                    url = repo["html_url"]
                    lines.append(f"### [{repo['full_name']}]({url})\n")
                    lines.append(summary)
                    lines.append("\n")
                    processed_repos += 1
                
                print(f"已处理 {processed_repos}/{total_repos} 个仓库")
                time.sleep(RATE_LIMIT_DELAY)  # 避免 API 限流
        
        # 写入文件
        with open(README_SUM_PATH, "w", encoding="utf-8") as f:
            f.write('\n'.join(lines))
        
        print(f"{README_SUM_PATH} 已生成，共处理了 {processed_repos} 个仓库。")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
