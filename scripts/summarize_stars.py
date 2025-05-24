import os
import time
import requests
import json
import concurrent.futures

GITHUB_USERNAME = "WuXiangM"
GITHUB_TOKEN = os.environ.get("STARRED_GITHUB_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY[:8])

YOUR_SITE_URL = "http://localhost:8088"
YOUR_SITE_NAME = "myGitStar"

README_SUM_PATH = "README-sum.md"

def get_starred_repos():
    print("Fetching starred repositories...")
    repos = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
    }
    while True:
        url = f"https://api.github.com/users/{GITHUB_USERNAME}/starred?per_page={per_page}&page={page}"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    print(f"Total starred repos: {len(repos)}")
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

def openrouter_summarize(repo):
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
        "model": "deepseek/deepseek-prover-v2:free",
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
            data=json.dumps(data)
        )
        if response.status_code == 429:
            raise requests.HTTPError("429 Too Many Requests")
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        return content
    except Exception as e:
        print(f"OpenRouter API 调用失败: {e}")
        return None  # 标记为失败

def summarize_batch(repos, old_summaries):
    results = [None] * len(repos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(openrouter_summarize, repo): idx
            for idx, repo in enumerate(repos)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            repo = repos[idx]
            try:
                summary = future.result()
                if summary is None:  # 429等失败
                    summary = old_summaries.get(repo["full_name"], "API生成失败或429")
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                summary = old_summaries.get(repo["full_name"], "API生成失败")
            results[idx] = summary
    return results

def classify_by_language(repos):
    classified = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified

def main():
    starred = get_starred_repos()
    classified = classify_by_language(starred)
    old_summaries = load_old_summaries()
    lines = ["# 我的 GitHub Star 项目AI总结（由 DeepSeek API 自动生成）\n"]
    for lang, repos in sorted(classified.items(), key=lambda x: -len(x[1])):
        lines.append(f"\n## {lang}（共{len(repos)}个）\n")
        for i in range(0, len(repos), 10):
            this_batch = repos[i:i+10]
            summaries = summarize_batch(this_batch, old_summaries)
            for repo, summary in zip(this_batch, summaries):
                url = repo["html_url"]
                lines.append(f"### [{repo['full_name']}]({url})\n")
                lines.append(summary)
                lines.append("\n")
            # 加一点总的间隔，防止每批过快
            time.sleep(5)
    with open(README_SUM_PATH, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    print(f"{README_SUM_PATH} 已生成。")

if __name__ == "__main__":
    main()
