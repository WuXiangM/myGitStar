import requests
import os
import time
import json

GITHUB_USERNAME = "WuXiangM"
GITHUB_TOKEN = os.environ.get("STARRED_GITHUB_TOKEN")
OPENROUTER_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_starred_repos():
    print("Fetching starred repositories...")
    repos = []
    page = 1
    per_page = 100
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    while True:
        url = f"https://api.github.com/users/{GITHUB_USERNAME}/starred?per_page={per_page}&page={page}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    print(f"Total starred repos: {len(repos)}")
    return repos

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
        "Content-Type": "application/json",
        # 可选参数
        #"HTTP-Referer": "<YOUR_SITE_URL>",
        #"X-Title": "<YOUR_SITE_NAME>",
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
        result = resp.json()
        summary = result["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        print(f"OpenRouter API 调用失败: {e}")
        return "API生成失败"
    
def classify_by_language(repos):
    classified = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified

def main():
    starred = get_starred_repos()
    classified = classify_by_language(starred)
    lines = ["# 我的 GitHub Star 项目AI总结（由 DeepSeek API 自动生成）\n"]
    for lang, repos in sorted(classified.items(), key=lambda x: -len(x[1])):
        lines.append(f"\n## {lang}（共{len(repos)}个）\n")
        for repo in repos:
            url = repo["html_url"]
            summary = openrouter_summarize(repo)
            lines.append(f"### [{repo['full_name']}]({url})\n")
            lines.append(summary)
            lines.append("\n")
            # OpenRouter 免费接口有限制，建议适当等待，防止被限流
            time.sleep(3)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    print("README.md 已生成。")

if __name__ == "__main__":
    main()
