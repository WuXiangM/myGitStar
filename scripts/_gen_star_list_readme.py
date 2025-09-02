# 无法通过api获取starlist
import os
import requests

USERNAME = "WuXiangM"
TOKEN = os.environ.get("STARRED_GITHUB_TOKEN")
API_BASE = "https://api.github.com"

if not TOKEN:
    raise RuntimeError("环境变量 STARRED_GITHUB_TOKEN 未设置")

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

def get_lists():
    url = f"{API_BASE}/users/{USERNAME}/starred/lists"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_list_repos(list_id):
    url = f"{API_BASE}/users/{USERNAME}/starred/lists/{list_id}/repos"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def main():
    lists = get_lists()
    readme = ["# 我的 GitHub 收藏夹整理\n"]
    for lst in lists:
        list_id = lst['id']
        list_name = lst['name']
        readme.append(f"## {list_name}\n")
        repos = get_list_repos(list_id)
        if not repos:
            readme.append("_（收藏夹为空）_\n")
        else:
            for repo in repos:
                desc = repo['description'] or ""
                readme.append(f"- [{repo['full_name']}]({repo['html_url']}) ⭐ {repo['stargazers_count']}<br>{desc}\n")
        readme.append("\n")
    with open("README.md", "w", encoding="utf-8") as f:
        f.writelines(readme)
    print("README.md 已生成")

if __name__ == "__main__":
    main()
