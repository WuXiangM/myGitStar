import time
from typing import Any, Dict, List, Optional

import requests

from scripts.core.throttle import SimpleThrottle


def get_starred_repos(
    github_token: str,
    github_username: str,
    throttle: Optional[SimpleThrottle] = None,
    timeout: float = 30.0,
    max_repos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not github_token:
        raise ValueError("缺少 GITHUB_TOKEN 环境变量")

    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"https://api.github.com/users/{github_username}/starred?per_page={per_page}&page={page}"
        try:
            if throttle:
                try:
                    throttle.wait()
                except Exception:
                    pass
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            repos.extend(data)
            print(f"已获取 {len(repos)} 个仓库... (第 {page} 页)")
            page += 1
            time.sleep(1)

            if max_repos and max_repos > 0 and len(repos) >= max_repos:
                repos = repos[:max_repos]
                break

        except requests.RequestException as e:
            print(f"获取星标仓库失败: {e}")
            break

    print(f"总共获取到 {len(repos)} 个星标仓库")
    return repos
