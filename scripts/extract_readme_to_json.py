import re
import json
import os

# 定义文件路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(BASE_DIR, 'README.md')
OUTPUT_PATH = os.path.join(BASE_DIR, 'repo_summaries_en.json')

# 加载现有 JSON 文件
def load_existing_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 读取 README.md 内容
with open(README_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 匹配所有分类
category_pattern = re.compile(r'## ([^\n]+) \(Total (\d+)\)', re.MULTILINE)
repo_pattern = re.compile(r'### 📌 \[([^\]]+)\]\(([^\)]+)\)\n+\n+((?:.|\n)*?)(?=\n---|\Z)', re.MULTILINE)

# 匹配repo字段
fields_pattern = re.compile(r'1\. \*\*Repository Name:\*\* ([^\n]+)\n2\. \*\*Brief Introduction:\*\* ([^\n]+)\n3\. \*\*Innovations:\*\* ([^\n]+)\n4\. \*\*Basic Usage:\*\* ([^\n]*)\n5\. \*\*Summary:\*\* ([^\n]+)', re.MULTILINE)

# Copilot 429/失败标记
fail_pattern = re.compile(r'Copilot API生成失败或429', re.MULTILINE)

# 加载现有数据
existing_data = load_existing_json(OUTPUT_PATH)
result = existing_data.copy()

for cat_match in category_pattern.finditer(content):
    cat_start = cat_match.end()
    next_cat = category_pattern.search(content, cat_start)
    cat_end = next_cat.start() if next_cat else len(content)
    cat_block = content[cat_start:cat_end]
    for repo_match in repo_pattern.finditer(cat_block):
        repo_title = repo_match.group(1).strip()
        repo_title = repo_title.replace("📝 ", "")  # 去掉表情
        repo_url = repo_match.group(2).strip()
        repo_block = repo_match.group(3)
        # 检查是否为失败/429
        if fail_pattern.search(repo_block):
            repo_obj = {
                "Repository Name": repo_title,
                "Repository URL": repo_url,
                "Brief Introduction": "",
                "Innovations": "",
                "Basic Usage": "",
                "Summary": ""
            }
        else:
            # 尝试提取字段
            fields = fields_pattern.search(repo_block)
            if fields:
                repo_obj = {
                    "Repository Name": fields.group(1).strip(),
                    "Repository URL": repo_url,
                    "Brief Introduction": fields.group(2).strip(),
                    "Innovations": fields.group(3).strip(),
                    "Basic Usage": fields.group(4).strip(),
                    "Summary": fields.group(5).strip()
                }
            else:
                # 兜底：只填repo名和url
                repo_obj = {
                    "Repository Name": repo_title,
                    "Repository URL": repo_url,
                    "Brief Introduction": "",
                    "Innovations": "",
                    "Basic Usage": "",
                    "Summary": ""
                }
        # 检查是否需要更新
        if repo_title not in result:
            result[repo_title] = repo_obj

# 保存为JSON
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f'已完成转换，输出到 {OUTPUT_PATH}')
