import re
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(BASE_DIR, 'README.md')
OUTPUT_PATH = os.path.join(BASE_DIR, 'repo_summaries_en.json')

def load_existing_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

with open(README_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

category_pattern = re.compile(r'## ([^\n]+) \(Total (\d+)\)', re.MULTILINE)
repo_pattern = re.compile(r'\*\*⭐ Stars:\*\* (.+?) \|.+?\*\*🍴 Forks:\*\* (.+?) \|.+?\*\*📅 Updated:\*\* (.+?)\n', re.MULTILINE)
url_pattern = re.compile(r'### 📌 \[([^\]]+)\]\(([^\)]+)\)')

fields_pattern = re.compile(r'1\. \*\*Repository Name:\*\* ([^\n]+)\n2\. \*\*Brief Introduction:\*\* ([^\n]+)\n3\. \*\*Innovations:\*\* ([^\n]+)\n4\. \*\*Basic Usage:\*\* ([^\n]*)\n5\. \*\*Summary:\*\* ([^\n]+)', re.MULTILINE)

fail_pattern = re.compile(r'Copilot API生成失败或429', re.MULTILINE)

existing_data = load_existing_json(OUTPUT_PATH)
result = existing_data.copy()

for cat_match in category_pattern.finditer(content):
    cat_start = cat_match.end()
    next_cat = category_pattern.search(content, cat_start)
    cat_end = next_cat.start() if next_cat else len(content)
    cat_block = content[cat_start:cat_end]

    for url_match in url_pattern.finditer(cat_block):
        repo_title = url_match.group(1).strip()
        repo_title = repo_title.replace("📝 ", "")
        repo_url = url_match.group(2).strip()

        repo_block_start = url_match.end()
        repo_block_end = repo_block_start + 2000
        repo_block = cat_block[repo_block_start:repo_block_end]

        stars_match = repo_pattern.search(repo_block)
        if stars_match:
            stars = int(stars_match.group(1).replace(",", ""))
            forks = int(stars_match.group(2).replace(",", ""))
            updated = stars_match.group(3).strip()
        else:
            stars = forks = 0
            updated = ""

        fields = fields_pattern.search(repo_block)
        if fail_pattern.search(repo_block):
            repo_obj = {
                "Repository Name": repo_title,
                "Repository URL": repo_url,
                "Stars": stars,
                "Forks": forks,
                "updated": updated,
                "Brief Introduction": "",
                "Innovations": "",
                "Basic Usage": "",
                "Summary": ""
            }
        elif fields:
            repo_obj = {
                "Repository Name": fields.group(1).strip(),
                "Repository URL": repo_url,
                "Stars": stars,
                "Forks": forks,
                "updated": updated,
                "Brief Introduction": fields.group(2).strip(),
                "Innovations": fields.group(3).strip(),
                "Basic Usage": fields.group(4).strip(),
                "Summary": fields.group(5).strip()
            }
        else:
            repo_obj = {
                "Repository Name": repo_title,
                "Repository URL": repo_url,
                "stars": stars,
                "forks": forks,
                "updated": updated,
                "Brief Introduction": "",
                "Innovations": "",
                "Basic Usage": "",
                "Summary": ""
            }
        result[repo_title] = repo_obj

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f'已完成转换，输出到 {OUTPUT_PATH}')
