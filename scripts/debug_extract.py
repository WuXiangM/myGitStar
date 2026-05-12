import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(BASE_DIR, 'README.md')
with open(README_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 完整匹配 - 使用 .+? 进行非贪婪匹配
pattern = re.compile(r'\*\*⭐ Stars:\*\* (.+?) \|.+?\*\*🍴 Forks:\*\* (.+?) \|.+?\*\*📅 Updated:\*\* (.+?)\n')
matches = pattern.findall(content)
print('Full pattern matches:', len(matches))
if matches:
    print('First:', matches[0])

# 简化：不使用emoji
pattern2 = re.compile(r'\*\*⭐ Stars:\*\* (.+?) \|.+?\*\*🍴 Forks:\*\* (.+?) \|.+?\*\*📅 Updated:\*\* (.+?)\n')
matches2 = pattern2.findall(content)
print('Pattern2 matches:', len(matches2))
