import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_LANGUAGE_CATEGORY_ALIASES = {
    "c#",
    "c++",
    "c",
    "cpp",
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "golang",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "dart",
    "r",
    "matlab",
    "shell",
    "bash",
    "powershell",
    "html",
    "css",
    "vue",
    "react",
    "jupyter notebook",
}


@dataclass
class Taxonomy:
    categories: List[Dict[str, Any]]


def _clean_inline_md(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s{2,}$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_repo_stats_or_meta_line(line: str) -> bool:
    s = _clean_inline_md(line)
    if not s:
        return True

    if s.startswith("```"):
        return True
    if s == "---" or s.startswith("---"):
        return True

    if s.startswith("**⭐") or s.startswith("**🍴") or s.startswith("**📅"):
        return True
    if "Stars:" in s or "Forks:" in s or "Updated:" in s:
        return True

    lower = s.lower()
    if lower.startswith("here's the summary") or lower.startswith("here is the summary"):
        return True
    if lower.startswith("repository url") or lower.startswith("**repository url"):
        return True
    if lower.startswith("repository name") or lower.startswith("**repository name"):
        return True

    if "仓库url" in s.lower() or "仓库链接" in s:
        return True
    if "仓库名称" in s or "repository name" in lower:
        return True

    if "生成失败" in s or "rate limit" in lower or "ratelimit" in lower:
        return True
    if re.search(r"\b429\b", s):
        return True

    return False


def _extract_structured_fields_from_block(block: str) -> Dict[str, str]:
    if not block:
        return {}

    patterns = {
        "brief_intro": [
            r"^2\.\s*\*\*Brief Introduction:\*\*\s*(?P<v>.+?)\s*$",
            r"^2\.\s*\*\*简要介绍：\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*Repository Description:\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*仓库描述：\*\*\s*(?P<v>.+?)\s*$",
        ],
        "innovations": [
            r"^3\.\s*\*\*Innovations:\*\*\s*(?P<v>.+?)\s*$",
            r"^3\.\s*\*\*创新点：\*\*\s*(?P<v>.+?)\s*$",
        ],
        "summary": [
            r"^5\.\s*\*\*Summary:\*\*\s*(?P<v>.+?)\s*$",
            r"^5\.\s*\*\*总结：\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*Summary:\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*总结：\*\*\s*(?P<v>.+?)\s*$",
        ],
    }

    out: Dict[str, str] = {}
    for key, pats in patterns.items():
        for ep in pats:
            mm = re.search(ep, block, flags=re.MULTILINE)
            if mm:
                val = _clean_inline_md(mm.group("v"))
                if val:
                    out[key] = val
                break

    return out


def _build_repo_content_text(repo: Dict[str, Any]) -> str:
    full_name = _clean_inline_md(str(repo.get("full_name") or ""))
    title = full_name.split("/")[-1] if "/" in full_name else full_name

    parts: List[str] = []
    if title:
        parts.append(title)

    for k in ("brief_intro", "innovations", "summary"):
        v = _clean_inline_md(str(repo.get(k) or ""))
        if v and not _is_repo_stats_or_meta_line(v):
            parts.append(v)

    desc = _clean_inline_md(str(repo.get("description") or ""))
    if desc and not _is_repo_stats_or_meta_line(desc):
        parts.append(desc)

    seen: set[str] = set()
    uniq: List[str] = []
    for p in parts:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    return " | ".join(uniq).strip()


def _extract_description_from_block(block: str) -> str:
    if not block:
        return ""

    extract_patterns = [
        r"^2\.\s*\*\*Brief Introduction:\*\*\s*(?P<d>.+?)\s*$",
        r"^2\.\s*\*\*简要介绍：\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*Repository Description:\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*仓库描述：\*\*\s*(?P<d>.+?)\s*$",
        r"^5\.\s*\*\*Summary:\*\*\s*(?P<d>.+?)\s*$",
        r"^5\.\s*\*\*总结：\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*Summary:\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*总结：\*\*\s*(?P<d>.+?)\s*$",
    ]
    for ep in extract_patterns:
        mm = re.search(ep, block, flags=re.MULTILINE)
        if mm:
            desc = _clean_inline_md(mm.group("d"))
            if desc:
                return desc

    for line in str(block).splitlines():
        s = _clean_inline_md(line)
        if not s:
            continue
        if _is_repo_stats_or_meta_line(s):
            continue

        if re.match(r"^\d+\.\s*\*\*Repository Name:\*\*", s, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d+\.\s*\*\*仓库名称：\*\*", s):
            continue

        s = re.sub(r"^\d+\.\s*", "", s)
        s = re.sub(r"^[-*]\s+", "", s)
        s = s.strip()
        if s:
            return s

    return ""


def _strip_leading_symbols(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^[^\w\u4e00-\u9fff]+\s*", "", s)
    return s.strip()


def _looks_like_language_category(name: str) -> bool:
    s = _strip_leading_symbols(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return False
    if s in _LANGUAGE_CATEGORY_ALIASES:
        return True
    head = s.split("(", 1)[0].strip()
    head = head.replace("项目", "").strip()
    if head in _LANGUAGE_CATEGORY_ALIASES:
        return True
    if any(s.startswith(lang + " ") for lang in _LANGUAGE_CATEGORY_ALIASES):
        return True
    return False


def _trim_repo_block_before_language_section(block: str) -> str:
    if not block:
        return ""

    kept: List[str] = []
    for line in str(block).splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[len("## ") :].strip()
            title = re.sub(r"\s*\(\s*total\s*\d+\s*\)\s*$", "", title, flags=re.IGNORECASE)
            title = re.sub(r"（\s*共\s*\d+\s*个\s*）\s*$", "", title)
            title = title.strip()
            if _looks_like_language_category(title):
                break
        kept.append(line)

    return ("\n".join(kept)).rstrip() + "\n"


def parse_repos_from_readme(
    readme_path: str,
    repo_root: str,
    max_repos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not os.path.isabs(readme_path):
        readme_path = os.path.join(repo_root, readme_path)

    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README not found: {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"^###\s+📌\s+\[(?P<full_name>[^\]]+)\]\((?P<url>https?://[^\)\s]+)\)\s*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))
    repos: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        heading_line = m.group(0)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block = content[start:end]
        block = _trim_repo_block_before_language_section(block)
        block_md = (heading_line + "\n" + block).strip() + "\n"

        full_name = _clean_inline_md(m.group("full_name"))
        url = _clean_inline_md(m.group("url"))

        fields = _extract_structured_fields_from_block(block)
        desc = _extract_description_from_block(block)

        repo_entry = {
            "id": idx + 1,
            "full_name": full_name,
            "description": desc,
            "html_url": url,
            "block_md": block_md,
            **fields,
        }

        repo_entry["content_text"] = _build_repo_content_text(repo_entry)

        repos.append(repo_entry)

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            break

    if not repos:
        raise ValueError(
            "No repositories found in README. Expected headings like: ### 📌 [owner/repo](https://github.com/owner/repo)"
        )

    return repos


def load_existing_categories(path: str) -> Tuple[Optional[Taxonomy], Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None, {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, {}
    if not isinstance(data, dict):
        return None, {}
    taxonomy_raw = data.get("taxonomy") or {}
    categories = taxonomy_raw.get("categories") if isinstance(taxonomy_raw, dict) else None
    if not isinstance(categories, list):
        return None, {}
    try:
        taxonomy = Taxonomy(
            categories=[
                {
                    "id": str(c.get("id") or "").strip(),
                    "name": str(c.get("name") or "").strip(),
                    "description": str(c.get("description") or "").strip(),
                }
                for c in categories
                if isinstance(c, dict)
            ]
        )
    except Exception:
        taxonomy = None
    assignments_map: Dict[str, str] = {}
    assignments = data.get("assignments")
    if isinstance(assignments, list):
        for a in assignments:
            if not isinstance(a, dict):
                continue
            full_name = str(a.get("full_name") or "").strip()
            category_id = str(a.get("category_id") or "").strip()
            if full_name and category_id:
                assignments_map[full_name] = category_id
    return taxonomy, assignments_map


def normalize_taxonomy(
    raw: Any,
    min_categories: int,
    max_categories: int,
) -> Taxonomy:
    if not isinstance(raw, dict) or "categories" not in raw or not isinstance(raw["categories"], list):
        raise ValueError("Invalid taxonomy JSON")

    cats = []
    for c in raw["categories"]:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        name = str(c.get("name") or "").strip()
        desc = str(c.get("description") or "").strip()
        if not cid or not name:
            continue
        cats.append({"id": cid, "name": name, "description": desc})

    cats = [c for c in cats if not _looks_like_language_category(str(c.get("name", "")))]
    cats = cats[:max_categories]

    if not any(str(c.get("name", "")).strip().lower() == "other" for c in cats):
        if len(cats) >= max_categories:
            cats[-1] = {"id": cats[-1]["id"], "name": "Other", "description": "Everything that does not fit other categories."}
        else:
            cats.append({"id": f"C{len(cats) + 1}", "name": "Other", "description": "Everything that does not fit other categories."})

    other_idx = next(
        (i for i, c in enumerate(cats) if str(c.get("name", "")).strip().lower() == "other"),
        len(cats) - 1,
    )
    while len(cats) < min_categories:
        cats.insert(
            other_idx,
            {
                "id": f"C{len(cats) + 1}",
                "name": f"Misc {len(cats) + 1}",
                "description": "Additional bucket to satisfy the minimum category count.",
            },
        )
        other_idx += 1

    normalized = []
    for idx, c in enumerate(cats, start=1):
        normalized.append({"id": f"C{idx}", "name": c["name"], "description": c.get("description", "")})

    return Taxonomy(categories=normalized)


def parse_assignments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict) or "assignments" not in raw or not isinstance(raw["assignments"], list):
        raise ValueError("Invalid assignments JSON")
    out: List[Dict[str, Any]] = []
    for a in raw["assignments"]:
        if not isinstance(a, dict):
            continue
        rid = a.get("id")
        cid = a.get("category_id")
        if rid is None or cid is None:
            continue
        out.append({"id": rid, "category_id": str(cid)})
    return out
