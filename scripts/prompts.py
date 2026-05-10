from typing import Any, Dict


def generate_summarize_prompt(repo: Dict[str, Any], language: str = "zh") -> str:
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo["html_url"]

    if language == "zh":
        return (
            f"请对以下 GitHub 仓库进行内容总结，按如下格式输出：\n"
            f"1. **仓库名称：** {repo_name}\n"
            f"2. **简要介绍：** （50字以内）\n"
            f"3. **创新点：** （简述本仓库最有特色的地方）\n"
            f"4. **简单用法：** （给出最简关键用法或调用示例，如无则略）\n"
            f"5. **总结：** （一句话总结它的用途/价值）\n"
            f"**仓库描述：** {desc}\n"
            f"**仓库地址：** {url}\n"
        )
    else:
        return (
            f"Please summarize the following GitHub repository in the specified format:\n"
            f"1. **Repository Name:** {repo_name}\n"
            f"2. **Brief Introduction:** (within 50 words)\n"
            f"3. **Innovations:** (Briefly describe the most distinctive features)\n"
            f"4. **Basic Usage:** (Provide the simplest key usage or example, omit if none)\n"
            f"5. **Summary:** (One sentence summarizing its purpose/value)\n"
            f"**Repository Description:** {desc}\n"
            f"**Repository URL:** {url}\n"
        )


def build_taxonomy_prompt(
    items: list,
    min_categories: int,
    max_categories: int,
) -> str:
    examples = []
    for r in items:
        examples.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "title": (
                    str(r.get("full_name") or "").split("/")[-1]
                    if str(r.get("full_name") or "").strip()
                    else ""
                ),
                "text": r.get("content_text") or r.get("description") or "",
            }
        )

    return (
        "You are a taxonomy designer.\n"
        "Task: create a content-based taxonomy to classify GitHub repositories, using repo title + a cleaned content summary text (not stats).\n"
        f"Constraints: create BETWEEN {min_categories} and {max_categories} categories total (inclusive), and include an 'Other' category.\n"
        "Rules:\n"
        "- Categories must be based on CONTENT/domains (e.g., LLM tooling, CV, data engineering), NOT programming languages.\n"
        "- DO NOT create categories named after programming languages (e.g., Python/C++/Java/C#/JS/TS/Rust/Go).\n"
        "- Category names should be short and clear.\n"
        "- Prefer broader categories over tiny niche buckets.\n"
        "- Design categories so that most repositories can fit a non-'Other' category; use 'Other' only as a true fallback.\n"
        "- Include an 'Other' category for anything that doesn't fit.\n"
        "Output strictly as JSON, with this shape:\n"
        "{\n"
        '  "categories": [\n'
        '    {"id": "C1", "name": "...", "description": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Here are example repositories (id, full_name, title, text):\n"
        + _json_dumps(examples, ensure_ascii=False, indent=2)
    )


def build_classification_prompt(taxonomy: Any, repos: list) -> str:
    items = []
    for r in repos:
        items.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "title": (
                    str(r.get("full_name") or "").split("/")[-1]
                    if str(r.get("full_name") or "").strip()
                    else ""
                ),
                "text": r.get("content_text") or r.get("description") or "",
            }
        )

    return (
        "You are a classifier.\n"
        "Classify each GitHub repository into exactly ONE category from the provided taxonomy.\n"
        "Use the repository title + content text (ignore stars/forks/updated).\n"
        "Pick the BEST matching category; use 'Other' only if none of the categories fit.\n"
        "Return STRICT JSON only.\n\n"
        "Taxonomy JSON:\n"
        + _json_dumps({"categories": taxonomy.categories}, ensure_ascii=False, indent=2)
        + "\n\n"
        "Repositories to classify (id, full_name, title, text):\n"
        + _json_dumps(items, ensure_ascii=False, indent=2)
        + "\n\n"
        "Output JSON shape:\n"
        "{\n"
        '  "assignments": [\n'
        '    {"id": 123, "category_id": "C1"},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )


def _json_dumps(obj: Any, **kwargs) -> str:
    import json
    return json.dumps(obj, **kwargs)
