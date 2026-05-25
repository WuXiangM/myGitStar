import json
import random
from typing import Any, Dict, List, Optional, Tuple

from scripts.classification.classification_parser import Taxonomy


config = None


def init_classify_helpers(cfg: Dict[str, Any]) -> None:
    global config
    config = cfg


def build_taxonomy_prompt(items: List[Dict[str, Any]], min_categories: int, max_categories: int) -> str:
    examples = []
    for r in items:
        examples.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "title": (str(r.get("full_name") or "").split("/")[-1] if str(r.get("full_name") or "").strip() else ""),
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
        "- CREATE DIVERSE, SPECIFIC categories that reflect actual project domains and use cases.\n"
        "- Aim for BALANCED category sizes; avoid creating a single dominant category.\n"
        "- Split broad topics into multiple specific sub-categories when repositories differ in their focus.\n"
        "- Design categories so that repositories are distributed across categories, NOT concentrated in one or two.\n"
        "- Include an 'Other' category for anything that doesn't fit, but it should contain few repositories.\n"
        "Output strictly as JSON, with this shape:\n"
        "{\n"
        "  \"categories\": [\n"
        "    {\"id\": \"C1\", \"name\": \"...\", \"description\": \"...\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Here are example repositories (id, full_name, title, text):\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
    )


def build_classification_prompt(taxonomy: Taxonomy, repos: List[Dict[str, Any]]) -> str:
    items = []
    for r in repos:
        items.append(
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "title": (str(r.get("full_name") or "").split("/")[-1] if str(r.get("full_name") or "").strip() else ""),
                "text": r.get("content_text") or r.get("description") or "",
            }
        )

    return (
        "You are a classifier.\n"
        "Classify each GitHub repository into exactly ONE category from the provided taxonomy.\n"
        "Use the repository title + content text (ignore stars/forks/updated).\n"
        "Pick the BEST matching category; use 'Other' ONLY if the repository genuinely does not relate to any category's domain.\n"
        "IMPORTANT: Avoid overusing 'Other'. If a repo mentions AI, agents, automation, LLMs, or similar topics, it almost certainly belongs in one of the specific categories, not 'Other'.\n"
        "When a repo could fit multiple categories, choose the ONE most central theme.\n\n"
        "Return STRICT JSON only.\n\n"
        "Taxonomy JSON:\n"
        + json.dumps({"categories": taxonomy.categories}, ensure_ascii=False, indent=2)
        + "\n\n"
        "Repositories to classify (id, full_name, title, text):\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


def _resolve_min_repos_per_category(args_value: Optional[int]) -> int:
    if args_value is not None:
        try:
            return max(0, int(args_value))
        except Exception:
            return 0
    try:
        if isinstance(config, dict) and config.get("content_min_repos_per_category") is not None:
            return max(0, int(config.get("content_min_repos_per_category")))
    except Exception:
        pass
    return 0


def _compute_effective_max_categories(total_repos: int, min_repos_per_category: int, min_categories: int, max_categories: int) -> int:
    if total_repos <= 0:
        return max_categories
    if min_repos_per_category <= 0:
        return max_categories
    upper = max(1, total_repos // min_repos_per_category)
    return max(min_categories, min(max_categories, upper))


def _sample_repos_for_taxonomy(repos: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    sample_size = max(1, min(int(sample_size or 0), len(repos)))
    strategy = "head"
    seed = None
    try:
        if isinstance(config, dict):
            strategy = str(config.get("content_taxonomy_sample_strategy", "head") or "head").strip().lower()
            seed_val = config.get("content_taxonomy_sample_seed")
            if seed_val is not None and str(seed_val).strip() != "":
                seed = int(seed_val)
    except Exception:
        strategy = "head"
        seed = None

    if strategy == "random" and len(repos) > sample_size:
        rng = random.Random(seed) if seed is not None else random.Random()
        return rng.sample(repos, k=sample_size)

    return repos[:sample_size]