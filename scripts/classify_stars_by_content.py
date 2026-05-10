import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from scripts.core.config import (
    load_config,
    get_int_config,
    get_float_config,
    env_truthy,
    normalize_update_mode,
    resolve_update_mode,
    get_model_choice,
)
from scripts.core.secrets import load_api_keys
from scripts.core.throttle import SimpleThrottle
from scripts.llm_caller import make_api_request, call_llm_json, RateLimitAbort
from scripts.classification_parser import (
    Taxonomy,
    parse_repos_from_readme,
    load_existing_categories,
    normalize_taxonomy,
    parse_assignments,
    _build_repo_content_text,
)
from scripts.markdown_renderer import (
    render_classified_readme,
    apply_min_repos_per_category,
    chunk_list,
)
from scripts.prompts import build_taxonomy_prompt, build_classification_prompt


config = load_config()

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

GITHUB_TOKEN, OPENROUTER_API_KEY, GEMINI_API_KEY = load_api_keys(config)

MODEL_CHOICE = get_model_choice(config)

DEFAULT_COPILOT_MODEL = config.get("default_copilot_model", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_MODEL = config.get("default_openrouter_model")
DEFAULT_GEMINI_MODEL = config.get("default_gemini_model", "gemini-2.0-flash")

update_mode = resolve_update_mode(config)

REQUEST_TIMEOUT = float(
    os.environ.get("MYGITSTAR_REQUEST_TIMEOUT", get_float_config(config, "request_timeout", 30.0))
)
REQUEST_RETRY_DELAY = float(
    os.environ.get("MYGITSTAR_REQUEST_RETRY_DELAY", get_float_config(config, "request_retry_delay", 2.0))
)
RETRY_ATTEMPTS = int(
    os.environ.get("MYGITSTAR_RETRY_ATTEMPTS", get_int_config(config, "retry_attempts", 1))
)
RATE_LIMIT_DELAY = float(
    os.environ.get("MYGITSTAR_RATE_LIMIT_DELAY", get_float_config(config, "rate_limit_delay", 3.0))
)
GLOBAL_QPS = float(
    os.environ.get("MYGITSTAR_GLOBAL_QPS", get_float_config(config, "global_qps", 0.5))
)

FALLBACK_ON_429 = env_truthy("MYGITSTAR_FALLBACK_ON_429")

DEBUG_API = env_truthy("DEBUG_API")

MAX_CONSECUTIVE_429 = int(os.environ.get("MYGITSTAR_MAX_CONSECUTIVE_429", "5") or "5")

THROTTLE = SimpleThrottle(GLOBAL_QPS)


def _api_request_with_throttle(url: str, headers: Dict[str, str], data: Dict[str, Any], retries: int, retry_delay: float) -> Optional[Dict[str, Any]]:
    return make_api_request(url, headers, data, retries, retry_delay, THROTTLE, REQUEST_TIMEOUT)


def get_github_username() -> str:
    gh = config.get("github_username") if isinstance(config, dict) else None
    if gh == "0" or gh == 0:
        actor = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
        if not actor:
            raise RuntimeError("github_username is 0 but GITHUB_ACTOR/GITHUB_USERNAME is not set.")
        return actor
    if not gh:
        raise RuntimeError("Missing config.github_username")
    return str(gh)


def get_starred_repos(max_repos: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GITHUB_TOKEN:
        raise RuntimeError("Missing STARRED_GITHUB_TOKEN")

    username = get_github_username()
    print(f"Fetching starred repos for: {username}")

    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"https://api.github.com/users/{username}/starred?per_page={per_page}&page={page}"
        THROTTLE.wait()
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
        if not data:
            break
        repos.extend(data)
        print(f"  got {len(repos)} repos... (page {page})")
        page += 1
        time.sleep(1)

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            repos = repos[:max_repos]
            break

    return repos


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


def _compute_effective_max_categories(
    total_repos: int, min_repos_per_category: int, min_categories: int, max_categories: int
) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify GitHub repositories into content categories using the selected LLM."
    )
    parser.add_argument(
        "--from-readme",
        type=str,
        default=None,
        help="Parse repositories from an existing generated README (e.g. README.md) instead of calling GitHub API.",
    )
    parser.add_argument("--max-repos", type=int, default=None, help="Limit number of repos (overrides config/env MAX_REPOS).")
    parser.add_argument("--min-categories", type=int, default=5, help="Min number of categories (default: 5).")
    parser.add_argument("--max-categories", type=int, default=8, help="Max number of categories (default: 8).")
    parser.add_argument(
        "--min-repos-per-category",
        type=int,
        default=None,
        help="Optional: enforce each non-'Other' category has at least X repos by pruning tiny categories into Other (default: read config 'content_min_repos_per_category', 0 disables).",
    )
    parser.add_argument("--sample-size", type=int, default=60, help="How many repos to sample to design taxonomy (default: 60).")
    parser.add_argument("--batch-size", type=int, default=30, help="Repos per LLM classification call (default: 30).")
    parser.add_argument("--out-json", type=str, default="repo_categories.json", help="Output JSON filename.")
    parser.add_argument("--out-md", type=str, default="README.md", help="Output Markdown filename (default: README.md).")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=None,
        help="Override update mode: missing_only or all (also supports env MYGITSTAR_UPDATE_MODE).",
    )

    args = parser.parse_args()

    if args.update_mode is not None:
        global update_mode
        update_mode = normalize_update_mode(args.update_mode)

    passed_batch = any(arg.startswith("--batch-size") for arg in sys.argv)
    if not passed_batch and isinstance(config, dict):
        classify_bs = config.get("classify_batch_size")
        if classify_bs is not None:
            try:
                args.batch_size = int(classify_bs)
            except Exception:
                pass
        else:
            batch_bs = config.get("batch_size")
            if batch_bs is not None:
                try:
                    args.batch_size = int(batch_bs)
                except Exception:
                    pass

    passed_sample = any(arg.startswith("--sample-size") for arg in sys.argv)
    if not passed_sample and isinstance(config, dict):
        sample_size = config.get("classify_sample_size")
        if sample_size is not None:
            try:
                args.sample_size = int(sample_size)
            except Exception:
                pass

    passed_outjson = any(arg.startswith("--out-json") for arg in sys.argv)
    if not passed_outjson and isinstance(config, dict):
        out_json = config.get("classify_out_json")
        if out_json:
            args.out_json = str(out_json)

    passed_outmd = any(arg.startswith("--out-md") for arg in sys.argv)
    if not passed_outmd and isinstance(config, dict):
        out_md = config.get("classify_out_md")
        if out_md:
            args.out_md = str(out_md)

    out_json_path = args.out_json
    if not os.path.isabs(out_json_path):
        out_json_path = os.path.join(REPO_ROOT, out_json_path)

    out_md_path = args.out_md
    if not os.path.isabs(out_md_path):
        out_md_path = os.path.join(REPO_ROOT, out_md_path)

    passed_min = "--min-categories" in sys.argv
    passed_max = "--max-categories" in sys.argv
    if isinstance(config, dict):
        if not passed_min and config.get("content_min_categories") is not None:
            try:
                v = config.get("content_min_categories")
                if v is not None:
                    args.min_categories = int(v)
            except Exception:
                pass
        if not passed_max and config.get("content_max_categories") is not None:
            try:
                v = config.get("content_max_categories")
                if v is not None:
                    args.max_categories = int(v)
            except Exception:
                pass

    if args.min_categories < 1:
        print("--min-categories must be >= 1")
        return 2
    if args.max_categories < args.min_categories:
        print("--max-categories must be >= --min-categories")
        return 2

    max_repos = args.max_repos
    if max_repos is None and not args.from_readme:
        env_mr = os.environ.get("MAX_REPOS")
        if env_mr:
            try:
                mr = int(env_mr)
                if mr > 0:
                    max_repos = mr
            except Exception:
                max_repos = None
        if max_repos is None:
            cfg_mr = config.get("max_repos") if isinstance(config, dict) else None
            try:
                if cfg_mr is not None and int(cfg_mr) > 0:
                    max_repos = int(cfg_mr)
            except Exception:
                pass

    if args.from_readme:
        try:
            repos = parse_repos_from_readme(args.from_readme, REPO_ROOT, max_repos=max_repos)
        except Exception as e:
            print(f"Failed to parse repos from README: {e}")
            return 2
    else:
        try:
            repos = get_starred_repos(max_repos=max_repos)
        except Exception as e:
            print(f"Failed to fetch starred repos: {e}")
            print("\nChecklist:")
            print("- Ensure you have set STARRED_GITHUB_TOKEN in environment variables, or in GitHub Actions Secrets.")
            print(
                "- If config.yaml sets github_username: 0, ensure GITHUB_ACTOR is available (Actions) or set github_username to your username locally."
            )
            return 2

    if not repos:
        print("No repos fetched.")
        return 2

    for r in repos:
        try:
            if not str(r.get("content_text") or "").strip():
                r["content_text"] = _build_repo_content_text(r)
        except Exception:
            pass

    min_repos_per_category = _resolve_min_repos_per_category(args.min_repos_per_category)
    effective_max_categories = _compute_effective_max_categories(
        total_repos=len(repos),
        min_repos_per_category=min_repos_per_category,
        min_categories=args.min_categories,
        max_categories=args.max_categories,
    )
    if effective_max_categories != args.max_categories:
        print(
            f"[info] Adjusted max_categories {args.max_categories} -> {effective_max_categories} due to content_min_repos_per_category={min_repos_per_category} (total_repos={len(repos)})"
        )
        args.max_categories = effective_max_categories

    taxonomy: Optional[Taxonomy] = None
    assignment_map: Dict[Any, str] = {}
    all_ids = {r.get("id") for r in repos}

    existing_taxonomy, existing_assignments = (None, {})
    if update_mode == "missing_only":
        existing_taxonomy, existing_assignments = load_existing_categories(out_json_path)

    repos_to_classify = repos
    if update_mode == "missing_only" and existing_taxonomy and existing_assignments:
        taxonomy = existing_taxonomy
        repos_to_classify = [r for r in repos if str(r.get("full_name") or "").strip() not in existing_assignments]
        for r in repos:
            full_name = str(r.get("full_name") or "").strip()
            cid = existing_assignments.get(full_name)
            if cid:
                assignment_map[r.get("id")] = cid

    if taxonomy is None:
        sample_size = max(5, min(args.sample_size, len(repos)))
        sample = _sample_repos_for_taxonomy(repos, sample_size)
        taxonomy_prompt = build_taxonomy_prompt(sample, min_categories=args.min_categories, max_categories=args.max_categories)
        print(
            f"Designing taxonomy from {len(sample)} sample repos (min_categories={args.min_categories}, max_categories={args.max_categories})..."
        )
        try:
            raw_tax = call_llm_json(
                taxonomy_prompt,
                config,
                GITHUB_TOKEN,
                OPENROUTER_API_KEY,
                GEMINI_API_KEY,
                DEFAULT_COPILOT_MODEL,
                DEFAULT_OPENROUTER_MODEL,
                DEFAULT_GEMINI_MODEL,
                _api_request_with_throttle,
                FALLBACK_ON_429,
                attempts=2,
            )
        except RateLimitAbort as e:
            print(f"[FATAL] {e}")
            return 4
        except Exception as e:
            print(f"Failed to call LLM for taxonomy: {e}")
            print("\nChecklist:")
            print("- model_choice=copilot requires STARRED_GITHUB_TOKEN")
            print("- model_choice=openrouter requires OPENROUTER_API_KEY")
            print("- model_choice=gemini requires GEMINI_API_KEY")
            return 3

        try:
            taxonomy = normalize_taxonomy(raw_tax, min_categories=args.min_categories, max_categories=args.max_categories)
            if len(taxonomy.categories) < args.min_categories:
                retry_prompt = (
                    taxonomy_prompt
                    + "\n\nIMPORTANT: Your previous output likely used programming-language categories. Redesign taxonomy strictly by CONTENT domains."
                )
                raw_tax2 = call_llm_json(
                    retry_prompt,
                    config,
                    GITHUB_TOKEN,
                    OPENROUTER_API_KEY,
                    GEMINI_API_KEY,
                    DEFAULT_COPILOT_MODEL,
                    DEFAULT_OPENROUTER_MODEL,
                    DEFAULT_GEMINI_MODEL,
                    _api_request_with_throttle,
                    FALLBACK_ON_429,
                    attempts=1,
                )
                taxonomy = normalize_taxonomy(raw_tax2, min_categories=args.min_categories, max_categories=args.max_categories)
        except Exception as e:
            print(f"Failed to normalize taxonomy JSON: {e}")
            return 3

        print("Taxonomy:")
        for c in taxonomy.categories:
            print(f"  {c['id']}: {c['name']}")

    if repos_to_classify:
        print(f"Classifying {len(repos_to_classify)} repos in batches of {args.batch_size}...")
        for i, batch in enumerate(chunk_list(repos_to_classify, args.batch_size), start=1):
            prompt = build_classification_prompt(taxonomy, batch)
            try:
                raw = call_llm_json(
                    prompt,
                    config,
                    GITHUB_TOKEN,
                    OPENROUTER_API_KEY,
                    GEMINI_API_KEY,
                    DEFAULT_COPILOT_MODEL,
                    DEFAULT_OPENROUTER_MODEL,
                    DEFAULT_GEMINI_MODEL,
                    _api_request_with_throttle,
                    FALLBACK_ON_429,
                    attempts=2,
                )
                assignments = parse_assignments(raw)
            except RateLimitAbort as e:
                print(f"[FATAL] {e}")
                return 4
            except Exception as e:
                print(f"[Batch {i}] failed to parse assignments: {e}")
                assignments = []

            valid_category_ids = {c["id"] for c in taxonomy.categories}
            other_id = next(
                (c["id"] for c in taxonomy.categories if c["name"].lower() == "other"),
                taxonomy.categories[-1]["id"],
            )
            for a in assignments:
                rid = a["id"]
                cid = a["category_id"]
                if cid not in valid_category_ids:
                    cid = other_id
                assignment_map[rid] = cid

            batch_ids = {r.get("id") for r in batch}
            for rid in batch_ids:
                if rid not in assignment_map:
                    assignment_map[rid] = other_id

            print(f"  batch {i}/{(len(repos_to_classify) + args.batch_size - 1) // args.batch_size} done")
            time.sleep(RATE_LIMIT_DELAY)

    other_id = next(
        (c["id"] for c in taxonomy.categories if c["name"].lower() == "other"),
        taxonomy.categories[-1]["id"],
    )
    for rid in all_ids:
        if rid not in assignment_map:
            assignment_map[rid] = other_id

    taxonomy, assignment_map = apply_min_repos_per_category(
        taxonomy,
        repos,
        assignment_map,
        min_repos_per_category=min_repos_per_category,
        min_categories=args.min_categories,
    )
    other_id = next(
        (c["id"] for c in taxonomy.categories if c["name"].lower() == "other"),
        taxonomy.categories[-1]["id"],
    )

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_choice": MODEL_CHOICE,
        "min_categories": args.min_categories,
        "max_categories": args.max_categories,
        "min_repos_per_category": min_repos_per_category,
        "taxonomy": {"categories": taxonomy.categories},
        "assignments": [
            {"id": r.get("id"), "full_name": r.get("full_name"), "category_id": assignment_map.get(r.get("id"), other_id)}
            for r in repos
        ],
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = render_classified_readme(taxonomy, repos, assignment_map, config, MODEL_CHOICE)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
