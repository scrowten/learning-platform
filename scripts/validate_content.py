#!/usr/bin/env python3
"""
Validates domain-registry.yaml and all module README.md frontmatter.
Exit 0 = all checks pass. Exit 1 = failures found.
Run from repo root: python scripts/validate_content.py
"""
import sys
import re
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "domain-registry.yaml"
DOMAINS_DIR = REPO_ROOT / "domains"

REQUIRED_REGISTRY_FIELDS = {"id", "name", "path", "tags"}
REQUIRED_MODULE_FIELDS = {
    "id", "domain", "title", "difficulty",
    "tags", "prerequisites", "estimated_hours",
    "last_reviewed", "sota_topics",
}
VALID_DIFFICULTIES = {
    "beginner", "beginner-intermediate", "intermediate",
    "intermediate-advanced", "advanced", "research",
}

errors = []


def err(msg: str) -> None:
    errors.append(msg)
    print(f"  ERROR: {msg}")


def extract_frontmatter(path: Path) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        err(f"{path}: invalid YAML frontmatter — {e}")
        return None


def validate_registry() -> list[str]:
    """Returns list of valid domain IDs."""
    print("\n[1] Validating domain-registry.yaml")
    if not REGISTRY_PATH.exists():
        err(f"domain-registry.yaml not found at {REGISTRY_PATH}")
        return []

    try:
        data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        err(f"domain-registry.yaml: invalid YAML — {e}")
        return []

    if "domains" not in data or not isinstance(data["domains"], list):
        err("domain-registry.yaml: missing or invalid 'domains' list")
        return []

    domain_ids = []
    for domain in data["domains"]:
        missing = REQUIRED_REGISTRY_FIELDS - set(domain.keys())
        if missing:
            err(f"domain '{domain.get('id', '?')}': missing fields {missing}")
            continue
        domain_path = REPO_ROOT / domain["path"]
        if not domain_path.is_dir():
            err(f"domain '{domain['id']}': path '{domain['path']}' does not exist")
        else:
            print(f"  OK: domain '{domain['id']}' → {domain['path']}")
        domain_ids.append(domain["id"])

    return domain_ids


def validate_modules(valid_domain_ids: list[str]) -> tuple[list[str], list[dict]]:
    """Validates all module READMEs. Returns tuple of (module_ids, frontmatters)."""
    print("\n[2] Validating module frontmatter")
    if not DOMAINS_DIR.exists():
        err(f"domains/ directory not found at {DOMAINS_DIR}")
        return [], []

    module_ids = []
    frontmatters = []
    readmes = sorted(DOMAINS_DIR.rglob("README.md"))

    # Skip domain-level READMEs (direct children of domains/)
    module_readmes = [
        r for r in readmes
        if r.parent.parent != DOMAINS_DIR
    ]

    if not module_readmes:
        err("No module README.md files found under domains/")
        return [], []

    for readme in module_readmes:
        fm = extract_frontmatter(readme)
        if fm is None:
            err(f"{readme.relative_to(REPO_ROOT)}: missing frontmatter block")
            continue

        missing = REQUIRED_MODULE_FIELDS - set(fm.keys())
        if missing:
            err(f"{readme.relative_to(REPO_ROOT)}: missing fields {missing}")
            continue

        if fm["difficulty"] not in VALID_DIFFICULTIES:
            err(
                f"{readme.relative_to(REPO_ROOT)}: invalid difficulty "
                f"'{fm['difficulty']}' — must be one of {VALID_DIFFICULTIES}"
            )
            continue

        if fm["domain"] not in valid_domain_ids:
            err(
                f"{readme.relative_to(REPO_ROOT)}: domain '{fm['domain']}' "
                f"not in domain-registry.yaml"
            )
            continue

        print(f"  OK: {fm['id']} ({fm['domain']})")
        module_ids.append(fm["id"])
        frontmatters.append(fm)

    return module_ids, frontmatters


def validate_prerequisites(module_ids: list[str], frontmatters: list[dict]) -> None:
    """Checks all prerequisite IDs reference valid module IDs."""
    print("\n[3] Validating prerequisite references")
    id_set = set(module_ids)

    for fm in frontmatters:
        prereqs = fm.get("prerequisites") or []
        for prereq in prereqs:
            if prereq not in id_set:
                err(
                    f"module '{fm['id']}': "
                    f"prerequisite '{prereq}' not found in any module"
                )

    if not errors:
        print("  OK: all prerequisite references are valid")


def main() -> None:
    print("=" * 60)
    print("Learning Platform — Content Validation")
    print("=" * 60)

    valid_domain_ids = validate_registry()
    module_ids, frontmatters = validate_modules(valid_domain_ids)
    validate_prerequisites(module_ids, frontmatters)

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s) found")
        sys.exit(1)
    else:
        print(f"PASSED — {len(module_ids)} modules validated across {len(valid_domain_ids)} domains")
        sys.exit(0)


if __name__ == "__main__":
    main()
