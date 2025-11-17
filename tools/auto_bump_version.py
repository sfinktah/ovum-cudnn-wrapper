import os
import re
from pathlib import Path
import argparse

# Resolve relative to this script's directory to be robust when run from git hooks
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parent.resolve()

VERSION_RE = re.compile(r'^(version\s*=\s*[\"\'])(\d+)\.(\d+)\.(\d+)([\"\'])\s*$', re.IGNORECASE)


def bump_patch(match):
    prefix, major, minor, patch, suffix = match.groups()
    try:
        p = int(patch) + 1
        return f"{prefix}{int(major)}.{int(minor)}.{p}{suffix}"
    except Exception:
        return match.group(0)


def process_file(path: Path) -> bool:
    if not path.exists():
        return False
    changed = False
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if changed:
            new_lines.append(line)
            continue
        m = VERSION_RE.match(line.strip())
        if m:
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            new_line = indent + bump_patch(m)
            if new_line != line:
                changed = True
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    if changed:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed


def resolve_target_file(repo_root: Path, cli_arg: str | None) -> Path:
    # Priority: CLI --file, env BUMP_TARGET_FILE, default to repo_root/pyproject.toml
    candidate = cli_arg or os.environ.get("BUMP_TARGET_FILE")
    if candidate:
        p = Path(candidate)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "pyproject.toml").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-bump patch version in a TOML file with version = 'X.Y.Z'.")
    parser.add_argument("--file", dest="file", help="Path to the target file (default: repo_root/pyproject.toml)")
    parser.add_argument("--repo-root", dest="repo_root", help="Repository root to resolve relative paths (default: inferred)")
    args = parser.parse_args()

    # Determine repo_root: prefer --repo-root, fall back to CWD, then script-based default
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    if not (repo_root / ".git").exists():
        # If CWD doesn't look like a repo root, fall back to script-derived default
        repo_root = REPO_ROOT_DEFAULT

    path = resolve_target_file(repo_root, args.file)

    if process_file(path):
        print("CHANGED=1")
        print(f"FILE={path}")
        return 1
    else:
        print("CHANGED=0")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())