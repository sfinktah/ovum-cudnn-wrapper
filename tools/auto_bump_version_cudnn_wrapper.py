import re
from pathlib import Path

# Single target: ovum-cudnn-wrapper
PROJECT_FILE = Path("custom_nodes/ovum-cudnn-wrapper/pyproject.toml")

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


def main() -> int:
    repo_root = Path.cwd()
    path = repo_root / PROJECT_FILE
    if process_file(path):
        print("CHANGED=1")
        print(f"FILE={path}")
        return 1
    else:
        print("CHANGED=0")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())