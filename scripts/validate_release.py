from __future__ import annotations
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
forbidden = [
    r"/project/", r"/90daydata/", r"ezekiel\.ahn", r"@gmail\.com", r"@usda\.gov",
    r"hotfix_backups",
]
allowed_path_docs = {ROOT / "docs" / "SCINET.md"}
failures = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.resolve() == Path(__file__).resolve():
        continue
    rel = path.relative_to(ROOT)
    if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
        continue
    if path.stat().st_size > 50 * 1024 * 1024:
        failures.append(f"file exceeds 50 MB: {rel}")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        failures.append(f"raw or rendered image included: {rel}")
    if path.suffix.lower() in {".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".toml", ".json", ".tsv", ".csv"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden:
            if re.search(pattern, text) and path not in allowed_path_docs:
                failures.append(f"forbidden pattern {pattern!r}: {rel}")
for path in (ROOT / "src").rglob("*.py"):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except Exception as exc:
        failures.append(f"compile failure {path.relative_to(ROOT)}: {exc}")
# Verify the tracked release manifest without considering runtime caches.
manifest_path = ROOT / "MANIFEST.sha256.tsv"
excluded_dirs = {".git", ".pytest_cache", "__pycache__", "dist"}

def manifest_included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path == manifest_path:
        return False
    if any(part in excluded_dirs or part.endswith(".egg-info") for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not manifest_path.is_file():
    failures.append("missing MANIFEST.sha256.tsv")
else:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}
    actual_paths = {path.relative_to(ROOT).as_posix(): path for path in ROOT.rglob("*") if manifest_included(path)}
    if set(manifest_rows) != set(actual_paths):
        missing = sorted(set(actual_paths) - set(manifest_rows))
        extra = sorted(set(manifest_rows) - set(actual_paths))
        failures.append(f"manifest path mismatch: missing={missing} extra={extra}")
    for rel, path in actual_paths.items():
        row = manifest_rows.get(rel)
        if row is None:
            continue
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            failures.append(f"manifest checksum mismatch: {rel}")

if failures:
    print("\n".join(sorted(set(failures))), file=sys.stderr)
    raise SystemExit(1)
print("release_validation=PASS")
