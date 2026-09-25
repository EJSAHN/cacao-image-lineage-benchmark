from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256.tsv"
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__", "dist"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path == MANIFEST:
        return False
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


rows = []
for path in sorted(path for path in ROOT.rglob("*") if included(path)):
    rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)))

with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["path", "bytes", "sha256"])
    writer.writerows(rows)

print(f"manifest_entries={len(rows)}")
