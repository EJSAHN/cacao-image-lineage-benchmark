from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RELEASE_NAME = "cacao-image-lineage-benchmark-v1.0.1"

IGNORE = shutil.ignore_patterns(
    ".git",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "dist",
    "work",
    "work-cache",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob(f"{RELEASE_NAME}*"):
        if old.is_file():
            old.unlink()
        else:
            shutil.rmtree(old)

    with tempfile.TemporaryDirectory(prefix="cacao-release-") as tmp:
        stage_root = Path(tmp) / RELEASE_NAME
        shutil.copytree(ROOT, stage_root, ignore=IGNORE)
        zip_path = Path(shutil.make_archive(str(DIST / RELEASE_NAME), "zip", root_dir=stage_root.parent, base_dir=stage_root.name))
        tar_path = Path(shutil.make_archive(str(DIST / RELEASE_NAME), "gztar", root_dir=stage_root.parent, base_dir=stage_root.name))

    checksum_path = DIST / f"{RELEASE_NAME}_SHA256SUMS.txt"
    checksum_path.write_text(
        f"{sha256(zip_path)}  {zip_path.name}\n{sha256(tar_path)}  {tar_path.name}\n",
        encoding="utf-8",
    )
    print(zip_path)
    print(tar_path)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
