#!/usr/bin/env python3
"""Download and verify the public cacao image archives with resume support.

The script does not extract anything. It writes full provenance and status TSVs
under the persistent project root while payloads stay under a user-provided working directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RemoteFile:
    source: str
    record_id: str
    remote_file_id: str
    name: str
    url: str
    expected_size: int | None
    checksum_algorithm: str | None
    checksum_value: str | None
    destination: str


def fetch_json(url: str, timeout: int = 120) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "cacao-image-lineage-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def safe_name(name: str) -> str:
    name = Path(name).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or "unnamed_download"


def checksum_file(path: Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_checksum(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    raw = raw.strip()
    if ":" in raw:
        algorithm, value = raw.split(":", 1)
        algorithm = algorithm.lower()
        if algorithm in hashlib.algorithms_available:
            return algorithm, value.lower()
        return None, value.lower()
    if re.fullmatch(r"[0-9a-fA-F]{32}", raw):
        return "md5", raw.lower()
    if re.fullmatch(r"[0-9a-fA-F]{64}", raw):
        return "sha256", raw.lower()
    return None, raw.lower()


def figshare_files(article_id: str, out_dir: Path, metadata_dir: Path) -> list[RemoteFile]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    record = fetch_json(url)
    (metadata_dir / f"figshare_{article_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result: list[RemoteFile] = []
    for item in record.get("files", []):
        md5 = item.get("supplied_md5") or item.get("computed_md5")
        alg, value = normalize_checksum(md5)
        name = safe_name(str(item.get("name") or item.get("id")))
        result.append(
            RemoteFile(
                source="figshare",
                record_id=str(article_id),
                remote_file_id=str(item.get("id", "")),
                name=name,
                url=str(item["download_url"]),
                expected_size=int(item["size"]) if item.get("size") is not None else None,
                checksum_algorithm=alg,
                checksum_value=value,
                destination=str(out_dir / name),
            )
        )
    if not result:
        raise RuntimeError(f"Figshare article {article_id} exposed no files")
    return result


def _zenodo_entries(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    raw = record.get("files", [])
    if isinstance(raw, dict):
        entries = raw.get("entries", raw)
        if isinstance(entries, dict):
            yield from entries.values()
        elif isinstance(entries, list):
            yield from entries
    elif isinstance(raw, list):
        yield from raw


def zenodo_files(record_id: str, out_dir: Path, metadata_dir: Path) -> list[RemoteFile]:
    url = f"https://zenodo.org/api/records/{record_id}"
    record = fetch_json(url)
    (metadata_dir / f"zenodo_{record_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result: list[RemoteFile] = []
    for item in _zenodo_entries(record):
        links = item.get("links", {}) or {}
        download_url = links.get("content") or links.get("self")
        if not download_url:
            continue
        key = str(item.get("key") or item.get("filename") or item.get("id") or "zenodo_file")
        name = safe_name(key)
        alg, value = normalize_checksum(item.get("checksum"))
        result.append(
            RemoteFile(
                source="zenodo",
                record_id=str(record_id),
                remote_file_id=str(item.get("id", key)),
                name=name,
                url=str(download_url),
                expected_size=int(item["size"]) if item.get("size") is not None else None,
                checksum_algorithm=alg,
                checksum_value=value,
                destination=str(out_dir / name),
            )
        )
    if not result:
        raise RuntimeError(f"Zenodo record {record_id} exposed no files")
    return result


def validate(path: Path, item: RemoteFile, compute_checksum: bool = True) -> tuple[bool, str, str | None]:
    if not path.is_file():
        return False, "missing", None
    actual_size = path.stat().st_size
    if item.expected_size is not None and actual_size != item.expected_size:
        return False, f"size_mismatch:{actual_size}!={item.expected_size}", None
    digest: str | None = None
    if compute_checksum and item.checksum_algorithm and item.checksum_value:
        digest = checksum_file(path, item.checksum_algorithm)
        if digest.lower() != item.checksum_value.lower():
            return False, f"checksum_mismatch:{digest}!={item.checksum_value}", digest
    return True, "valid", digest


def run_curl(item: RemoteFile, max_attempts: int = 3) -> tuple[bool, str, str | None]:
    destination = Path(item.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    valid, _, existing_digest = validate(destination, item)
    if valid:
        return True, "skipped_existing_valid", existing_digest
    if destination.exists():
        quarantine = destination.with_name(destination.name + f".invalid_{int(time.time())}")
        destination.rename(quarantine)

    curl = shutil.which("curl")
    if not curl:
        return False, "curl_not_found", None

    last_message = ""
    for attempt in range(1, max_attempts + 1):
        command = [
            curl,
            "--fail",
            "--location",
            "--retry", "8",
            "--retry-delay", "10",
            "--connect-timeout", "60",
            "--speed-time", "300",
            "--speed-limit", "1024",
            "--continue-at", "-",
            "--output", str(partial),
            item.url,
        ]
        print(f"DOWNLOAD source={item.source} file={item.name} attempt={attempt}", flush=True)
        proc = subprocess.run(command, text=True)
        if proc.returncode == 0:
            valid, reason, digest = validate(partial, item)
            if valid:
                os.replace(partial, destination)
                return True, "downloaded_and_validated", digest
            last_message = reason
        else:
            last_message = f"curl_exit_{proc.returncode}"
        time.sleep(min(30 * attempt, 90))
    return False, last_message, None


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--figshare-id", required=True)
    parser.add_argument("--zenodo-id", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    big = args.big.resolve()
    metadata_dir = root / "manifests" / "source_metadata"
    figshare_dir = big / "raw" / f"figshare_{args.figshare_id}"
    zenodo_dir = big / "raw" / f"zenodo_{args.zenodo_id}"
    for directory in (metadata_dir, figshare_dir, zenodo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    items = figshare_files(args.figshare_id, figshare_dir, metadata_dir)
    items.extend(zenodo_files(args.zenodo_id, zenodo_dir, metadata_dir))

    manifest_rows = [asdict(item) for item in items]
    manifest_fields = list(asdict(items[0]).keys())
    write_tsv(root / "manifests" / "download_manifest.tsv", manifest_rows, manifest_fields)

    status_rows: list[dict[str, Any]] = []
    failures = 0
    for item in items:
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        ok, message, digest = run_curl(item)
        destination = Path(item.destination)
        actual_size = destination.stat().st_size if destination.exists() else None
        status_rows.append(
            {
                **asdict(item),
                "started_at": started,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ok": "YES" if ok else "NO",
                "status": message,
                "actual_size": actual_size,
                "actual_checksum": digest,
            }
        )
        write_tsv(
            root / "manifests" / "download_status.tsv",
            status_rows,
            manifest_fields + ["started_at", "finished_at", "ok", "status", "actual_size", "actual_checksum"],
        )
        if not ok:
            failures += 1

    print(f"download_items={len(items)} failures={failures}")
    print(
        "raw_bytes="
        f"{sum(Path(i.destination).stat().st_size for i in items if Path(i.destination).exists())}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
