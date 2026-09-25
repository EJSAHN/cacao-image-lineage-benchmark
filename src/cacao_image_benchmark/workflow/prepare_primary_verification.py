#!/usr/bin/env python3
"""Prepare primary geometric verification primary-shortlist and frozen sentinel-control pair tables.

primary geometric verification verifies the 6,000 evidence-gated pairs selected in evidence reranking with
exactly the geometric/photometric verifier used in geometric verification.  Archive, split,
and label flags are retained as scientific priorities only; they do not count
as image-lineage evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

VERIFIER_SHA256 = "a497c9a923901f6c016e60d6bed475720e1496fd9a77766607092b4bfe2603b1"


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def set_join(values: Iterable[Any], keep_unspecified: bool = False) -> str:
    skip = {"", "UNMATCHED"}
    if not keep_unspecified:
        skip.add("unspecified")
    return "|".join(sorted({str(v) for v in values if str(v) not in skip}))


def unordered_pair_key(a: Any, b: Any) -> str:
    x, y = sorted((str(a), str(b)))
    return f"{x}||{y}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--hash-table", required=True, type=Path)
    parser.add_argument("--stage2b-controls", required=True, type=Path)
    parser.add_argument("--frozen-verifier", required=True, type=Path)
    parser.add_argument("--stage2b-verifier", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-path-check", action="store_true", help="Package-validation only; never use on SCINet production runs.")
    args = parser.parse_args()

    for path in (args.shortlist, args.hash_table, args.stage2b_controls, args.frozen_verifier):
        if not path.is_file():
            raise FileNotFoundError(path)

    frozen_sha = sha256_file(args.frozen_verifier)
    if frozen_sha != VERIFIER_SHA256:
        raise RuntimeError(f"Frozen verifier checksum mismatch: {frozen_sha} != {VERIFIER_SHA256}")
    stage2b_sha = ""
    stage2b_match = "NOT_CHECKED"
    if args.stage2b_verifier is not None and args.stage2b_verifier.is_file():
        stage2b_sha = sha256_file(args.stage2b_verifier)
        stage2b_match = "YES" if stage2b_sha == frozen_sha else "NO"

    shortlist = pd.read_csv(args.shortlist, sep="\t", keep_default_na=False, low_memory=False)
    hashes = pd.read_csv(args.hash_table, sep="\t", keep_default_na=False, low_memory=False)
    controls = pd.read_csv(args.stage2b_controls, sep="\t", keep_default_na=False, low_memory=False)

    required_shortlist = {
        "pair_id", "component_a", "component_b", "absolute_path_a", "absolute_path_b",
        "path_a", "path_b", "archive_a", "archive_b", "label_a", "label_b",
        "split_a", "split_b", "source_origin_key_a", "source_origin_key_b",
        "cross_archive", "cross_split", "label_conflict", "evidence_tier",
        "scientific_risk", "evidence_score", "shortlist_selection_reason",
    }
    missing = required_shortlist - set(shortlist.columns)
    if missing:
        raise RuntimeError(f"evidence reranking shortlist missing columns: {sorted(missing)}")
    if shortlist.empty:
        raise RuntimeError("evidence reranking shortlist is empty")

    shortlist = shortlist.copy()
    shortlist["component_a"] = shortlist["component_a"].astype(str)
    shortlist["component_b"] = shortlist["component_b"].astype(str)
    if shortlist["pair_id"].duplicated().any():
        raise RuntimeError("evidence reranking pair_id values are not unique")
    if shortlist["component_a"].eq(shortlist["component_b"]).any():
        bad = shortlist.loc[shortlist["component_a"].eq(shortlist["component_b"]), "pair_id"].head().tolist()
        raise RuntimeError(f"Same-component candidate pairs found: {bad}")
    shortlist["unordered_pair_key"] = [unordered_pair_key(a, b) for a, b in zip(shortlist["component_a"], shortlist["component_b"])]
    if shortlist["unordered_pair_key"].duplicated().any():
        bad = shortlist.loc[shortlist["unordered_pair_key"].duplicated(False), ["pair_id", "unordered_pair_key"]].head().to_dict("records")
        raise RuntimeError(f"Duplicate unordered candidate pairs found: {bad}")

    hash_required = {
        "exact_component_id", "source_archive", "source_dataset_id", "corrected_label",
        "corrected_split", "source_origin_key", "representative_relative_path",
        "representative_absolute_path", "width", "height", "gray_entropy",
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    }
    missing_hash = hash_required - set(hashes.columns)
    if missing_hash:
        raise RuntimeError(f"perceptual-candidate discovery hash table missing columns: {sorted(missing_hash)}")
    if hashes["exact_component_id"].duplicated().any():
        raise RuntimeError("perceptual-candidate discovery hash table contains duplicate exact_component_id values")
    hash_meta = hashes.set_index("exact_component_id")

    missing_components = sorted((set(shortlist["component_a"]) | set(shortlist["component_b"])) - set(hash_meta.index.astype(str)))
    if missing_components:
        raise RuntimeError(f"Shortlist components absent from perceptual-candidate discovery hash table: {missing_components[:10]}")

    candidate = shortlist.copy()
    candidate = candidate.rename(columns={"pair_id": "stage2c1_pair_id"})
    candidate.insert(0, "pair_id", [f"S2D{i:06d}" for i in range(1, len(candidate) + 1)])
    candidate.insert(1, "pair_origin", "STAGE2C1_PRIMARY_SHORTLIST")
    candidate.insert(2, "expected_relation", "UNKNOWN")
    candidate.insert(3, "synthetic_transform", "")
    candidate["candidate_strength"] = candidate["evidence_tier"]
    candidate["archives"] = [a if a == b else set_join([a, b], keep_unspecified=True) for a, b in zip(candidate["archive_a"], candidate["archive_b"])]
    candidate["labels"] = [set_join([a, b], keep_unspecified=True) for a, b in zip(candidate["label_a"], candidate["label_b"])]
    candidate["splits"] = [set_join([a, b], keep_unspecified=False) for a, b in zip(candidate["split_a"], candidate["split_b"])]
    candidate["shared_origin_keys"] = [a if a and a == b else "" for a, b in zip(candidate["source_origin_key_a"], candidate["source_origin_key_b"])]
    for col in ("min_phash_hamming", "min_dhash_hamming", "min_whash_hamming"):
        candidate[col] = ""

    meta_cols = ["width", "height", "gray_entropy", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    for side in ("a", "b"):
        joined = candidate[[f"component_{side}"]].join(hash_meta[meta_cols], on=f"component_{side}")
        for col in meta_cols:
            candidate[f"{col}_{side}"] = joined[col].values

    # Confirm that evidence reranking metadata still agrees with the immutable perceptual-candidate discovery component table.
    mismatch_rows = []
    for row in candidate.itertuples(index=False):
        for side in ("a", "b"):
            component = getattr(row, f"component_{side}")
            meta = hash_meta.loc[component]
            observed = {
                "archive": getattr(row, f"archive_{side}"),
                "label": getattr(row, f"label_{side}"),
                "split": getattr(row, f"split_{side}"),
                "absolute_path": getattr(row, f"absolute_path_{side}"),
            }
            expected = {
                "archive": str(meta["source_archive"]),
                "label": str(meta["corrected_label"]),
                "split": str(meta["corrected_split"]),
                "absolute_path": str(meta["representative_absolute_path"]),
            }
            for field in observed:
                if str(observed[field]) != str(expected[field]):
                    mismatch_rows.append({
                        "pair_id": row.pair_id,
                        "side": side,
                        "component": component,
                        "field": field,
                        "stage2c1_value": observed[field],
                        "stage2a_value": expected[field],
                    })
    mismatch = pd.DataFrame(mismatch_rows)
    if len(mismatch):
        raise RuntimeError(f"evidence reranking / perceptual-candidate discovery component metadata mismatch, first rows: {mismatch.head().to_dict('records')}")

    required_control = {"pair_id", "pair_origin", "expected_relation", "synthetic_transform", "component_a", "component_b", "absolute_path_a", "absolute_path_b"}
    missing_control = required_control - set(controls.columns)
    if missing_control:
        raise RuntimeError(f"geometric verification control table missing columns: {sorted(missing_control)}")
    if controls["pair_id"].duplicated().any():
        raise RuntimeError("geometric verification control pair IDs are not unique")

    missing_paths = []
    if not args.skip_path_check:
        for table_name, table in (("candidate", candidate), ("control", controls)):
            for row in table.itertuples(index=False):
                transform = str(getattr(row, "synthetic_transform", ""))
                path_a = str(getattr(row, "absolute_path_a", ""))
                path_b = str(getattr(row, "absolute_path_b", ""))
                if not path_a or not Path(path_a).is_file():
                    missing_paths.append((table_name, getattr(row, "pair_id"), "a", path_a))
                if not transform and (not path_b or not Path(path_b).is_file()):
                    missing_paths.append((table_name, getattr(row, "pair_id"), "b", path_b))
        if missing_paths:
            raise RuntimeError(f"Missing image paths, first examples: {missing_paths[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "stage2d_primary_candidate_pairs.tsv"
    controls_path = args.output_dir / "stage2d_sentinel_control_pairs.tsv"
    write_tsv(candidate.drop(columns=["unordered_pair_key"]), candidate_path)
    write_tsv(controls, controls_path)
    write_tsv(mismatch, args.output_dir / "stage2d_component_metadata_mismatch_audit.tsv")

    provenance = pd.DataFrame([
        {
            "shortlist_path": str(args.shortlist),
            "shortlist_sha256": sha256_file(args.shortlist),
            "hash_table_path": str(args.hash_table),
            "hash_table_sha256": sha256_file(args.hash_table),
            "stage2b_control_path": str(args.stage2b_controls),
            "stage2b_control_sha256": sha256_file(args.stage2b_controls),
            "frozen_verifier_path": str(args.frozen_verifier),
            "frozen_verifier_sha256": frozen_sha,
            "stage2b_verifier_path": str(args.stage2b_verifier or ""),
            "stage2b_verifier_sha256": stage2b_sha,
            "stage2b_verifier_matches_frozen": stage2b_match,
            "candidate_pairs": len(candidate),
            "sentinel_controls": len(controls),
            "path_check_skipped": "YES" if args.skip_path_check else "NO",
        }
    ])
    write_tsv(provenance, args.output_dir / "stage2d_prepare_provenance.tsv")

    summary_rows = [
        {"table": "candidate", "category": "ALL", "count": len(candidate)},
        {"table": "control", "category": "ALL", "count": len(controls)},
    ]
    for category, count in candidate["evidence_tier"].value_counts().sort_index().items():
        summary_rows.append({"table": "candidate", "category": f"evidence_tier:{category}", "count": int(count)})
    for category, count in candidate["scientific_risk"].value_counts().sort_index().items():
        summary_rows.append({"table": "candidate", "category": f"scientific_risk:{category}", "count": int(count)})
    for category, count in controls["pair_origin"].value_counts().sort_index().items():
        summary_rows.append({"table": "control", "category": f"pair_origin:{category}", "count": int(count)})
    summary = pd.DataFrame(summary_rows)
    write_tsv(summary, args.output_dir / "stage2d_prepare_summary.tsv")
    (args.output_dir / ".stage2d_pairs_ready").write_text("OK\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(provenance.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
