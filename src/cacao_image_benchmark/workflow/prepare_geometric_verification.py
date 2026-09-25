#!/usr/bin/env python3
"""Prepare geometric verification candidate and control pair tables."""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SEED = 20260729
SYNTHETIC_TRANSFORMS = [
    "jpeg_q45",
    "brightness_065",
    "brightness_135",
    "center_crop_15",
    "corner_crop_20",
    "rotate90",
    "mirror_horizontal",
    "resize_045_jpeg60",
    "combined_crop_brightness_jpeg",
]


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def bit_distance_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 999


def choose_exact_control_pairs(canonical: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = canonical[
        canonical["image_read_ok"].eq("YES")
        & canonical["exact_component_id"].ne("NO_HASH")
    ].copy()
    groups = []
    for component_id, group in valid.groupby("exact_component_id", sort=True):
        group = group.drop_duplicates("absolute_path")
        if len(group) < 2:
            continue
        groups.append((component_id, group))
    groups.sort(
        key=lambda item: (
            -item[1]["source_archive"].nunique(),
            -len(item[1]),
            item[0],
        )
    )
    per_archive = Counter()
    for component_id, group in groups:
        # Prefer paths from different archives when possible.
        selected = None
        archives = sorted(group["source_archive"].unique())
        if len(archives) > 1:
            first = group[group["source_archive"].eq(archives[0])].sort_values("absolute_path").iloc[0]
            second = group[~group["source_archive"].eq(archives[0])].sort_values("absolute_path").iloc[0]
            selected = (first, second)
        else:
            ordered = group.sort_values(["source_archive", "relative_path"])
            selected = (ordered.iloc[0], ordered.iloc[1])
        a, b = selected
        archive_key = str(a["source_archive"])
        if len(archives) == 1 and per_archive[archive_key] >= max(20, limit // 5):
            continue
        per_archive[archive_key] += 1
        rows.append(
            {
                "pair_origin": "EXACT_POSITIVE_CONTROL",
                "expected_relation": "DERIVATIVE",
                "synthetic_transform": "",
                "component_a": component_id,
                "component_b": component_id,
                "absolute_path_a": a["absolute_path"],
                "absolute_path_b": b["absolute_path"],
                "path_a": a["relative_path"],
                "path_b": b["relative_path"],
                "archives": "|".join(sorted({str(a["source_archive"]), str(b["source_archive"])})),
                "labels": "|".join(sorted({str(a["corrected_label"]), str(b["corrected_label"])})),
                "splits": "|".join(sorted({str(a["corrected_split"]), str(b["corrected_split"])} - {"", "unspecified"})),
                "cross_archive": "YES" if a["source_archive"] != b["source_archive"] else "NO",
                "cross_split": "YES" if a["corrected_split"] != b["corrected_split"] else "NO",
                "label_conflict": "YES" if a["corrected_label"] != b["corrected_label"] else "NO",
                "candidate_strength": "CONTROL",
                "candidate_reasons": "same_exact_sha256",
                "min_phash_hamming": 0,
                "min_dhash_hamming": 0,
                "min_whash_hamming": 0,
                "shared_origin_keys": "",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def stratified_representatives(canonical: pd.DataFrame) -> pd.DataFrame:
    reps = canonical[
        canonical["exact_representative"].eq("YES")
        & canonical["image_read_ok"].eq("YES")
        & canonical["exact_component_id"].ne("NO_HASH")
    ].copy()
    reps["stratum"] = reps["source_archive"].astype(str) + "||" + reps["corrected_label"].astype(str)
    return reps.sort_values(["stratum", "exact_component_id"])


def choose_synthetic_controls(canonical: pd.DataFrame, per_transform: int) -> list[dict[str, Any]]:
    reps = stratified_representatives(canonical)
    strata = {key: group.to_dict("records") for key, group in reps.groupby("stratum", sort=True)}
    keys = sorted(strata)
    pointers = {key: 0 for key in keys}
    rows: list[dict[str, Any]] = []
    key_cursor = 0
    for transform in SYNTHETIC_TRANSFORMS:
        selected = 0
        attempts = 0
        while selected < per_transform and attempts < per_transform * max(20, len(keys) * 3):
            key = keys[key_cursor % len(keys)]
            key_cursor += 1
            attempts += 1
            items = strata[key]
            if not items:
                continue
            row = items[pointers[key] % len(items)]
            pointers[key] += 1
            rows.append(
                {
                    "pair_origin": "SYNTHETIC_POSITIVE_CONTROL",
                    "expected_relation": "DERIVATIVE",
                    "synthetic_transform": transform,
                    "component_a": row["exact_component_id"],
                    "component_b": row["exact_component_id"],
                    "absolute_path_a": row["absolute_path"],
                    "absolute_path_b": "",
                    "path_a": row["relative_path"],
                    "path_b": f"SYNTHETIC:{transform}",
                    "archives": row["source_archive"],
                    "labels": row["corrected_label"],
                    "splits": row["corrected_split"],
                    "cross_archive": "NO",
                    "cross_split": "NO",
                    "label_conflict": "NO",
                    "candidate_strength": "CONTROL",
                    "candidate_reasons": f"synthetic_{transform}",
                    "min_phash_hamming": "",
                    "min_dhash_hamming": "",
                    "min_whash_hamming": "",
                    "shared_origin_keys": row["source_origin_key"],
                }
            )
            selected += 1
    return rows


def choose_random_negative_controls(
    canonical: pd.DataFrame,
    hashes: pd.DataFrame,
    candidates: pd.DataFrame,
    limit: int,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    reps = stratified_representatives(canonical).copy()
    hash_meta = hashes.set_index("exact_component_id")[["phash_primary", "gray_entropy"]].to_dict("index")
    reps["aspect_ratio"] = pd.to_numeric(reps["width"], errors="coerce") / pd.to_numeric(reps["height"], errors="coerce")
    reps["aspect_bin"] = (reps["aspect_ratio"].fillna(1.0) * 10).round().astype(int)
    reps["megapixels"] = pd.to_numeric(reps["width"], errors="coerce") * pd.to_numeric(reps["height"], errors="coerce") / 1e6
    reps["mp_bin"] = pd.cut(reps["megapixels"], bins=[-1, 0.5, 1.5, 3, 8, 20, 1e9], labels=False).fillna(-1).astype(int)
    reps["negative_stratum"] = (
        reps["source_archive"].astype(str)
        + "||" + reps["corrected_label"].astype(str)
        + "||" + reps["aspect_bin"].astype(str)
        + "||" + reps["mp_bin"].astype(str)
    )
    forbidden = {tuple(sorted((str(a), str(b)))) for a, b in zip(candidates["component_a"], candidates["component_b"])}
    groups = [group.to_dict("records") for _, group in reps.groupby("negative_stratum", sort=True) if len(group) >= 2]
    rng.shuffle(groups)
    rows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    component_use = Counter()
    attempts = 0
    while len(rows) < limit and attempts < limit * 500:
        attempts += 1
        group = groups[attempts % len(groups)]
        a, b = rng.sample(group, 2)
        ca, cb = str(a["exact_component_id"]), str(b["exact_component_id"])
        pair = tuple(sorted((ca, cb)))
        if pair in forbidden or pair in used_pairs or ca == cb:
            continue
        if a["source_origin_key"] and a["source_origin_key"] == b["source_origin_key"]:
            continue
        ha = hash_meta.get(ca, {}).get("phash_primary", "")
        hb = hash_meta.get(cb, {}).get("phash_primary", "")
        if bit_distance_hex(str(ha), str(hb)) < 12:
            continue
        if component_use[ca] >= 3 or component_use[cb] >= 3:
            continue
        used_pairs.add(pair)
        component_use[ca] += 1
        component_use[cb] += 1
        rows.append(
            {
                "pair_origin": "RANDOM_NEGATIVE_CONTROL",
                "expected_relation": "NON_DERIVATIVE",
                "synthetic_transform": "",
                "component_a": ca,
                "component_b": cb,
                "absolute_path_a": a["absolute_path"],
                "absolute_path_b": b["absolute_path"],
                "path_a": a["relative_path"],
                "path_b": b["relative_path"],
                "archives": a["source_archive"],
                "labels": a["corrected_label"],
                "splits": "|".join(sorted({str(a["corrected_split"]), str(b["corrected_split"])} - {"", "unspecified"})),
                "cross_archive": "NO",
                "cross_split": "YES" if a["corrected_split"] != b["corrected_split"] else "NO",
                "label_conflict": "NO",
                "candidate_strength": "CONTROL",
                "candidate_reasons": "same_source_label_dimension_bin_random_negative",
                "min_phash_hamming": bit_distance_hex(str(ha), str(hb)),
                "min_dhash_hamming": "",
                "min_whash_hamming": "",
                "shared_origin_keys": "",
            }
        )
    if len(rows) < limit:
        raise RuntimeError(f"Could construct only {len(rows)} of {limit} random negative controls")
    return rows


def enrich_entropy(table: pd.DataFrame, hashes: pd.DataFrame) -> pd.DataFrame:
    meta = hashes.set_index("exact_component_id")[["gray_entropy", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]]
    for side in ("a", "b"):
        joined = table[[f"component_{side}"]].join(meta, on=f"component_{side}")
        for col in meta.columns:
            table[f"{col}_{side}"] = joined[col].values
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--candidate-table", required=True, type=Path)
    parser.add_argument("--hash-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exact-controls", type=int, default=240)
    parser.add_argument("--synthetic-per-transform", type=int, default=30)
    parser.add_argument("--negative-controls", type=int, default=600)
    args = parser.parse_args()

    canonical = pd.read_csv(args.canonical_manifest, sep="\t", keep_default_na=False, low_memory=False)
    candidates = pd.read_csv(args.candidate_table, sep="\t", keep_default_na=False, low_memory=False)
    hashes = pd.read_csv(args.hash_table, sep="\t", keep_default_na=False, low_memory=False)

    required = {"component_a", "component_b", "absolute_path_a", "absolute_path_b"}
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate table missing columns: {sorted(missing)}")

    candidate = candidates.copy()
    candidate.insert(0, "pair_origin", "STAGE2A_CANDIDATE")
    candidate.insert(1, "expected_relation", "UNKNOWN")
    candidate.insert(2, "synthetic_transform", "")
    # Critical pairs first, then stronger/smaller-distance pairs.
    strength_rank = {"HIGH_CANDIDATE": 0, "MEDIUM_CANDIDATE": 1, "ORIGIN_ONLY_CANDIDATE": 2, "PHASH_ONLY_CANDIDATE": 3}
    candidate["_priority"] = (
        candidate["cross_split"].eq("YES").astype(int) * 1000
        + candidate["cross_archive"].eq("YES").astype(int) * 500
        + candidate["label_conflict"].eq("YES").astype(int) * 250
        + (10 - candidate["candidate_strength"].map(strength_rank).fillna(9)) * 10
        + (64 - pd.to_numeric(candidate["min_phash_hamming"], errors="coerce").fillna(64))
    )
    candidate = candidate.sort_values(["_priority", "component_a", "component_b"], ascending=[False, True, True]).drop(columns="_priority")
    candidate.insert(0, "pair_id", [f"CAND{i:06d}" for i in range(1, len(candidate) + 1)])
    candidate = enrich_entropy(candidate, hashes)

    control_rows = []
    control_rows.extend(choose_exact_control_pairs(canonical, args.exact_controls))
    control_rows.extend(choose_synthetic_controls(canonical, args.synthetic_per_transform))
    control_rows.extend(choose_random_negative_controls(canonical, hashes, candidates, args.negative_controls))
    controls = pd.DataFrame(control_rows)
    controls.insert(0, "pair_id", [f"CTRL{i:06d}" for i in range(1, len(controls) + 1)])
    controls = enrich_entropy(controls, hashes)

    # Path existence is a strict preflight for all real-image sides.
    missing_paths = []
    for table_name, table in (("candidate", candidate), ("control", controls)):
        for side in ("a", "b"):
            col = f"absolute_path_{side}"
            for pair_id, path, transform in zip(table["pair_id"], table[col], table["synthetic_transform"]):
                if side == "b" and transform:
                    continue
                if not path or not Path(path).is_file():
                    missing_paths.append((table_name, pair_id, side, path))
    if missing_paths:
        raise RuntimeError(f"Missing pair paths, first examples: {missing_paths[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(candidate, args.output_dir / "stage2b_candidate_pairs.tsv")
    write_tsv(controls, args.output_dir / "stage2b_control_pairs.tsv")

    summary_rows = []
    for name, table in (("candidate", candidate), ("control", controls)):
        summary_rows.append({"table": name, "category": "ALL", "count": len(table)})
        category_col = "candidate_strength" if name == "candidate" else "pair_origin"
        for category, count in table[category_col].value_counts().sort_index().items():
            summary_rows.append({"table": name, "category": category, "count": int(count)})
    summary = pd.DataFrame(summary_rows)
    write_tsv(summary, args.output_dir / "stage2b_prepare_summary.tsv")
    (args.output_dir / ".stage2b_pairs_ready").write_text("OK\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
