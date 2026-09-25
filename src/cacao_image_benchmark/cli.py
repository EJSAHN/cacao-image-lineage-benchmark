from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from importlib.resources import as_file, files

COMMANDS = {
    "aggregate_control_analyses": "aggregate_control_analyses.py",
    "aggregate_design_confirmation": "aggregate_design_confirmation.py",
    "aggregate_end_to_end_sensitivity": "aggregate_end_to_end_sensitivity.py",
    "aggregate_frozen_benchmark": "aggregate_frozen_benchmark.py",
    "aggregate_primary_verification": "aggregate_primary_verification.py",
    "build_archive_manifest": "build_archive_manifest.py",
    "build_canonical_manifest": "build_canonical_manifest.py",
    "build_initial_lineages": "build_initial_lineages.py",
    "cache_training_images": "cache_training_images.py",
    "compute_perceptual_hashes": "compute_perceptual_hashes.py",
    "discover_perceptual_candidates": "discover_perceptual_candidates.py",
    "download_public_data": "download_public_data.py",
    "extract_archive": "extract_archive.py",
    "extract_image_embeddings": "extract_image_embeddings.py",
    "extract_shortcut_features": "extract_shortcut_features.py",
    "freeze_lineages_and_split_blocks": "freeze_lineages_and_split_blocks.py",
    "harmonize_labels": "harmonize_labels.py",
    "inventory_archives": "inventory_archives.py",
    "prepare_design_confirmation": "prepare_design_confirmation.py",
    "prepare_end_to_end_sensitivity": "prepare_end_to_end_sensitivity.py",
    "prepare_frozen_benchmark": "prepare_frozen_benchmark.py",
    "prepare_geometric_verification": "prepare_geometric_verification.py",
    "prepare_matched_and_shortcut_controls": "prepare_matched_and_shortcut_controls.py",
    "prepare_primary_verification": "prepare_primary_verification.py",
    "prepare_reserve_verification": "prepare_reserve_verification.py",
    "rank_embedding_candidates": "rank_embedding_candidates.py",
    "retrieve_embedding_candidates": "retrieve_embedding_candidates.py",
    "run_design_confirmation": "run_design_confirmation.py",
    "run_frozen_benchmark": "run_frozen_benchmark.py",
    "run_matched_removal": "run_matched_removal.py",
    "run_shortcut_benchmark": "run_shortcut_benchmark.py",
    "train_end_to_end_sensitivity": "train_end_to_end_sensitivity.py",
    "verify_image_pairs": "verify_image_pairs.py",
    "verify_primary_pairs": "verify_primary_pairs.py",
    "verify_reserve_pairs": "verify_reserve_pairs.py"
}


def workflow_path(filename: str):
    return files("cacao_image_benchmark.workflow").joinpath(filename)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cacao-benchmark",
        description="Lineage- and scene-aware reconstruction and benchmarking of public cacao disease image collections.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List reproducible workflow commands.")
    run = sub.add_parser("run", help="Run one workflow command and pass the remaining arguments to it.")
    run.add_argument("command", choices=sorted(COMMANDS))
    help_cmd = sub.add_parser("step-help", help="Show the original command-line options for one workflow command.")
    help_cmd.add_argument("command", choices=sorted(COMMANDS))
    sub.add_parser("validate-install", help="Compile every installed workflow module.")
    ns, remainder = parser.parse_known_args()

    if ns.action == "list":
        for name in sorted(COMMANDS):
            print(name)
        return 0
    if ns.action == "validate-install":
        failures = []
        for filename in COMMANDS.values():
            with as_file(workflow_path(filename)) as path:
                try:
                    py_compile.compile(str(path), doraise=True)
                except Exception as exc:
                    failures.append((filename, str(exc)))
        if failures:
            for filename, message in failures:
                print(f"{filename}: {message}", file=sys.stderr)
            return 1
        print(f"compiled={len(COMMANDS)}")
        return 0

    filename = COMMANDS[ns.command]
    forwarded = list(remainder)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if ns.action == "step-help":
        forwarded = ["--help"]
    with as_file(workflow_path(filename)) as path:
        completed = subprocess.run([sys.executable, str(path), *forwarded])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
