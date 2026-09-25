"""Aggregate geometric verification, controls, and lineage families."""
from __future__ import annotations
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

class UnionFind:

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = (self.find(a), self.find(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False, na_rep='')

def finite(value: Any, default: float=float('nan')) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))

def is_capture_sequence_pair(row: pd.Series) -> bool:
    """Conservative hint for separate frames from the CocoaMonilia phone sequence.

    These pairs may be highly overlapping biological scenes, but differing timestamped
    camera captures are not called export/augmentation descendants. They remain grouped
    later in leakage-safe scene families.
    """
    archives = str(row.get('archives', ''))
    if archives != 'CocoaMoniliaDataSet.zip':
        return False
    if str(row.get('shared_origin_keys', '')).strip():
        return False
    name_a = Path(str(row.get('path_a', ''))).name.lower()
    name_b = Path(str(row.get('path_b', ''))).name.lower()
    return name_a.startswith('img_') and name_b.startswith('img_') and (name_a != name_b) and (str(row.get('component_a', '')) != str(row.get('component_b', '')))

def classify(row: pd.Series) -> tuple[str, str, float]:
    entropy = min(finite(row.get('gray_entropy_a'), 0.0), finite(row.get('gray_entropy_b'), 0.0))
    direct_ncc = finite(row.get('direct_ncc'), -1.0)
    direct_grad = finite(row.get('direct_gradient_ncc'), -1.0)
    direct_ssim = finite(row.get('direct_ssim'), -1.0)
    inliers = int(finite(row.get('best_inliers'), 0.0))
    ratio = finite(row.get('best_inlier_ratio'), 0.0)
    reproj = finite(row.get('best_median_reprojection_error'), 999.0)
    cov_a = finite(row.get('best_coverage_a'), 0.0)
    cov_b = finite(row.get('best_coverage_b'), 0.0)
    cov_min, cov_max = (min(cov_a, cov_b), max(cov_a, cov_b))
    overlap = finite(row.get('best_overlap_fraction_smaller'), 0.0)
    aligned_ncc = finite(row.get('best_aligned_ncc'), -1.0)
    aligned_grad = finite(row.get('best_aligned_gradient_ncc'), -1.0)
    aligned_ssim = finite(row.get('best_aligned_ssim'), -1.0)
    direct_strong = entropy >= 4.0 and direct_ncc >= 0.965 and (direct_grad >= 0.9) and (direct_ssim >= 0.78)
    direct_moderate = entropy >= 4.5 and direct_ncc >= 0.94 and (direct_grad >= 0.82) and (direct_ssim >= 0.68)
    geom_strong = inliers >= 18 and ratio >= 0.35 and (cov_min >= 0.02) and (cov_max >= 0.08) and (reproj <= 4.5) and (overlap >= 0.1)
    photo_strong = aligned_ncc >= 0.7 and aligned_grad >= 0.55 and (aligned_ssim >= 0.4)
    geom_moderate = inliers >= 12 and ratio >= 0.25 and (cov_min >= 0.008) and (cov_max >= 0.05) and (reproj <= 6.5) and (overlap >= 0.06)
    photo_moderate = aligned_ncc >= 0.58 and aligned_grad >= 0.4 and (aligned_ssim >= 0.28)
    same_scene_geometry = inliers >= 18 and ratio >= 0.3 and (cov_min >= 0.015) and (cov_max >= 0.08) and (reproj <= 5.0) and (overlap >= 0.08)
    capture_sequence = is_capture_sequence_pair(row)
    if capture_sequence and same_scene_geometry:
        relation = 'GEOMETRIC_SAME_SCENE_ONLY'
        reason = 'timestamped_cocoa_capture_sequence_with_geometric_overlap'
    elif direct_strong:
        relation = 'VERIFIED_DERIVATIVE_STRONG'
        reason = 'direct_orientation_normalized_photometric_match'
    elif geom_strong and photo_strong:
        relation = 'VERIFIED_DERIVATIVE_STRONG'
        reason = 'strong_geometry_and_aligned_photometry'
    elif direct_moderate:
        relation = 'VERIFIED_DERIVATIVE_MODERATE'
        reason = 'moderate_direct_photometric_match'
    elif geom_moderate and photo_moderate:
        relation = 'VERIFIED_DERIVATIVE_MODERATE'
        reason = 'moderate_geometry_and_aligned_photometry'
    elif same_scene_geometry:
        relation = 'GEOMETRIC_SAME_SCENE_ONLY'
        reason = 'geometry_passed_but_photometry_below_derivative_threshold'
    elif inliers >= 8 or direct_ncc >= 0.9:
        relation = 'AMBIGUOUS_MANUAL_REVIEW'
        reason = 'partial_geometric_or_photometric_support'
    else:
        relation = 'REJECTED_NOT_DERIVATIVE'
        reason = 'insufficient_geometric_and_photometric_support'
    direct_score = (clip01((direct_ncc - 0.7) / 0.3) + clip01((direct_grad - 0.5) / 0.5) + clip01((direct_ssim - 0.4) / 0.6)) / 3.0
    geometry_score = (clip01(inliers / 40.0) + clip01(ratio / 0.6) + clip01(cov_min / 0.08) + clip01(cov_max / 0.25) + clip01((7.0 - reproj) / 7.0) + clip01(overlap / 0.35)) / 6.0
    photo_score = (clip01((aligned_ncc - 0.3) / 0.7) + clip01((aligned_grad - 0.2) / 0.8) + clip01((aligned_ssim - 0.15) / 0.85)) / 3.0
    score = 100.0 * max(direct_score, 0.6 * geometry_score + 0.4 * photo_score)
    return (relation, reason, score)

def load_task_tables(directory: Path, prefix: str) -> pd.DataFrame:
    files = sorted((path for path in directory.glob(f'{prefix}_task_*.tsv') if not path.name.endswith('_status.tsv')))
    if not files:
        raise RuntimeError(f'No task tables found in {directory} with prefix {prefix}')
    return pd.concat([pd.read_csv(path, sep='\t', keep_default_na=False, low_memory=False) for path in files], ignore_index=True)

def verify_completeness(prepared: pd.DataFrame, raw: pd.DataFrame, label: str) -> None:
    expected = set(prepared['pair_id'].astype(str))
    observed = list(raw['pair_id'].astype(str))
    duplicates = [item for item, count in Counter(observed).items() if count > 1]
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    failed = raw[~raw['verification_status'].eq('COMPLETED')]
    if duplicates or missing or unexpected or len(failed):
        raise RuntimeError(f"{label} verification incomplete: duplicates={duplicates[:5]} missing={missing[:5]} unexpected={unexpected[:5]} failed={failed[['pair_id', 'verification_error']].head().to_dict('records')}")

def relation_is_derivative(series: pd.Series) -> pd.Series:
    return series.isin(['VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE'])

def control_summary(controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for (origin, transform), group in controls.groupby(['pair_origin', 'synthetic_transform'], dropna=False, sort=True):
        predicted = relation_is_derivative(group['verification_relation'])
        expected_positive = group['expected_relation'].eq('DERIVATIVE')
        rows.append({'pair_origin': origin, 'synthetic_transform': transform, 'n': len(group), 'expected_derivative': int(expected_positive.sum()), 'predicted_derivative': int(predicted.sum()), 'sensitivity_if_positive': float(predicted[expected_positive].mean()) if expected_positive.any() else float('nan'), 'false_positive_rate_if_negative': float(predicted[~expected_positive].mean()) if (~expected_positive).any() else float('nan'), 'strong': int(group['verification_relation'].eq('VERIFIED_DERIVATIVE_STRONG').sum()), 'moderate': int(group['verification_relation'].eq('VERIFIED_DERIVATIVE_MODERATE').sum()), 'same_scene_only': int(group['verification_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY').sum()), 'ambiguous': int(group['verification_relation'].eq('AMBIGUOUS_MANUAL_REVIEW').sum()), 'rejected': int(group['verification_relation'].eq('REJECTED_NOT_DERIVATIVE').sum())})
    summary = pd.DataFrame(rows)
    expected_positive = controls['expected_relation'].eq('DERIVATIVE')
    predicted = relation_is_derivative(controls['verification_relation'])
    overall = {'positive_sensitivity': float(predicted[expected_positive].mean()), 'negative_false_positive_rate': float(predicted[~expected_positive].mean()), 'exact_positive_sensitivity': float(predicted[controls['pair_origin'].eq('EXACT_POSITIVE_CONTROL')].mean()), 'synthetic_positive_sensitivity': float(predicted[controls['pair_origin'].eq('SYNTHETIC_POSITIVE_CONTROL')].mean()), 'n_controls': len(controls)}
    qc_rows = [{'criterion': 'exact_positive_sensitivity_ge_0.95', 'value': overall['exact_positive_sensitivity'], 'pass': 'YES' if overall['exact_positive_sensitivity'] >= 0.95 else 'NO'}, {'criterion': 'synthetic_positive_sensitivity_ge_0.75', 'value': overall['synthetic_positive_sensitivity'], 'pass': 'YES' if overall['synthetic_positive_sensitivity'] >= 0.75 else 'NO'}, {'criterion': 'random_negative_fpr_le_0.01', 'value': overall['negative_false_positive_rate'], 'pass': 'YES' if overall['negative_false_positive_rate'] <= 0.01 else 'NO'}]
    qc = pd.DataFrame(qc_rows)
    overall['qc_pass'] = 'YES' if qc['pass'].eq('YES').all() else 'NO'
    return (summary, qc, overall)

def set_join(values: Iterable[Any]) -> str:
    return '|'.join(sorted({str(value) for value in values if str(value) and str(value) not in {'unspecified', 'UNMATCHED'}}))

def build_lineages(canonical: pd.DataFrame, candidate: pd.DataFrame, include_moderate: bool, prefix: str, include_same_scene: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = sorted(set(canonical.loc[canonical['exact_component_id'].ne('NO_HASH'), 'exact_component_id'].astype(str)))
    uf = UnionFind(nodes)
    allowed = {'VERIFIED_DERIVATIVE_STRONG'}
    if include_moderate:
        allowed.add('VERIFIED_DERIVATIVE_MODERATE')
    if include_same_scene:
        allowed.add('GEOMETRIC_SAME_SCENE_ONLY')
    edges = candidate[candidate['verification_relation'].isin(allowed)]
    for row in edges.itertuples(index=False):
        if row.component_a in uf.parent and row.component_b in uf.parent:
            uf.union(row.component_a, row.component_b)
    root_to_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for node in nodes:
        root_to_nodes[uf.find(node)].append(node)
    ordered = sorted(root_to_nodes.values(), key=lambda members: (min(members), len(members)))
    node_to_lineage = {}
    for index, members in enumerate(ordered, 1):
        lineage_id = f'{prefix}{index:06d}'
        for node in members:
            node_to_lineage[node] = lineage_id
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid['lineage_id'] = valid['exact_component_id'].map(node_to_lineage)
    component_rows = []
    for lineage_id, group in valid.groupby('lineage_id', sort=True):
        component_ids = sorted(group['exact_component_id'].unique())
        component_rows.append({'lineage_id': lineage_id, 'n_exact_components': len(component_ids), 'n_photo_paths': len(group), 'archives': set_join(group['source_archive']), 'archive_count': group['source_archive'].nunique(), 'labels': set_join(group['corrected_label']), 'label_count': group['corrected_label'].nunique(), 'splits': set_join(group['corrected_split']), 'split_count': len({v for v in group['corrected_split'] if v not in {'', 'unspecified', 'UNMATCHED'}}), 'exact_component_ids': '|'.join(component_ids), 'cross_archive': 'YES' if group['source_archive'].nunique() > 1 else 'NO', 'label_conflict': 'YES' if group['corrected_label'].nunique() > 1 else 'NO', 'cross_split': 'YES' if len({v for v in group['corrected_split'] if v not in {'', 'unspecified', 'UNMATCHED'}}) > 1 else 'NO'})
    components = pd.DataFrame(component_rows)
    member_cols = ['lineage_id', 'exact_component_id', 'photo_path_id', 'source_archive', 'source_dataset_id', 'corrected_label', 'corrected_split', 'relative_path', 'absolute_path', 'sha256', 'pixel_sha256', 'supervised_eligible', 'exclusion_reason']
    members = valid[member_cols].sort_values(['lineage_id', 'exact_component_id', 'source_archive', 'relative_path'])
    return (components, members)

def source_overlap_from_lineages(components: pd.DataFrame) -> pd.DataFrame:
    archives = sorted({archive for value in components['archives'] for archive in str(value).split('|') if archive})
    lineage_sets = {archive: set(components.loc[components['archives'].str.split('|').apply(lambda xs: archive in xs), 'lineage_id']) for archive in archives}
    rows = []
    for a in archives:
        for b in archives:
            shared = len(lineage_sets[a] & lineage_sets[b])
            rows.append({'archive_a': a, 'archive_b': b, 'lineage_units_a': len(lineage_sets[a]), 'lineage_units_b': len(lineage_sets[b]), 'shared_lineage_units': shared, 'pct_a_shared': 100.0 * shared / max(1, len(lineage_sets[a])), 'pct_b_shared': 100.0 * shared / max(1, len(lineage_sets[b]))})
    return pd.DataFrame(rows)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--candidate-task-dir', required=True, type=Path)
    parser.add_argument('--control-task-dir', required=True, type=Path)
    parser.add_argument('--canonical-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    prepared_candidates = pd.read_csv(args.prepared_dir / 'stage2b_candidate_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    prepared_controls = pd.read_csv(args.prepared_dir / 'stage2b_control_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    candidate = load_task_tables(args.candidate_task_dir, 'candidate')
    controls = load_task_tables(args.control_task_dir, 'control')
    verify_completeness(prepared_candidates, candidate, 'candidate')
    verify_completeness(prepared_controls, controls, 'control')
    for table in (candidate, controls):
        classifications = table.apply(classify, axis=1, result_type='expand')
        classifications.columns = ['verification_relation', 'verification_reason', 'verification_score']
        table[['verification_relation', 'verification_reason', 'verification_score']] = classifications
        table['manual_review_priority'] = np.where(table['cross_archive'].eq('YES') | table['cross_split'].eq('YES') | table['label_conflict'].eq('YES'), 'YES', 'NO')
    control_metrics, control_qc, overall = control_summary(controls)
    candidate_summary = candidate.groupby(['verification_relation', 'cross_archive', 'cross_split', 'label_conflict'], dropna=False).size().reset_index(name='pair_count').sort_values(['verification_relation', 'cross_archive', 'cross_split', 'label_conflict'])
    canonical = pd.read_csv(args.canonical_manifest, sep='\t', keep_default_na=False, low_memory=False)
    strict_components, strict_members = build_lineages(canonical, candidate, include_moderate=False, prefix='STRICTL')
    extended_components, extended_members = build_lineages(canonical, candidate, include_moderate=True, prefix='EXTL')
    leakage_components, leakage_members = build_lineages(canonical, candidate, include_moderate=True, prefix='SAFEL', include_same_scene=True)
    strict_overlap = source_overlap_from_lineages(strict_components)
    extended_overlap = source_overlap_from_lineages(extended_components)
    leakage_overlap = source_overlap_from_lineages(leakage_components)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(candidate.sort_values(['verification_relation', 'verification_score'], ascending=[True, False]), args.output_dir / 'stage2b_candidate_pair_verification.tsv')
    write_tsv(controls.sort_values(['pair_origin', 'synthetic_transform', 'pair_id']), args.output_dir / 'stage2b_control_pair_verification.tsv')
    write_tsv(candidate_summary, args.output_dir / 'stage2b_candidate_verification_summary.tsv')
    write_tsv(control_metrics, args.output_dir / 'stage2b_control_performance.tsv')
    write_tsv(control_qc, args.output_dir / 'stage2b_control_qc.tsv')
    write_tsv(strict_components, args.output_dir / 'stage2b_strict_lineage_components.tsv')
    write_tsv(strict_members, args.output_dir / 'stage2b_strict_lineage_members.tsv')
    write_tsv(extended_components, args.output_dir / 'stage2b_extended_lineage_components.tsv')
    write_tsv(extended_members, args.output_dir / 'stage2b_extended_lineage_members.tsv')
    write_tsv(leakage_components, args.output_dir / 'stage2b_leakage_safe_family_components.tsv')
    write_tsv(leakage_members, args.output_dir / 'stage2b_leakage_safe_family_members.tsv')
    write_tsv(strict_overlap, args.output_dir / 'stage2b_strict_source_overlap.tsv')
    write_tsv(extended_overlap, args.output_dir / 'stage2b_extended_source_overlap.tsv')
    write_tsv(leakage_overlap, args.output_dir / 'stage2b_leakage_safe_source_overlap.tsv')
    derivative = relation_is_derivative(candidate['verification_relation'])
    write_tsv(candidate[derivative], args.output_dir / 'stage2b_verified_derivative_pairs.tsv')
    write_tsv(candidate[derivative & candidate['cross_archive'].eq('YES')], args.output_dir / 'stage2b_verified_cross_archive_pairs.tsv')
    write_tsv(candidate[derivative & candidate['cross_split'].eq('YES')], args.output_dir / 'stage2b_verified_cross_split_pairs.tsv')
    write_tsv(candidate[derivative & candidate['label_conflict'].eq('YES')], args.output_dir / 'stage2b_verified_label_conflict_pairs.tsv')
    same_scene = candidate['verification_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY')
    write_tsv(candidate[same_scene], args.output_dir / 'stage2b_geometric_same_scene_pairs.tsv')
    write_tsv(candidate[same_scene & candidate['cross_split'].eq('YES')], args.output_dir / 'stage2b_cross_split_same_scene_pairs.tsv')
    write_tsv(candidate[candidate['verification_relation'].isin(['GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW']) & candidate['manual_review_priority'].eq('YES')], args.output_dir / 'stage2b_priority_manual_review.tsv')
    relation_counts = candidate['verification_relation'].value_counts()
    summary_lines = ['# Geometric-verification summary', '', f'- Perceptual candidate pairs evaluated: {len(candidate):,}', f"- Strong verified derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_STRONG', 0)):,}", f"- Moderate verified derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_MODERATE', 0)):,}", f"- Geometric same-scene-only pairs: {int(relation_counts.get('GEOMETRIC_SAME_SCENE_ONLY', 0)):,}", f"- Ambiguous manual-review pairs: {int(relation_counts.get('AMBIGUOUS_MANUAL_REVIEW', 0)):,}", f"- Rejected candidate pairs: {int(relation_counts.get('REJECTED_NOT_DERIVATIVE', 0)):,}", '', '## Critical verified relations', f"- Verified cross-archive derivatives: {int((derivative & candidate['cross_archive'].eq('YES')).sum()):,}", f"- Verified cross-split derivatives: {int((derivative & candidate['cross_split'].eq('YES')).sum()):,}", f"- Verified label-conflict derivatives: {int((derivative & candidate['label_conflict'].eq('YES')).sum()):,}", '', '## Verification controls', f"- Exact-positive sensitivity: {overall['exact_positive_sensitivity']:.4f}", f"- Synthetic-positive sensitivity: {overall['synthetic_positive_sensitivity']:.4f}", f"- Random-negative false-positive rate: {overall['negative_false_positive_rate']:.4f}", f"- Control QC pass: {overall['qc_pass']}", '', '## Lineage-family reconstruction', f'- Strict lineage families: {len(strict_components):,}', f"- Strict multi-exact-component families: {int((strict_components['n_exact_components'] > 1).sum()):,}", f"- Strict cross-archive families: {int(strict_components['cross_archive'].eq('YES').sum()):,}", f"- Strict cross-split families: {int(strict_components['cross_split'].eq('YES').sum()):,}", f"- Strict label-conflict families: {int(strict_components['label_conflict'].eq('YES').sum()):,}", f'- Leakage-safe families (derivatives plus same-scene): {len(leakage_components):,}', f"- Leakage-safe cross-split families: {int(leakage_components['cross_split'].eq('YES').sum()):,}", '', '## Claim boundary', 'Strong and moderate derivative classes require both geometric support and aligned photometric support,', 'or exceptionally strong orientation-normalized direct photometric identity. Geometric same-scene-only pairs', 'are not merged into strict derivative lineages, but they are merged into a separate leakage-safe scene-family', 'definition for benchmark splitting. Ambiguous pairs require human review. Geometric verification remains', 'candidate-driven; an all-image embedding sweep is needed to estimate lineages missed by perceptual hashing.']
    (args.output_dir / 'stage2b_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    claim_status = pd.DataFrame([{'control_qc_pass': overall['qc_pass'], 'candidate_lineage_calls_publication_ready': 'YES' if overall['qc_pass'] == 'YES' else 'NO', 'claim_boundary': 'Candidate-derived lineage calls may be interpreted only with the control-QC table; same-scene-only pairs are not derivative lineages.'}])
    write_tsv(claim_status, args.output_dir / 'stage2b_claim_status.tsv')
    if overall['qc_pass'] == 'YES':
        (args.output_dir / '.stage2b_complete').write_text('OK\n', encoding='utf-8')
    else:
        (args.output_dir / '.stage2b_qc_failed').write_text('CONTROL_QC_FAILED\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
