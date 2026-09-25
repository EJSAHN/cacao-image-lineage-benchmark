"""Aggregate primary geometric verification and construct provisional lineage-safe units.

Primary verification reuses the frozen geometric and photometric verifier and its
pre-specified relation thresholds.  The 6,000 evidence reranking pairs are treated as a
risk-prioritized discovery tranche.  Verified edges are integrated with Stage
2B families, but the resulting family freeze remains provisional until the
reserve-tranche decision is resolved.
"""
from __future__ import annotations
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
RELATION_ORDER = ['VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE', 'GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW', 'REJECTED_NOT_DERIVATIVE']
DERIVATIVE_RELATIONS = {'VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE'}
SAFE_RELATIONS = DERIVATIVE_RELATIONS | {'GEOMETRIC_SAME_SCENE_ONLY'}

class UnionFind:

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {str(item): str(item) for item in items}
        self.rank = {str(item): 0 for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> bool:
        a, b = (str(a), str(b))
        ra, rb = (self.find(a), self.find(b))
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

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
    archives = str(row.get('archives', ''))
    if archives != 'CocoaMoniliaDataSet.zip':
        return False
    if str(row.get('shared_origin_keys', '')).strip():
        return False
    name_a = Path(str(row.get('path_a', ''))).name.lower()
    name_b = Path(str(row.get('path_b', ''))).name.lower()
    return name_a.startswith('img_') and name_b.startswith('img_') and (name_a != name_b) and (str(row.get('component_a', '')) != str(row.get('component_b', '')))

def classify(row: pd.Series) -> tuple[str, str, float]:
    """Frozen geometric verification relation classifier; thresholds intentionally unchanged."""
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
    observed_list = list(raw['pair_id'].astype(str))
    duplicates = [item for item, count in Counter(observed_list).items() if count > 1]
    missing = sorted(expected - set(observed_list))
    unexpected = sorted(set(observed_list) - expected)
    failed = raw[~raw['verification_status'].eq('COMPLETED')]
    if duplicates or missing or unexpected or len(failed):
        raise RuntimeError(f"{label} verification incomplete: duplicates={duplicates[:5]} missing={missing[:5]} unexpected={unexpected[:5]} failed={failed[['pair_id', 'verification_error']].head().to_dict('records')}")

def relation_is_derivative(series: pd.Series) -> pd.Series:
    return series.isin(DERIVATIVE_RELATIONS)

def relation_is_safe(series: pd.Series) -> pd.Series:
    return series.isin(SAFE_RELATIONS)

def control_summary(controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for (origin, transform), group in controls.groupby(['pair_origin', 'synthetic_transform'], dropna=False, sort=True):
        predicted = relation_is_derivative(group['verification_relation'])
        expected_positive = group['expected_relation'].eq('DERIVATIVE')
        rows.append({'pair_origin': origin, 'synthetic_transform': transform, 'n': len(group), 'expected_positive': 'YES' if expected_positive.all() else 'NO', 'strong': int(group['verification_relation'].eq('VERIFIED_DERIVATIVE_STRONG').sum()), 'moderate': int(group['verification_relation'].eq('VERIFIED_DERIVATIVE_MODERATE').sum()), 'same_scene_only': int(group['verification_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY').sum()), 'ambiguous': int(group['verification_relation'].eq('AMBIGUOUS_MANUAL_REVIEW').sum()), 'rejected': int(group['verification_relation'].eq('REJECTED_NOT_DERIVATIVE').sum()), 'sensitivity_if_positive': float(predicted.mean()) if expected_positive.all() else float('nan'), 'false_positive_rate_if_negative': float(predicted.mean()) if (~expected_positive).all() else float('nan')})
    summary = pd.DataFrame(rows)
    predicted = relation_is_derivative(controls['verification_relation'])
    exact_mask = controls['pair_origin'].eq('EXACT_POSITIVE_CONTROL')
    synthetic_mask = controls['pair_origin'].eq('SYNTHETIC_POSITIVE_CONTROL')
    negative_mask = controls['pair_origin'].eq('RANDOM_NEGATIVE_CONTROL')
    overall = {'exact_positive_sensitivity': float(predicted[exact_mask].mean()), 'synthetic_positive_sensitivity': float(predicted[synthetic_mask].mean()), 'negative_false_positive_rate': float(predicted[negative_mask].mean()), 'n_controls': len(controls)}
    qc = pd.DataFrame([{'criterion': 'exact_positive_sensitivity_ge_0.95', 'value': overall['exact_positive_sensitivity'], 'pass': 'YES' if overall['exact_positive_sensitivity'] >= 0.95 else 'NO'}, {'criterion': 'synthetic_positive_sensitivity_ge_0.75', 'value': overall['synthetic_positive_sensitivity'], 'pass': 'YES' if overall['synthetic_positive_sensitivity'] >= 0.75 else 'NO'}, {'criterion': 'random_negative_fpr_le_0.01', 'value': overall['negative_false_positive_rate'], 'pass': 'YES' if overall['negative_false_positive_rate'] <= 0.01 else 'NO'}])
    overall['qc_pass'] = 'YES' if qc['pass'].eq('YES').all() else 'NO'
    return (summary, qc, overall)

def set_join(values: Iterable[Any], keep_unspecified: bool=False) -> str:
    skip = {'', 'UNMATCHED'}
    if not keep_unspecified:
        skip.add('unspecified')
    return '|'.join(sorted({str(v) for v in values if str(v) not in skip}))

def initialize_from_family_members(nodes: list[str], members: pd.DataFrame, family_column: str) -> UnionFind:
    uf = UnionFind(nodes)
    for _, group in members.groupby(family_column, sort=False):
        component_ids = [str(v) for v in group['exact_component_id'].unique() if str(v) in uf.parent]
        if component_ids:
            anchor = component_ids[0]
            for component in component_ids[1:]:
                uf.union(anchor, component)
    return uf

def hidden_same_scene_edges(hidden: pd.DataFrame) -> pd.DataFrame:
    if hidden.empty:
        return hidden.copy()
    return hidden[hidden['stage2b_control_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY') & hidden['recommended_action'].eq('ADD_TO_LEAKAGE_SAFE_FAMILY_WITHOUT_REVERIFICATION')].copy()

def evidence_sorted(candidate: pd.DataFrame) -> pd.DataFrame:
    table = candidate.copy()
    for col in ('tier_order', 'efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score'):
        table[col] = pd.to_numeric(table[col], errors='coerce')
    table = table.sort_values(['tier_order', 'efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score', 'pair_id'], ascending=[True, False, False, False, True]).reset_index(drop=True)
    table['evidence_rank'] = np.arange(1, len(table) + 1)
    table['evidence_decile'] = np.minimum(10, np.ceil(table['evidence_rank'] * 10.0 / max(1, len(table))).astype(int))
    return table

def mark_incremental_merges(candidate: pd.DataFrame, nodes: list[str], baseline_strict_members: pd.DataFrame, baseline_safe_members: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    ordered = evidence_sorted(candidate)
    strict_uf = initialize_from_family_members(nodes, baseline_strict_members, 'lineage_id')
    safe_uf = initialize_from_family_members(nodes, baseline_safe_members, 'lineage_id')
    hidden_edges = hidden_same_scene_edges(hidden)
    hidden_merges = 0
    for row in hidden_edges.itertuples(index=False):
        if str(row.component_a) in safe_uf.parent and str(row.component_b) in safe_uf.parent:
            hidden_merges += int(safe_uf.union(str(row.component_a), str(row.component_b)))
    strict_flags, safe_flags = ([], [])
    for row in ordered.itertuples(index=False):
        a, b = (str(row.component_a), str(row.component_b))
        relation = str(row.verification_relation)
        strict_new = False
        safe_new = False
        if relation == 'VERIFIED_DERIVATIVE_STRONG' and a in strict_uf.parent and (b in strict_uf.parent):
            strict_new = strict_uf.union(a, b)
        if relation in SAFE_RELATIONS and a in safe_uf.parent and (b in safe_uf.parent):
            safe_new = safe_uf.union(a, b)
        strict_flags.append('YES' if strict_new else 'NO')
        safe_flags.append('YES' if safe_new else 'NO')
    ordered['incremental_new_strict_family_merge'] = strict_flags
    ordered['incremental_new_leakage_safe_family_merge'] = safe_flags
    ordered['hidden_same_scene_merges_before_stage2d'] = hidden_merges
    return ordered

def family_membership_from_uf(canonical: pd.DataFrame, uf: UnionFind, prefix: str, family_id_column: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    root_to_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for node in sorted(uf.parent):
        root_to_nodes[uf.find(node)].append(node)
    ordered = sorted(root_to_nodes.values(), key=lambda members: (min(members), len(members)))
    component_to_family: dict[str, str] = {}
    for index, component_ids in enumerate(ordered, 1):
        family_id = f'{prefix}{index:06d}'
        for component in component_ids:
            component_to_family[component] = family_id
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid[family_id_column] = valid['exact_component_id'].map(component_to_family)
    component_rows = []
    for family_id, group in valid.groupby(family_id_column, sort=True):
        components = sorted(group['exact_component_id'].astype(str).unique())
        labels = sorted({str(v) for v in group['corrected_label'] if str(v)})
        atomic_splits = sorted({str(v) for v in group['corrected_split'] if str(v) not in {'', 'unspecified', 'UNMATCHED'}})
        archives = sorted({str(v) for v in group['source_archive'] if str(v)})
        all_cacao = group['is_cacao'].astype(str).eq('YES').all()
        all_readable = group['image_read_ok'].astype(str).eq('YES').all()
        reasons = []
        if not all_cacao:
            reasons.append('NON_CACAO_MEMBER')
        if not all_readable:
            reasons.append('UNREADABLE_IMAGE_MEMBER')
        if len(labels) != 1:
            reasons.append('FAMILY_LABEL_CONFLICT' if len(labels) > 1 else 'MISSING_FAMILY_LABEL')
        if labels and any((label.lower() in {'unknown', 'unresolved'} for label in labels)):
            reasons.append('UNRESOLVED_FAMILY_LABEL')
        eligible = 'YES' if not reasons else 'NO'
        component_rows.append({family_id_column: family_id, 'n_exact_components': len(components), 'n_photo_paths': len(group), 'archives': '|'.join(archives), 'archive_count': len(archives), 'labels': '|'.join(labels), 'label_count': len(labels), 'family_label': labels[0] if len(labels) == 1 else '', 'splits': '|'.join(atomic_splits), 'split_count': len(atomic_splits), 'exact_component_ids': '|'.join(components), 'cross_archive': 'YES' if len(archives) > 1 else 'NO', 'label_conflict': 'YES' if len(labels) > 1 else 'NO', 'cross_split': 'YES' if len(atomic_splits) > 1 else 'NO', 'family_supervised_eligible': eligible, 'family_exclusion_reason': '|'.join(reasons)})
    components = pd.DataFrame(component_rows)
    members = valid.merge(components[[family_id_column, 'family_label', 'family_supervised_eligible', 'family_exclusion_reason']], on=family_id_column, how='left')
    members = members.sort_values([family_id_column, 'exact_component_id', 'source_archive', 'relative_path'])
    return (components, members, component_to_family)

def construct_provisional_families(canonical: pd.DataFrame, candidate: pd.DataFrame, baseline_strict_members: pd.DataFrame, baseline_safe_members: pd.DataFrame, hidden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = sorted(set(canonical.loc[canonical['exact_component_id'].ne('NO_HASH'), 'exact_component_id'].astype(str)))
    strict_uf = initialize_from_family_members(nodes, baseline_strict_members, 'lineage_id')
    safe_uf = initialize_from_family_members(nodes, baseline_safe_members, 'lineage_id')
    for row in hidden_same_scene_edges(hidden).itertuples(index=False):
        a, b = (str(row.component_a), str(row.component_b))
        if a in safe_uf.parent and b in safe_uf.parent:
            safe_uf.union(a, b)
    for row in candidate.itertuples(index=False):
        a, b = (str(row.component_a), str(row.component_b))
        relation = str(row.verification_relation)
        if relation == 'VERIFIED_DERIVATIVE_STRONG' and a in strict_uf.parent and (b in strict_uf.parent):
            strict_uf.union(a, b)
        if relation in SAFE_RELATIONS and a in safe_uf.parent and (b in safe_uf.parent):
            safe_uf.union(a, b)
    strict_components, strict_members, strict_map = family_membership_from_uf(canonical, strict_uf, 'S2DSTRICT', 'provisional_strict_lineage_id')
    safe_components, safe_members, safe_map = family_membership_from_uf(canonical, safe_uf, 'S2DSAFE', 'provisional_leakage_safe_family_id')
    manifest = canonical.copy()
    manifest['provisional_strict_lineage_id'] = manifest['exact_component_id'].map(strict_map).fillna('')
    manifest['provisional_leakage_safe_family_id'] = manifest['exact_component_id'].map(safe_map).fillna('')
    safe_fields = safe_components.set_index('provisional_leakage_safe_family_id')[['family_label', 'family_supervised_eligible', 'family_exclusion_reason', 'n_exact_components', 'n_photo_paths', 'archive_count', 'cross_archive', 'cross_split', 'label_conflict']]
    manifest = manifest.join(safe_fields, on='provisional_leakage_safe_family_id', rsuffix='_family')
    manifest['recommended_split_group_id'] = manifest['provisional_leakage_safe_family_id']
    eligible_components = safe_components[safe_components['family_supervised_eligible'].eq('YES')].copy()
    representative_rows = []
    for family_id in eligible_components['provisional_leakage_safe_family_id']:
        group = manifest[manifest['provisional_leakage_safe_family_id'].eq(family_id)].copy()
        group['_rep_rank'] = np.where(group['exact_representative'].astype(str).eq('YES'), 0, 1)
        group = group.sort_values(['_rep_rank', 'bytes', 'source_archive', 'relative_path'], ascending=[True, False, True, True])
        row = group.iloc[0]
        family = eligible_components.loc[eligible_components['provisional_leakage_safe_family_id'].eq(family_id)].iloc[0]
        representative_rows.append({'provisional_leakage_safe_family_id': family_id, 'family_label': family['family_label'], 'n_exact_components': family['n_exact_components'], 'n_photo_paths': family['n_photo_paths'], 'archive_count': family['archive_count'], 'archives': family['archives'], 'representative_photo_path_id': row['photo_path_id'], 'representative_source_archive': row['source_archive'], 'representative_relative_path': row['relative_path'], 'representative_absolute_path': row['absolute_path'], 'recommended_split_group_id': family_id})
    supervised_units = pd.DataFrame(representative_rows)
    return (strict_components, strict_members, safe_components, safe_members, manifest, supervised_units)

def source_overlap_from_families(components: pd.DataFrame, family_col: str) -> pd.DataFrame:
    archives = sorted({archive for value in components['archives'] for archive in str(value).split('|') if archive})
    family_sets = {}
    for archive in archives:
        family_sets[archive] = set(components.loc[components['archives'].str.split('|').apply(lambda xs: archive in xs), family_col])
    rows = []
    for a in archives:
        for b in archives:
            shared = len(family_sets[a] & family_sets[b])
            rows.append({'archive_a': a, 'archive_b': b, 'family_units_a': len(family_sets[a]), 'family_units_b': len(family_sets[b]), 'shared_family_units': shared, 'pct_a_shared': 100.0 * shared / max(1, len(family_sets[a])), 'pct_b_shared': 100.0 * shared / max(1, len(family_sets[b]))})
    return pd.DataFrame(rows)

def wilson_interval(k: int, n: int, z: float=1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def discovery_and_recommendation(candidate: pd.DataFrame, control_qc_pass: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = evidence_sorted(candidate)
    ordered['leakage_safe_edge'] = relation_is_safe(ordered['verification_relation'])
    ordered['strict_edge'] = ordered['verification_relation'].eq('VERIFIED_DERIVATIVE_STRONG')
    ordered['critical_risk_edge'] = ordered['leakage_safe_edge'] & (ordered['cross_split'].eq('YES') | ordered['label_conflict'].eq('YES'))
    rows = []
    for decile, group in ordered.groupby('evidence_decile', sort=True):
        safe_n = int(group['leakage_safe_edge'].sum())
        strict_n = int(group['strict_edge'].sum())
        new_safe = int(group['incremental_new_leakage_safe_family_merge'].eq('YES').sum())
        new_strict = int(group['incremental_new_strict_family_merge'].eq('YES').sum())
        critical = int(group['critical_risk_edge'].sum())
        lo, hi = wilson_interval(safe_n, len(group))
        rows.append({'evidence_decile': int(decile), 'n_pairs': len(group), 'strict_edges': strict_n, 'leakage_safe_edges': safe_n, 'leakage_safe_rate': safe_n / max(1, len(group)), 'leakage_safe_rate_wilson_low': lo, 'leakage_safe_rate_wilson_high': hi, 'critical_cross_split_or_label_edges': critical, 'incremental_new_strict_family_merges': new_strict, 'incremental_new_leakage_safe_family_merges': new_safe})
    deciles = pd.DataFrame(rows)
    bottom = ordered[ordered['evidence_decile'].ge(9)]
    bottom_safe = int(relation_is_safe(bottom['verification_relation']).sum())
    bottom_n = len(bottom)
    bottom_rate = bottom_safe / max(1, bottom_n)
    bottom_lo, bottom_hi = wilson_interval(bottom_safe, bottom_n)
    bottom_critical = int((relation_is_safe(bottom['verification_relation']) & (bottom['cross_split'].eq('YES') | bottom['label_conflict'].eq('YES'))).sum())
    bottom_merges = int(bottom['incremental_new_leakage_safe_family_merge'].eq('YES').sum())
    if control_qc_pass != 'YES':
        recommendation = 'HALT_QC_FAILURE'
        reason = 'Sentinel-control QC failed; candidate classifications cannot be interpreted.'
    elif bottom_safe >= 12 or bottom_rate >= 0.01 or bottom_critical >= 5 or (bottom_merges >= 5):
        recommendation = 'FULL_RESERVE'
        reason = 'The weakest 20% of the primary tranche still yielded material lineage/same-scene discovery.'
    elif bottom_safe >= 3 or bottom_critical >= 1 or bottom_merges >= 1 or (bottom_hi >= 0.01):
        recommendation = 'TARGETED_RESERVE_ONLY'
        reason = 'Discovery persists near the boundary, but evidence does not justify verifying all reserve pairs indiscriminately.'
    else:
        recommendation = 'STOP_AFTER_PRIMARY'
        reason = 'The weakest 20% showed negligible discovery with a low upper confidence bound and no critical-risk hits.'
    recommendation_df = pd.DataFrame([{'recommendation': recommendation, 'bottom_20_percent_pairs': bottom_n, 'bottom_20_percent_leakage_safe_edges': bottom_safe, 'bottom_20_percent_leakage_safe_rate': bottom_rate, 'bottom_20_percent_wilson_low': bottom_lo, 'bottom_20_percent_wilson_high': bottom_hi, 'bottom_20_percent_critical_edges': bottom_critical, 'bottom_20_percent_incremental_safe_merges': bottom_merges, 'decision_rule': 'FULL_RESERVE if bottom safe edges>=12, rate>=1%, critical hits>=5, or new safe merges>=5; TARGETED if weaker evidence persists; otherwise STOP.', 'reason': reason}])
    return (deciles, recommendation_df)

def yield_summaries(candidate: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table = candidate.copy()
    table['source_pair'] = [' || '.join(sorted((str(a), str(b)))) for a, b in zip(table['archive_a'], table['archive_b'])]
    table['leakage_safe_edge'] = relation_is_safe(table['verification_relation'])
    table['derivative_edge'] = relation_is_derivative(table['verification_relation'])
    table['efficientnet_bin'] = pd.cut(pd.to_numeric(table['efficientnet_b0_similarity'], errors='coerce'), bins=[-np.inf, 0.75, 0.78, 0.8, 0.85, 0.9, 0.95, np.inf], labels=['<0.75', '0.75-<0.78', '0.78-<0.80', '0.80-<0.85', '0.85-<0.90', '0.90-<0.95', '>=0.95'], right=False).astype(str)
    outputs = {}
    for name, group_cols in {'relation_risk': ['verification_relation', 'evidence_tier', 'scientific_risk'], 'evidence_tier': ['evidence_tier'], 'scientific_risk': ['scientific_risk'], 'selection_reason': ['shortlist_selection_reason'], 'source_pair': ['source_pair'], 'similarity_bin': ['evidence_tier', 'efficientnet_bin']}.items():
        rows = []
        for keys, group in table.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(group_cols, keys)}
            safe = int(group['leakage_safe_edge'].sum())
            deriv = int(group['derivative_edge'].sum())
            row.update({'n_pairs': len(group), 'derivative_edges': deriv, 'leakage_safe_edges': safe, 'derivative_rate': deriv / max(1, len(group)), 'leakage_safe_rate': safe / max(1, len(group)), 'new_strict_family_merges': int(group['incremental_new_strict_family_merge'].eq('YES').sum()), 'new_leakage_safe_family_merges': int(group['incremental_new_leakage_safe_family_merge'].eq('YES').sum())})
            rows.append(row)
        outputs[name] = pd.DataFrame(rows)
    return outputs

def reserve_projection(primary: pd.DataFrame, reserve: pd.DataFrame) -> pd.DataFrame:
    if reserve.empty:
        return pd.DataFrame()
    primary = primary.copy()
    primary['leakage_safe_edge'] = relation_is_safe(primary['verification_relation'])
    rows = []
    groups = sorted(set(zip(reserve['evidence_tier'], reserve['scientific_risk'])))
    for tier, risk in groups:
        p = primary[primary['evidence_tier'].eq(tier) & primary['scientific_risk'].eq(risk)]
        r = reserve[reserve['evidence_tier'].eq(tier) & reserve['scientific_risk'].eq(risk)]
        k = int(p['leakage_safe_edge'].sum())
        n = len(p)
        rate = k / n if n else float('nan')
        lo, hi = wilson_interval(k, n)
        rows.append({'evidence_tier': tier, 'scientific_risk': risk, 'primary_pairs': n, 'primary_leakage_safe_edges': k, 'primary_leakage_safe_rate': rate, 'primary_rate_wilson_low': lo, 'primary_rate_wilson_high': hi, 'reserve_pairs': len(r), 'heuristic_expected_reserve_edges': rate * len(r) if n else float('nan'), 'heuristic_low': lo * len(r) if n else float('nan'), 'heuristic_high': hi * len(r) if n else float('nan'), 'warning': 'Projection is heuristic because primary shortlist selection was risk- and score-prioritized within strata.'})
    return pd.DataFrame(rows)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--candidate-task-dir', type=Path)
    parser.add_argument('--control-task-dir', type=Path)
    parser.add_argument('--canonical-manifest', type=Path)
    parser.add_argument('--stage2b-candidate-verification', type=Path)
    parser.add_argument('--stage2b-strict-members', type=Path)
    parser.add_argument('--stage2b-safe-members', type=Path)
    parser.add_argument('--hidden-control-relations', type=Path)
    parser.add_argument('--reserve-candidates', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert classify(pd.Series({'gray_entropy_a': 7, 'gray_entropy_b': 7, 'direct_ncc': 1, 'direct_gradient_ncc': 1, 'direct_ssim': 1}))[0] == 'VERIFIED_DERIVATIVE_STRONG'
        lo, hi = wilson_interval(0, 600)
        assert 0 <= lo <= hi < 0.01
        uf = UnionFind(['a', 'b', 'c'])
        assert uf.union('a', 'b') and (not uf.union('a', 'b'))
        print('PRIMARY_VERIFICATION_SELF_TEST=OK')
        return 0
    required_production = {
        'prepared_dir': args.prepared_dir,
        'candidate_task_dir': args.candidate_task_dir,
        'control_task_dir': args.control_task_dir,
        'canonical_manifest': args.canonical_manifest,
        'stage2b_candidate_verification': args.stage2b_candidate_verification,
        'stage2b_strict_members': args.stage2b_strict_members,
        'stage2b_safe_members': args.stage2b_safe_members,
        'hidden_control_relations': args.hidden_control_relations,
        'reserve_candidates': args.reserve_candidates,
        'output_dir': args.output_dir,
    }
    missing_production = [name for name, value in required_production.items() if value is None]
    if missing_production:
        raise RuntimeError(f'Missing required production arguments: {missing_production}')
    prepared_candidates = pd.read_csv(args.prepared_dir / 'stage2d_primary_candidate_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    prepared_controls = pd.read_csv(args.prepared_dir / 'stage2d_sentinel_control_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    candidate = load_task_tables(args.candidate_task_dir, 'candidate')
    controls = load_task_tables(args.control_task_dir, 'control')
    verify_completeness(prepared_candidates, candidate, 'candidate')
    verify_completeness(prepared_controls, controls, 'control')
    for table in (candidate, controls):
        classifications = table.apply(classify, axis=1, result_type='expand')
        classifications.columns = ['verification_relation', 'verification_reason', 'verification_score']
        table[['verification_relation', 'verification_reason', 'verification_score']] = classifications
        table['manual_review_priority'] = np.where(table['cross_archive'].eq('YES') | table['cross_split'].eq('YES') | table['label_conflict'].eq('YES'), 'YES', 'NO')
    control_metrics, control_qc, control_overall = control_summary(controls)
    canonical = pd.read_csv(args.canonical_manifest, sep='\t', keep_default_na=False, low_memory=False)
    baseline_candidate = pd.read_csv(args.stage2b_candidate_verification, sep='\t', keep_default_na=False, low_memory=False)
    baseline_strict_members = pd.read_csv(args.stage2b_strict_members, sep='\t', keep_default_na=False, low_memory=False)
    baseline_safe_members = pd.read_csv(args.stage2b_safe_members, sep='\t', keep_default_na=False, low_memory=False)
    hidden = pd.read_csv(args.hidden_control_relations, sep='\t', keep_default_na=False, low_memory=False)
    reserve = pd.read_csv(args.reserve_candidates, sep='\t', compression='gzip', keep_default_na=False, low_memory=False)
    nodes = sorted(set(canonical.loc[canonical['exact_component_id'].ne('NO_HASH'), 'exact_component_id'].astype(str)))
    candidate = mark_incremental_merges(candidate, nodes, baseline_strict_members, baseline_safe_members, hidden)
    strict_components, strict_members, safe_components, safe_members, benchmark_manifest, supervised_units = construct_provisional_families(canonical, candidate, baseline_strict_members, baseline_safe_members, hidden)
    safe_overlap = source_overlap_from_families(safe_components, 'provisional_leakage_safe_family_id')
    yield_tables = yield_summaries(candidate)
    decile_summary, reserve_recommendation = discovery_and_recommendation(candidate, control_overall['qc_pass'])
    projection = reserve_projection(candidate, reserve)
    hidden_edges = hidden_same_scene_edges(hidden)
    family_summary = pd.DataFrame([{'unit_definition': 'Physical photograph paths', 'unit_count': len(canonical)}, {'unit_definition': 'Exact SHA-256 components', 'unit_count': len(nodes)}, {'unit_definition': 'initial strict lineages', 'unit_count': baseline_strict_members['lineage_id'].nunique()}, {'unit_definition': 'provisional strict lineages', 'unit_count': len(strict_components)}, {'unit_definition': 'initial leakage-safe families', 'unit_count': baseline_safe_members['lineage_id'].nunique()}, {'unit_definition': 'provisional leakage-safe families', 'unit_count': len(safe_components)}, {'unit_definition': 'eligible supervised families', 'unit_count': int(safe_components['family_supervised_eligible'].eq('YES').sum())}])
    relation_summary = candidate.groupby(['verification_relation', 'evidence_tier', 'scientific_risk'], dropna=False).size().reset_index(name='pair_count').sort_values(['verification_relation', 'evidence_tier', 'scientific_risk'])
    derivative_mask = relation_is_derivative(candidate['verification_relation'])
    safe_mask = relation_is_safe(candidate['verification_relation'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(candidate.sort_values(['verification_relation', 'verification_score'], ascending=[True, False]), args.output_dir / 'stage2d_candidate_pair_verification.tsv')
    write_tsv(controls.sort_values(['pair_origin', 'synthetic_transform', 'pair_id']), args.output_dir / 'stage2d_control_pair_verification.tsv')
    write_tsv(control_metrics, args.output_dir / 'stage2d_control_performance.tsv')
    write_tsv(control_qc, args.output_dir / 'stage2d_control_qc.tsv')
    write_tsv(relation_summary, args.output_dir / 'stage2d_relation_summary.tsv')
    for name, table in yield_tables.items():
        write_tsv(table, args.output_dir / f'stage2d_yield_by_{name}.tsv')
    write_tsv(decile_summary, args.output_dir / 'stage2d_discovery_deciles.tsv')
    write_tsv(reserve_recommendation, args.output_dir / 'stage2d_reserve_recommendation.tsv')
    write_tsv(projection, args.output_dir / 'stage2d_reserve_yield_projection.tsv')
    write_tsv(family_summary, args.output_dir / 'stage2d_family_unit_summary.tsv')
    write_tsv(candidate[derivative_mask], args.output_dir / 'stage2d_verified_derivative_pairs.tsv')
    write_tsv(candidate[safe_mask], args.output_dir / 'stage2d_verified_leakage_safe_edges.tsv')
    write_tsv(candidate[safe_mask & candidate['cross_archive'].eq('YES')], args.output_dir / 'stage2d_verified_cross_archive_edges.tsv')
    write_tsv(candidate[safe_mask & candidate['cross_split'].eq('YES')], args.output_dir / 'stage2d_verified_cross_split_edges.tsv')
    write_tsv(candidate[safe_mask & candidate['label_conflict'].eq('YES')], args.output_dir / 'stage2d_verified_label_conflict_edges.tsv')
    write_tsv(candidate[candidate['incremental_new_strict_family_merge'].eq('YES')], args.output_dir / 'stage2d_incremental_new_strict_family_merges.tsv')
    write_tsv(candidate[candidate['incremental_new_leakage_safe_family_merge'].eq('YES')], args.output_dir / 'stage2d_incremental_new_leakage_safe_family_merges.tsv')
    write_tsv(candidate[candidate['verification_relation'].eq('AMBIGUOUS_MANUAL_REVIEW') & candidate['manual_review_priority'].eq('YES')], args.output_dir / 'stage2d_priority_ambiguous_manual_review.tsv')
    write_tsv(hidden_edges, args.output_dir / 'stage2d_hidden_control_same_scene_edges_added.tsv')
    write_tsv(strict_components, args.output_dir / 'stage2d_provisional_strict_lineage_components.tsv')
    write_tsv(strict_members, args.output_dir / 'stage2d_provisional_strict_lineage_members.tsv')
    write_tsv(safe_components, args.output_dir / 'stage2d_provisional_leakage_safe_family_components.tsv')
    write_tsv(safe_members, args.output_dir / 'stage2d_provisional_leakage_safe_family_members.tsv')
    write_tsv(safe_overlap, args.output_dir / 'stage2d_provisional_leakage_safe_source_overlap.tsv')
    write_tsv(benchmark_manifest, args.output_dir / 'stage2d_provisional_benchmark_manifest.tsv')
    write_tsv(supervised_units, args.output_dir / 'stage2d_provisional_supervised_family_units.tsv')
    relation_counts = candidate['verification_relation'].value_counts()
    rec = reserve_recommendation.iloc[0]
    summary_lines = ['# Primary-tranche geometric-verification summary', '', f'- Primary evidence reranking pairs verified: {len(candidate):,}', f"- Strong verified derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_STRONG', 0)):,}", f"- Moderate verified derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_MODERATE', 0)):,}", f"- Geometric same-scene-only pairs: {int(relation_counts.get('GEOMETRIC_SAME_SCENE_ONLY', 0)):,}", f"- Ambiguous manual-review pairs: {int(relation_counts.get('AMBIGUOUS_MANUAL_REVIEW', 0)):,}", f"- Rejected semantic/look-alike candidates: {int(relation_counts.get('REJECTED_NOT_DERIVATIVE', 0)):,}", '', '## Frozen-verifier controls', f"- Exact-positive sensitivity: {control_overall['exact_positive_sensitivity']:.4f}", f"- Synthetic-positive sensitivity: {control_overall['synthetic_positive_sensitivity']:.4f}", f"- Random-negative derivative FPR: {control_overall['negative_false_positive_rate']:.4f}", f"- Control QC pass: {control_overall['qc_pass']}", '', '## New critical relations', f"- Leakage-safe cross-archive edges: {int((safe_mask & candidate['cross_archive'].eq('YES')).sum()):,}", f"- Leakage-safe cross-split edges: {int((safe_mask & candidate['cross_split'].eq('YES')).sum()):,}", f"- Leakage-safe label-conflict edges: {int((safe_mask & candidate['label_conflict'].eq('YES')).sum()):,}", f"- Incremental new strict family merges: {int(candidate['incremental_new_strict_family_merge'].eq('YES').sum()):,}", f"- Incremental new leakage-safe family merges: {int(candidate['incremental_new_leakage_safe_family_merge'].eq('YES').sum()):,}", f'- Hidden-control same-scene edges added to leakage-safe grouping: {len(hidden_edges):,}', '', '## Provisional family reconstruction', f'- Provisional strict lineages: {len(strict_components):,}', f'- Provisional leakage-safe families: {len(safe_components):,}', f"- Provisional family-level supervised units: {int(safe_components['family_supervised_eligible'].eq('YES').sum()):,}", f"- Provisional label-conflict families excluded from supervision: {int(safe_components['label_conflict'].eq('YES').sum()):,}", '', '## Reserve decision', f"- Recommendation: {rec['recommendation']}", f"- Weakest 20% leakage-safe yield: {int(rec['bottom_20_percent_leakage_safe_edges'])}/{int(rec['bottom_20_percent_pairs'])} ({float(rec['bottom_20_percent_leakage_safe_rate']):.4f}; Wilson 95% {float(rec['bottom_20_percent_wilson_low']):.4f}-{float(rec['bottom_20_percent_wilson_high']):.4f})", f"- Reason: {rec['reason']}", '', '## Claim boundary', 'Primary candidate pairs are interpreted only after the unchanged geometric verification verifier thresholds and sentinel-control QC.', 'Strong derivatives enter provisional strict lineages. Strong, moderate, and geometric same-scene relations enter', 'provisional leakage-safe split groups. Archive, split, label, and embedding flags are prioritization variables,', 'not lineage evidence. The family freeze remains provisional until the reserve recommendation is resolved.']
    (args.output_dir / 'stage2d_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    claim_status = pd.DataFrame([{'frozen_verifier_control_qc_pass': control_overall['qc_pass'], 'stage2d_verified_pair_calls_interpretable': 'YES' if control_overall['qc_pass'] == 'YES' else 'NO', 'family_level_supervised_eligibility_recomputed': 'YES', 'stage2d_family_freeze_final': 'NO', 'reserve_recommendation': rec['recommendation'], 'claim_boundary': 'Embedding and risk flags selected pairs but did not establish lineage. Only frozen geometric/photometric relation classes are merged; same-scene-only edges affect split grouping, not strict derivative ancestry.'}])
    write_tsv(claim_status, args.output_dir / 'stage2d_claim_status.tsv')
    (args.output_dir / '.stage2d_complete').write_text('OK\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    print(claim_status.to_string(index=False))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
