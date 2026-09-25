"""Aggregate the full evidence reranking reserve and freeze hierarchy-aware benchmark units.

final lineage freeze verifies every evidence-gated reserve pair with the byte-identical
geometric and photometric verifier. Strong derivative relations define
strict image lineages; moderate relations form an extended sensitivity layer;
capture-sequence same-scene relations form provisional capture bursts; and all
verified same-scene/context relations define conservative split blocks only.
Label eligibility is evaluated at strict-lineage level. Mixed-label split
blocks are allowed and remain together during train/validation/test partitioning.
"""
from __future__ import annotations
import hashlib
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
RELATION_PRIORITY = {'VERIFIED_DERIVATIVE_STRONG': 5, 'VERIFIED_DERIVATIVE_MODERATE': 4, 'GEOMETRIC_SAME_SCENE_ONLY': 3, 'AMBIGUOUS_MANUAL_REVIEW': 2, 'REJECTED_NOT_DERIVATIVE': 1}
INVALID_LABELS = {'', 'unknown', 'unresolved', 'none', 'nan', 'na', 'not_applicable'}
COARSE3_LABELS = {'healthy', 'black_pod', 'frosty_pod'}
CAUSAL5_LABELS = {'healthy', 'black_pod', 'frosty_pod', 'mirid_damage', 'pod_borer_damage'}

def unordered_pair_key(a: Any, b: Any) -> str:
    x, y = sorted((str(a), str(b)))
    return f'{x}||{y}'

def stable_group_id(prefix: str, members: Iterable[Any]) -> str:
    ordered = sorted({str(value) for value in members})
    digest = hashlib.sha256('\n'.join(ordered).encode('utf-8')).hexdigest()[:16]
    return f'{prefix}_{digest}'

def atomic_values(values: Iterable[Any], skip_unspecified: bool=False) -> list[str]:
    skip = {'', 'UNMATCHED'}
    if skip_unspecified:
        skip.add('unspecified')
    output: set[str] = set()
    for value in values:
        for token in str(value).split('|'):
            token = token.strip()
            if token and token not in skip:
                output.add(token)
    return sorted(output)

def coarse_label(value: Any) -> str:
    label = str(value).strip()
    mapping = {'healthy': 'healthy', 'black_pod': 'black_pod', 'frosty_pod': 'frosty_pod', 'frosty_pod_stage_m1': 'frosty_pod', 'frosty_pod_stage_m2': 'frosty_pod', 'frosty_pod_stage_m3': 'frosty_pod', 'mirid_damage': 'mirid_damage', 'pod_borer_damage': 'pod_borer_damage', 'coffee_healthy': 'coffee_healthy'}
    return mapping.get(label, label if label else '')

def stage_label(value: Any) -> str:
    label = str(value).strip()
    return {'healthy': 'healthy', 'frosty_pod_stage_m1': 'm1_hump', 'frosty_pod_stage_m2': 'm2_spot', 'frosty_pod_stage_m3': 'm3_sporulation'}.get(label, '')

def component_representatives(canonical: pd.DataFrame) -> pd.DataFrame:
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid['_rep_rank'] = np.where(valid['exact_representative'].astype(str).eq('YES'), 0, 1)
    valid['_area'] = pd.to_numeric(valid['width'], errors='coerce').fillna(0) * pd.to_numeric(valid['height'], errors='coerce').fillna(0)
    valid['_bytes'] = pd.to_numeric(valid['bytes'], errors='coerce').fillna(0)
    reps = valid.sort_values(['exact_component_id', '_rep_rank', '_area', '_bytes', 'source_archive', 'relative_path'], ascending=[True, True, False, False, True, True]).drop_duplicates('exact_component_id', keep='first')
    return reps.set_index('exact_component_id')

def derive_risk(row: pd.Series) -> str:
    risks = []
    if str(row.get('cross_split', 'NO')) == 'YES':
        risks.append('CROSS_SPLIT')
    if str(row.get('cross_archive', 'NO')) == 'YES':
        risks.append('CROSS_ARCHIVE')
    if str(row.get('label_conflict', 'NO')) == 'YES':
        risks.append('LABEL_CONFLICT')
    return '|'.join(risks) if risks else 'GENERAL'

def normalize_edge_table(table: pd.DataFrame, stage: str, reps: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    required = {'pair_id', 'component_a', 'component_b', 'verification_relation'}
    missing = required - set(output.columns)
    if missing:
        raise RuntimeError(f'{stage} edge table missing columns: {sorted(missing)}')
    output['component_a'] = output['component_a'].astype(str)
    output['component_b'] = output['component_b'].astype(str)
    output['edge_key'] = [unordered_pair_key(a, b) for a, b in zip(output['component_a'], output['component_b'])]
    output['edge_stage'] = stage
    output['edge_pair_id'] = output['pair_id'].astype(str)
    output['relation_priority'] = output['verification_relation'].map(RELATION_PRIORITY).fillna(0).astype(int)
    for side in ('a', 'b'):
        component_col = f'component_{side}'
        joined = output[[component_col]].join(reps[['source_archive', 'corrected_label', 'corrected_split', 'relative_path', 'absolute_path', 'source_origin_key', 'source_dataset_id']], on=component_col)
        fill_map = {f'archive_{side}': 'source_archive', f'label_{side}': 'corrected_label', f'split_{side}': 'corrected_split', f'path_{side}': 'relative_path', f'absolute_path_{side}': 'absolute_path', f'source_origin_key_{side}': 'source_origin_key', f'source_dataset_id_{side}': 'source_dataset_id'}
        for target, source in fill_map.items():
            if target not in output.columns:
                output[target] = joined[source].values
            else:
                mask = output[target].astype(str).eq('')
                output.loc[mask, target] = joined.loc[mask, source].values
    for flag, left, right in (('cross_archive', 'archive_a', 'archive_b'), ('cross_split', 'split_a', 'split_b'), ('label_conflict', 'label_a', 'label_b')):
        if flag not in output.columns:
            values = []
            for a, b in zip(output[left], output[right]):
                a_text, b_text = (str(a), str(b))
                if flag == 'cross_split':
                    valid_a = a_text not in {'', 'unspecified', 'UNMATCHED'}
                    valid_b = b_text not in {'', 'unspecified', 'UNMATCHED'}
                    values.append('YES' if valid_a and valid_b and (a_text != b_text) else 'NO')
                else:
                    values.append('YES' if a_text and b_text and (a_text != b_text) else 'NO')
            output[flag] = values
    if 'scientific_risk' not in output.columns:
        output['scientific_risk'] = output.apply(derive_risk, axis=1)
    if 'evidence_tier' not in output.columns:
        output['evidence_tier'] = stage
    if 'evidence_score' not in output.columns:
        output['evidence_score'] = pd.to_numeric(output.get('verification_score', 0.0), errors='coerce').fillna(0.0)
    if 'candidate_reasons' not in output.columns:
        output['candidate_reasons'] = ''
    return output

def hidden_edge_table(hidden: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    if hidden.empty:
        return pd.DataFrame()
    keep = hidden[hidden['stage2b_control_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY') & hidden['recommended_action'].eq('ADD_TO_LEAKAGE_SAFE_FAMILY_WITHOUT_REVERIFICATION')].copy()
    if keep.empty:
        return keep
    keep['verification_relation'] = 'GEOMETRIC_SAME_SCENE_ONLY'
    keep['verification_reason'] = 'verified_hidden_control_same_scene_relation'
    keep['verification_score'] = np.nan
    keep['pair_origin'] = 'STAGE2B_HIDDEN_CONTROL_RELATION'
    return normalize_edge_table(keep, 'HIDDEN_CONTROL', reps)

def deduplicate_edge_ledger(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ledger.empty:
        return (ledger.copy(), pd.DataFrame())
    audit_rows = []
    for edge_key, group in ledger.groupby('edge_key', sort=True):
        relations = sorted(set(group['verification_relation'].astype(str)))
        stages = sorted(set(group['edge_stage'].astype(str)))
        if len(group) > 1:
            audit_rows.append({'edge_key': edge_key, 'n_records': len(group), 'relations': '|'.join(relations), 'stages': '|'.join(stages), 'relation_disagreement': 'YES' if len(relations) > 1 else 'NO', 'pair_ids': '|'.join(sorted(group['edge_pair_id'].astype(str)))})
    ordered = ledger.sort_values(['edge_key', 'relation_priority', 'edge_stage', 'edge_pair_id'], ascending=[True, False, True, True])
    best = ordered.drop_duplicates('edge_key', keep='first').copy()
    return (best, pd.DataFrame(audit_rows))

def capture_burst_same_scene_mask(edges: pd.DataFrame) -> pd.Series:
    """Conservative subset of same-scene edges interpreted as capture bursts.

    Only timestamp-style CocoaMonilia image pairs are admitted. These groups are
    reported separately and are not used to determine label eligibility.
    """
    if edges.empty:
        return pd.Series(dtype=bool, index=edges.index)
    same_scene = edges['verification_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY')
    archive_match = edges.get('archive_a', '').astype(str).eq('CocoaMoniliaDataSet.zip') & edges.get('archive_b', '').astype(str).eq('CocoaMoniliaDataSet.zip')
    name_a = edges.get('path_a', '').astype(str).map(lambda value: Path(value).name.lower())
    name_b = edges.get('path_b', '').astype(str).map(lambda value: Path(value).name.lower())
    timestamp_style = name_a.str.startswith('img_') & name_b.str.startswith('img_') & name_a.ne(name_b)
    explicit_reason = edges.get('verification_reason', '').astype(str).eq('timestamped_cocoa_capture_sequence_with_geometric_overlap')
    return same_scene & (explicit_reason | archive_match & timestamp_style)

def initialize_capture_burst_uf(nodes: list[str], edges: pd.DataFrame) -> UnionFind:
    uf = initialize_uf(nodes, edges, DERIVATIVE_RELATIONS)
    if not edges.empty:
        for row in edges[capture_burst_same_scene_mask(edges)].itertuples(index=False):
            a, b = (str(row.component_a), str(row.component_b))
            if a in uf.parent and b in uf.parent:
                uf.union(a, b)
    return uf

def initialize_uf(nodes: list[str], edges: pd.DataFrame, relations: set[str]) -> UnionFind:
    uf = UnionFind(nodes)
    if edges.empty:
        return uf
    for row in edges[edges['verification_relation'].isin(relations)].itertuples(index=False):
        a, b = (str(row.component_a), str(row.component_b))
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    return uf

def groups_from_uf(uf: UnionFind, prefix: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    root_to_members: defaultdict[str, list[str]] = defaultdict(list)
    for node in sorted(uf.parent):
        root_to_members[uf.find(node)].append(node)
    component_to_group: dict[str, str] = {}
    group_to_components: dict[str, list[str]] = {}
    for members in sorted(root_to_members.values(), key=lambda x: (x[0], len(x))):
        group_id = stable_group_id(prefix, members)
        group_to_components[group_id] = sorted(members)
        for component in members:
            component_to_group[component] = group_id
    return (component_to_group, group_to_components)

def evidence_sort(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    for col in ('tier_order', 'efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score'):
        if col not in output.columns:
            output[col] = np.nan
        output[col] = pd.to_numeric(output[col], errors='coerce')
    output = output.sort_values(['tier_order', 'efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score', 'pair_id'], ascending=[True, False, False, False, True]).reset_index(drop=True)
    output['reserve_evidence_rank'] = np.arange(1, len(output) + 1)
    return output

def mark_reserve_incremental_merges(reserve: pd.DataFrame, nodes: list[str], baseline_edges: pd.DataFrame) -> pd.DataFrame:
    ordered = evidence_sort(reserve)
    strict_uf = initialize_uf(nodes, baseline_edges, {'VERIFIED_DERIVATIVE_STRONG'})
    extended_uf = initialize_uf(nodes, baseline_edges, DERIVATIVE_RELATIONS)
    split_uf = initialize_uf(nodes, baseline_edges, SAFE_RELATIONS)
    strict_flags, extended_flags, split_flags = ([], [], [])
    strict_cum, extended_cum, split_cum = (0, 0, 0)
    strict_cums, extended_cums, split_cums = ([], [], [])
    for row in ordered.itertuples(index=False):
        a, b = (str(row.component_a), str(row.component_b))
        relation = str(row.verification_relation)
        strict_new = False
        extended_new = False
        split_new = False
        if relation == 'VERIFIED_DERIVATIVE_STRONG' and a in strict_uf.parent and (b in strict_uf.parent):
            strict_new = strict_uf.union(a, b)
        if relation in DERIVATIVE_RELATIONS and a in extended_uf.parent and (b in extended_uf.parent):
            extended_new = extended_uf.union(a, b)
        if relation in SAFE_RELATIONS and a in split_uf.parent and (b in split_uf.parent):
            split_new = split_uf.union(a, b)
        strict_cum += int(strict_new)
        extended_cum += int(extended_new)
        split_cum += int(split_new)
        strict_flags.append('YES' if strict_new else 'NO')
        extended_flags.append('YES' if extended_new else 'NO')
        split_flags.append('YES' if split_new else 'NO')
        strict_cums.append(strict_cum)
        extended_cums.append(extended_cum)
        split_cums.append(split_cum)
    ordered['incremental_new_strict_merge'] = strict_flags
    ordered['incremental_new_extended_merge'] = extended_flags
    ordered['incremental_new_split_block_merge'] = split_flags
    ordered['cumulative_new_strict_merges'] = strict_cums
    ordered['cumulative_new_extended_merges'] = extended_cums
    ordered['cumulative_new_split_block_merges'] = split_cums
    return ordered

def lineage_table(canonical: pd.DataFrame, component_to_group: dict[str, str], group_to_components: dict[str, list[str]], group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid[group_col] = valid['exact_component_id'].map(component_to_group).fillna('')
    valid['coarse_label'] = valid['corrected_label'].map(coarse_label)
    valid['stage_label'] = valid['corrected_label'].map(stage_label)
    rows = []
    for group_id, group in valid.groupby(group_col, sort=True):
        fine_labels = sorted({str(v) for v in group['corrected_label'] if str(v).lower() not in INVALID_LABELS})
        coarse_labels = sorted({str(v) for v in group['coarse_label'] if str(v).lower() not in INVALID_LABELS})
        archives = sorted({str(v) for v in group['source_archive'] if str(v)})
        splits = sorted({str(v) for v in group['corrected_split'] if str(v) not in {'', 'unspecified', 'UNMATCHED'}})
        stage_members = group[group['source_archive'].eq('CocoaMoniliaDataSet.zip')]
        stage_labels = sorted({str(v) for v in stage_members['stage_label'] if str(v)})
        all_cacao = group['is_cacao'].astype(str).eq('YES').all()
        all_readable = group['image_read_ok'].astype(str).eq('YES').all()
        base_reasons = []
        if not all_cacao:
            base_reasons.append('NON_CACAO_MEMBER')
        if not all_readable:
            base_reasons.append('UNREADABLE_IMAGE_MEMBER')
        if not coarse_labels:
            base_reasons.append('MISSING_COARSE_LABEL')
        if any((label.lower() in INVALID_LABELS for label in coarse_labels)):
            base_reasons.append('UNRESOLVED_COARSE_LABEL')
        if len(coarse_labels) > 1:
            base_reasons.append('HARD_COARSE_LABEL_CONFLICT')
        coarse_eligible = 'YES' if not base_reasons and len(coarse_labels) == 1 else 'NO'
        fine_reasons = list(base_reasons)
        if len(fine_labels) != 1:
            fine_reasons.append('FINE_LABEL_CONFLICT' if len(fine_labels) > 1 else 'MISSING_FINE_LABEL')
        fine_eligible = 'YES' if not fine_reasons else 'NO'
        stage_reasons = []
        if not all_cacao or not all_readable:
            stage_reasons.extend(base_reasons)
        if len(stage_members) == 0:
            stage_reasons.append('NO_COCOAMONILIA_MEMBER')
        if len(stage_labels) != 1:
            stage_reasons.append('STAGE_LABEL_CONFLICT' if len(stage_labels) > 1 else 'MISSING_STAGE_LABEL')
        stage_eligible = 'YES' if not stage_reasons else 'NO'
        rows.append({group_col: group_id, 'n_exact_components': len(group_to_components[group_id]), 'n_photo_paths': len(group), 'fine_labels': '|'.join(fine_labels), 'fine_label_count': len(fine_labels), 'coarse_labels': '|'.join(coarse_labels), 'coarse_label_count': len(coarse_labels), 'coarse_label': coarse_labels[0] if len(coarse_labels) == 1 else '', 'stage_labels': '|'.join(stage_labels), 'stage_label_count': len(stage_labels), 'stage_label': stage_labels[0] if len(stage_labels) == 1 else '', 'archives': '|'.join(archives), 'archive_count': len(archives), 'splits': '|'.join(splits), 'split_count': len(splits), 'cross_archive': 'YES' if len(archives) > 1 else 'NO', 'cross_split': 'YES' if len(splits) > 1 else 'NO', 'hard_coarse_label_conflict': 'YES' if len(coarse_labels) > 1 else 'NO', 'fine_label_conflict': 'YES' if len(fine_labels) > 1 else 'NO', 'all_cacao': 'YES' if all_cacao else 'NO', 'all_images_readable': 'YES' if all_readable else 'NO', 'coarse_supervised_eligible': coarse_eligible, 'coarse_exclusion_reason': '|'.join(base_reasons), 'fine_supervised_eligible': fine_eligible, 'fine_exclusion_reason': '|'.join(fine_reasons), 'stage4_supervised_eligible': stage_eligible, 'stage4_exclusion_reason': '|'.join(stage_reasons), 'exact_component_ids': '|'.join(group_to_components[group_id])})
    components = pd.DataFrame(rows)
    members = valid.merge(components, on=group_col, how='left', suffixes=('', '_group'))
    members = members.sort_values([group_col, 'exact_component_id', 'source_archive', 'relative_path'])
    return (components, members)

def build_split_block_table(canonical: pd.DataFrame, strict_map: dict[str, str], extended_map: dict[str, str], capture_map: dict[str, str], split_map: dict[str, str], safe_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid['final_strict_lineage_id'] = valid['exact_component_id'].map(strict_map).fillna('')
    valid['final_extended_lineage_id'] = valid['exact_component_id'].map(extended_map).fillna('')
    valid['final_capture_burst_id'] = valid['exact_component_id'].map(capture_map).fillna('')
    valid['final_split_block_id'] = valid['exact_component_id'].map(split_map).fillna('')
    valid['coarse_label'] = valid['corrected_label'].map(coarse_label)
    edge_counts = defaultdict(Counter)
    for row in safe_edges.itertuples(index=False):
        block = split_map.get(str(row.component_a), '')
        if block and block == split_map.get(str(row.component_b), ''):
            edge_counts[block][str(row.verification_relation)] += 1
    rows = []
    for block_id, group in valid.groupby('final_split_block_id', sort=True):
        strict_ids = sorted(set(group['final_strict_lineage_id'].astype(str)))
        extended_ids = sorted(set(group['final_extended_lineage_id'].astype(str)))
        capture_ids = sorted(set(group['final_capture_burst_id'].astype(str)))
        components = sorted(set(group['exact_component_id'].astype(str)))
        labels = sorted({str(v) for v in group['corrected_label'] if str(v).lower() not in INVALID_LABELS})
        coarse_labels = sorted({str(v) for v in group['coarse_label'] if str(v).lower() not in INVALID_LABELS})
        archives = sorted(set(group['source_archive'].astype(str)))
        splits = sorted({str(v) for v in group['corrected_split'] if str(v) not in {'', 'unspecified', 'UNMATCHED'}})
        counts = edge_counts[block_id]
        audit_reasons = []
        if len(strict_ids) >= 10:
            audit_reasons.append('LARGE_BLOCK_GE_10_STRICT_LINEAGES')
        if len(coarse_labels) > 1:
            audit_reasons.append('MIXED_COARSE_LABELS_ALLOWED_FOR_SPLIT_BLOCK')
        if len(splits) > 1:
            audit_reasons.append('CROSSES_ORIGINAL_SPLITS')
        if len(archives) > 2:
            audit_reasons.append('SPANS_GT_2_ARCHIVES')
        rows.append({'final_split_block_id': block_id, 'n_strict_lineages': len(strict_ids), 'n_extended_lineages': len(extended_ids), 'n_capture_bursts': len(capture_ids), 'n_exact_components': len(components), 'n_photo_paths': len(group), 'fine_labels': '|'.join(labels), 'fine_label_count': len(labels), 'coarse_labels': '|'.join(coarse_labels), 'coarse_label_count': len(coarse_labels), 'archives': '|'.join(archives), 'archive_count': len(archives), 'splits': '|'.join(splits), 'split_count': len(splits), 'strong_edge_count': int(counts['VERIFIED_DERIVATIVE_STRONG']), 'moderate_edge_count': int(counts['VERIFIED_DERIVATIVE_MODERATE']), 'same_scene_context_edge_count': int(counts['GEOMETRIC_SAME_SCENE_ONLY']), 'mixed_label_block': 'YES' if len(coarse_labels) > 1 else 'NO', 'cross_archive_block': 'YES' if len(archives) > 1 else 'NO', 'cross_split_block': 'YES' if len(splits) > 1 else 'NO', 'audit_required': 'YES' if audit_reasons else 'NO', 'audit_reason': '|'.join(audit_reasons), 'interpretation': 'CONSERVATIVE_ACQUISITION_SPLIT_BLOCK_NOT_BIOLOGICAL_SPECIMEN_COUNT', 'strict_lineage_ids': '|'.join(strict_ids), 'capture_burst_ids': '|'.join(capture_ids), 'exact_component_ids': '|'.join(components)})
    blocks = pd.DataFrame(rows)
    members = valid.merge(blocks, on='final_split_block_id', how='left', suffixes=('', '_block'))
    members = members.sort_values(['final_split_block_id', 'final_strict_lineage_id', 'relative_path'])
    return (blocks, members)

def graph_bridges(nodes: list[str], edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
    adjacency: defaultdict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        if a == b:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
    timer = 0
    tin: dict[str, int] = {}
    low: dict[str, int] = {}
    visited: set[str] = set()
    bridges: set[tuple[str, str]] = set()

    def dfs(v: str, parent: str | None) -> None:
        nonlocal timer
        visited.add(v)
        timer += 1
        tin[v] = low[v] = timer
        for to in adjacency.get(v, []):
            if to == parent:
                continue
            if to in visited:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.add(tuple(sorted((v, to))))
    for node in nodes:
        if node not in visited:
            dfs(node, None)
    return bridges

def split_block_bridge_audit(split_blocks: pd.DataFrame, split_map: dict[str, str], safe_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    safe = safe_edges.copy()
    safe['final_split_block_id'] = safe['component_a'].map(split_map).fillna('')
    for block in split_blocks.itertuples(index=False):
        if int(block.n_exact_components) < 3:
            continue
        group = safe[safe['final_split_block_id'].eq(block.final_split_block_id)].copy()
        if group.empty:
            continue
        nodes = str(block.exact_component_ids).split('|')
        bridges = graph_bridges(nodes, [(str(a), str(b)) for a, b in zip(group['component_a'], group['component_b'])])
        if not bridges:
            continue
        for edge in bridges:
            matches = group[group['edge_key'].eq(unordered_pair_key(*edge))]
            if matches.empty:
                continue
            row = matches.sort_values(['relation_priority', 'verification_score'], ascending=[False, False]).iloc[0].to_dict()
            row.update({'bridge_edge': 'YES', 'block_n_exact_components': block.n_exact_components, 'block_n_strict_lineages': block.n_strict_lineages, 'block_coarse_labels': block.coarse_labels, 'block_audit_required': block.audit_required})
            rows.append(row)
    return pd.DataFrame(rows)

def precompute_representatives(valid: pd.DataFrame, group_col: str, preferred_archive: str | None=None) -> pd.DataFrame:
    """Select one deterministic representative per group with one global sort.

    A global sort/drop-duplicates avoids thousands of tiny DataFrame copies and
    sorts, which are disproportionately slow on large manifest tables.
    """
    if valid.empty:
        return pd.DataFrame().set_index(pd.Index([], name=group_col))
    table = valid.copy()
    table['_read_rank'] = np.where(table['image_read_ok'].astype(str).eq('YES'), 0, 1)
    if preferred_archive is None:
        table['_pref_rank'] = 0
    else:
        table['_pref_rank'] = np.where(table['source_archive'].astype(str).eq(preferred_archive), 0, 1)
    table['_exact_rank'] = np.where(table['exact_representative'].astype(str).eq('YES'), 0, 1)
    table['_area'] = pd.to_numeric(table['width'], errors='coerce').fillna(0) * pd.to_numeric(table['height'], errors='coerce').fillna(0)
    table['_bytes'] = pd.to_numeric(table['bytes'], errors='coerce').fillna(0)
    entropy_source = table['gray_entropy'] if 'gray_entropy' in table.columns else pd.Series(0.0, index=table.index)
    table['_entropy'] = pd.to_numeric(entropy_source, errors='coerce').fillna(0)
    table = table.sort_values([group_col, '_read_rank', '_pref_rank', '_exact_rank', '_area', '_entropy', '_bytes', 'source_archive', 'relative_path'], ascending=[True, True, True, True, False, False, False, True, True])
    return table.drop_duplicates(group_col, keep='first').set_index(group_col)

def build_analysis_units(canonical: pd.DataFrame, strict_components: pd.DataFrame, strict_map: dict[str, str], split_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    valid['final_strict_lineage_id'] = valid['exact_component_id'].map(strict_map).fillna('')
    valid['final_split_block_id'] = valid['exact_component_id'].map(split_map).fillna('')
    split_counts = valid.groupby('final_strict_lineage_id')['final_split_block_id'].nunique()
    bad_nested = split_counts[split_counts.ne(1)]
    if len(bad_nested):
        raise RuntimeError(f'Strict lineages are not nested in one split block: {bad_nested.head().to_dict()}')
    strict_to_split = valid.groupby('final_strict_lineage_id')['final_split_block_id'].first()
    reps = precompute_representatives(valid, 'final_strict_lineage_id')
    stage_valid = valid[valid['source_archive'].eq('CocoaMoniliaDataSet.zip')].copy()
    stage_reps = precompute_representatives(stage_valid, 'final_strict_lineage_id', preferred_archive='CocoaMoniliaDataSet.zip')
    summary = strict_components.copy().set_index('final_strict_lineage_id', drop=False)
    summary['final_split_block_id'] = summary.index.map(strict_to_split).fillna('')
    representative_columns = ['photo_path_id', 'source_archive', 'source_dataset_id', 'relative_path', 'absolute_path', 'width', 'height', 'bytes']
    missing_rep = [col for col in representative_columns if col not in reps.columns]
    if missing_rep:
        raise RuntimeError(f'Representative table missing columns: {missing_rep}')
    reps_small = reps[representative_columns].rename(columns={col: f'representative_{col}' for col in representative_columns})
    summary = summary.join(reps_small, how='left')
    units: list[dict[str, Any]] = []
    coarse3_rows: list[dict[str, Any]] = []
    causal5_rows: list[dict[str, Any]] = []
    stage4_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        lineage_id = str(row.final_strict_lineage_id)
        base = {'analysis_unit_id': lineage_id, 'final_strict_lineage_id': lineage_id, 'final_split_block_id': str(row.final_split_block_id), 'recommended_split_group_id': str(row.final_split_block_id), 'n_exact_components': int(row.n_exact_components), 'n_photo_paths': int(row.n_photo_paths), 'fine_labels': row.fine_labels, 'fine_label_count': int(row.fine_label_count), 'coarse_labels': row.coarse_labels, 'coarse_label_count': int(row.coarse_label_count), 'coarse_label': row.coarse_label, 'stage_labels': row.stage_labels, 'stage_label_count': int(row.stage_label_count), 'stage_label': row.stage_label, 'archives': row.archives, 'archive_count': int(row.archive_count), 'splits': row.splits, 'split_count': int(row.split_count), 'hard_coarse_label_conflict': row.hard_coarse_label_conflict, 'fine_label_conflict': row.fine_label_conflict, 'coarse_supervised_eligible': row.coarse_supervised_eligible, 'coarse_exclusion_reason': row.coarse_exclusion_reason, 'fine_supervised_eligible': row.fine_supervised_eligible, 'fine_exclusion_reason': row.fine_exclusion_reason, 'stage4_supervised_eligible': row.stage4_supervised_eligible, 'stage4_exclusion_reason': row.stage4_exclusion_reason, 'representative_photo_path_id': row.representative_photo_path_id, 'representative_source_archive': row.representative_source_archive, 'representative_source_dataset_id': row.representative_source_dataset_id, 'representative_relative_path': row.representative_relative_path, 'representative_absolute_path': row.representative_absolute_path, 'representative_width': row.representative_width, 'representative_height': row.representative_height, 'representative_bytes': row.representative_bytes}
        units.append(base)
        if row.coarse_supervised_eligible == 'YES' and row.coarse_label in COARSE3_LABELS:
            task_row = dict(base)
            task_row['task_name'] = 'cacao_coarse_three_class'
            task_row['task_label'] = row.coarse_label
            coarse3_rows.append(task_row)
        if row.coarse_supervised_eligible == 'YES' and row.coarse_label in CAUSAL5_LABELS:
            task_row = dict(base)
            task_row['task_name'] = 'cacao_causal_five_class'
            task_row['task_label'] = row.coarse_label
            causal5_rows.append(task_row)
        if row.stage4_supervised_eligible == 'YES' and lineage_id in stage_reps.index:
            stage_rep = stage_reps.loc[lineage_id]
            task_row = dict(base)
            task_row['task_name'] = 'cocoamonilia_four_stage'
            task_row['task_label'] = row.stage_label
            task_row['representative_photo_path_id'] = stage_rep['photo_path_id']
            task_row['representative_source_archive'] = stage_rep['source_archive']
            task_row['representative_source_dataset_id'] = stage_rep['source_dataset_id']
            task_row['representative_relative_path'] = stage_rep['relative_path']
            task_row['representative_absolute_path'] = stage_rep['absolute_path']
            task_row['representative_width'] = stage_rep['width']
            task_row['representative_height'] = stage_rep['height']
            task_row['representative_bytes'] = stage_rep['bytes']
            stage4_rows.append(task_row)
    return (pd.DataFrame(units), pd.DataFrame(coarse3_rows), pd.DataFrame(causal5_rows), pd.DataFrame(stage4_rows))

def build_final_manifest(canonical: pd.DataFrame, strict_map: dict[str, str], extended_map: dict[str, str], capture_map: dict[str, str], split_map: dict[str, str], strict_components: pd.DataFrame, split_blocks: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    manifest = canonical.copy()
    manifest['fine_label'] = manifest['corrected_label'].astype(str)
    manifest['coarse_label'] = manifest['corrected_label'].map(coarse_label)
    manifest['stage_label'] = manifest['corrected_label'].map(stage_label)
    manifest['final_strict_lineage_id'] = manifest['exact_component_id'].map(strict_map).fillna('')
    manifest['final_extended_lineage_id'] = manifest['exact_component_id'].map(extended_map).fillna('')
    manifest['final_capture_burst_id'] = manifest['exact_component_id'].map(capture_map).fillna('')
    manifest['final_split_block_id'] = manifest['exact_component_id'].map(split_map).fillna('')
    manifest['analysis_unit_id'] = manifest['final_strict_lineage_id']
    manifest['recommended_split_group_id'] = manifest['final_split_block_id']
    strict_fields = strict_components.set_index('final_strict_lineage_id')
    strict_keep = ['fine_labels', 'fine_label_count', 'coarse_labels', 'coarse_label_count', 'stage_labels', 'stage_label_count', 'hard_coarse_label_conflict', 'fine_label_conflict', 'coarse_supervised_eligible', 'coarse_exclusion_reason', 'fine_supervised_eligible', 'fine_exclusion_reason', 'stage4_supervised_eligible', 'stage4_exclusion_reason']
    manifest = manifest.join(strict_fields[strict_keep], on='final_strict_lineage_id', rsuffix='_strict')
    block_fields = split_blocks.set_index('final_split_block_id')
    block_keep = ['n_strict_lineages', 'n_extended_lineages', 'n_capture_bursts', 'n_exact_components', 'n_photo_paths', 'mixed_label_block', 'cross_archive_block', 'cross_split_block', 'audit_required', 'audit_reason', 'interpretation']
    manifest = manifest.join(block_fields[block_keep], on='final_split_block_id', rsuffix='_split_block')
    representative_ids = set(units['representative_photo_path_id'].astype(str)) if not units.empty else set()
    manifest['analysis_unit_representative'] = np.where(manifest['photo_path_id'].astype(str).isin(representative_ids), 'YES', 'NO')
    manifest['coarse3_task_eligible'] = np.where(manifest['coarse_supervised_eligible'].eq('YES') & manifest['coarse_label'].isin(COARSE3_LABELS), 'YES', 'NO')
    manifest['causal5_task_eligible'] = np.where(manifest['coarse_supervised_eligible'].eq('YES') & manifest['coarse_label'].isin(CAUSAL5_LABELS), 'YES', 'NO')
    manifest['stage4_task_eligible'] = np.where(manifest['stage4_supervised_eligible'].eq('YES') & manifest['source_archive'].eq('CocoaMoniliaDataSet.zip'), 'YES', 'NO')
    return manifest

def source_overlap_from_mapping(canonical: pd.DataFrame, mapping: dict[str, str], group_col: str) -> pd.DataFrame:
    table = canonical[canonical['exact_component_id'].ne('NO_HASH')].copy()
    table[group_col] = table['exact_component_id'].map(mapping).fillna('')
    archives = sorted(set(table['source_archive'].astype(str)))
    sets = {archive: set(table.loc[table['source_archive'].eq(archive), group_col]) for archive in archives}
    rows = []
    for a in archives:
        for b in archives:
            shared = len(sets[a] & sets[b])
            rows.append({'archive_a': a, 'archive_b': b, 'units_a': len(sets[a]), 'units_b': len(sets[b]), 'shared_units': shared, 'pct_a_shared': 100.0 * shared / max(1, len(sets[a])), 'pct_b_shared': 100.0 * shared / max(1, len(sets[b]))})
    return pd.DataFrame(rows)

def task_original_split_audit(task: pd.DataFrame) -> pd.DataFrame:
    if task.empty:
        return pd.DataFrame()
    rows = []
    for block_id, group in task.groupby('final_split_block_id', sort=True):
        splits = atomic_values(group['splits'], skip_unspecified=True)
        rows.append({'task_name': str(group['task_name'].iloc[0]), 'final_split_block_id': block_id, 'n_analysis_units': len(group), 'labels': '|'.join(sorted(set(group['task_label'].astype(str)))), 'original_splits': '|'.join(splits), 'original_split_count': len(splits), 'crosses_original_splits': 'YES' if len(splits) > 1 else 'NO'})
    return pd.DataFrame(rows)

def wilson_interval(k: int, n: int, z: float=1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def reserve_yield_tables(reserve: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table = reserve.copy()
    table['leakage_safe_edge'] = relation_is_safe(table['verification_relation'])
    table['derivative_edge'] = relation_is_derivative(table['verification_relation'])
    table['source_pair'] = [' || '.join(sorted((str(a), str(b)))) for a, b in zip(table['archive_a'], table['archive_b'])]
    outputs: dict[str, pd.DataFrame] = {}
    for name, cols in {'evidence_tier': ['evidence_tier'], 'scientific_risk': ['scientific_risk'], 'relation_tier_risk': ['verification_relation', 'evidence_tier', 'scientific_risk'], 'source_pair': ['source_pair']}.items():
        rows = []
        for keys, group in table.groupby(cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(cols, keys)}
            safe = int(group['leakage_safe_edge'].sum())
            deriv = int(group['derivative_edge'].sum())
            lo, hi = wilson_interval(safe, len(group))
            row.update({'n_pairs': len(group), 'derivative_edges': deriv, 'leakage_safe_edges': safe, 'derivative_rate': deriv / max(1, len(group)), 'leakage_safe_rate': safe / max(1, len(group)), 'leakage_safe_wilson_low': lo, 'leakage_safe_wilson_high': hi, 'new_strict_merges': int(group['incremental_new_strict_merge'].eq('YES').sum()), 'new_extended_merges': int(group['incremental_new_extended_merge'].eq('YES').sum()), 'new_split_block_merges': int(group['incremental_new_split_block_merge'].eq('YES').sum())})
            rows.append(row)
        outputs[name] = pd.DataFrame(rows)
    decile_rows = []
    for tier, tier_group in table.groupby('evidence_tier', sort=True):
        group = tier_group.copy()
        for col in ('efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score'):
            group[col] = pd.to_numeric(group[col], errors='coerce')
        group = group.sort_values(['efficientnet_b0_similarity', 'resnet18_similarity', 'evidence_score', 'pair_id'], ascending=[False, False, False, True]).reset_index(drop=True)
        group['within_tier_rank'] = np.arange(1, len(group) + 1)
        group['within_tier_decile'] = np.minimum(10, np.ceil(group['within_tier_rank'] * 10.0 / max(1, len(group))).astype(int))
        for decile, part in group.groupby('within_tier_decile', sort=True):
            safe = int(part['leakage_safe_edge'].sum())
            lo, hi = wilson_interval(safe, len(part))
            decile_rows.append({'evidence_tier': tier, 'within_tier_decile': int(decile), 'n_pairs': len(part), 'derivative_edges': int(part['derivative_edge'].sum()), 'leakage_safe_edges': safe, 'leakage_safe_rate': safe / max(1, len(part)), 'leakage_safe_wilson_low': lo, 'leakage_safe_wilson_high': hi, 'new_strict_merges': int(part['incremental_new_strict_merge'].eq('YES').sum()), 'new_extended_merges': int(part['incremental_new_extended_merge'].eq('YES').sum()), 'new_split_block_merges': int(part['incremental_new_split_block_merge'].eq('YES').sum())})
    outputs['within_tier_deciles'] = pd.DataFrame(decile_rows)
    return outputs

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--reserve-task-dir', type=Path)
    parser.add_argument('--control-task-dir', type=Path)
    parser.add_argument('--canonical-manifest', type=Path)
    parser.add_argument('--stage2b-candidate-verification', type=Path)
    parser.add_argument('--stage2d-candidate-verification', type=Path)
    parser.add_argument('--hidden-control-relations', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert classify(pd.Series({'gray_entropy_a': 7, 'gray_entropy_b': 7, 'direct_ncc': 1, 'direct_gradient_ncc': 1, 'direct_ssim': 1}))[0] == 'VERIFIED_DERIVATIVE_STRONG'
        assert coarse_label('frosty_pod_stage_m2') == 'frosty_pod'
        assert stage_label('frosty_pod_stage_m3') == 'm3_sporulation'
        uf = UnionFind(['a', 'b', 'c'])
        uf.union('a', 'b')
        mapping, groups = groups_from_uf(uf, 'TEST')
        assert mapping['a'] == mapping['b'] and mapping['a'] != mapping['c']
        assert tuple(sorted(('a', 'b'))) in graph_bridges(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')])
        print('LINEAGE_FREEZE_SELF_TEST=OK')
        return 0
    required_production = {'prepared_dir': args.prepared_dir, 'reserve_task_dir': args.reserve_task_dir, 'control_task_dir': args.control_task_dir, 'canonical_manifest': args.canonical_manifest, 'stage2b_candidate_verification': args.stage2b_candidate_verification, 'stage2d_candidate_verification': args.stage2d_candidate_verification, 'hidden_control_relations': args.hidden_control_relations, 'output_dir': args.output_dir}
    missing_production = [name for name, value in required_production.items() if value is None]
    if missing_production:
        raise RuntimeError(f'Missing required production arguments: {missing_production}')
    prepared_reserve = pd.read_csv(args.prepared_dir / 'stage2e_reserve_candidate_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    prepared_controls = pd.read_csv(args.prepared_dir / 'stage2e_sentinel_control_pairs.tsv', sep='\t', keep_default_na=False, low_memory=False)
    reserve_raw = load_task_tables(args.reserve_task_dir, 'reserve')
    controls = load_task_tables(args.control_task_dir, 'control')
    verify_completeness(prepared_reserve, reserve_raw, 'reserve')
    verify_completeness(prepared_controls, controls, 'control')
    for table in (reserve_raw, controls):
        classifications = table.apply(classify, axis=1, result_type='expand')
        classifications.columns = ['verification_relation', 'verification_reason', 'verification_score']
        table[['verification_relation', 'verification_reason', 'verification_score']] = classifications
        table['manual_review_priority'] = np.where(table.get('cross_archive', 'NO').astype(str).eq('YES') | table.get('cross_split', 'NO').astype(str).eq('YES') | table.get('label_conflict', 'NO').astype(str).eq('YES'), 'YES', 'NO')
    control_metrics, control_qc, control_overall = control_summary(controls)
    canonical = pd.read_csv(args.canonical_manifest, sep='\t', keep_default_na=False, low_memory=False)
    canonical = canonical.copy()
    canonical['coarse_label'] = canonical['corrected_label'].map(coarse_label)
    canonical['stage_label'] = canonical['corrected_label'].map(stage_label)
    nodes = sorted(set(canonical.loc[canonical['exact_component_id'].ne('NO_HASH'), 'exact_component_id'].astype(str)))
    reps = component_representatives(canonical)
    stage2b_raw = pd.read_csv(args.stage2b_candidate_verification, sep='\t', keep_default_na=False, low_memory=False)
    stage2d_raw = pd.read_csv(args.stage2d_candidate_verification, sep='\t', keep_default_na=False, low_memory=False)
    hidden_raw = pd.read_csv(args.hidden_control_relations, sep='\t', keep_default_na=False, low_memory=False)
    stage2b = normalize_edge_table(stage2b_raw, 'STAGE2B', reps)
    stage2d = normalize_edge_table(stage2d_raw, 'STAGE2D', reps)
    reserve = normalize_edge_table(reserve_raw, 'STAGE2E', reps)
    hidden = hidden_edge_table(hidden_raw, reps)
    baseline_ledger = pd.concat([stage2b, stage2d, hidden], ignore_index=True, sort=False)
    baseline_best, baseline_duplicate_audit = deduplicate_edge_ledger(baseline_ledger)
    reserve = mark_reserve_incremental_merges(reserve, nodes, baseline_best)
    full_ledger = pd.concat([stage2b, stage2d, reserve, hidden], ignore_index=True, sort=False)
    best_edges, duplicate_audit = deduplicate_edge_ledger(full_ledger)
    safe_edges = best_edges[best_edges['verification_relation'].isin(SAFE_RELATIONS)].copy()
    strict_uf = initialize_uf(nodes, best_edges, {'VERIFIED_DERIVATIVE_STRONG'})
    extended_uf = initialize_uf(nodes, best_edges, DERIVATIVE_RELATIONS)
    capture_uf = initialize_capture_burst_uf(nodes, best_edges)
    split_uf = initialize_uf(nodes, best_edges, SAFE_RELATIONS)
    strict_map, strict_groups = groups_from_uf(strict_uf, 'CIFSTRICT')
    extended_map, extended_groups = groups_from_uf(extended_uf, 'CIFEXT')
    capture_map, capture_groups = groups_from_uf(capture_uf, 'CIFCAPTURE')
    split_map, split_groups = groups_from_uf(split_uf, 'CIFBLOCK')
    strict_components, strict_members = lineage_table(canonical, strict_map, strict_groups, 'final_strict_lineage_id')
    extended_components, extended_members = lineage_table(canonical, extended_map, extended_groups, 'final_extended_lineage_id')
    capture_components, capture_members = lineage_table(canonical, capture_map, capture_groups, 'final_capture_burst_id')
    split_blocks, split_members = build_split_block_table(canonical, strict_map, extended_map, capture_map, split_map, safe_edges)
    bridge_audit = split_block_bridge_audit(split_blocks, split_map, safe_edges)
    units, coarse3_units, causal5_units, stage4_units = build_analysis_units(canonical, strict_components, strict_map, split_map)
    manifest = build_final_manifest(canonical, strict_map, extended_map, capture_map, split_map, strict_components, split_blocks, units)
    strict_overlap = source_overlap_from_mapping(canonical, strict_map, 'final_strict_lineage_id')
    split_overlap = source_overlap_from_mapping(canonical, split_map, 'final_split_block_id')
    split_audits = []
    for task in (coarse3_units, causal5_units, stage4_units):
        audit = task_original_split_audit(task)
        if not audit.empty:
            split_audits.append(audit)
    task_split_audit = pd.concat(split_audits, ignore_index=True) if split_audits else pd.DataFrame()
    yield_tables = reserve_yield_tables(reserve)
    reserve_relation_summary = reserve.groupby(['verification_relation', 'evidence_tier', 'scientific_risk'], dropna=False).size().reset_index(name='pair_count').sort_values(['verification_relation', 'evidence_tier', 'scientific_risk'])
    stage_contribution = best_edges.groupby(['edge_stage', 'verification_relation'], dropna=False).size().reset_index(name='edge_count').sort_values(['edge_stage', 'verification_relation'])
    hard_conflicts = strict_components[strict_components['hard_coarse_label_conflict'].eq('YES')].copy()
    fine_only_conflicts = strict_components[strict_components['fine_label_conflict'].eq('YES') & strict_components['hard_coarse_label_conflict'].eq('NO')].copy()
    mixed_label_blocks = split_blocks[split_blocks['mixed_label_block'].eq('YES')].copy()
    large_blocks = split_blocks[split_blocks['n_strict_lineages'].astype(int).ge(10) | split_blocks['audit_required'].eq('YES')].sort_values(['n_strict_lineages', 'n_photo_paths'], ascending=False)
    hierarchy_summary = pd.DataFrame([{'unit_definition': 'Physical photograph paths', 'unit_count': len(canonical)}, {'unit_definition': 'Exact SHA-256 components', 'unit_count': len(nodes)}, {'unit_definition': 'Final strict image lineages (strong only)', 'unit_count': len(strict_components)}, {'unit_definition': 'Final extended image lineages (strong + moderate)', 'unit_count': len(extended_components)}, {'unit_definition': 'Provisional timestamp capture-burst groups', 'unit_count': len(capture_components)}, {'unit_definition': 'Final conservative split blocks', 'unit_count': len(split_blocks)}, {'unit_definition': 'Coarse-label eligible strict analysis units', 'unit_count': int(strict_components['coarse_supervised_eligible'].eq('YES').sum())}, {'unit_definition': 'Coarse three-class task units', 'unit_count': len(coarse3_units)}, {'unit_definition': 'Causal five-class task units', 'unit_count': len(causal5_units)}, {'unit_definition': 'CocoaMonilia four-stage task units', 'unit_count': len(stage4_units)}])
    task_summary = pd.DataFrame([{'task_name': 'cacao_coarse_three_class', 'analysis_units': len(coarse3_units), 'split_blocks': coarse3_units['final_split_block_id'].nunique() if len(coarse3_units) else 0, 'labels': '|'.join(sorted(set(coarse3_units.get('task_label', pd.Series(dtype=str)).astype(str))))}, {'task_name': 'cacao_causal_five_class', 'analysis_units': len(causal5_units), 'split_blocks': causal5_units['final_split_block_id'].nunique() if len(causal5_units) else 0, 'labels': '|'.join(sorted(set(causal5_units.get('task_label', pd.Series(dtype=str)).astype(str))))}, {'task_name': 'cocoamonilia_four_stage', 'analysis_units': len(stage4_units), 'split_blocks': stage4_units['final_split_block_id'].nunique() if len(stage4_units) else 0, 'labels': '|'.join(sorted(set(stage4_units.get('task_label', pd.Series(dtype=str)).astype(str))))}])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(reserve.sort_values(['verification_relation', 'verification_score'], ascending=[True, False]), args.output_dir / 'stage2e_reserve_pair_verification.tsv')
    write_tsv(controls.sort_values(['pair_origin', 'synthetic_transform', 'pair_id']), args.output_dir / 'stage2e_control_pair_verification.tsv')
    write_tsv(control_metrics, args.output_dir / 'stage2e_control_performance.tsv')
    write_tsv(control_qc, args.output_dir / 'stage2e_control_qc.tsv')
    write_tsv(reserve_relation_summary, args.output_dir / 'stage2e_reserve_relation_summary.tsv')
    write_tsv(stage_contribution, args.output_dir / 'stage2e_edge_stage_contribution.tsv')
    for name, table in yield_tables.items():
        write_tsv(table, args.output_dir / f'stage2e_reserve_yield_by_{name}.tsv')
    write_tsv(reserve[reserve['verification_relation'].isin(DERIVATIVE_RELATIONS)], args.output_dir / 'stage2e_reserve_verified_derivative_edges.tsv')
    write_tsv(reserve[reserve['verification_relation'].isin(SAFE_RELATIONS)], args.output_dir / 'stage2e_reserve_verified_split_block_edges.tsv')
    write_tsv(reserve[reserve['verification_relation'].isin(SAFE_RELATIONS) & reserve['cross_archive'].eq('YES')], args.output_dir / 'stage2e_reserve_verified_cross_archive_edges.tsv')
    write_tsv(reserve[reserve['verification_relation'].isin(SAFE_RELATIONS) & reserve['cross_split'].eq('YES')], args.output_dir / 'stage2e_reserve_verified_cross_split_edges.tsv')
    write_tsv(reserve[reserve['verification_relation'].isin(SAFE_RELATIONS) & reserve['label_conflict'].eq('YES')], args.output_dir / 'stage2e_reserve_verified_label_conflict_edges.tsv')
    write_tsv(reserve[reserve['verification_relation'].eq('AMBIGUOUS_MANUAL_REVIEW')], args.output_dir / 'stage2e_reserve_ambiguous_manual_review.tsv')
    write_tsv(reserve[reserve['incremental_new_strict_merge'].eq('YES')], args.output_dir / 'stage2e_incremental_new_strict_merges.tsv')
    write_tsv(reserve[reserve['incremental_new_extended_merge'].eq('YES')], args.output_dir / 'stage2e_incremental_new_extended_merges.tsv')
    write_tsv(reserve[reserve['incremental_new_split_block_merge'].eq('YES')], args.output_dir / 'stage2e_incremental_new_split_block_merges.tsv')
    best_edges.to_csv(args.output_dir / 'stage2e_final_edge_ledger.tsv.gz', sep='\t', index=False, compression='gzip', na_rep='')
    write_tsv(safe_edges, args.output_dir / 'stage2e_final_verified_split_block_edge_ledger.tsv')
    write_tsv(duplicate_audit, args.output_dir / 'stage2e_duplicate_edge_audit.tsv')
    write_tsv(baseline_duplicate_audit, args.output_dir / 'stage2e_baseline_duplicate_edge_audit.tsv')
    write_tsv(strict_components, args.output_dir / 'stage2e_final_strict_lineage_components.tsv')
    write_tsv(strict_members, args.output_dir / 'stage2e_final_strict_lineage_members.tsv')
    write_tsv(extended_components, args.output_dir / 'stage2e_final_extended_lineage_components.tsv')
    write_tsv(extended_members, args.output_dir / 'stage2e_final_extended_lineage_members.tsv')
    write_tsv(capture_components, args.output_dir / 'stage2e_provisional_capture_burst_components.tsv')
    write_tsv(capture_members, args.output_dir / 'stage2e_provisional_capture_burst_members.tsv')
    write_tsv(best_edges[capture_burst_same_scene_mask(best_edges)], args.output_dir / 'stage2e_capture_burst_same_scene_edges.tsv')
    write_tsv(split_blocks, args.output_dir / 'stage2e_final_split_blocks.tsv')
    write_tsv(split_members, args.output_dir / 'stage2e_final_split_block_members.tsv')
    write_tsv(bridge_audit, args.output_dir / 'stage2e_split_block_bridge_edge_audit.tsv')
    write_tsv(large_blocks, args.output_dir / 'stage2e_large_or_mixed_split_block_audit.tsv')
    write_tsv(mixed_label_blocks, args.output_dir / 'stage2e_mixed_label_split_blocks_allowed.tsv')
    write_tsv(hard_conflicts, args.output_dir / 'stage2e_hard_coarse_label_conflict_strict_lineages.tsv')
    write_tsv(fine_only_conflicts, args.output_dir / 'stage2e_fine_label_granularity_conflict_strict_lineages.tsv')
    write_tsv(strict_overlap, args.output_dir / 'stage2e_final_strict_source_overlap.tsv')
    write_tsv(split_overlap, args.output_dir / 'stage2e_final_split_block_source_overlap.tsv')
    write_tsv(manifest, args.output_dir / 'stage2e_final_benchmark_manifest.tsv')
    write_tsv(units, args.output_dir / 'stage2e_final_analysis_units.tsv')
    write_tsv(coarse3_units, args.output_dir / 'stage2e_task_cacao_coarse_three_class_units.tsv')
    write_tsv(causal5_units, args.output_dir / 'stage2e_task_cacao_causal_five_class_units.tsv')
    write_tsv(stage4_units, args.output_dir / 'stage2e_task_cocoamonilia_four_stage_units.tsv')
    write_tsv(task_summary, args.output_dir / 'stage2e_task_unit_summary.tsv')
    write_tsv(task_split_audit, args.output_dir / 'stage2e_original_split_contamination_by_task.tsv')
    write_tsv(hierarchy_summary, args.output_dir / 'stage2e_final_hierarchy_summary.tsv')
    relation_counts = reserve['verification_relation'].value_counts()
    reserve_complete = len(reserve) == len(prepared_reserve)
    qc_pass = control_overall['qc_pass']
    final_interpretable = 'YES' if reserve_complete and qc_pass == 'YES' else 'NO'
    claim_status = pd.DataFrame([{'frozen_verifier_control_qc_pass': qc_pass, 'reserve_pairs_expected': len(prepared_reserve), 'reserve_pairs_verified': len(reserve), 'evidence_gated_reserve_exhausted': 'YES' if reserve_complete else 'NO', 'conservative_strict_lineage_freeze_final': final_interpretable, 'extended_lineage_freeze_final': final_interpretable, 'capture_burst_layer_provisional': 'YES', 'conservative_split_block_freeze_final': final_interpretable, 'analysis_unit_eligibility_based_on_strict_lineage': 'YES', 'same_scene_context_edges_used_only_for_split_blocking': 'YES', 'mixed_label_split_blocks_automatically_excluded': 'NO', 'ambiguous_edges_merged': 'NO', 'model_split_assignments_created': 'NO', 'claim_boundary': 'Final IDs are conservative within the pre-specified exact, perceptual, and two-encoder evidence-gated search universe. Strong derivative edges define strict lineages; moderate edges define an extended sensitivity layer; timestamp-style CocoaMonilia same-scene edges form a provisional capture-burst layer; all geometric same-scene/context edges only block data splits. Split blocks are not biological specimen counts, and ambiguous pairs are not merged.'}])
    write_tsv(claim_status, args.output_dir / 'stage2e_claim_status.tsv')
    summary_lines = ['# final lineage freeze full-reserve verification and hierarchy-aware freeze', '', f'- Full evidence reranking reserve pairs verified: {len(reserve):,} / {len(prepared_reserve):,}', f"- Strong reserve derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_STRONG', 0)):,}", f"- Moderate reserve derivatives: {int(relation_counts.get('VERIFIED_DERIVATIVE_MODERATE', 0)):,}", f"- Reserve geometric same-scene/context pairs: {int(relation_counts.get('GEOMETRIC_SAME_SCENE_ONLY', 0)):,}", f"- Reserve ambiguous pairs: {int(relation_counts.get('AMBIGUOUS_MANUAL_REVIEW', 0)):,}", f"- Reserve rejected pairs: {int(relation_counts.get('REJECTED_NOT_DERIVATIVE', 0)):,}", '', '## Final frozen-verifier controls', f"- Exact-positive sensitivity: {control_overall['exact_positive_sensitivity']:.4f}", f"- Synthetic-positive sensitivity: {control_overall['synthetic_positive_sensitivity']:.4f}", f"- Random-negative derivative FPR: {control_overall['negative_false_positive_rate']:.4f}", f'- Control QC pass: {qc_pass}', '', '## Final hierarchy', f'- Physical photograph paths: {len(canonical):,}', f'- Exact SHA-256 components: {len(nodes):,}', f'- Strict image lineages, strong-only: {len(strict_components):,}', f'- Extended image lineages, strong plus moderate: {len(extended_components):,}', f'- Provisional timestamp capture-burst groups: {len(capture_components):,}', f'- Conservative acquisition split blocks: {len(split_blocks):,}', f"- Coarse-label eligible strict analysis units: {int(strict_components['coarse_supervised_eligible'].eq('YES').sum()):,}", f'- Hard coarse-label conflict strict lineages: {len(hard_conflicts):,}', f'- Fine-label granularity conflicts without coarse conflict: {len(fine_only_conflicts):,}', f'- Mixed-label split blocks retained for grouping: {len(mixed_label_blocks):,}', '', '## Task-ready analysis units', f'- Coarse healthy/black-pod/frosty-pod units: {len(coarse3_units):,}', f'- Causal five-class units including insect damage: {len(causal5_units):,}', f'- CocoaMonilia four-stage units: {len(stage4_units):,}', '', '## Claim boundary', 'The reserve is exhausted only within the pre-specified evidence-gated candidate universe. Strong derivative edges define strict image lineage.', 'Moderate derivative edges are retained as an extended sensitivity layer. Timestamp-style CocoaMonilia same-scene edges form a provisional capture-burst layer. All geometric same-scene/context edges are used only as conservative', 'train/validation/test blocking relations and are not counted as biological specimens. Eligibility is decided at strict-lineage level; a mixed-label', 'split block is allowed when its member strict lineages are individually label-consistent. Ambiguous pairs are not merged.']
    (args.output_dir / 'stage2e_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    (args.output_dir / '.stage2e_complete').write_text('OK\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    print(claim_status.to_string(index=False))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
