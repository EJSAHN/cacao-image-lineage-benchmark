"""Evidence-first recalibration and deterministic primary geometric verification shortlisting.

This analysis reuses the frozen image embeddings. It does not create lineage
edges. Scientific-risk flags (cross archive, split, or label) affect priority
only after a pair passes an embedding-evidence gate calibrated against geometric verification.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
MODEL_NAMES = ('resnet18', 'efficientnet_b0')
UNKNOWN_SPLITS = {'', 'unspecified', 'UNMATCHED', 'unknown', 'none', 'nan'}
POSITIVE_GROUPS = ('VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE', 'GEOMETRIC_SAME_SCENE_ONLY')
ALL_CALIBRATION_GROUPS = ('VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE', 'GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW', 'REJECTED_NOT_DERIVATIVE', 'RANDOM_NEGATIVE_CONTROL')

def write_tsv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep='\t', index=False, na_rep='')

def write_tsv_gz(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep='\t', index=False, na_rep='', compression='gzip')

def sha256_file(path: Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def normalize_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def clean_origin(value: str) -> str:
    value = str(value or '').lower().strip()
    value = re.sub('\\.rf\\.[0-9a-f]+$', '', value)
    value = re.sub('_jpg$', '', value)
    value = re.sub('\\.(jpg|jpeg|png)$', '', value)
    return value

def timestamp_key(value: str) -> tuple[str, float] | None:
    value = clean_origin(value)
    match = re.search('(?:img|pxl)_(\\d{8})_(\\d{6})(\\d{0,3})', value)
    if not match:
        return None
    day, hhmmss, millis = match.groups()
    try:
        moment = datetime.strptime(day + hhmmss, '%Y%m%d%H%M%S').timestamp()
        if millis:
            moment += int(millis.ljust(3, '0')[:3]) / 1000.0
    except ValueError:
        return None
    prefix = value[:match.start()] + f'img_{day}_'
    return (prefix, moment)

def sequence_key(value: str) -> tuple[str, int] | None:
    value = clean_origin(value)
    match = re.search('^(.*?)(\\d{1,7})$', value)
    if not match:
        return None
    prefix, number = match.groups()
    if len(prefix) < 2:
        return None
    return (prefix, int(number))

def safe_topk(scores: np.ndarray, mask: np.ndarray, k: int, minimum_similarity: float | None=None) -> list[tuple[int, float]]:
    valid = np.flatnonzero(mask & np.isfinite(scores))
    if valid.size == 0 or k <= 0:
        return []
    take = min(k, len(valid))
    values = scores[valid]
    if take < len(valid):
        local = np.argpartition(values, -take)[-take:]
        selected = valid[local]
    else:
        selected = valid
    selected = selected[np.argsort(scores[selected])[::-1]]
    output: list[tuple[int, float]] = []
    for index in selected:
        similarity = float(scores[index])
        if minimum_similarity is not None and similarity < minimum_similarity:
            continue
        output.append((int(index), similarity))
    return output

def add_record(records: dict[tuple[int, int], dict[str, Any]], i: int, j: int, reason: str, model: str | None=None, rank: int | None=None) -> None:
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    record = records.setdefault((a, b), {'i': a, 'j': b, 'reasons': set(), 'resnet18_best_rank': np.nan, 'efficientnet_b0_best_rank': np.nan, 'stage2b_control_relation': ''})
    record['reasons'].add(reason)
    if model is not None and rank is not None:
        key = f'{model}_best_rank'
        prior = record[key]
        if not np.isfinite(prior) or rank < prior:
            record[key] = int(rank)

def build_pair_universe(embeddings: dict[str, np.ndarray], meta: pd.DataFrame, top_k_overall: int, top_k_constrained: int, resnet_prefilter: float, efficientnet_prefilter: float, query_chunk: int) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    archives = meta['source_archive'].astype(str).to_numpy()
    labels = meta['corrected_label'].astype(str).to_numpy()
    splits = meta['corrected_split'].astype(str).to_numpy()
    split_lower = np.char.lower(splits.astype(str))
    known_split_mask = ~np.isin(split_lower, list(UNKNOWN_SPLITS))
    n = len(meta)
    for model_name, raw_matrix in embeddings.items():
        matrix = np.ascontiguousarray(np.asarray(raw_matrix, dtype=np.float32))
        if matrix.shape[0] != n:
            raise RuntimeError(f'{model_name} row mismatch: {matrix.shape[0]} vs {n}')
        if not np.isfinite(matrix).all():
            raise RuntimeError(f'{model_name} contains non-finite values')
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=0.0003):
            raise RuntimeError(f'{model_name} embeddings are not L2 normalized')
        matrix_t = np.ascontiguousarray(matrix.T)
        constrained_floor = efficientnet_prefilter if model_name == 'efficientnet_b0' else resnet_prefilter
        print(f'pair_universe model={model_name} rows={n} dim={matrix.shape[1]} overall_k={top_k_overall} constrained_k={top_k_constrained} constrained_floor={constrained_floor}', flush=True)
        for start in range(0, n, query_chunk):
            end = min(n, start + query_chunk)
            similarities = matrix[start:end] @ matrix_t
            for local, i in enumerate(range(start, end)):
                scores = similarities[local]
                scores[i] = -np.inf
                base = np.ones(n, dtype=bool)
                base[i] = False
                for rank, (j, _) in enumerate(safe_topk(scores, base, top_k_overall), 1):
                    add_record(records, i, j, f'{model_name}_overall_top{top_k_overall}', model_name, rank)
                masks = {'cross_archive': base & (archives != archives[i]), 'cross_label': base & (labels != labels[i])}
                if known_split_mask[i]:
                    masks['cross_split'] = base & known_split_mask & (splits != splits[i])
                for mode, mask in masks.items():
                    for rank, (j, _) in enumerate(safe_topk(scores, mask, top_k_constrained, constrained_floor), 1):
                        add_record(records, i, j, f'{model_name}_{mode}_top{top_k_constrained}', model_name, rank)
            print(f'pair_universe_progress model={model_name} rows={end}/{n} pairs={len(records)}', flush=True)
    return records

def add_sequence_candidates(records: dict[tuple[int, int], dict[str, Any]], meta: pd.DataFrame) -> dict[str, int]:
    working = meta.copy()
    working['origin_clean'] = working['source_origin_key'].where(working['source_origin_key'].astype(str).str.len() > 0, working['relative_path'].map(lambda value: Path(str(value)).stem)).map(clean_origin)
    timestamp_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []
    for row in working.itertuples(index=False):
        timestamp = timestamp_key(row.origin_clean)
        if timestamp:
            prefix, moment = timestamp
            timestamp_rows.append({'embedding_row': int(row.embedding_row), 'group': f'{row.source_archive}||{prefix}', 'moment': moment})
        sequence = sequence_key(row.origin_clean)
        if sequence:
            prefix, number = sequence
            numeric_rows.append({'embedding_row': int(row.embedding_row), 'group': f'{row.source_archive}||{prefix}', 'number': number})
    counts = Counter()
    if timestamp_rows:
        table = pd.DataFrame(timestamp_rows)
        for _, group in table.groupby('group', sort=False):
            values = group.sort_values('moment').to_dict('records')
            for left, right in zip(values, values[1:]):
                delta = float(right['moment']) - float(left['moment'])
                if 0 < delta <= 15.0:
                    add_record(records, int(left['embedding_row']), int(right['embedding_row']), 'adjacent_timestamp_le_15s')
                    counts['adjacent_timestamp_le_15s'] += 1
                    if delta <= 2.0:
                        add_record(records, int(left['embedding_row']), int(right['embedding_row']), 'adjacent_timestamp_le_2s')
                        counts['adjacent_timestamp_le_2s'] += 1
    if numeric_rows:
        table = pd.DataFrame(numeric_rows)
        for _, group in table.groupby('group', sort=False):
            values = group.sort_values('number').to_dict('records')
            for left, right in zip(values, values[1:]):
                difference = int(right['number']) - int(left['number'])
                if difference == 1:
                    add_record(records, int(left['embedding_row']), int(right['embedding_row']), 'adjacent_numeric_plus1')
                    counts['adjacent_numeric_plus1'] += 1
    return dict(counts)

def add_hidden_control_relations(records: dict[tuple[int, int], dict[str, Any]], control: pd.DataFrame, component_to_index: dict[str, int]) -> int:
    selected = control[control['pair_origin'].eq('RANDOM_NEGATIVE_CONTROL') & control['verification_relation'].isin(['GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW'])]
    added = 0
    for row in selected.itertuples(index=False):
        component_a = str(row.component_a)
        component_b = str(row.component_b)
        if component_a not in component_to_index or component_b not in component_to_index:
            continue
        i = component_to_index[component_a]
        j = component_to_index[component_b]
        key = (i, j) if i < j else (j, i)
        before = key in records
        add_record(records, i, j, 'stage2b_hidden_control_relation')
        records[key]['stage2b_control_relation'] = str(row.verification_relation)
        added += int(not before)
    return added

def score_pair_chunks(records: dict[tuple[int, int], dict[str, Any]], embeddings: dict[str, np.ndarray], chunk_size: int=50000) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    pairs = sorted(records)
    i_values = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    j_values = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    scores: dict[str, np.ndarray] = {name: np.empty(len(pairs), dtype=np.float32) for name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        matrix = embeddings[model_name]
        for start in range(0, len(pairs), chunk_size):
            end = min(len(pairs), start + chunk_size)
            left = np.asarray(matrix[i_values[start:end]], dtype=np.float32)
            right = np.asarray(matrix[j_values[start:end]], dtype=np.float32)
            scores[model_name][start:end] = np.einsum('ij,ij->i', left, right)
        print(f'pair_scoring model={model_name} pairs={len(pairs)}', flush=True)
    return (scores['resnet18'], scores['efficientnet_b0'], pairs)

def gate_masks(table: pd.DataFrame, primary_eff: float, dual_eff: float, dual_res: float, sequence_eff: float, sequence_res: float) -> dict[str, pd.Series]:
    eff = pd.to_numeric(table['efficientnet_b0_similarity'], errors='coerce')
    res = pd.to_numeric(table['resnet18_similarity'], errors='coerce')
    primary = eff >= primary_eff
    dual = ~primary & (eff >= dual_eff) & (res >= dual_res)
    sequence_reason = table['is_sequence_candidate'].eq('YES')
    sequence = ~primary & ~dual & sequence_reason & (eff >= sequence_eff) & (res >= sequence_res)
    return {'PRIMARY_EFFICIENTNET': primary, 'DUAL_MODEL_RESCUE': dual, 'SEQUENCE_RESCUE': sequence, 'PRIMARY_OR_DUAL': primary | dual, 'ANY_GATE': primary | dual | sequence}

def calibration_performance(calibration: pd.DataFrame, primary_eff: float, dual_eff: float, dual_res: float, sequence_eff: float, sequence_res: float) -> pd.DataFrame:
    working = calibration.copy()
    working['is_sequence_candidate'] = 'NO'
    masks = gate_masks(working, primary_eff, dual_eff, dual_res, sequence_eff, sequence_res)
    rows: list[dict[str, object]] = []
    for gate_name in ('PRIMARY_EFFICIENTNET', 'DUAL_MODEL_RESCUE', 'PRIMARY_OR_DUAL'):
        mask = masks[gate_name]
        for group_name in ALL_CALIBRATION_GROUPS:
            group_mask = working['calibration_group'].eq(group_name)
            n = int(group_mask.sum())
            value = float(mask[group_mask].mean()) if n else np.nan
            rows.append({'gate': gate_name, 'calibration_group': group_name, 'n': n, 'pass_rate': value, 'pairs_passing': int(mask[group_mask].sum())})
    return pd.DataFrame(rows)

def build_gate_qc(performance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    def value(group: str) -> float:
        subset = performance[performance['gate'].eq('PRIMARY_OR_DUAL') & performance['calibration_group'].eq(group)]
        return float(subset.iloc[0]['pass_rate']) if not subset.empty else np.nan
    rules = [('strong_sensitivity_ge_0.975', 'VERIFIED_DERIVATIVE_STRONG', 0.975, '>='), ('moderate_sensitivity_ge_0.75', 'VERIFIED_DERIVATIVE_MODERATE', 0.75, '>='), ('same_scene_sensitivity_ge_0.95', 'GEOMETRIC_SAME_SCENE_ONLY', 0.95, '>='), ('rejected_pair_fpr_le_0.01', 'REJECTED_NOT_DERIVATIVE', 0.01, '<='), ('random_negative_fpr_le_0.01', 'RANDOM_NEGATIVE_CONTROL', 0.01, '<=')]
    rows = []
    for criterion, group, threshold, direction in rules:
        observed = value(group)
        if direction == '>=':
            passed = np.isfinite(observed) and observed >= threshold
        else:
            passed = np.isfinite(observed) and observed <= threshold
        rows.append({'criterion': criterion, 'calibration_group': group, 'observed': observed, 'threshold': threshold, 'direction': direction, 'pass': 'YES' if passed else 'NO'})
    qc = pd.DataFrame(rows)
    all_pass = bool(qc['pass'].eq('YES').all())
    claims = pd.DataFrame([{'item': 'evidence_first_gate_qc_pass', 'value': 'YES' if all_pass else 'NO', 'interpretation': 'The calibrated embedding gate is suitable for generating a primary geometric-verification tranche.' if all_pass else 'Do not treat the primary shortlist as a calibrated verification tranche; revise the embedding gate.'}, {'item': 'stage2c1_pairs_are_verified_lineage_edges', 'value': 'NO', 'interpretation': 'Every new evidence reranking pair still requires geometric and photometric verification.'}, {'item': 'cross_archive_split_or_label_flags_are_inclusion_evidence', 'value': 'NO', 'interpretation': 'Scientific-risk flags affect ranking only after an embedding-evidence gate is passed.'}])
    return (qc, claims)

def relation_risk(row: pd.Series) -> tuple[str, int]:
    labels: list[str] = []
    score = 0
    if row['cross_split'] == 'YES':
        labels.append('CROSS_SPLIT')
        score += 4
    if row['cross_archive'] == 'YES':
        labels.append('CROSS_ARCHIVE')
        score += 2
    if row['label_conflict'] == 'YES':
        labels.append('LABEL_CONFLICT')
        score += 1
    return ('|'.join(labels) if labels else 'GENERAL', score)

def select_shortlist(gated: pd.DataFrame, max_shortlist: int, max_degree: int, quota_cross_split: int, quota_cross_archive: int, quota_label_conflict: int, quota_sequence: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if gated.empty:
        raise RuntimeError('No candidates passed the evidence-first gate')
    working = gated.sort_values(['tier_order', 'evidence_score', 'risk_score', 'component_a', 'component_b'], ascending=[True, False, False, True, True]).reset_index(drop=True)
    selected: set[int] = set()
    degree: Counter[str] = Counter()
    selection_reason: dict[int, str] = {}

    def take_from(mask: pd.Series, limit: int, reason: str, degree_limit: int) -> None:
        if limit <= 0 or len(selected) >= max_shortlist:
            return
        taken = 0
        for index, row in working[mask].iterrows():
            if index in selected:
                continue
            a = str(row['component_a'])
            b = str(row['component_b'])
            if degree[a] >= degree_limit or degree[b] >= degree_limit:
                continue
            selected.add(int(index))
            selection_reason[int(index)] = reason
            degree[a] += 1
            degree[b] += 1
            taken += 1
            if taken >= limit or len(selected) >= max_shortlist:
                break
    take_from(working['cross_split'].eq('YES'), quota_cross_split, 'RISK_RESERVE_CROSS_SPLIT', max_degree * 2)
    take_from(working['cross_archive'].eq('YES'), quota_cross_archive, 'RISK_RESERVE_CROSS_ARCHIVE', max_degree * 2)
    take_from(working['label_conflict'].eq('YES'), quota_label_conflict, 'RISK_RESERVE_LABEL_CONFLICT', max_degree * 2)
    take_from(working['evidence_tier'].eq('C_SEQUENCE_RESCUE'), quota_sequence, 'SEQUENCE_RESERVE', max_degree * 2)
    take_from(pd.Series(True, index=working.index), max_shortlist, 'EVIDENCE_FILL', max_degree)
    selected_indices = sorted(selected)
    shortlist = working.loc[selected_indices].copy()
    shortlist['shortlist_selection_reason'] = [selection_reason[i] for i in selected_indices]
    shortlist = shortlist.sort_values(['tier_order', 'evidence_score', 'risk_score'], ascending=[True, False, False]).reset_index(drop=True)
    shortlist.insert(0, 'pair_id', [f'S2C1P{i:06d}' for i in range(1, len(shortlist) + 1)])
    reserve = working.drop(index=selected_indices).copy().reset_index(drop=True)
    reserve.insert(0, 'pair_id', [f'S2C1R{i:07d}' for i in range(1, len(reserve) + 1)])
    degree_rows = [{'exact_component_id': component, 'shortlist_degree': count} for component, count in sorted(degree.items())]
    degree_table = pd.DataFrame(degree_rows)
    return (shortlist, reserve, degree_table)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-manifest', required=True, type=Path)
    parser.add_argument('--resnet18-embeddings', required=True, type=Path)
    parser.add_argument('--efficientnet-b0-embeddings', required=True, type=Path)
    parser.add_argument('--stage2b-candidates', required=True, type=Path)
    parser.add_argument('--stage2b-controls', required=True, type=Path)
    parser.add_argument('--stage2c-calibration', required=True, type=Path)
    parser.add_argument('--stage2c-provenance', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--top-k-overall', type=int, default=30)
    parser.add_argument('--top-k-constrained', type=int, default=12)
    parser.add_argument('--resnet-prefilter', type=float, default=0.75)
    parser.add_argument('--efficientnet-prefilter', type=float, default=0.6)
    parser.add_argument('--primary-efficientnet', type=float, default=0.8)
    parser.add_argument('--dual-efficientnet', type=float, default=0.78)
    parser.add_argument('--dual-resnet', type=float, default=0.85)
    parser.add_argument('--sequence-efficientnet', type=float, default=0.7)
    parser.add_argument('--sequence-resnet', type=float, default=0.85)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--max-shortlist', type=int, default=6000)
    parser.add_argument('--max-degree', type=int, default=30)
    parser.add_argument('--quota-cross-split', type=int, default=1500)
    parser.add_argument('--quota-cross-archive', type=int, default=2500)
    parser.add_argument('--quota-label-conflict', type=int, default=1500)
    parser.add_argument('--quota-sequence', type=int, default=500)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.embedding_manifest, sep='\t', keep_default_na=False, low_memory=False)
    required_meta = {'embedding_row', 'exact_component_id', 'source_archive', 'corrected_label', 'corrected_split', 'source_origin_key', 'relative_path', 'absolute_path'}
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise RuntimeError(f'Embedding manifest is missing columns: {sorted(missing_meta)}')
    observed_rows = pd.to_numeric(meta['embedding_row'], errors='raise').to_numpy(dtype=np.int64)
    if not np.array_equal(observed_rows, np.arange(len(meta), dtype=np.int64)):
        raise RuntimeError('Embedding manifest row order is not contiguous')
    embeddings = {'resnet18': np.load(args.resnet18_embeddings, mmap_mode='r'), 'efficientnet_b0': np.load(args.efficientnet_b0_embeddings, mmap_mode='r')}
    for model_name, matrix in embeddings.items():
        if matrix.shape[0] != len(meta):
            raise RuntimeError(f'{model_name} row mismatch')
    stage2b_candidate = pd.read_csv(args.stage2b_candidates, sep='\t', keep_default_na=False, low_memory=False)
    stage2b_control = pd.read_csv(args.stage2b_controls, sep='\t', keep_default_na=False, low_memory=False)
    calibration = pd.read_csv(args.stage2c_calibration, sep='\t', keep_default_na=False, low_memory=False)
    provenance = pd.read_csv(args.stage2c_provenance, sep='\t', keep_default_na=False)
    provenance_map = dict(zip(provenance['item'].astype(str), provenance['value'].astype(str)))
    current_hashes = {'embedding_manifest_sha256': sha256_file(args.embedding_manifest), 'resnet18_embedding_sha256': sha256_file(args.resnet18_embeddings), 'efficientnet_b0_embedding_sha256': sha256_file(args.efficientnet_b0_embeddings), 'stage2b_candidate_sha256': sha256_file(args.stage2b_candidates), 'stage2b_control_sha256': sha256_file(args.stage2b_controls)}
    mismatches = [key for key, value in current_hashes.items() if key in provenance_map and provenance_map[key] != value]
    if mismatches:
        raise RuntimeError(f'embedding retrieval input checksum mismatch: {mismatches}')
    component_to_index = dict(zip(meta['exact_component_id'].astype(str), meta['embedding_row'].astype(int)))
    records = build_pair_universe(embeddings, meta, args.top_k_overall, args.top_k_constrained, args.resnet_prefilter, args.efficientnet_prefilter, args.query_chunk)
    sequence_counts = add_sequence_candidates(records, meta)
    hidden_added = add_hidden_control_relations(records, stage2b_control, component_to_index)
    pair_universe_count = len(records)
    existing_pairs = {normalize_pair(str(a), str(b)) for a, b in zip(stage2b_candidate['component_a'], stage2b_candidate['component_b'])}
    excluded_existing = 0
    for key in list(records):
        a, b = key
        pair = normalize_pair(str(meta.iloc[a]['exact_component_id']), str(meta.iloc[b]['exact_component_id']))
        if pair in existing_pairs:
            del records[key]
            excluded_existing += 1
    res_scores, eff_scores, pairs = score_pair_chunks(records, embeddings)
    rows: list[dict[str, object]] = []
    hidden_rows: list[dict[str, object]] = []
    for position, (i, j) in enumerate(pairs):
        record = records[i, j]
        left = meta.iloc[i]
        right = meta.iloc[j]
        reasons = sorted(record['reasons'])
        archive_a, archive_b = (str(left['source_archive']), str(right['source_archive']))
        label_a, label_b = (str(left['corrected_label']), str(right['corrected_label']))
        split_a, split_b = (str(left['corrected_split']), str(right['corrected_split']))
        cross_archive = archive_a != archive_b
        label_conflict = label_a != label_b
        cross_split = split_a.lower() not in UNKNOWN_SPLITS and split_b.lower() not in UNKNOWN_SPLITS and (split_a != split_b)
        is_sequence = any((reason.startswith('adjacent_') for reason in reasons))
        hidden_relation = str(record['stage2b_control_relation'])
        common = {'component_a': str(left['exact_component_id']), 'component_b': str(right['exact_component_id']), 'candidate_reasons': '|'.join(reasons), 'stage2b_control_relation': hidden_relation, 'resnet18_similarity': float(res_scores[position]), 'efficientnet_b0_similarity': float(eff_scores[position]), 'resnet18_best_rank': record['resnet18_best_rank'], 'efficientnet_b0_best_rank': record['efficientnet_b0_best_rank'], 'cross_archive': 'YES' if cross_archive else 'NO', 'cross_split': 'YES' if cross_split else 'NO', 'label_conflict': 'YES' if label_conflict else 'NO', 'is_sequence_candidate': 'YES' if is_sequence else 'NO', 'archive_a': archive_a, 'archive_b': archive_b, 'label_a': label_a, 'label_b': label_b, 'split_a': split_a, 'split_b': split_b, 'path_a': str(left['relative_path']), 'path_b': str(right['relative_path']), 'absolute_path_a': str(left['absolute_path']), 'absolute_path_b': str(right['absolute_path']), 'source_origin_key_a': str(left['source_origin_key']), 'source_origin_key_b': str(right['source_origin_key'])}
        if hidden_relation:
            hidden_rows.append(common)
        else:
            rows.append(common)
    universe = pd.DataFrame(rows)
    hidden = pd.DataFrame(hidden_rows)
    if universe.empty:
        raise RuntimeError('Pair universe is empty after excluding existing geometric verification pairs')
    masks = gate_masks(universe, args.primary_efficientnet, args.dual_efficientnet, args.dual_resnet, args.sequence_efficientnet, args.sequence_resnet)
    universe['evidence_tier'] = 'D_DROPPED_BELOW_GATE'
    universe.loc[masks['PRIMARY_EFFICIENTNET'], 'evidence_tier'] = 'A_EFFICIENTNET_PRIMARY'
    universe.loc[masks['DUAL_MODEL_RESCUE'], 'evidence_tier'] = 'B_DUAL_MODEL_RESCUE'
    universe.loc[masks['SEQUENCE_RESCUE'], 'evidence_tier'] = 'C_SEQUENCE_RESCUE'
    tier_order_map = {'A_EFFICIENTNET_PRIMARY': 1, 'B_DUAL_MODEL_RESCUE': 2, 'C_SEQUENCE_RESCUE': 3, 'D_DROPPED_BELOW_GATE': 9}
    universe['tier_order'] = universe['evidence_tier'].map(tier_order_map).astype(int)
    risks = universe.apply(relation_risk, axis=1)
    universe['scientific_risk'] = [item[0] for item in risks]
    universe['risk_score'] = [item[1] for item in risks]
    eff = pd.to_numeric(universe['efficientnet_b0_similarity'], errors='coerce')
    res = pd.to_numeric(universe['resnet18_similarity'], errors='coerce')
    universe['efficientnet_primary_margin'] = eff - args.primary_efficientnet
    universe['dual_margin'] = np.minimum(eff - args.dual_efficientnet, res - args.dual_resnet)
    universe['evidence_score'] = eff * 1000.0 + res * 500.0 + universe['risk_score'] * 10.0 + universe['is_sequence_candidate'].eq('YES').astype(int) * 5.0
    gated = universe[masks['ANY_GATE']].copy()
    dropped = universe[~masks['ANY_GATE']].copy()
    gated = gated.sort_values(['tier_order', 'evidence_score', 'risk_score', 'component_a', 'component_b'], ascending=[True, False, False, True, True]).reset_index(drop=True)
    gated.insert(0, 'gated_pair_id', [f'S2C1G{i:07d}' for i in range(1, len(gated) + 1)])
    shortlist, reserve, degree_table = select_shortlist(gated, args.max_shortlist, args.max_degree, args.quota_cross_split, args.quota_cross_archive, args.quota_label_conflict, args.quota_sequence)
    performance = calibration_performance(calibration, args.primary_efficientnet, args.dual_efficientnet, args.dual_resnet, args.sequence_efficientnet, args.sequence_resnet)
    gate_qc, claims = build_gate_qc(performance)
    thresholds = pd.DataFrame([{'parameter': 'primary_efficientnet_b0_similarity', 'value': args.primary_efficientnet, 'role': 'primary evidence gate'}, {'parameter': 'dual_efficientnet_b0_similarity', 'value': args.dual_efficientnet, 'role': 'dual-model rescue'}, {'parameter': 'dual_resnet18_similarity', 'value': args.dual_resnet, 'role': 'dual-model rescue'}, {'parameter': 'sequence_efficientnet_b0_similarity', 'value': args.sequence_efficientnet, 'role': 'sequence-only relaxed rescue'}, {'parameter': 'sequence_resnet18_similarity', 'value': args.sequence_resnet, 'role': 'sequence-only relaxed rescue'}, {'parameter': 'max_primary_shortlist', 'value': args.max_shortlist, 'role': 'primary geometric-verification tranche cap'}, {'parameter': 'max_component_degree', 'value': args.max_degree, 'role': 'semantic-cluster domination guard'}])
    if not hidden.empty:
        hidden.insert(0, 'pair_id', [f'S2C1H{i:04d}' for i in range(1, len(hidden) + 1)])
        hidden['recommended_action'] = np.where(hidden['stage2b_control_relation'].eq('GEOMETRIC_SAME_SCENE_ONLY'), 'ADD_TO_LEAKAGE_SAFE_FAMILY_WITHOUT_REVERIFICATION', 'MANUAL_ADJUDICATION_ONLY')
    write_tsv(thresholds, args.output_dir / 'stage2c1_calibrated_thresholds.tsv')
    write_tsv(performance, args.output_dir / 'stage2c1_gate_performance.tsv')
    write_tsv(gate_qc, args.output_dir / 'stage2c1_gate_qc.tsv')
    write_tsv(claims, args.output_dir / 'stage2c1_claim_status.tsv')
    write_tsv(shortlist, args.output_dir / 'stage2c1_stage2d_primary_shortlist.tsv')
    write_tsv_gz(gated, args.output_dir / 'stage2c1_all_evidence_gated_candidates.tsv.gz')
    write_tsv_gz(reserve, args.output_dir / 'stage2c1_stage2d_reserve_candidates.tsv.gz')
    write_tsv(degree_table, args.output_dir / 'stage2c1_shortlist_component_degrees.tsv')
    if not hidden.empty:
        write_tsv(hidden, args.output_dir / 'stage2c1_hidden_control_relations.tsv')
    else:
        write_tsv(pd.DataFrame(columns=['pair_id', 'component_a', 'component_b']), args.output_dir / 'stage2c1_hidden_control_relations.tsv')
    source_counts = shortlist.assign(source_pair=[' || '.join(sorted((a, b))) for a, b in zip(shortlist['archive_a'], shortlist['archive_b'])]).groupby(['source_pair', 'evidence_tier', 'scientific_risk'], dropna=False).size().reset_index(name='pair_count').sort_values('pair_count', ascending=False)
    write_tsv(source_counts, args.output_dir / 'stage2c1_shortlist_source_pair_counts.tsv')
    funnel = pd.DataFrame([{'stage': 'PAIR_UNIVERSE', 'count': pair_universe_count}, {'stage': 'EXCLUDED_EXISTING_STAGE2B', 'count': excluded_existing}, {'stage': 'HIDDEN_CONTROL_RELATIONS', 'count': len(hidden)}, {'stage': 'PRIMARY_GATE', 'count': int(masks['PRIMARY_EFFICIENTNET'].sum())}, {'stage': 'DUAL_RESCUE', 'count': int(masks['DUAL_MODEL_RESCUE'].sum())}, {'stage': 'SEQUENCE_RESCUE', 'count': int(masks['SEQUENCE_RESCUE'].sum())}, {'stage': 'ALL_EVIDENCE_GATED', 'count': len(gated)}, {'stage': 'PRIMARY_STAGE2D_SHORTLIST', 'count': len(shortlist)}, {'stage': 'RESERVE', 'count': len(reserve)}, {'stage': 'DROPPED_BELOW_GATE', 'count': len(dropped)}])
    write_tsv(funnel, args.output_dir / 'stage2c1_candidate_funnel.tsv')
    tier_summary = shortlist.groupby(['evidence_tier', 'scientific_risk'], dropna=False).size().reset_index(name='pair_count').sort_values(['evidence_tier', 'pair_count'], ascending=[True, False])
    write_tsv(tier_summary, args.output_dir / 'stage2c1_shortlist_summary.tsv')
    provenance_rows = [{'item': key, 'value': value} for key, value in current_hashes.items()]
    provenance_rows.extend([{'item': 'stage2c_calibration_sha256', 'value': sha256_file(args.stage2c_calibration)}, {'item': 'top_k_overall', 'value': args.top_k_overall}, {'item': 'top_k_constrained', 'value': args.top_k_constrained}, {'item': 'resnet_prefilter', 'value': args.resnet_prefilter}, {'item': 'efficientnet_prefilter', 'value': args.efficientnet_prefilter}, {'item': 'pair_universe_count', 'value': pair_universe_count}, {'item': 'excluded_existing_stage2b', 'value': excluded_existing}, {'item': 'hidden_control_added', 'value': hidden_added}, {'item': 'sequence_candidate_counts', 'value': str(sequence_counts)}, {'item': 'elapsed_seconds', 'value': time.time() - started}])
    write_tsv(pd.DataFrame(provenance_rows), args.output_dir / 'stage2c1_run_provenance.tsv')
    boundary = shortlist.assign(threshold_distance=np.minimum(np.abs(shortlist['efficientnet_b0_similarity'] - args.primary_efficientnet), np.minimum(np.abs(shortlist['efficientnet_b0_similarity'] - args.dual_efficientnet), np.abs(shortlist['resnet18_similarity'] - args.dual_resnet)))).sort_values(['threshold_distance', 'risk_score'], ascending=[True, False])
    qc_pass = claims.loc[claims['item'].eq('evidence_first_gate_qc_pass'), 'value'].iloc[0]
    summary_lines = ['# Evidence-first embedding recalibration', '', f'- Exact-component representatives: {len(meta):,}', f'- Pair universe before excluding previously verified pairs: {pair_universe_count:,}', f'- Previously verified pairs removed: {excluded_existing:,}', f'- Primary EfficientNet gate: similarity >= {args.primary_efficientnet:.3f}', f'- Dual-model rescue: EfficientNet >= {args.dual_efficientnet:.3f} and ResNet18 >= {args.dual_resnet:.3f}', f'- Sequence-only rescue: EfficientNet >= {args.sequence_efficientnet:.3f} and ResNet18 >= {args.sequence_resnet:.3f}', f'- Evidence-gated candidates: {len(gated):,}', f'- Primary geometric-verification shortlist: {len(shortlist):,}', f'- Reserve candidates: {len(reserve):,}', f'- Gate QC pass: {qc_pass}', '', '## Evidence-first correction to embedding retrieval', 'Cross-archive, cross-split, and cross-label status no longer causes automatic inclusion.', 'Those variables are scientific-risk priorities only after a pair passes an embedding-evidence gate.', 'The earlier broad retrieval cap is not used as the primary geometric-verification input.', '', '## Claim boundary', 'No new pair in this stage is a verified lineage or same-scene edge. The primary shortlist', 'must be processed by the geometric and photometric verifier. The reserve is retained', 'for a second verification tranche if the lower-evidence end of the primary tranche has a', 'non-negligible verified-edge yield.']
    (args.output_dir / 'stage2c1_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    (args.output_dir / '.stage2c1_complete').write_text('OK\n', encoding='utf-8')
    print(funnel.to_string(index=False), flush=True)
    print(gate_qc.to_string(index=False), flush=True)
    print(tier_summary.to_string(index=False), flush=True)
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
