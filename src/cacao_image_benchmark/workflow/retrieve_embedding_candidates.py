"""All-image frozen-embedding nearest-neighbor sweep for cacao image-lineage recall."""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
MODEL_NAMES = ('resnet18', 'efficientnet_b0')
CALIBRATION_K = 20
VALID_UNKNOWN_SPLITS = {'', 'unspecified', 'UNMATCHED'}

def write_tsv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep='\t', index=False, na_rep='')

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

def add_record(records: dict[tuple[int, int], dict[str, Any]], i: int, j: int, reason: str, model: str | None=None, similarity: float | None=None, rank: int | None=None) -> None:
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    record = records.setdefault((a, b), {'i': a, 'j': b, 'reasons': set(), 'resnet18_similarity': np.nan, 'efficientnet_b0_similarity': np.nan, 'resnet18_rank': np.nan, 'efficientnet_b0_rank': np.nan, 'stage2b_control_relation': ''})
    record['reasons'].add(reason)
    if model and similarity is not None:
        similarity_key = f'{model}_similarity'
        rank_key = f'{model}_rank'
        prior_similarity = record.get(similarity_key, np.nan)
        if not np.isfinite(prior_similarity) or similarity > prior_similarity:
            record[similarity_key] = float(similarity)
        if rank is not None:
            prior_rank = record.get(rank_key, np.nan)
            if not np.isfinite(prior_rank) or rank < prior_rank:
                record[rank_key] = int(rank)

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

def build_neighbor_candidates(embeddings: dict[str, np.ndarray], meta: pd.DataFrame, top_k: int, top_k_cross_archive: int, top_k_cross_split: int, top_k_cross_label: int, cross_archive_min: float, cross_split_min: float, cross_label_min: float, query_chunk: int, threads: int) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, dict[tuple[int, int], int]]]:
    del threads
    records: dict[tuple[int, int], dict[str, Any]] = {}
    rank_maps: dict[str, dict[tuple[int, int], int]] = {name: {} for name in embeddings}
    archives = meta['source_archive'].astype(str).to_numpy()
    labels = meta['corrected_label'].astype(str).to_numpy()
    splits = meta['corrected_split'].astype(str).to_numpy()
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
        print(f'neighbor_sweep model={model_name} rows={n} dim={matrix.shape[1]}', flush=True)
        for start in range(0, n, query_chunk):
            end = min(n, start + query_chunk)
            similarities = matrix[start:end] @ matrix_t
            for local, i in enumerate(range(start, end)):
                scores = similarities[local]
                scores[i] = -np.inf
                base_mask = np.ones(n, dtype=bool)
                base_mask[i] = False
                overall = safe_topk(scores, base_mask, max(CALIBRATION_K, top_k))
                for rank, (j, similarity) in enumerate(overall, 1):
                    pair = (i, j) if i < j else (j, i)
                    prior = rank_maps[model_name].get(pair)
                    if prior is None or rank < prior:
                        rank_maps[model_name][pair] = rank
                    if rank <= top_k:
                        add_record(records, i, j, f'{model_name}_overall_top{top_k}', model_name, similarity, rank)
                cross_archive_mask = base_mask & (archives != archives[i])
                for rank, (j, similarity) in enumerate(safe_topk(scores, cross_archive_mask, top_k_cross_archive, cross_archive_min), 1):
                    add_record(records, i, j, f'{model_name}_cross_archive_top{top_k_cross_archive}', model_name, similarity, rank)
                if splits[i] not in VALID_UNKNOWN_SPLITS:
                    cross_split_mask = base_mask & (splits != splits[i]) & ~np.isin(splits, list(VALID_UNKNOWN_SPLITS))
                    for rank, (j, similarity) in enumerate(safe_topk(scores, cross_split_mask, top_k_cross_split, cross_split_min), 1):
                        add_record(records, i, j, f'{model_name}_cross_split_top{top_k_cross_split}', model_name, similarity, rank)
                cross_label_mask = base_mask & (labels != labels[i])
                for rank, (j, similarity) in enumerate(safe_topk(scores, cross_label_mask, top_k_cross_label, cross_label_min), 1):
                    add_record(records, i, j, f'{model_name}_cross_label_top{top_k_cross_label}', model_name, similarity, rank)
            print(f'neighbor_progress model={model_name} rows={end}/{n} pairs={len(records)}', flush=True)
    return (records, rank_maps)

def add_sequence_candidates(records: dict[tuple[int, int], dict[str, Any]], meta: pd.DataFrame) -> int:
    added_before = len(records)
    working = meta.copy()
    working['origin_clean'] = working['source_origin_key'].where(working['source_origin_key'].astype(str).str.len() > 0, working['relative_path'].map(lambda x: Path(str(x)).stem)).map(clean_origin)
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
    if timestamp_rows:
        table = pd.DataFrame(timestamp_rows)
        for _, group in table.groupby('group', sort=False):
            values = group.sort_values('moment').to_dict('records')
            for a, b in zip(values, values[1:]):
                if 0 < float(b['moment']) - float(a['moment']) <= 15.0:
                    add_record(records, int(a['embedding_row']), int(b['embedding_row']), 'adjacent_timestamp_sequence')
    if numeric_rows:
        table = pd.DataFrame(numeric_rows)
        for _, group in table.groupby('group', sort=False):
            values = group.sort_values('number').to_dict('records')
            for a, b in zip(values, values[1:]):
                difference = int(b['number']) - int(a['number'])
                if 0 < difference <= 1:
                    add_record(records, int(a['embedding_row']), int(b['embedding_row']), 'adjacent_numeric_sequence')
    return len(records) - added_before

def enrich_hidden_control_discoveries(records: dict[tuple[int, int], dict[str, Any]], control: pd.DataFrame, component_to_index: dict[str, int]) -> int:
    selected = control[control['pair_origin'].eq('RANDOM_NEGATIVE_CONTROL') & control['verification_relation'].isin(['GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW'])]
    added = 0
    for row in selected.itertuples(index=False):
        component_a = str(row.component_a)
        component_b = str(row.component_b)
        if component_a not in component_to_index or component_b not in component_to_index:
            continue
        i, j = (component_to_index[component_a], component_to_index[component_b])
        before = len(records)
        add_record(records, i, j, 'stage2b_control_hidden_related')
        key = (i, j) if i < j else (j, i)
        records[key]['stage2b_control_relation'] = str(row.verification_relation)
        added += int(len(records) > before)
    return added

def pair_similarity(matrix: np.ndarray, i: int, j: int) -> float:
    return float(np.dot(matrix[i], matrix[j]))

def build_calibration(stage2b_candidate: pd.DataFrame, stage2b_control: pd.DataFrame, embeddings: dict[str, np.ndarray], component_to_index: dict[str, int], rank_maps: dict[str, dict[tuple[int, int], int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = stage2b_candidate.copy()
    candidate['calibration_group'] = candidate['verification_relation']
    candidate['calibration_origin'] = 'STAGE2B_CANDIDATE'
    controls = stage2b_control[stage2b_control['pair_origin'].eq('RANDOM_NEGATIVE_CONTROL')].copy()
    controls['calibration_group'] = 'RANDOM_NEGATIVE_CONTROL'
    controls['calibration_origin'] = 'STAGE2B_CONTROL'
    combined = pd.concat([candidate, controls], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for row in combined.itertuples(index=False):
        component_a = str(row.component_a)
        component_b = str(row.component_b)
        if component_a not in component_to_index or component_b not in component_to_index or component_a == component_b:
            continue
        i, j = (component_to_index[component_a], component_to_index[component_b])
        pair = (i, j) if i < j else (j, i)
        output: dict[str, object] = {'pair_id': str(row.pair_id), 'calibration_origin': str(row.calibration_origin), 'calibration_group': str(row.calibration_group), 'component_a': component_a, 'component_b': component_b}
        for model_name, matrix in embeddings.items():
            output[f'{model_name}_similarity'] = pair_similarity(matrix, i, j)
            output[f'{model_name}_rank'] = rank_maps[model_name].get(pair, np.nan)
        rows.append(output)
    table = pd.DataFrame(rows)
    metric_rows: list[dict[str, object]] = []
    order = ['VERIFIED_DERIVATIVE_STRONG', 'VERIFIED_DERIVATIVE_MODERATE', 'GEOMETRIC_SAME_SCENE_ONLY', 'AMBIGUOUS_MANUAL_REVIEW', 'REJECTED_NOT_DERIVATIVE', 'RANDOM_NEGATIVE_CONTROL']
    for group_name in order:
        group = table[table['calibration_group'].eq(group_name)]
        if group.empty:
            continue
        metric: dict[str, object] = {'calibration_group': group_name, 'n': len(group)}
        for model_name in embeddings:
            values = pd.to_numeric(group[f'{model_name}_similarity'], errors='coerce').dropna()
            metric[f'{model_name}_similarity_median'] = float(values.median())
            metric[f'{model_name}_similarity_q05'] = float(values.quantile(0.05))
            metric[f'{model_name}_similarity_q95'] = float(values.quantile(0.95))
            ranks = pd.to_numeric(group[f'{model_name}_rank'], errors='coerce')
            for k in (1, 5, 10, 20):
                metric[f'{model_name}_recall_at_{k}'] = float((ranks <= k).mean())
        for k in (1, 5, 10, 20):
            union = np.zeros(len(group), dtype=bool)
            for model_name in embeddings:
                ranks = pd.to_numeric(group[f'{model_name}_rank'], errors='coerce').to_numpy()
                union |= np.isfinite(ranks) & (ranks <= k)
            metric[f'union_recall_at_{k}'] = float(union.mean())
        metric_rows.append(metric)
    return (table, pd.DataFrame(metric_rows))

def build_retrieval_qc(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = [('strong_union_recall_at_20_ge_0.95', 'VERIFIED_DERIVATIVE_STRONG', 'union_recall_at_20', 0.95, '>='), ('moderate_union_recall_at_20_ge_0.75', 'VERIFIED_DERIVATIVE_MODERATE', 'union_recall_at_20', 0.75, '>='), ('same_scene_union_recall_at_20_ge_0.50', 'GEOMETRIC_SAME_SCENE_ONLY', 'union_recall_at_20', 0.5, '>=')]
    rows = []
    for criterion, group, column, threshold, direction in rules:
        subset = metrics[metrics['calibration_group'].eq(group)]
        value = float(subset.iloc[0][column]) if not subset.empty else np.nan
        passed = np.isfinite(value) and value >= threshold
        rows.append({'criterion': criterion, 'calibration_group': group, 'metric': column, 'value': value, 'threshold': threshold, 'pass': 'YES' if passed else 'NO'})
    qc = pd.DataFrame(rows)
    all_pass = bool(qc['pass'].eq('YES').all())
    status = pd.DataFrame([{'item': 'embedding_candidate_sweep_retrieval_qc_pass', 'value': 'YES' if all_pass else 'NO', 'interpretation': 'Embedding sweep can be used as the primary recall-expansion layer, with geometric verification still required.' if all_pass else 'Embedding sweep remains exploratory; add a stronger self-supervised embedding before claiming broad recall.'}, {'item': 'new_candidate_pairs_are_verified_lineage_edges', 'value': 'NO', 'interpretation': 'Every new pair requires geometric and photometric verification.'}])
    return (qc, status)

def source_pair(a: str, b: str) -> str:
    return ' || '.join(sorted((a, b)))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-manifest', required=True, type=Path)
    parser.add_argument('--resnet18-embeddings', required=True, type=Path)
    parser.add_argument('--efficientnet-b0-embeddings', required=True, type=Path)
    parser.add_argument('--stage2b-candidates', required=True, type=Path)
    parser.add_argument('--stage2b-controls', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--top-k-cross-archive', type=int, default=6)
    parser.add_argument('--top-k-cross-split', type=int, default=6)
    parser.add_argument('--top-k-cross-label', type=int, default=6)
    parser.add_argument('--cross-archive-min', type=float, default=0.7)
    parser.add_argument('--cross-split-min', type=float, default=0.7)
    parser.add_argument('--cross-label-min', type=float, default=0.75)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--max-candidates', type=int, default=150000)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.embedding_manifest, sep='\t', keep_default_na=False, low_memory=False)
    observed_rows = pd.to_numeric(meta['embedding_row'], errors='raise').to_numpy(dtype=np.int64)
    if not np.array_equal(observed_rows, np.arange(len(meta), dtype=np.int64)):
        raise RuntimeError('Embedding manifest row order is not contiguous/deterministic')
    embeddings = {'resnet18': np.load(args.resnet18_embeddings, mmap_mode='r'), 'efficientnet_b0': np.load(args.efficientnet_b0_embeddings, mmap_mode='r')}
    stage2b_candidate = pd.read_csv(args.stage2b_candidates, sep='\t', keep_default_na=False, low_memory=False)
    stage2b_control = pd.read_csv(args.stage2b_controls, sep='\t', keep_default_na=False, low_memory=False)
    component_to_index = dict(zip(meta['exact_component_id'].astype(str), meta['embedding_row'].astype(int)))
    records, rank_maps = build_neighbor_candidates(embeddings=embeddings, meta=meta, top_k=args.top_k, top_k_cross_archive=args.top_k_cross_archive, top_k_cross_split=args.top_k_cross_split, top_k_cross_label=args.top_k_cross_label, cross_archive_min=args.cross_archive_min, cross_split_min=args.cross_split_min, cross_label_min=args.cross_label_min, query_chunk=args.query_chunk, threads=args.threads)
    sequence_added = add_sequence_candidates(records, meta)
    hidden_added = enrich_hidden_control_discoveries(records, stage2b_control, component_to_index)
    existing_pairs = {normalize_pair(str(a), str(b)) for a, b in zip(stage2b_candidate['component_a'], stage2b_candidate['component_b'])}
    rows: list[dict[str, object]] = []
    for record in records.values():
        a, b = (int(record['i']), int(record['j']))
        row_a, row_b = (meta.iloc[a], meta.iloc[b])
        component_pair = normalize_pair(str(row_a['exact_component_id']), str(row_b['exact_component_id']))
        if component_pair in existing_pairs:
            continue
        reasons = sorted(record['reasons'])
        similarities = [float(value) for value in (record['resnet18_similarity'], record['efficientnet_b0_similarity']) if np.isfinite(value)]
        max_similarity = max(similarities) if similarities else float('nan')
        both_models = any((reason.startswith('resnet18_') for reason in reasons)) and any((reason.startswith('efficientnet_b0_') for reason in reasons))
        archive_a, archive_b = (str(row_a['source_archive']), str(row_b['source_archive']))
        label_a, label_b = (str(row_a['corrected_label']), str(row_b['corrected_label']))
        split_a, split_b = (str(row_a['corrected_split']), str(row_b['corrected_split']))
        cross_archive = archive_a != archive_b
        cross_label = label_a != label_b
        cross_split = split_a not in VALID_UNKNOWN_SPLITS and split_b not in VALID_UNKNOWN_SPLITS and (split_a != split_b)
        hidden = 'stage2b_control_hidden_related' in reasons
        sequence = any((reason.startswith('adjacent_') for reason in reasons))
        if hidden or cross_split or cross_archive or cross_label:
            strength = 'CRITICAL_EMBEDDING_CANDIDATE'
        elif both_models or (np.isfinite(max_similarity) and max_similarity >= 0.9):
            strength = 'HIGH_EMBEDDING_CANDIDATE'
        elif np.isfinite(max_similarity) and max_similarity >= 0.8:
            strength = 'MEDIUM_EMBEDDING_CANDIDATE'
        else:
            strength = 'TOPK_EMBEDDING_CANDIDATE'
        priority = int(hidden) * 20000 + int(cross_split) * 10000 + int(cross_archive) * 5000 + int(cross_label) * 2500 + int(sequence) * 1200 + int(both_models) * 500 + (max_similarity if np.isfinite(max_similarity) else 0.0) * 100.0
        rows.append({'component_a': row_a['exact_component_id'], 'component_b': row_b['exact_component_id'], 'pair_origin': 'STAGE2C_EMBEDDING_SWEEP', 'expected_relation': 'UNKNOWN', 'synthetic_transform': '', 'candidate_strength': strength, 'candidate_reasons': '|'.join(reasons), 'stage2b_control_relation': record['stage2b_control_relation'], 'resnet18_similarity': record['resnet18_similarity'], 'efficientnet_b0_similarity': record['efficientnet_b0_similarity'], 'resnet18_rank': record['resnet18_rank'], 'efficientnet_b0_rank': record['efficientnet_b0_rank'], 'max_embedding_similarity': max_similarity, 'both_models_retrieved': 'YES' if both_models else 'NO', 'cross_archive': 'YES' if cross_archive else 'NO', 'cross_split': 'YES' if cross_split else 'NO', 'label_conflict': 'YES' if cross_label else 'NO', 'archive_a': archive_a, 'archive_b': archive_b, 'archives': '|'.join(sorted({archive_a, archive_b})), 'label_a': label_a, 'label_b': label_b, 'labels': '|'.join(sorted({label_a, label_b})), 'split_a': split_a, 'split_b': split_b, 'splits': '|'.join(sorted({split for split in (split_a, split_b) if split not in VALID_UNKNOWN_SPLITS})), 'path_a': row_a['relative_path'], 'path_b': row_b['relative_path'], 'absolute_path_a': row_a['absolute_path'], 'absolute_path_b': row_b['absolute_path'], 'source_origin_key_a': row_a['source_origin_key'], 'source_origin_key_b': row_b['source_origin_key'], 'priority_score': priority})
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError('Embedding sweep produced no new candidates')
    pre_cap = len(candidates)
    candidates = candidates.sort_values(['priority_score', 'max_embedding_similarity', 'component_a', 'component_b'], ascending=[False, False, True, True])
    if len(candidates) > args.max_candidates:
        candidates = candidates.head(args.max_candidates).copy()
    candidates.insert(0, 'pair_id', [f'EMB{i:07d}' for i in range(1, len(candidates) + 1)])
    write_tsv(candidates, args.output_dir / 'stage2c_new_embedding_candidates.tsv')
    calibration, calibration_metrics = build_calibration(stage2b_candidate=stage2b_candidate, stage2b_control=stage2b_control, embeddings=embeddings, component_to_index=component_to_index, rank_maps=rank_maps)
    write_tsv(calibration, args.output_dir / 'stage2c_calibration_pair_similarities.tsv')
    write_tsv(calibration_metrics, args.output_dir / 'stage2c_calibration_metrics.tsv')
    retrieval_qc, claim_status = build_retrieval_qc(calibration_metrics)
    write_tsv(retrieval_qc, args.output_dir / 'stage2c_retrieval_qc.tsv')
    write_tsv(claim_status, args.output_dir / 'stage2c_claim_status.tsv')
    summary_rows = [{'category': 'ALL', 'count': len(candidates)}, {'category': 'PRE_CAP_ALL', 'count': pre_cap}, {'category': 'SEQUENCE_PAIRS_ADDED_TO_UNION', 'count': sequence_added}, {'category': 'HIDDEN_CONTROL_PAIRS_ADDED_TO_UNION', 'count': hidden_added}, {'category': 'CROSS_ARCHIVE', 'count': int(candidates['cross_archive'].eq('YES').sum())}, {'category': 'CROSS_SPLIT', 'count': int(candidates['cross_split'].eq('YES').sum())}, {'category': 'LABEL_CONFLICT', 'count': int(candidates['label_conflict'].eq('YES').sum())}, {'category': 'BOTH_MODELS', 'count': int(candidates['both_models_retrieved'].eq('YES').sum())}]
    for category, count in candidates['candidate_strength'].value_counts().sort_index().items():
        summary_rows.append({'category': category, 'count': int(count)})
    candidate_summary = pd.DataFrame(summary_rows)
    write_tsv(candidate_summary, args.output_dir / 'stage2c_candidate_summary.tsv')
    source_counts = candidates.assign(source_pair=[source_pair(a, b) for a, b in zip(candidates['archive_a'], candidates['archive_b'])]).groupby(['source_pair', 'cross_archive', 'cross_split', 'label_conflict']).size().reset_index(name='pair_count').sort_values('pair_count', ascending=False)
    write_tsv(source_counts, args.output_dir / 'stage2c_source_pair_candidate_counts.tsv')
    strong = calibration_metrics[calibration_metrics['calibration_group'].eq('VERIFIED_DERIVATIVE_STRONG')]
    moderate = calibration_metrics[calibration_metrics['calibration_group'].eq('VERIFIED_DERIVATIVE_MODERATE')]
    same_scene = calibration_metrics[calibration_metrics['calibration_group'].eq('GEOMETRIC_SAME_SCENE_ONLY')]
    summary_lines = ['# embedding retrieval all-image frozen-embedding sweep', '', f'- Exact representatives embedded: {len(meta):,}', '- Models: ResNet18 and EfficientNet-B0 with frozen ImageNet weights', '- Similarity search: exact blockwise cosine search with normalized NumPy matrix multiplication', f'- New pair union before cap: {pre_cap:,}', f'- New candidate pairs retained: {len(candidates):,}', f"- Cross-archive new candidates: {int(candidates['cross_archive'].eq('YES').sum()):,}", f"- Cross-split new candidates: {int(candidates['cross_split'].eq('YES').sum()):,}", f"- Label-conflict new candidates: {int(candidates['label_conflict'].eq('YES').sum()):,}", f'- Adjacent filename/timestamp sequence additions: {sequence_added:,}', f'- geometric verification hidden-control discoveries added: {hidden_added:,}', '', '## Calibration against geometric verification']
    for label, table in (('Strong-pair', strong), ('Moderate-pair', moderate), ('Same-scene', same_scene)):
        if not table.empty:
            summary_lines.append(f"- {label} union recall@20: {float(table.iloc[0]['union_recall_at_20']):.4f}")
    summary_lines.extend([f"- Retrieval QC pass: {claim_status.iloc[0]['value']}", '', '## Claim boundary', 'These are nearest-neighbor candidates, not verified lineage edges. They must pass the', 'geometric and photometric verifier before they can modify strict or leakage-safe families.', 'The candidate cap and calibration recall table must be reported when interpreting sensitivity.'])
    (args.output_dir / 'stage2c_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    provenance = pd.DataFrame([{'item': 'embedding_manifest_sha256', 'value': sha256_file(args.embedding_manifest)}, {'item': 'resnet18_embedding_sha256', 'value': sha256_file(args.resnet18_embeddings)}, {'item': 'efficientnet_b0_embedding_sha256', 'value': sha256_file(args.efficientnet_b0_embeddings)}, {'item': 'stage2b_candidate_sha256', 'value': sha256_file(args.stage2b_candidates)}, {'item': 'stage2b_control_sha256', 'value': sha256_file(args.stage2b_controls)}, {'item': 'top_k', 'value': args.top_k}, {'item': 'top_k_cross_archive', 'value': args.top_k_cross_archive}, {'item': 'top_k_cross_split', 'value': args.top_k_cross_split}, {'item': 'top_k_cross_label', 'value': args.top_k_cross_label}, {'item': 'max_candidates', 'value': args.max_candidates}, {'item': 'elapsed_seconds', 'value': time.time() - started}])
    write_tsv(provenance, args.output_dir / 'stage2c_run_provenance.tsv')
    (args.output_dir / '.stage2c_complete').write_text('OK\n', encoding='utf-8')
    print(candidate_summary.to_string(index=False), flush=True)
    print(calibration_metrics.to_string(index=False), flush=True)
    print(retrieval_qc.to_string(index=False), flush=True)
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
