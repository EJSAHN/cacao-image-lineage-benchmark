"""perceptual-candidate discovery aggregation: perceptual candidates and metadata-only controls.

Perceptual pairs are candidate edges only. They are not promoted to biological-photo
lineage until geometric verification in geometric verification.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class BKNode:
    __slots__ = ('value', 'children')

    def __init__(self, value: int) -> None:
        self.value = value
        self.children: dict[int, 'BKNode'] = {}

class BKTree:

    def __init__(self) -> None:
        self.root: BKNode | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value)
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value)
                return
            node = child

    def search(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        results: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                results.append(node.value)
            low = distance - radius
            high = distance + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return results

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False, na_rep='')

def parse_hashes(value: object) -> tuple[int, ...]:
    text = str(value) if value is not None else ''
    values: list[int] = []
    for token in text.split('|'):
        token = token.strip()
        if token:
            values.append(int(token, 16))
    return tuple(sorted(set(values)))

def min_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if not a or not b:
        return 999
    return min(((x ^ y).bit_count() for x in a for y in b))

def set_join(values: Iterable[object]) -> str:
    return '|'.join(sorted({str(x) for x in values if str(x)}))

def build_candidates(hashes: pd.DataFrame, canonical: pd.DataFrame, radius: int, max_pairs: int) -> pd.DataFrame:
    representative_rows = canonical[canonical['exact_representative'].eq('YES')].set_index('exact_component_id')[['relative_path', 'absolute_path']].rename(columns={'relative_path': 'representative_path', 'absolute_path': 'representative_absolute_path'})
    component_meta = canonical.groupby('exact_component_id').agg(archives=('source_archive', set_join), archive_count=('source_archive', 'nunique'), labels=('corrected_label', set_join), label_count=('corrected_label', 'nunique'), splits=('corrected_split', lambda x: set_join((v for v in x if v not in {'', 'unspecified', 'UNMATCHED'}))), split_count=('corrected_split', lambda x: len({v for v in x if v not in {'', 'unspecified', 'UNMATCHED'}})), origin_keys=('source_origin_key', set_join)).join(representative_rows).reset_index()
    meta = component_meta.set_index('exact_component_id').to_dict('index')
    records: dict[str, dict[str, Any]] = {}
    value_to_ids: defaultdict[int, set[str]] = defaultdict(set)
    for row in hashes.to_dict('records'):
        component_id = row['exact_component_id']
        phash = parse_hashes(row.get('phash_variants', ''))
        dhash = parse_hashes(row.get('dhash_variants', ''))
        whash = parse_hashes(row.get('whash_variants', ''))
        records[component_id] = {**row, 'phash_values': phash, 'dhash_values': dhash, 'whash_values': whash}
        for value in phash:
            value_to_ids[value].add(component_id)
    unique_values = list(value_to_ids)
    random.Random(20260729).shuffle(unique_values)
    tree = BKTree()
    for value in unique_values:
        tree.add(value)
    candidate_pairs: set[tuple[str, str]] = set()
    component_ids = sorted(records)
    for index, component_id in enumerate(component_ids, 1):
        for query in records[component_id]['phash_values']:
            for neighbor_value in tree.search(query, radius):
                for other_id in value_to_ids[neighbor_value]:
                    if other_id == component_id:
                        continue
                    pair = tuple(sorted((component_id, other_id)))
                    candidate_pairs.add(pair)
                    if len(candidate_pairs) > max_pairs:
                        raise RuntimeError(f'Perceptual candidate guard exceeded: {len(candidate_pairs):,} > {max_pairs:,}')
        if index % 1000 == 0 or index == len(component_ids):
            print(f'phash_candidate_progress={index}/{len(component_ids)} pairs={len(candidate_pairs)}')
    origin_to_components: defaultdict[str, set[str]] = defaultdict(set)
    for row in canonical[canonical['exact_representative'].eq('YES')].itertuples(index=False):
        origin_to_components[row.source_origin_key].add(row.exact_component_id)
    origin_pair_set: set[tuple[str, str]] = set()
    for origin, ids in origin_to_components.items():
        if len(ids) < 2:
            continue
        if len(ids) > 100:
            continue
        for a, b in combinations(sorted(ids), 2):
            origin_pair_set.add((a, b))
            candidate_pairs.add((a, b))
    rows: list[dict[str, Any]] = []
    for a, b in sorted(candidate_pairs):
        rec_a = records.get(a)
        rec_b = records.get(b)
        if rec_a is None or rec_b is None:
            continue
        ph = min_distance(rec_a['phash_values'], rec_b['phash_values'])
        dh = min_distance(rec_a['dhash_values'], rec_b['dhash_values'])
        wh = min_distance(rec_a['whash_values'], rec_b['whash_values'])
        origins_a = set(meta[a]['origin_keys'].split('|'))
        origins_b = set(meta[b]['origin_keys'].split('|'))
        shared_origins = sorted((origins_a & origins_b) - {''})
        same_origin = bool(shared_origins)
        reasons = []
        if ph <= radius:
            reasons.append(f'phash_le_{radius}')
        if same_origin:
            reasons.append('shared_source_origin_key')
        if ph <= 2 and min(dh, wh) <= 6:
            strength = 'HIGH_CANDIDATE'
        elif ph <= radius and min(dh, wh) <= 8:
            strength = 'MEDIUM_CANDIDATE'
        elif same_origin:
            strength = 'ORIGIN_ONLY_CANDIDATE'
        else:
            strength = 'PHASH_ONLY_CANDIDATE'
        archives = sorted(set(meta[a]['archives'].split('|')) | set(meta[b]['archives'].split('|')))
        labels = sorted(set(meta[a]['labels'].split('|')) | set(meta[b]['labels'].split('|')))
        splits = sorted((set(meta[a]['splits'].split('|')) | set(meta[b]['splits'].split('|'))) - {''})
        rows.append({'component_a': a, 'component_b': b, 'candidate_strength': strength, 'candidate_reasons': '|'.join(reasons), 'min_phash_hamming': ph, 'min_dhash_hamming': dh, 'min_whash_hamming': wh, 'shared_origin_keys': '|'.join(shared_origins), 'cross_archive': 'YES' if len(archives) > 1 else 'NO', 'cross_split': 'YES' if len(splits) > 1 else 'NO', 'label_conflict': 'YES' if len(labels) > 1 else 'NO', 'archives': '|'.join(archives), 'labels': '|'.join(labels), 'splits': '|'.join(splits), 'path_a': meta[a]['representative_path'], 'path_b': meta[b]['representative_path'], 'absolute_path_a': meta[a]['representative_absolute_path'], 'absolute_path_b': meta[b]['representative_absolute_path'], 'width_a': rec_a.get('width', ''), 'height_a': rec_a.get('height', ''), 'width_b': rec_b.get('width', ''), 'height_b': rec_b.get('height', '')})
    return pd.DataFrame(rows)

def metadata_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = df.copy()
    for col in ['width', 'height', 'bytes']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['aspect_ratio'] = data['width'] / data['height']
    data['megapixels'] = data['width'] * data['height'] / 1000000.0
    data['log_bytes'] = np.log1p(data['bytes'])
    numeric = ['width', 'height', 'aspect_ratio', 'megapixels', 'log_bytes']
    categorical = ['image_format', 'exif_make', 'exif_model']
    return (data, numeric, categorical)

def run_grouped_model(data: pd.DataFrame, target_col: str, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, numeric, categorical = metadata_features(data)
    y = data[target_col].astype(str)
    groups = data['exact_component_id'].astype(str)
    min_class = y.value_counts().min()
    n_splits = min(5, int(min_class))
    if n_splits < 2:
        raise RuntimeError(f'Not enough samples for grouped CV task {task_name}')
    preprocess = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
    model = Pipeline([('preprocess', preprocess), ('classifier', LogisticRegression(max_iter=3000, class_weight='balanced', solver='lbfgs'))])
    predictions = pd.Series(index=data.index, dtype=object)
    folds = pd.Series(index=data.index, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260729)
    for fold, (train_idx, test_idx) in enumerate(cv.split(data, y, groups), 1):
        fitted = clone(model)
        fitted.fit(data.iloc[train_idx], y.iloc[train_idx])
        predictions.iloc[test_idx] = fitted.predict(data.iloc[test_idx])
        folds.iloc[test_idx] = fold
    labels = sorted(y.unique())
    cm = confusion_matrix(y, predictions, labels=labels)
    metrics = pd.DataFrame([{'task': task_name, 'n_photo_paths': len(data), 'n_exact_components': groups.nunique(), 'n_classes': len(labels), 'n_folds': n_splits, 'accuracy': accuracy_score(y, predictions), 'balanced_accuracy': balanced_accuracy_score(y, predictions), 'macro_f1': f1_score(y, predictions, average='macro'), 'chance_balanced_accuracy': 1.0 / len(labels), 'features': 'width|height|aspect_ratio|megapixels|log_bytes|image_format|EXIF_make|EXIF_model', 'grouping': 'exact_component_id'}])
    prediction_rows = pd.DataFrame({'task': task_name, 'photo_path_id': data['photo_path_id'].values, 'exact_component_id': data['exact_component_id'].values, 'true_label': y.values, 'predicted_label': predictions.values, 'fold': folds.values})
    confusion_rows: list[dict[str, Any]] = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            confusion_rows.append({'task': task_name, 'true_label': true_label, 'predicted_label': predicted_label, 'count': int(cm[i, j])})
    return (metrics, prediction_rows, pd.DataFrame(confusion_rows))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical-dir', required=True, type=Path)
    parser.add_argument('--hash-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--phash-radius', type=int, default=4)
    parser.add_argument('--max-candidate-pairs', type=int, default=5000000)
    args = parser.parse_args()
    canonical_path = args.canonical_dir / 'stage1_1_canonical_photo_manifest.tsv'
    ready = args.canonical_dir / '.stage2a_canonical_ready'
    if not canonical_path.is_file() or not ready.is_file():
        raise RuntimeError('manifest reconstruction.1 canonical outputs are not ready')
    canonical = pd.read_csv(canonical_path, sep='\t', low_memory=False, keep_default_na=False)
    statuses = []
    parts = []
    for status_path in sorted(args.hash_dir.glob('task_*_status.tsv')):
        statuses.append(pd.read_csv(status_path, sep='\t', keep_default_na=False))
    if statuses:
        status = pd.concat(statuses, ignore_index=True)
    else:
        raise RuntimeError('No hash status files found')
    if len(status) != 6 or not status['status'].eq('COMPLETED').all():
        raise RuntimeError(f'Hash tasks incomplete:\n{status.to_string(index=False)}')
    for part_path in sorted(args.hash_dir.glob('task_*_hashes.tsv')):
        parts.append(pd.read_csv(part_path, sep='\t', low_memory=False, keep_default_na=False))
    hashes = pd.concat(parts, ignore_index=True)
    expected = canonical[canonical['exact_representative'].eq('YES') & canonical['image_read_ok'].eq('YES') & canonical['exact_component_id'].ne('NO_HASH')]['exact_component_id'].nunique()
    if len(hashes) != expected or hashes['exact_component_id'].nunique() != expected:
        raise RuntimeError(f"Hash row mismatch: expected={expected} rows={len(hashes)} unique={hashes['exact_component_id'].nunique()}")
    if not hashes['hash_status'].eq('OK').all():
        raise RuntimeError('One or more representative hashes failed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(status, args.output_dir / 'stage2a_hash_task_status.tsv')
    write_tsv(hashes, args.output_dir / 'stage2a_perceptual_hashes.tsv')
    candidates = build_candidates(hashes=hashes, canonical=canonical, radius=args.phash_radius, max_pairs=args.max_candidate_pairs)
    write_tsv(candidates, args.output_dir / 'stage2a_perceptual_candidate_pairs.tsv')
    if len(candidates):
        strength_rank = {'HIGH_CANDIDATE': 0, 'MEDIUM_CANDIDATE': 1, 'ORIGIN_ONLY_CANDIDATE': 2, 'PHASH_ONLY_CANDIDATE': 3}
        priority = candidates.copy()
        priority['_priority_risk'] = priority['cross_split'].eq('YES').astype(int) * 8 + priority['label_conflict'].eq('YES').astype(int) * 4 + priority['cross_archive'].eq('YES').astype(int) * 2 + priority['shared_origin_keys'].ne('').astype(int)
        priority['_strength_rank'] = priority['candidate_strength'].map(strength_rank).fillna(9)
        priority = priority.sort_values(['_priority_risk', '_strength_rank', 'min_phash_hamming', 'min_dhash_hamming', 'min_whash_hamming', 'component_a', 'component_b'], ascending=[False, True, True, True, True, True, True]).drop(columns=['_priority_risk', '_strength_rank'])
        write_tsv(priority.head(50000), args.output_dir / 'stage2a_perceptual_candidate_pairs_priority_top50000.tsv')
        candidate_summary = candidates.groupby(['candidate_strength', 'cross_archive', 'cross_split', 'label_conflict'], as_index=False).size().rename(columns={'size': 'pair_count'})
    else:
        write_tsv(pd.DataFrame(columns=candidates.columns), args.output_dir / 'stage2a_perceptual_candidate_pairs_priority_top50000.tsv')
        candidate_summary = pd.DataFrame(columns=['candidate_strength', 'cross_archive', 'cross_split', 'label_conflict', 'pair_count'])
    write_tsv(candidate_summary, args.output_dir / 'stage2a_perceptual_candidate_summary.tsv')
    model_metrics = []
    model_predictions = []
    model_confusions = []
    source_data = canonical[canonical['image_read_ok'].eq('YES')].copy()
    metrics, predictions, confusion = run_grouped_model(source_data, 'source_archive', 'source_archive_from_metadata')
    model_metrics.append(metrics)
    model_predictions.append(predictions)
    model_confusions.append(confusion)
    main3 = canonical[canonical['supervised_eligible'].eq('YES') & canonical['corrected_label'].isin(['healthy', 'black_pod', 'frosty_pod'])].copy()
    metrics, predictions, confusion = run_grouped_model(main3, 'corrected_label', 'three_class_pathology_from_metadata')
    model_metrics.append(metrics)
    model_predictions.append(predictions)
    model_confusions.append(confusion)
    cm_stage = canonical[canonical['source_dataset_id'].eq('zen_cocoamonilia') & canonical['supervised_eligible'].eq('YES') & canonical['corrected_label'].isin(['healthy', 'frosty_pod_stage_m1', 'frosty_pod_stage_m2', 'frosty_pod_stage_m3'])].copy()
    metrics, predictions, confusion = run_grouped_model(cm_stage, 'corrected_label', 'cocoamonilia_stage_from_metadata')
    model_metrics.append(metrics)
    model_predictions.append(predictions)
    model_confusions.append(confusion)
    metrics_df = pd.concat(model_metrics, ignore_index=True)
    predictions_df = pd.concat(model_predictions, ignore_index=True)
    confusion_df = pd.concat(model_confusions, ignore_index=True)
    write_tsv(metrics_df, args.output_dir / 'stage2a_metadata_baseline_metrics.tsv')
    write_tsv(predictions_df, args.output_dir / 'stage2a_metadata_baseline_predictions.tsv')
    write_tsv(confusion_df, args.output_dir / 'stage2a_metadata_baseline_confusion.tsv')
    overlap = pd.read_csv(args.canonical_dir / 'stage1_1_pairwise_source_overlap.tsv', sep='\t', keep_default_na=False)
    exact_components = pd.read_csv(args.canonical_dir / 'stage1_1_exact_components.tsv', sep='\t', keep_default_na=False)
    summary_lines = ['# perceptual-candidate discovery perceptual-candidate and metadata-control summary', '', f'- Exact representatives hashed: {len(hashes):,}', f'- Perceptual/source-origin candidate pairs: {len(candidates):,}', f"- Cross-archive candidate pairs: {(int(candidates['cross_archive'].eq('YES').sum()) if len(candidates) else 0):,}", f"- Cross-split candidate pairs: {(int(candidates['cross_split'].eq('YES').sum()) if len(candidates) else 0):,}", f"- Candidate label conflicts: {(int(candidates['label_conflict'].eq('YES').sum()) if len(candidates) else 0):,}", '', '## Metadata-only grouped baselines', '']
    for row in metrics_df.itertuples(index=False):
        summary_lines.append(f'- {row.task}: balanced accuracy={row.balanced_accuracy:.3f}; macro-F1={row.macro_f1:.3f}; chance={row.chance_balanced_accuracy:.3f}; n={row.n_photo_paths:,}')
    summary_lines.extend(['', '## Strict claim boundary', '', 'Exact components are proven identical file payloads. perceptual-candidate discovery perceptual pairs are only', 'candidate transformed lineages. They must pass geometric verification ORB/RANSAC or equivalent geometric', 'verification before being counted as near-duplicate biological-photo families.', ''])
    (args.output_dir / 'stage2a_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    (args.output_dir / '.stage2a_aggregate_ready').write_text('OK\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
