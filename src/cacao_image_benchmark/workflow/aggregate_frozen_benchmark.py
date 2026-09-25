"""Aggregate frozen-feature benchmark model outputs, compute clustered uncertainty, and make figures."""
from __future__ import annotations
import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from frozen_benchmark_common import PRIMARY_CLASSIFIER, PRIMARY_FEATURE_SET, TASK_LABEL_ORDERS, ProjectPaths, aggregate_predictions_by_unit, config_seed, decision_matrix_from_prediction_frame, multiclass_metrics, per_class_metrics, probabilities_from_prediction_frame, read_tsv, sha256_file, write_tsv
CV_ORDER = ['PATH_RANDOM_5FOLD', 'EXACT_COMPONENT_5FOLD', 'STRICT_LINEAGE_5FOLD', 'VERIFIED_SCENE_BLOCK_5FOLD', 'AMBIGUITY_SENS_BLOCK_5FOLD']
CV_LABELS = {'PATH_RANDOM_5FOLD': 'Path random', 'EXACT_COMPONENT_5FOLD': 'Exact grouped', 'STRICT_LINEAGE_5FOLD': 'Lineage grouped', 'VERIFIED_SCENE_BLOCK_5FOLD': 'Scene blocked', 'AMBIGUITY_SENS_BLOCK_5FOLD': 'Ambiguity sensitivity'}
SOURCE_ORDER = ['fig_ghana_balanced', 'fig_roboflow_mixed', 'fig_spanish_yolov4']
SOURCE_LABELS = {'fig_ghana_balanced': 'Ghana', 'fig_roboflow_mixed': 'Roboflow', 'fig_spanish_yolov4': 'Spanish YOLOv4'}
SOURCE_DESIGN_ORDER = ['SOURCE_HOLDOUT_NAIVE', 'SOURCE_HOLDOUT_STRICT_SAFE', 'SOURCE_HOLDOUT_SCENE_SAFE']
SOURCE_DESIGN_LABELS = {'SOURCE_HOLDOUT_NAIVE': 'Archive held out', 'SOURCE_HOLDOUT_STRICT_SAFE': 'Lineage safe', 'SOURCE_HOLDOUT_SCENE_SAFE': 'Scene safe'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--big', required=True)
    parser.add_argument('--expected-configs', type=int, default=176)
    parser.add_argument('--bootstrap-replicates', type=int, default=500)
    return parser.parse_args()

def cm_metrics(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else float('nan')
    row_sum = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    balanced = float(np.mean(recalls))
    col_sum = cm.sum(axis=0)
    precision = np.divide(np.diag(cm), col_sum, out=np.zeros_like(col_sum), where=col_sum > 0)
    f1 = np.divide(2 * precision * recalls, precision + recalls, out=np.zeros_like(precision), where=precision + recalls > 0)
    return {'accuracy': accuracy, 'balanced_accuracy': balanced, 'macro_f1': float(np.mean(f1))}

def bootstrap_confusion_metrics(frame: pd.DataFrame, labels: list[str], cluster_column: str, replicates: int, seed: int) -> pd.DataFrame:
    if cluster_column not in frame.columns:
        raise RuntimeError(f'Bootstrap cluster column is missing: {cluster_column}')
    group_matrices: list[np.ndarray] = []
    for _, group in frame.groupby(cluster_column, sort=False):
        group_matrices.append(confusion_matrix(group['y_true'], group['y_pred'], labels=labels))
    stack = np.stack(group_matrices, axis=0).astype(np.int64)
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(replicates, dtype=float) for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']}
    n_groups = len(stack)
    for idx in range(replicates):
        sampled = rng.integers(0, n_groups, size=n_groups)
        metrics = cm_metrics(stack[sampled].sum(axis=0))
        for metric, value in metrics.items():
            values[metric][idx] = value
    rows = []
    for metric, array in values.items():
        rows.append({'metric': metric, 'bootstrap_replicates': replicates, 'cluster_column': cluster_column, 'n_clusters': n_groups, 'ci_low': float(np.quantile(array, 0.025)), 'ci_high': float(np.quantile(array, 0.975)), 'bootstrap_median': float(np.quantile(array, 0.5))})
    return pd.DataFrame(rows)

def metric_row(frame: pd.DataFrame, config: pd.Series, scope: str, labels: list[str]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    classifier = config['classifier']
    probabilities = probabilities_from_prediction_frame(frame, labels, classifier)
    metrics = multiclass_metrics(frame['y_true'], frame['y_pred'], labels, probabilities)
    row = {'config_id': config['config_id'], 'task_name': config['task_name'], 'design_family': config['design_family'], 'evaluation_design': config['evaluation_design'], 'feature_set': config['feature_set'], 'classifier': classifier, 'heldout_source': config.get('heldout_source', ''), 'metric_scope': scope, 'n_evaluation_rows': len(frame), **metrics}
    per_class = per_class_metrics(frame['y_true'], frame['y_pred'], labels)
    for key, value in row.items():
        per_class[key] = value
    cm = confusion_matrix(frame['y_true'], frame['y_pred'], labels=labels)
    cm_rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_rows.append({**{key: value for key, value in row.items() if key not in metrics}, 'true_label': true_label, 'predicted_label': pred_label, 'count': int(cm[i, j])})
    return (row, per_class, pd.DataFrame(cm_rows))

def bootstrap_cluster_for_config(config: pd.Series, scope: str) -> str:
    if scope == 'strict_lineage_balanced':
        return 'analysis_unit_id'
    design = config['evaluation_design']
    mapping = {'PATH_RANDOM_5FOLD': 'sample_id', 'EXACT_COMPONENT_5FOLD': 'exact_component_id', 'STRICT_LINEAGE_5FOLD': 'final_strict_lineage_id', 'VERIFIED_SCENE_BLOCK_5FOLD': 'final_split_block_id', 'AMBIGUITY_SENS_BLOCK_5FOLD': 'ambiguity_sensitive_block_id'}
    return mapping.get(design, 'final_split_block_id')

def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    started = time.time()
    configs = read_tsv(paths.prepared / 'stage3a_config_table.tsv')
    status_files = sorted(paths.task_results.glob('stage3a_S3A*_status.tsv'))
    if len(configs) != args.expected_configs:
        raise RuntimeError(f'Expected {args.expected_configs} configs, table has {len(configs)}')
    if len(status_files) != args.expected_configs:
        raise RuntimeError(f'Expected {args.expected_configs} status files, found {len(status_files)}')
    statuses = pd.concat([read_tsv(path) for path in status_files], ignore_index=True)
    if len(statuses) != args.expected_configs or not statuses['status'].eq('COMPLETED').all():
        failed = statuses[~statuses['status'].eq('COMPLETED')]
        raise RuntimeError(f'frozen-feature benchmark model tasks are incomplete or failed:\n{failed.to_string(index=False)}')
    metric_rows: list[dict[str, object]] = []
    per_class_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    primary_prediction_frames: list[pd.DataFrame] = []
    output_index_rows: list[dict[str, object]] = []
    config_lookup = configs.set_index('config_id', drop=False)
    for status in statuses.sort_values('config_id').itertuples(index=False):
        config = config_lookup.loc[status.config_id]
        prediction_path = Path(status.prediction_path)
        fold_path = Path(status.fold_metric_path)
        predictions = read_tsv(prediction_path)
        fold_metric_frames.append(read_tsv(fold_path))
        labels = TASK_LABEL_ORDERS[config['task_name']]
        path_row, path_per_class, path_cm = metric_row(predictions, config, 'physical_path_weighted', labels)
        metric_rows.append(path_row)
        per_class_frames.append(path_per_class)
        confusion_frames.append(path_cm)
        unit_predictions = aggregate_predictions_by_unit(predictions, labels, config['classifier'], unit_column='analysis_unit_id')
        unit_row, unit_per_class, unit_cm = metric_row(unit_predictions, config, 'strict_lineage_balanced', labels)
        metric_rows.append(unit_row)
        per_class_frames.append(unit_per_class)
        confusion_frames.append(unit_cm)
        if config['primary_model'] == 'YES':
            primary_prediction_frames.append(predictions)
            for scope, frame in [('physical_path_weighted', predictions), ('strict_lineage_balanced', unit_predictions)]:
                cluster = bootstrap_cluster_for_config(config, scope)
                ci = bootstrap_confusion_metrics(frame, labels, cluster, args.bootstrap_replicates, config_seed(config['config_id'], scope))
                for key in ['config_id', 'task_name', 'design_family', 'evaluation_design', 'feature_set', 'classifier', 'heldout_source']:
                    ci[key] = config.get(key, '')
                ci['metric_scope'] = scope
                bootstrap_frames.append(ci)
        output_index_rows.append({'config_id': config['config_id'], 'prediction_path': str(prediction_path), 'prediction_sha256': sha256_file(prediction_path), 'prediction_rows': len(predictions), 'fold_metric_path': str(fold_path), 'fold_metric_sha256': sha256_file(fold_path)})
    metrics = pd.DataFrame(metric_rows)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    confusion_long = pd.concat(confusion_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    bootstrap_ci = pd.concat(bootstrap_frames, ignore_index=True)
    primary_predictions = pd.concat(primary_prediction_frames, ignore_index=True)
    output_index = pd.DataFrame(output_index_rows)
    write_tsv(metrics, paths.aggregate / 'stage3a_aggregate_metrics.tsv')
    write_tsv(per_class, paths.aggregate / 'stage3a_per_class_metrics.tsv')
    write_tsv(confusion_long, paths.aggregate / 'stage3a_confusion_matrices_long.tsv')
    write_tsv(fold_metrics, paths.aggregate / 'stage3a_fold_metrics.tsv')
    write_tsv(bootstrap_ci, paths.aggregate / 'stage3a_primary_bootstrap_ci.tsv')
    write_tsv(primary_predictions, paths.aggregate / 'stage3a_primary_predictions.tsv.gz')
    write_tsv(output_index, paths.aggregate / 'stage3a_task_output_index.tsv')
    write_tsv(statuses, paths.aggregate / 'stage3a_model_task_status.tsv')
    primary = metrics[metrics['feature_set'].eq(PRIMARY_FEATURE_SET) & metrics['classifier'].eq(PRIMARY_CLASSIFIER) & metrics['metric_scope'].eq('physical_path_weighted')].copy()
    delta_rows: list[dict[str, object]] = []
    for task_name in ['cacao_coarse_three_class', 'cocoamonilia_four_stage']:
        subset = primary[primary['task_name'].eq(task_name) & primary['design_family'].eq('CV')]
        baseline = subset[subset['evaluation_design'].eq('PATH_RANDOM_5FOLD')]
        if len(baseline) == 1:
            baseline = baseline.iloc[0]
            for row in subset.itertuples(index=False):
                delta_rows.append({'comparison_family': 'CV_WATERFALL', 'task_name': task_name, 'heldout_source': '', 'baseline_design': 'PATH_RANDOM_5FOLD', 'comparison_design': row.evaluation_design, 'balanced_accuracy_delta': float(row.balanced_accuracy) - float(baseline.balanced_accuracy), 'macro_f1_delta': float(row.macro_f1) - float(baseline.macro_f1)})
    loso = primary[primary['design_family'].eq('SOURCE_HOLDOUT')]
    for source in SOURCE_ORDER:
        subset = loso[loso['heldout_source'].eq(source)]
        baseline = subset[subset['evaluation_design'].eq('SOURCE_HOLDOUT_NAIVE')]
        if len(baseline) == 1:
            baseline = baseline.iloc[0]
            for row in subset.itertuples(index=False):
                delta_rows.append({'comparison_family': 'SOURCE_HOLDOUT', 'task_name': row.task_name, 'heldout_source': source, 'baseline_design': 'SOURCE_HOLDOUT_NAIVE', 'comparison_design': row.evaluation_design, 'balanced_accuracy_delta': float(row.balanced_accuracy) - float(baseline.balanced_accuracy), 'macro_f1_delta': float(row.macro_f1) - float(baseline.macro_f1)})
    deltas = pd.DataFrame(delta_rows)
    write_tsv(deltas, paths.aggregate / 'stage3a_primary_performance_deltas.tsv')
    cv_audit = read_tsv(paths.prepared / 'stage3a_cv_split_integrity_audit.tsv')
    fixed_audit = read_tsv(paths.prepared / 'stage3a_original_split_integrity_audit.tsv')
    source_audit = read_tsv(paths.prepared / 'stage3a_source_holdout_integrity_audit.tsv')
    write_tsv(cv_audit, paths.aggregate / 'stage3a_cv_split_integrity_audit.tsv')
    write_tsv(fixed_audit, paths.aggregate / 'stage3a_original_split_integrity_audit.tsv')
    write_tsv(source_audit, paths.aggregate / 'stage3a_source_holdout_integrity_audit.tsv')
    split_integrity = cv_audit['intended_group_crossings'].astype(int).eq(0).all() and cv_audit['all_labels_present_in_every_test_fold'].eq('YES').all() and source_audit[source_audit['evaluation_design'].eq('SOURCE_HOLDOUT_STRICT_SAFE')]['strict_lineage_overlap'].astype(int).eq(0).all() and source_audit[source_audit['evaluation_design'].eq('SOURCE_HOLDOUT_SCENE_SAFE')]['verified_scene_block_overlap'].astype(int).eq(0).all()
    claim = pd.DataFrame([{'all_model_configs_completed': 'YES', 'expected_model_configs': args.expected_configs, 'observed_model_configs': len(statuses), 'deterministic_split_integrity_pass': 'YES' if split_integrity else 'NO', 'metadata_models_are_negative_controls': 'YES', 'frozen_embeddings_only_no_finetuning': 'YES', 'primary_model_prespecified': f'{PRIMARY_FEATURE_SET}|{PRIMARY_CLASSIFIER}', 'ambiguous_edges_used_only_for_split_sensitivity': 'YES', 'causal_five_class_loso_performed': 'NO', 'source_holdout_test_paths_are_heldout_source_only': 'YES', 'original_public_splits_claim_scene_independence': 'NO', 'claim_boundary': 'frozen-feature benchmark quantifies performance under pre-specified physical-path, exact-component, strict-lineage, verified-scene, and ambiguity-sensitive partitions using frozen ImageNet features and linear classifiers. Metadata is a confounding control. No fine-tuning or field generalization claim is made. Source-held-out results are reported separately from grouped within-pool evaluation, and the five-class panel is not used for unrestricted LOSO.'}])
    write_tsv(claim, paths.aggregate / 'stage3a_claim_status.tsv')
    lines = ['# frozen-feature benchmark leakage-aware frozen-feature benchmark summary', '', f'- Model configurations completed: {len(statuses)} / {args.expected_configs}', f'- Primary model: {PRIMARY_FEATURE_SET} + {PRIMARY_CLASSIFIER}', f'- Bootstrap confidence intervals: {args.bootstrap_replicates} cluster resamples for primary configurations', f"- Split integrity pass: {('YES' if split_integrity else 'NO')}", '', '## Coarse three-class primary waterfall']
    coarse_primary = primary[primary['task_name'].eq('cacao_coarse_three_class') & primary['design_family'].eq('CV')].set_index('evaluation_design')
    for design in CV_ORDER:
        if design in coarse_primary.index:
            row = coarse_primary.loc[design]
            lines.append(f"- {design}: balanced accuracy={float(row['balanced_accuracy']):.4f}; macro-F1={float(row['macro_f1']):.4f}")
    lines.extend(['', '## CocoaMonilia primary results'])
    stage_primary = primary[primary['task_name'].eq('cocoamonilia_four_stage')].set_index('evaluation_design')
    for design in ['ORIGINAL_TRAIN_TO_VALIDATION', 'ORIGINAL_TRAINVAL_TO_TEST', *CV_ORDER]:
        if design in stage_primary.index:
            row = stage_primary.loc[design]
            lines.append(f"- {design}: balanced accuracy={float(row['balanced_accuracy']):.4f}; macro-F1={float(row['macro_f1']):.4f}")
    lines.extend(['', '## Coarse three-class source holdout'])
    for source in SOURCE_ORDER:
        lines.append(f'- {source}:')
        part = loso[loso['heldout_source'].eq(source)].set_index('evaluation_design')
        for design in SOURCE_DESIGN_ORDER:
            if design in part.index:
                row = part.loc[design]
                lines.append(f"  - {design}: balanced accuracy={float(row['balanced_accuracy']):.4f}; macro-F1={float(row['macro_f1']):.4f}")
    lines.extend(['', '## Claim boundary', claim.iloc[0]['claim_boundary']])
    summary_path = paths.aggregate / 'stage3a_summary.md'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (paths.aggregate / '.stage3a_complete').touch()
    print('\n'.join(lines))
    print(f'\nAggregation elapsed seconds: {time.time() - started:.1f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
