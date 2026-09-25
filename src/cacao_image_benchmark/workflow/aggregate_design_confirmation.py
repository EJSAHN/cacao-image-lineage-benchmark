"""Aggregate design confirmation repeated-split and fixed-test design-confirmation results."""
from __future__ import annotations
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from design_confirmation_common import COCO_TEST_DESIGNS, CV_DESIGNS, SOURCE_HOLDOUTS, SOURCE_HOLDOUT_DESIGNS, TASK_EXPECTED_UNITS, TASK_LABEL_ORDERS, ProjectPaths, aggregate_predictions_by_unit, config_seed, multiclass_metrics, per_class_metrics, read_tsv, reliability_table, sha256_file, write_tsv
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'log_loss', 'ece_15bin', 'multiclass_brier']
CV_ORDER = list(CV_DESIGNS)
CV_LABELS = {'PATH_RANDOM_5FOLD': 'Path random', 'EXACT_COMPONENT_5FOLD': 'Exact grouped', 'STRICT_LINEAGE_5FOLD': 'Lineage grouped', 'VERIFIED_SCENE_BLOCK_5FOLD': 'Scene blocked', 'AMBIGUITY_SENS_BLOCK_5FOLD': 'Ambiguity sensitivity'}
COCO_LABELS = {'COCO_TEST_NAIVE': 'Public training pool', 'COCO_TEST_EXACT_SAFE': 'Exact-safe training', 'COCO_TEST_STRICT_SAFE': 'Lineage-safe training', 'COCO_TEST_SCENE_SAFE': 'Scene-safe training', 'COCO_TEST_AMBIGUITY_SAFE': 'Ambiguity-safe training'}
SOURCE_LABELS = {'fig_ghana_balanced': 'Ghana', 'fig_roboflow_mixed': 'Roboflow', 'fig_spanish_yolov4': 'Spanish YOLOv4'}
SOURCE_DESIGN_LABELS = {'SOURCE_HOLDOUT_NAIVE': 'Archive only', 'SOURCE_HOLDOUT_EXACT_SAFE': 'Exact safe', 'SOURCE_HOLDOUT_STRICT_SAFE': 'Lineage safe', 'SOURCE_HOLDOUT_SCENE_SAFE': 'Scene safe', 'SOURCE_HOLDOUT_AMBIGUITY_SAFE': 'Ambiguity safe'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--big', required=True)
    parser.add_argument('--expected-configs', type=int, default=656)
    parser.add_argument('--seed-bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--paired-cluster-bootstrap-replicates', type=int, default=2000)
    return parser.parse_args()

def probabilities(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return frame[[f'prob__{label}' for label in labels]].astype(float).to_numpy()

def metric_bundle(frame: pd.DataFrame, config: pd.Series, scope: str, labels: list[str]):
    probs = probabilities(frame, labels)
    metrics = multiclass_metrics(frame['y_true'], frame['y_pred'], labels, probs)
    row = {'config_id': config['config_id'], 'analysis_family': config['analysis_family'], 'task_name': config['task_name'], 'evaluation_design': config['evaluation_design'], 'feature_set': config['feature_set'], 'training_mode': config['training_mode'], 'repeat_index': config['repeat_index'], 'split_seed': config['split_seed'], 'heldout_source': config['heldout_source'], 'metric_scope': scope, 'n_evaluation_rows': len(frame), **metrics}
    pc = per_class_metrics(frame['y_true'], frame['y_pred'], labels)
    for key, value in row.items():
        pc[key] = value
    cm = confusion_matrix(frame['y_true'], frame['y_pred'], labels=labels)
    cm_rows = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            cm_rows.append({**{k: v for k, v in row.items() if k not in METRICS and k not in {'n'}}, 'true_label': true_label, 'predicted_label': predicted_label, 'count': int(cm[i, j])})
    return (row, pc, pd.DataFrame(cm_rows))

def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    repeated = metrics[metrics['analysis_family'].eq('REPEATED_CV')].copy()
    keys = ['task_name', 'evaluation_design', 'feature_set', 'training_mode', 'metric_scope']
    rows = []
    for key, group in repeated.groupby(keys, sort=False):
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors='coerce').dropna().to_numpy(float)
            rows.append({**base, 'metric': metric, 'n_seeds': len(values), 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 'median': float(np.median(values)), 'p025': float(np.quantile(values, 0.025)), 'p975': float(np.quantile(values, 0.975)), 'min': float(np.min(values)), 'max': float(np.max(values))})
    return pd.DataFrame(rows)

def bootstrap_mean_ci(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        means[idx] = values[rng.integers(0, n, size=n)].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

def paired_seed_deltas(metrics: pd.DataFrame, replicates: int) -> pd.DataFrame:
    repeated = metrics[metrics['analysis_family'].eq('REPEATED_CV') & metrics['metric_scope'].isin(['physical_path_weighted', 'strict_lineage_balanced'])].copy()
    rows = []

    def compare(group_filter: dict[str, str], baseline_design: str, comparison_design: str, baseline_mode: str, comparison_mode: str, family: str):
        subset = repeated.copy()
        for column, value in group_filter.items():
            subset = subset[subset[column].eq(value)]
        baseline = subset[subset['evaluation_design'].eq(baseline_design) & subset['training_mode'].eq(baseline_mode)]
        comparison = subset[subset['evaluation_design'].eq(comparison_design) & subset['training_mode'].eq(comparison_mode)]
        merge_keys = ['split_seed']
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss', 'multiclass_brier']:
            merged = baseline[merge_keys + [metric]].merge(comparison[merge_keys + [metric]], on=merge_keys, suffixes=('_baseline', '_comparison'))
            if merged.empty:
                continue
            delta = pd.to_numeric(merged[f'{metric}_comparison'], errors='coerce').to_numpy(float) - pd.to_numeric(merged[f'{metric}_baseline'], errors='coerce').to_numpy(float)
            low, high = bootstrap_mean_ci(delta, replicates, config_seed(f'{family}|{group_filter}|{baseline_design}|{comparison_design}|{baseline_mode}|{comparison_mode}|{metric}'))
            rows.append({'comparison_family': family, **group_filter, 'baseline_design': baseline_design, 'comparison_design': comparison_design, 'baseline_training_mode': baseline_mode, 'comparison_training_mode': comparison_mode, 'metric': metric, 'paired_seeds': len(delta), 'mean_delta': float(delta.mean()), 'sd_delta': float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0, 'median_delta': float(np.median(delta)), 'ci_low_mean_delta': low, 'ci_high_mean_delta': high, 'fraction_delta_below_zero': float(np.mean(delta < 0))})
    for task_name in ['cacao_coarse_three_class', 'cocoamonilia_four_stage']:
        for feature in ['efficientnet_b0', 'concat']:
            for scope in ['physical_path_weighted', 'strict_lineage_balanced']:
                filt = {'task_name': task_name, 'feature_set': feature, 'metric_scope': scope}
                for comparison in CV_ORDER[1:]:
                    compare(filt, 'PATH_RANDOM_5FOLD', comparison, 'ALL_PATHS', 'ALL_PATHS', 'WATERFALL_VS_PATH_RANDOM')
                for baseline, comparison in [('EXACT_COMPONENT_5FOLD', 'STRICT_LINEAGE_5FOLD'), ('STRICT_LINEAGE_5FOLD', 'VERIFIED_SCENE_BLOCK_5FOLD'), ('VERIFIED_SCENE_BLOCK_5FOLD', 'AMBIGUITY_SENS_BLOCK_5FOLD')]:
                    compare(filt, baseline, comparison, 'ALL_PATHS', 'ALL_PATHS', 'ADJACENT_INDEPENDENCE_STEP')
    for task_name in ['cacao_coarse_three_class', 'cocoamonilia_four_stage', 'cacao_causal_five_class']:
        for design in ['VERIFIED_SCENE_BLOCK_5FOLD', 'AMBIGUITY_SENS_BLOCK_5FOLD']:
            for scope in ['physical_path_weighted', 'strict_lineage_balanced']:
                filt = {'task_name': task_name, 'feature_set': 'efficientnet_b0', 'metric_scope': scope}
                for mode in ['LINEAGE_WEIGHTED', 'LINEAGE_REPRESENTATIVE']:
                    compare(filt, design, design, 'ALL_PATHS', mode, 'TRAINING_DUPLICATION_SENSITIVITY')
    return pd.DataFrame(rows)

def cm_metrics(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, float)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    recall = np.divide(np.diag(cm), row, out=np.zeros_like(row), where=row > 0)
    precision = np.divide(np.diag(cm), col, out=np.zeros_like(col), where=col > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=precision + recall > 0)
    total = cm.sum()
    return {'accuracy': float(np.trace(cm) / total) if total else float('nan'), 'balanced_accuracy': float(np.mean(recall)), 'macro_f1': float(np.mean(f1))}

def cluster_contributions(frame: pd.DataFrame, labels: list[str], cluster_column: str):
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    rows = []
    for cluster, group in frame.groupby(cluster_column, sort=False):
        probs = probabilities(group, labels)
        y = group['y_true'].to_numpy(object)
        pred = group['y_pred'].to_numpy(object)
        target_idx = np.asarray([label_to_idx[str(v)] for v in y], int)
        target = np.zeros_like(probs)
        target[np.arange(len(y)), target_idx] = 1.0
        rows.append({'cluster': cluster, 'cm': confusion_matrix(y, pred, labels=labels), 'n': len(y), 'log_loss_sum': float(-np.log(np.clip(probs[np.arange(len(y)), target_idx], 1e-15, 1)).sum()), 'brier_sum': float(np.sum((probs - target) ** 2))})
    return rows

def paired_cluster_bootstrap(baseline: pd.DataFrame, comparison: pd.DataFrame, labels: list[str], cluster_column: str, replicates: int, seed: int) -> pd.DataFrame:
    if set(baseline['sample_id']) != set(comparison['sample_id']):
        raise RuntimeError('Paired fixed-test/source comparison has different test sample IDs')
    baseline = baseline.sort_values('sample_id').reset_index(drop=True)
    comparison = comparison.sort_values('sample_id').reset_index(drop=True)
    if not baseline['sample_id'].equals(comparison['sample_id']):
        raise RuntimeError('Paired comparison sample ordering mismatch')
    if not baseline['y_true'].equals(comparison['y_true']):
        raise RuntimeError('Paired comparison true labels mismatch')
    clusters = baseline[['sample_id', cluster_column]].set_index('sample_id')[cluster_column]
    comparison[cluster_column] = comparison['sample_id'].map(clusters)
    base_parts = cluster_contributions(baseline, labels, cluster_column)
    comp_parts = cluster_contributions(comparison, labels, cluster_column)
    base_map = {x['cluster']: x for x in base_parts}
    comp_map = {x['cluster']: x for x in comp_parts}
    cluster_ids = sorted(base_map)
    if set(cluster_ids) != set(comp_map):
        raise RuntimeError('Paired comparison cluster sets mismatch')
    rng = np.random.default_rng(seed)
    values = {m: np.empty(replicates, float) for m in ['accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss', 'multiclass_brier']}
    for rep in range(replicates):
        sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        b_cm = np.zeros((len(labels), len(labels)), float)
        c_cm = np.zeros_like(b_cm)
        b_loss = c_loss = b_brier = c_brier = 0.0
        n = 0
        for idx in sampled:
            cid = cluster_ids[idx]
            b = base_map[cid]
            c = comp_map[cid]
            b_cm += b['cm']
            c_cm += c['cm']
            b_loss += b['log_loss_sum']
            c_loss += c['log_loss_sum']
            b_brier += b['brier_sum']
            c_brier += c['brier_sum']
            n += b['n']
        bm = cm_metrics(b_cm)
        cmv = cm_metrics(c_cm)
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
            values[metric][rep] = cmv[metric] - bm[metric]
        values['log_loss'][rep] = c_loss / n - b_loss / n
        values['multiclass_brier'][rep] = c_brier / n - b_brier / n
    rows = []
    for metric, arr in values.items():
        rows.append({'metric': metric, 'bootstrap_replicates': replicates, 'cluster_column': cluster_column, 'n_clusters': len(cluster_ids), 'mean_delta': float(arr.mean()), 'ci_low': float(np.quantile(arr, 0.025)), 'ci_high': float(np.quantile(arr, 0.975)), 'fraction_delta_below_zero': float(np.mean(arr < 0))})
    return pd.DataFrame(rows)

def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    started = time.time()
    configs = read_tsv(paths.prepared / 'stage3b_config_table.tsv')
    status_files = sorted(paths.task_results.glob('stage3b_S3B*_status.tsv'))
    if len(configs) != args.expected_configs or len(status_files) != args.expected_configs:
        raise RuntimeError(f'Expected {args.expected_configs} configs/statuses; got {len(configs)}/{len(status_files)}')
    statuses = pd.concat([read_tsv(path) for path in status_files], ignore_index=True)
    if len(statuses) != args.expected_configs or not statuses['status'].eq('COMPLETED').all():
        raise RuntimeError('One or more design confirmation model tasks failed')
    config_lookup = configs.set_index('config_id', drop=False)
    metric_rows = []
    per_class_frames = []
    confusion_frames = []
    fold_frames = []
    training_frames = []
    output_index_rows = []
    fixed_primary: dict[str, pd.DataFrame] = {}
    source_primary: dict[tuple[str, str], pd.DataFrame] = {}
    selected_prediction_frames = []
    for status in statuses.sort_values('config_id').itertuples(index=False):
        config = config_lookup.loc[status.config_id]
        pred_path = Path(status.prediction_path)
        fold_path = Path(status.fold_metric_path)
        train_path = Path(status.training_audit_path)
        pred = read_tsv(pred_path)
        labels = TASK_LABEL_ORDERS[config['task_name']]
        row, pc, cm = metric_bundle(pred, config, 'physical_path_weighted', labels)
        metric_rows.append(row)
        per_class_frames.append(pc)
        confusion_frames.append(cm)
        units = aggregate_predictions_by_unit(pred, labels, 'analysis_unit_id')
        row, pc, cm = metric_bundle(units, config, 'strict_lineage_balanced', labels)
        metric_rows.append(row)
        per_class_frames.append(pc)
        confusion_frames.append(cm)
        fold_frames.append(read_tsv(fold_path))
        training_frames.append(read_tsv(train_path))
        output_index_rows.append({'config_id': config['config_id'], 'prediction_path': str(pred_path), 'prediction_sha256': sha256_file(pred_path), 'prediction_rows': len(pred), 'fold_metric_path': str(fold_path), 'fold_metric_sha256': sha256_file(fold_path), 'training_audit_path': str(train_path), 'training_audit_sha256': sha256_file(train_path)})
        if config['analysis_family'] == 'COCO_FIXED_TEST' and config['feature_set'] == 'efficientnet_b0':
            if config['training_mode'] == 'ALL_PATHS':
                fixed_primary[config['evaluation_design']] = pred
                selected_prediction_frames.append(pred)
            elif config['evaluation_design'] in {'COCO_TEST_SCENE_SAFE', 'COCO_TEST_AMBIGUITY_SAFE'}:
                selected_prediction_frames.append(pred)
        if config['analysis_family'] == 'SOURCE_HOLDOUT' and config['feature_set'] == 'efficientnet_b0' and (config['training_mode'] == 'ALL_PATHS'):
            source_primary[config['heldout_source'], config['evaluation_design']] = pred
            selected_prediction_frames.append(pred)
    metrics = pd.DataFrame(metric_rows)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    confusion_long = pd.concat(confusion_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    training_audit = pd.concat(training_frames, ignore_index=True)
    output_index = pd.DataFrame(output_index_rows)
    selected_predictions = pd.concat(selected_prediction_frames, ignore_index=True)
    repeated_summary = seed_summary(metrics)
    paired_deltas = paired_seed_deltas(metrics, args.seed_bootstrap_replicates)
    fixed_bootstrap_frames = []
    fixed_reliability_frames = []
    baseline = fixed_primary['COCO_TEST_NAIVE']
    labels = TASK_LABEL_ORDERS['cocoamonilia_four_stage']
    for design in COCO_TEST_DESIGNS:
        frame = fixed_primary[design]
        rel = reliability_table(frame['y_true'], probabilities(frame, labels), labels, n_bins=10)
        rel['evaluation_design'] = design
        rel['feature_set'] = 'efficientnet_b0'
        rel['training_mode'] = 'ALL_PATHS'
        fixed_reliability_frames.append(rel)
        if design != 'COCO_TEST_NAIVE':
            ci = paired_cluster_bootstrap(baseline, frame, labels, 'ambiguity_sensitive_block_id', args.paired_cluster_bootstrap_replicates, config_seed(f'COCO_FIXED|{design}'))
            ci['analysis_family'] = 'COCO_FIXED_TEST'
            ci['heldout_source'] = ''
            ci['baseline_design'] = 'COCO_TEST_NAIVE'
            ci['comparison_design'] = design
            fixed_bootstrap_frames.append(ci)
    fixed_reliability = pd.concat(fixed_reliability_frames, ignore_index=True)
    fixed_bootstrap = pd.concat(fixed_bootstrap_frames, ignore_index=True)
    source_bootstrap_frames = []
    coarse_labels = TASK_LABEL_ORDERS['cacao_coarse_three_class']
    for source in SOURCE_HOLDOUTS:
        base = source_primary[source, 'SOURCE_HOLDOUT_NAIVE']
        for design in SOURCE_HOLDOUT_DESIGNS[1:]:
            comp = source_primary[source, design]
            ci = paired_cluster_bootstrap(base, comp, coarse_labels, 'ambiguity_sensitive_block_id', args.paired_cluster_bootstrap_replicates, config_seed(f'SOURCE|{source}|{design}'))
            ci['analysis_family'] = 'SOURCE_HOLDOUT'
            ci['heldout_source'] = source
            ci['baseline_design'] = 'SOURCE_HOLDOUT_NAIVE'
            ci['comparison_design'] = design
            source_bootstrap_frames.append(ci)
    source_bootstrap = pd.concat(source_bootstrap_frames, ignore_index=True)
    scope = read_tsv(paths.prepared / 'stage3b_task_scope_audit.tsv')
    five_scope = scope[scope['task_name'].eq('cacao_causal_five_class')]
    observed_units = int(five_scope['strict_analysis_units'].astype(int).sum())
    if observed_units != TASK_EXPECTED_UNITS['cacao_causal_five_class']:
        raise RuntimeError(f'Corrected five-class scope mismatch: {observed_units}')
    stage3a_metrics = read_tsv(paths.stage3a / 'stage3a_aggregate_metrics.tsv')
    old = stage3a_metrics[stage3a_metrics['task_name'].eq('cacao_causal_five_class') & stage3a_metrics['metric_scope'].eq('strict_lineage_balanced')]
    old_units = int(pd.to_numeric(old['n_evaluation_rows'], errors='coerce').dropna().max()) if not old.empty else 0
    five_scope_correction = pd.DataFrame([{'stage3a_provisional_strict_units': old_units, 'stage3b_corrected_strict_units': observed_units, 'units_restored': observed_units - old_units, 'correction': 'design confirmation uses final lineage freeze causal task_label/coarse_label; CocoaMonilia m1-m3 lineages map to frosty_pod instead of being excluded by stage-specific fine labels.'}])
    repeated_integrity = read_tsv(paths.prepared / 'stage3b_repeated_split_integrity_audit.tsv')
    fixed_integrity = read_tsv(paths.prepared / 'stage3b_fixed_test_integrity_audit.tsv')
    source_integrity = read_tsv(paths.prepared / 'stage3b_source_holdout_integrity_audit.tsv')
    split_pass = repeated_integrity['intended_group_crossings'].astype(int).eq(0).all() and repeated_integrity['all_labels_present_in_every_test_fold'].eq('YES').all() and fixed_integrity[fixed_integrity['evaluation_design'].eq('COCO_TEST_AMBIGUITY_SAFE')]['ambiguity_block_overlap'].astype(int).eq(0).all() and source_integrity[source_integrity['evaluation_design'].eq('SOURCE_HOLDOUT_AMBIGUITY_SAFE')]['ambiguity_block_overlap'].astype(int).eq(0).all()
    claim = pd.DataFrame([{'all_model_configs_completed': 'YES', 'expected_model_configs': args.expected_configs, 'observed_model_configs': len(statuses), 'repeated_split_seeds_primary': 20, 'repeated_split_seeds_concat': 10, 'repeated_split_integrity_pass': 'YES' if split_pass else 'NO', 'corrected_fiveclass_scope_units': observed_units, 'corrected_fiveclass_scope_matches_stage2e': 'YES' if observed_units == 9132 else 'NO', 'coco_fixed_test_sample_ids_constant': 'YES', 'lineage_balanced_training_tested': 'YES', 'ambiguity_safe_source_holdout_tested': 'YES', 'neural_network_finetuning_performed': 'NO', 'claim_boundary': 'design confirmation confirms frozen-feature benchmark with repeated grouped partitions, corrects the secondary five-class scope, holds the CocoaMonilia public test set fixed while progressively removing contaminated training data, and evaluates lineage-balanced training plus ambiguity-safe source holdout. Frozen ImageNet embeddings and logistic regression are used; no field-generalization or fine-tuning claim is made.'}])
    write_tsv(metrics, paths.aggregate / 'stage3b_config_metrics.tsv')
    write_tsv(repeated_summary, paths.aggregate / 'stage3b_repeated_seed_summary.tsv')
    write_tsv(paired_deltas, paths.aggregate / 'stage3b_paired_seed_deltas.tsv')
    write_tsv(per_class, paths.aggregate / 'stage3b_per_class_metrics.tsv')
    write_tsv(confusion_long, paths.aggregate / 'stage3b_confusion_matrices_long.tsv')
    write_tsv(fold_metrics, paths.aggregate / 'stage3b_fold_metrics.tsv')
    write_tsv(training_audit, paths.aggregate / 'stage3b_training_mode_audit.tsv')
    write_tsv(fixed_bootstrap, paths.aggregate / 'stage3b_coco_fixed_test_paired_bootstrap.tsv')
    write_tsv(fixed_reliability, paths.aggregate / 'stage3b_coco_fixed_test_reliability_bins.tsv')
    write_tsv(source_bootstrap, paths.aggregate / 'stage3b_source_holdout_paired_bootstrap.tsv')
    write_tsv(five_scope_correction, paths.aggregate / 'stage3b_fiveclass_scope_correction.tsv')
    write_tsv(selected_predictions, paths.aggregate / 'stage3b_selected_fixed_and_source_predictions.tsv.gz')
    write_tsv(output_index, paths.aggregate / 'stage3b_task_output_index.tsv')
    write_tsv(statuses, paths.aggregate / 'stage3b_model_task_status.tsv')
    write_tsv(repeated_integrity, paths.aggregate / 'stage3b_repeated_split_integrity_audit.tsv')
    write_tsv(fixed_integrity, paths.aggregate / 'stage3b_fixed_test_integrity_audit.tsv')
    write_tsv(source_integrity, paths.aggregate / 'stage3b_source_holdout_integrity_audit.tsv')
    write_tsv(claim, paths.aggregate / 'stage3b_claim_status.tsv')

    def repeated_value(task: str, design: str, mode: str='ALL_PATHS') -> float:
        row = repeated_summary[repeated_summary['task_name'].eq(task) & repeated_summary['evaluation_design'].eq(design) & repeated_summary['feature_set'].eq('efficientnet_b0') & repeated_summary['training_mode'].eq(mode) & repeated_summary['metric_scope'].eq('physical_path_weighted') & repeated_summary['metric'].eq('balanced_accuracy')]
        return float(row.iloc[0]['mean'])
    fixed_point = metrics[metrics['analysis_family'].eq('COCO_FIXED_TEST') & metrics['feature_set'].eq('efficientnet_b0') & metrics['training_mode'].eq('ALL_PATHS') & metrics['metric_scope'].eq('physical_path_weighted')]
    lines = ['# Design-confirmation summary', '', f'- Model configurations completed: {len(statuses)} / {args.expected_configs}', '- Primary repeated splits: 20 seeds × 5 folds; concatenated-feature robustness: 10 seeds × 5 folds.', f'- Corrected causal five-class strict units: {observed_units} (frozen-feature benchmark provisional scope: {old_units}).', '', '## Repeated primary balanced accuracy (mean across split seeds)']
    for task in ['cacao_coarse_three_class', 'cocoamonilia_four_stage']:
        lines.append(f'### {task}')
        for design in CV_ORDER:
            lines.append(f'- {design}: {repeated_value(task, design):.4f}')
    lines.extend(['', '## CocoaMonilia fixed public test set'])
    for design in COCO_TEST_DESIGNS:
        row = fixed_point[fixed_point['evaluation_design'].eq(design)].iloc[0]
        lines.append(f"- {design}: balanced accuracy={float(row['balanced_accuracy']):.4f}; macro-F1={float(row['macro_f1']):.4f}; log loss={float(row['log_loss']):.4f}")
    lines.extend(['', '## Claim boundary', claim.iloc[0]['claim_boundary']])
    (paths.aggregate / 'stage3b_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (paths.aggregate / '.stage3b_complete').touch()
    print('\n'.join(lines))
    print(f'\nAggregation elapsed seconds: {time.time() - started:.2f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
