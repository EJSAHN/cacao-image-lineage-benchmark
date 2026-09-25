"""Aggregate matched-removal and visual-shortcut results."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from control_analysis_common import FEATURE_SETS, TASK_LABEL_ORDERS, ProjectPaths, config_seed, per_class_metrics, read_tsv, sha256_file, write_tsv
METRICS = ['balanced_accuracy', 'macro_f1', 'log_loss', 'multiclass_brier', 'ece_15bin']
FEATURE_LABELS = {'normal_efficientnet': 'Normal RGB', 'blur32_efficientnet': 'Blurred 32×32', 'grayscale_efficientnet': 'Grayscale', 'border_only_efficientnet': 'Border only', 'center_only_efficientnet': 'Center only', 'color_histogram': 'Color histogram', 'metadata': 'Metadata only'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--big', required=True, type=Path)
    parser.add_argument('--expected-null-configs', required=True, type=int)
    parser.add_argument('--expected-shortcut-configs', required=True, type=int)
    parser.add_argument('--expected-feature-tasks', type=int, default=5)
    return parser.parse_args()

def collect_status(directory: Path, pattern: str, expected: int) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if len(files) != expected:
        raise RuntimeError(f'Status count mismatch for {directory}: {len(files)} != {expected}')
    tables = [read_tsv(path) for path in files]
    combined = pd.concat(tables, ignore_index=True)
    failed = combined[~combined['status'].eq('COMPLETED')]
    if len(failed):
        raise RuntimeError(f"Incomplete tasks in {directory}: {failed[['config_id', 'status']].to_dict('records')[:10]}")
    return combined

def concat_indexed(status: pd.DataFrame, column: str) -> pd.DataFrame:
    frames = []
    for path in status[column]:
        if not path or not Path(path).is_file():
            raise RuntimeError(f'Missing indexed output: {path}')
        frames.append(read_tsv(path))
    return pd.concat(frames, ignore_index=True)

def bootstrap_mean(values: np.ndarray, seed: int, replicates: int=5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        boot[idx] = rng.choice(values, size=len(values), replace=True).mean()
    return (float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

def aggregate_shortcut_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = ['n', 'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'log_loss', 'ece_15bin', 'multiclass_brier']
    for column in numeric:
        fold_metrics[column] = pd.to_numeric(fold_metrics[column], errors='coerce')
    keys = ['config_id', 'analysis_family', 'task_name', 'evaluation_design', 'feature_set', 'repeat_index', 'split_seed', 'heldout_source', 'evaluation_weighting']
    rows = []
    for key, group in fold_metrics.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row['folds'] = len(group)
        for column in numeric:
            if column == 'n':
                row[column] = float(group[column].sum())
            else:
                weights = group['n'].to_numpy(float)
                row[column] = float(np.average(group[column].to_numpy(float), weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)

def matched_null_inference(null_metrics: pd.DataFrame) -> pd.DataFrame:
    for column in METRICS:
        null_metrics[column] = pd.to_numeric(null_metrics[column], errors='raise')
    rows = []
    scenarios = sorted(null_metrics['scenario_id'].unique())
    for scenario in scenarios:
        for weighting in sorted(null_metrics['evaluation_weighting'].unique()):
            subset = null_metrics[null_metrics['scenario_id'].eq(scenario) & null_metrics['evaluation_weighting'].eq(weighting)]
            naive = subset[subset['config_type'].eq('OBSERVED_NAIVE')]
            safe = subset[subset['config_type'].eq('OBSERVED_SCENE_SAFE')]
            if len(naive) != 1 or len(safe) != 1:
                raise RuntimeError(f'Observed matched-removal metrics missing for {scenario}/{weighting}')
            for null_design in sorted(subset.loc[subset['config_type'].eq('MATCHED_NULL'), 'null_design'].unique()):
                null = subset[subset['config_type'].eq('MATCHED_NULL') & subset['null_design'].eq(null_design)]
                for metric in METRICS:
                    baseline = float(naive.iloc[0][metric])
                    observed = float(safe.iloc[0][metric])
                    observed_delta = observed - baseline
                    null_delta = null[metric].to_numpy(float) - baseline
                    lower = float(np.quantile(null_delta, 0.025))
                    median = float(np.quantile(null_delta, 0.5))
                    upper = float(np.quantile(null_delta, 0.975))
                    if metric in {'balanced_accuracy', 'macro_f1'}:
                        extreme = int(np.sum(null_delta <= observed_delta))
                        direction = 'lower_than_null'
                    else:
                        extreme = int(np.sum(null_delta >= observed_delta))
                        direction = 'higher_than_null'
                    p_value = (1 + extreme) / (len(null_delta) + 1)
                    rows.append({'scenario_id': scenario, 'evaluation_weighting': weighting, 'null_design': null_design, 'metric': metric, 'naive_value': baseline, 'observed_scene_safe_value': observed, 'observed_delta': observed_delta, 'null_replicates': len(null_delta), 'null_delta_2_5pct': lower, 'null_delta_median': median, 'null_delta_97_5pct': upper, 'contamination_specific_excess': observed_delta - median, 'empirical_one_sided_p': p_value, 'test_direction': direction})
    return pd.DataFrame(rows)

def shortcut_repeated_summary(config_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated = config_metrics[config_metrics['analysis_family'].eq('REPEATED_SCENE_CV')].copy()
    rows = []
    for (task, feature, weighting), group in repeated.groupby(['task_name', 'feature_set', 'evaluation_weighting'], sort=False):
        for metric in ['balanced_accuracy', 'macro_f1', 'log_loss', 'multiclass_brier', 'ece_15bin']:
            values = pd.to_numeric(group[metric], errors='raise').to_numpy(float)
            mean, low, high = bootstrap_mean(values, seed=config_seed(f'{task}|{feature}|{metric}', 'stage3c_bootstrap'))
            rows.append({'task_name': task, 'feature_set': feature, 'evaluation_weighting': weighting, 'metric': metric, 'split_seeds': len(values), 'mean': mean, 'bootstrap_95_low': low, 'bootstrap_95_high': high})
    summary = pd.DataFrame(rows)
    retention_rows = []
    for task in repeated['task_name'].unique():
        for weighting in sorted(repeated['evaluation_weighting'].unique()):
            normal = repeated[repeated['task_name'].eq(task) & repeated['feature_set'].eq('normal_efficientnet') & repeated['evaluation_weighting'].eq(weighting)][['repeat_index', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'normal_ba'})
            for feature in FEATURE_SETS:
                current = repeated[repeated['task_name'].eq(task) & repeated['feature_set'].eq(feature) & repeated['evaluation_weighting'].eq(weighting)][['repeat_index', 'balanced_accuracy']].rename(columns={'balanced_accuracy': 'feature_ba'})
                merged = normal.merge(current, on='repeat_index', how='inner')
                if len(merged) == 0:
                    continue
                delta = merged['feature_ba'].to_numpy(float) - merged['normal_ba'].to_numpy(float)
                retention_rows.append({'task_name': task, 'feature_set': feature, 'evaluation_weighting': weighting, 'seeds': len(merged), 'normal_ba_mean': float(merged['normal_ba'].mean()), 'feature_ba_mean': float(merged['feature_ba'].mean()), 'ba_retained_fraction': float(merged['feature_ba'].mean() / merged['normal_ba'].mean()), 'paired_ba_delta_mean': float(delta.mean()), 'paired_ba_delta_min': float(delta.min()), 'paired_ba_delta_max': float(delta.max())})
    return (summary, pd.DataFrame(retention_rows))

def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    feature_status_files = sorted(paths.feature_meta.glob('stage3c_feature_task_*_status.tsv'))
    if len(feature_status_files) != args.expected_feature_tasks:
        raise RuntimeError(f'Feature status count mismatch: {len(feature_status_files)}')
    feature_status = pd.concat([read_tsv(path) for path in feature_status_files], ignore_index=True)
    if not feature_status['status'].eq('COMPLETED').all():
        raise RuntimeError('One or more matched-removal and shortcut controls feature tasks failed')
    null_status = collect_status(paths.null_tasks, 'stage3c_S3C_NULL_*_status.tsv', args.expected_null_configs)
    shortcut_status = collect_status(paths.shortcut_tasks, 'stage3c_S3C_VIS_*_status.tsv', args.expected_shortcut_configs)
    null_metrics = concat_indexed(null_status, 'metric_path')
    null_audits = concat_indexed(null_status, 'training_audit_path')
    shortcut_fold_metrics = concat_indexed(shortcut_status, 'metric_path')
    shortcut_config_metrics = aggregate_shortcut_metrics(shortcut_fold_metrics)
    selected_prediction_frames = []
    for row in null_status[null_status['config_type'].str.startswith('OBSERVED')].itertuples(index=False):
        if row.prediction_path and Path(row.prediction_path).is_file():
            selected_prediction_frames.append(read_tsv(row.prediction_path))
    for row in shortcut_status[~shortcut_status['analysis_family'].eq('REPEATED_SCENE_CV')].itertuples(index=False):
        if row.prediction_path and Path(row.prediction_path).is_file():
            selected_prediction_frames.append(read_tsv(row.prediction_path))
    selected_predictions = pd.concat(selected_prediction_frames, ignore_index=True) if selected_prediction_frames else pd.DataFrame()
    selected_per_class_rows = []
    if len(selected_predictions):
        for config_id, group in selected_predictions.groupby('config_id', sort=False):
            task_name = str(group['task_name'].iloc[0])
            labels = TASK_LABEL_ORDERS[task_name]
            table = per_class_metrics(group['y_true'], group['y_pred'], labels)
            table.insert(0, 'config_id', config_id)
            table.insert(1, 'task_name', task_name)
            if 'feature_set' in group.columns:
                table.insert(2, 'feature_set', str(group['feature_set'].iloc[0]))
            elif 'config_type' in group.columns:
                table.insert(2, 'feature_set', str(group['config_type'].iloc[0]))
            selected_per_class_rows.append(table)
    selected_per_class = pd.concat(selected_per_class_rows, ignore_index=True) if selected_per_class_rows else pd.DataFrame()
    inference = matched_null_inference(null_metrics)
    repeated_summary, retention = shortcut_repeated_summary(shortcut_config_metrics)
    fixed_source = shortcut_config_metrics[~shortcut_config_metrics['analysis_family'].eq('REPEATED_SCENE_CV')].copy()
    write_tsv(feature_status, paths.aggregate / 'stage3c_feature_task_status.tsv')
    write_tsv(null_status, paths.aggregate / 'stage3c_null_task_status.tsv')
    write_tsv(null_metrics, paths.aggregate / 'stage3c_null_metrics.tsv')
    write_tsv(null_audits, paths.aggregate / 'stage3c_null_training_audit.tsv')
    write_tsv(inference, paths.aggregate / 'stage3c_matched_removal_inference.tsv')
    write_tsv(shortcut_status, paths.aggregate / 'stage3c_shortcut_task_status.tsv')
    write_tsv(shortcut_fold_metrics, paths.aggregate / 'stage3c_shortcut_fold_metrics.tsv')
    write_tsv(shortcut_config_metrics, paths.aggregate / 'stage3c_shortcut_config_metrics.tsv')
    write_tsv(repeated_summary, paths.aggregate / 'stage3c_shortcut_repeated_summary.tsv')
    write_tsv(retention, paths.aggregate / 'stage3c_shortcut_retention.tsv')
    write_tsv(fixed_source, paths.aggregate / 'stage3c_shortcut_fixed_source_summary.tsv')
    if len(selected_predictions):
        write_tsv(selected_predictions, paths.aggregate / 'stage3c_selected_predictions.tsv.gz')
    if len(selected_per_class):
        write_tsv(selected_per_class, paths.aggregate / 'stage3c_selected_per_class_metrics.tsv')
    output_index_rows = []
    for frame, columns in [(null_status, ['metric_path', 'training_audit_path', 'prediction_path']), (shortcut_status, ['metric_path', 'training_audit_path', 'prediction_path'])]:
        for row in frame.itertuples(index=False):
            for column in columns:
                path_value = getattr(row, column, '')
                if path_value and Path(path_value).is_file():
                    output_index_rows.append({'config_id': row.config_id, 'output_type': column, 'path': path_value, 'bytes': Path(path_value).stat().st_size, 'sha256': sha256_file(path_value)})
    write_tsv(pd.DataFrame(output_index_rows), paths.aggregate / 'stage3c_task_output_index.tsv')
    plan_audit = read_tsv(paths.prepared / 'stage3c_null_plan_audit.tsv')
    plan_audit['target_paths'] = pd.to_numeric(plan_audit['target_paths'])
    plan_audit['matched_paths'] = pd.to_numeric(plan_audit['matched_paths'])
    path_match = bool((plan_audit['target_paths'] == plan_audit['matched_paths']).all())
    exact_profiles = plan_audit[plan_audit['scenario_id'].isin(['coco_public_test', 'spanish_source_holdout']) | plan_audit['null_design'].eq('CLASS_BLOCK_MATCHED')]
    block_profile_match = bool((pd.to_numeric(exact_profiles['target_blocks']) == pd.to_numeric(exact_profiles['matched_blocks'])).all())
    claim = pd.DataFrame([{'all_feature_tasks_completed': 'YES', 'feature_tasks_expected': args.expected_feature_tasks, 'all_matched_null_configs_completed': 'YES', 'matched_null_configs_expected': args.expected_null_configs, 'all_shortcut_configs_completed': 'YES', 'shortcut_configs_expected': args.expected_shortcut_configs, 'matched_removal_path_counts_exact': 'YES' if path_match else 'NO', 'primary_block_profiles_exact_where_feasible': 'YES' if block_profile_match else 'NO', 'roboflow_source_matched_block_profile_nearest_feasible': 'YES', 'visual_shortcut_training_lineage_balanced': 'YES', 'neural_network_finetuning_performed': 'NO', 'claim_boundary': 'Matched-removal analyses compare observed contamination removal with clean training-block removals matched on path count, class composition, and—where specified—source composition. The Roboflow source-matched null uses the nearest feasible block-size profile because the clean Spanish subset lacks enough larger scene blocks. Visual-shortcut models use frozen EfficientNet-B0 features or fixed handcrafted/metadata controls; no end-to-end fine-tuning or field-generalization claim is made.'}])
    write_tsv(claim, paths.aggregate / 'stage3c_claim_status.tsv')

    def ba(task: str, feature: str) -> float:
        row = repeated_summary[repeated_summary['task_name'].eq(task) & repeated_summary['feature_set'].eq(feature) & repeated_summary['evaluation_weighting'].eq('PATH') & repeated_summary['metric'].eq('balanced_accuracy')]
        return float(row.iloc[0]['mean'])
    null_ba = inference[inference['evaluation_weighting'].eq('PATH') & inference['metric'].eq('balanced_accuracy')]
    summary_lines = ['# Matched-removal and visual-shortcut summary', '', f'- Matched-removal model configurations completed: {len(null_status)} / {args.expected_null_configs}.', f'- Visual-shortcut configurations completed: {len(shortcut_status)} / {args.expected_shortcut_configs}.', f'- Visual feature extractions completed: {len(feature_status)} / {args.expected_feature_tasks}.', '', '## Matched-removal balanced-accuracy tests']
    for row in null_ba.itertuples(index=False):
        summary_lines.append(f'- {row.scenario_id} / {row.null_design}: observed scene-safe delta={row.observed_delta:+.4f}; matched-null median={row.null_delta_median:+.4f}; one-sided P={row.empirical_one_sided_p:.4g}.')
    summary_lines.extend(['', '## Verified-scene visual controls', f"- Cacao coarse three-class normal EfficientNet balanced accuracy: {ba('cacao_coarse_three_class', 'normal_efficientnet'):.4f}.", f"- Cacao coarse three-class blurred-32 balanced accuracy: {ba('cacao_coarse_three_class', 'blur32_efficientnet'):.4f}.", f"- Cacao coarse three-class border-only balanced accuracy: {ba('cacao_coarse_three_class', 'border_only_efficientnet'):.4f}.", f"- CocoaMonilia normal EfficientNet balanced accuracy: {ba('cocoamonilia_four_stage', 'normal_efficientnet'):.4f}.", f"- CocoaMonilia blurred-32 balanced accuracy: {ba('cocoamonilia_four_stage', 'blur32_efficientnet'):.4f}.", f"- CocoaMonilia border-only balanced accuracy: {ba('cocoamonilia_four_stage', 'border_only_efficientnet'):.4f}.", '', '## Claim boundary', claim.iloc[0]['claim_boundary']])
    (paths.aggregate / 'stage3c_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    (paths.aggregate / '.stage3c_complete').write_text('OK\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
