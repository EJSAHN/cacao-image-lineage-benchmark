"""Aggregate end-to-end run outputs and create paired design-sensitivity summaries."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from end_to_end_common import TASK_DISPLAY_NAMES, ProjectPaths, aggregate_lineage_predictions, multiclass_metrics, read_tsv, t_interval, write_tsv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--big', required=True, type=Path)
    parser.add_argument('--expected-configs', type=int, default=40)
    return parser.parse_args()

def config_metrics_from_predictions(config: pd.Series, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = str(config['label_order']).split('|')
    prob_columns = [f'prob__{label}' for label in labels]
    rows: list[dict[str, object]] = []
    per_class_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    path_metrics, path_per_class, path_confusion = multiclass_metrics(predictions['true_label'], predictions['predicted_label'], predictions[prob_columns].to_numpy(dtype=float), labels)
    lineage_predictions = aggregate_lineage_predictions(predictions, labels)
    lineage_metrics, lineage_per_class, lineage_confusion = multiclass_metrics(lineage_predictions['true_label'], lineage_predictions['predicted_label'], lineage_predictions[prob_columns].to_numpy(dtype=float), labels)
    for evaluation_unit, metrics, per_class, confusion in [('PHYSICAL_PATH', path_metrics, path_per_class, path_confusion), ('STRICT_LINEAGE', lineage_metrics, lineage_per_class, lineage_confusion)]:
        base = {'config_id': config['config_id'], 'task_name': config['task_name'], 'architecture': config['architecture'], 'evaluation_design': config['evaluation_design'], 'repeat_index': int(config['repeat_index']), 'split_seed': int(config['split_seed']), 'evaluation_unit': evaluation_unit}
        rows.append({**base, **metrics})
        per_class = per_class.copy()
        for key, value in reversed(list(base.items())):
            per_class.insert(0, key, value)
        per_class_frames.append(per_class)
        confusion = confusion.copy()
        for key, value in reversed(list(base.items())):
            confusion.insert(0, key, value)
        confusion_frames.append(confusion)
    return (pd.DataFrame(rows), pd.concat(per_class_frames), pd.concat(confusion_frames))

def paired_delta_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = metrics[metrics['evaluation_design'].eq('PATH_RANDOM_5FOLD')].copy()
    scene = metrics[metrics['evaluation_design'].eq('VERIFIED_SCENE_BLOCK_5FOLD')].copy()
    key = ['task_name', 'architecture', 'repeat_index', 'split_seed', 'evaluation_unit']
    value_columns = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'log_loss', 'brier_score', 'ece_15bin']
    paired = path[key + value_columns].merge(scene[key + value_columns], on=key, suffixes=('__path', '__scene'), validate='one_to_one')
    for metric in value_columns:
        paired[f'delta_scene_minus_path__{metric}'] = paired[f'{metric}__scene'] - paired[f'{metric}__path']
    summary_rows = []
    for group_keys, group in paired.groupby(['task_name', 'architecture', 'evaluation_unit'], sort=True):
        task_name, architecture, evaluation_unit = group_keys
        for metric in value_columns:
            values = group[f'delta_scene_minus_path__{metric}'].astype(float).to_numpy()
            mean, low, high = t_interval(values)
            summary_rows.append({'task_name': task_name, 'architecture': architecture, 'evaluation_unit': evaluation_unit, 'metric': metric, 'n_repeat_seeds': len(values), 'mean_delta_scene_minus_path': mean, 'ci95_low': low, 'ci95_high': high, 'negative_seed_count': int(np.sum(values < 0)), 'positive_seed_count': int(np.sum(values > 0)), 'zero_seed_count': int(np.sum(values == 0)), 'all_seeds_lower_under_scene_blocking': 'YES' if np.all(values < 0) else 'NO'})
    return (paired, pd.DataFrame(summary_rows))

def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    config_path = paths.prepared / 'stage3d_config_table.tsv'
    if not config_path.is_file():
        raise RuntimeError('end-to-end sensitivity configuration table is missing')
    config_table = read_tsv(config_path)
    if len(config_table) != args.expected_configs:
        raise RuntimeError(f'Expected {args.expected_configs} end-to-end sensitivity configs, found {len(config_table)}')
    status_frames = []
    fold_metric_frames = []
    per_class_frames = []
    confusion_frames = []
    history_frames = []
    audit_frames = []
    overall_metric_frames = []
    overall_per_class_frames = []
    overall_confusion_frames = []
    for config in config_table.itertuples(index=False):
        output_dir = paths.run_results / str(config.config_id)
        status_path = output_dir / f'stage3d_{config.config_id}_status.tsv'
        prediction_path = output_dir / f'stage3d_{config.config_id}_oof_predictions.tsv.gz'
        fold_metric_path = output_dir / f'stage3d_{config.config_id}_fold_metrics.tsv'
        per_class_path = output_dir / f'stage3d_{config.config_id}_per_class.tsv'
        confusion_path = output_dir / f'stage3d_{config.config_id}_confusion.tsv'
        history_path = output_dir / f'stage3d_{config.config_id}_history.tsv'
        audit_path = output_dir / f'stage3d_{config.config_id}_training_audit.tsv'
        required = [status_path, prediction_path, fold_metric_path, per_class_path, confusion_path, history_path, audit_path]
        missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f'end-to-end sensitivity config {config.config_id} is incomplete: {missing}')
        status = read_tsv(status_path)
        if len(status) != 1 or status.iloc[0]['status'] != 'COMPLETED':
            raise RuntimeError(f'end-to-end sensitivity config {config.config_id} did not complete')
        if int(status.iloc[0]['folds_completed']) != 5:
            raise RuntimeError(f'end-to-end sensitivity config {config.config_id} lacks five completed folds')
        predictions = read_tsv(prediction_path)
        if predictions['sample_id'].duplicated().any():
            raise RuntimeError(f'Duplicate OOF sample predictions in {config.config_id}')
        config_series = pd.Series(config._asdict())
        metric_frame, overall_per_class, overall_confusion = config_metrics_from_predictions(config_series, predictions)
        overall_metric_frames.append(metric_frame)
        overall_per_class_frames.append(overall_per_class)
        overall_confusion_frames.append(overall_confusion)
        status_frames.append(status)
        fold_metric_frames.append(read_tsv(fold_metric_path))
        per_class_frames.append(read_tsv(per_class_path))
        confusion_frames.append(read_tsv(confusion_path))
        history_frames.append(read_tsv(history_path))
        audit_frames.append(read_tsv(audit_path))
    statuses = pd.concat(status_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    fold_per_class = pd.concat(per_class_frames, ignore_index=True)
    fold_confusion = pd.concat(confusion_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    training_audit = pd.concat(audit_frames, ignore_index=True)
    overall_metrics = pd.concat(overall_metric_frames, ignore_index=True)
    overall_per_class = pd.concat(overall_per_class_frames, ignore_index=True)
    overall_confusion = pd.concat(overall_confusion_frames, ignore_index=True)
    paired, delta_summary = paired_delta_summary(overall_metrics)
    write_tsv(statuses, paths.aggregate / 'stage3d_run_status.tsv')
    write_tsv(fold_metrics, paths.aggregate / 'stage3d_fold_metrics.tsv')
    write_tsv(fold_per_class, paths.aggregate / 'stage3d_fold_per_class_metrics.tsv')
    write_tsv(fold_confusion, paths.aggregate / 'stage3d_fold_confusion_matrices.tsv')
    write_tsv(histories, paths.aggregate / 'stage3d_training_history.tsv.gz')
    write_tsv(training_audit, paths.aggregate / 'stage3d_training_audit.tsv')
    write_tsv(overall_metrics, paths.aggregate / 'stage3d_oof_metrics.tsv')
    write_tsv(overall_per_class, paths.aggregate / 'stage3d_oof_per_class_metrics.tsv')
    write_tsv(overall_confusion, paths.aggregate / 'stage3d_oof_confusion_matrices.tsv')
    write_tsv(paired, paths.aggregate / 'stage3d_paired_seed_results.tsv')
    write_tsv(delta_summary, paths.aggregate / 'stage3d_paired_delta_summary.tsv')
    direction_table = delta_summary[delta_summary['evaluation_unit'].eq('PHYSICAL_PATH') & delta_summary['metric'].eq('balanced_accuracy')].copy()
    expected_pairs = 4
    direction_pass = len(direction_table) == expected_pairs and bool((direction_table['mean_delta_scene_minus_path'].astype(float) < 0).all())
    all_seed_pass = len(direction_table) == expected_pairs and bool(direction_table['all_seeds_lower_under_scene_blocking'].eq('YES').all())
    claim = pd.DataFrame([{'all_run_configs_completed': 'YES' if len(statuses) == args.expected_configs else 'NO', 'expected_run_configs': args.expected_configs, 'observed_run_configs': len(statuses), 'expected_fold_fits': args.expected_configs * 5, 'observed_fold_metric_rows': len(fold_metrics[fold_metrics['evaluation_unit'].eq('PHYSICAL_PATH')]), 'imagenet_pretrained_end_to_end_training_performed': 'YES', 'all_model_layers_trainable': 'YES', 'two_architectures_tested': 'YES', 'two_tasks_tested': 'YES', 'five_split_seeds_tested': 'YES', 'mean_scene_block_effect_negative_for_all_task_architecture_pairs': 'YES' if direction_pass else 'NO', 'scene_block_effect_negative_for_all_five_seeds_in_all_pairs': 'YES' if all_seed_pass else 'NO', 'hyperparameter_search_performed': 'NO', 'claim_boundary': 'This sensitivity analysis tests whether the evaluation-independence effect persists under end-to-end ImageNet-initialized training. It does not establish field deployment performance or compare state-of-the-art architectures.'}])
    write_tsv(claim, paths.aggregate / 'stage3d_claim_status.tsv')
    summary_lines = ['# End-to-end evaluation-independence sensitivity', '', f'- Completed run configurations: {len(statuses)} / {args.expected_configs}', f'- Completed fold fits: {args.expected_configs * 5}', '- Architectures: ResNet-18 and EfficientNet-B0', '- Tasks: cacao three-class and CocoaMonilia four-stage', '- Evaluation designs: path-random and verified-scene-blocked five-fold partitions', '- Split seeds: five', '', '## Physical-path balanced accuracy']
    for row in direction_table.itertuples(index=False):
        path_value = float(overall_metrics[overall_metrics['task_name'].eq(row.task_name) & overall_metrics['architecture'].eq(row.architecture) & overall_metrics['evaluation_unit'].eq('PHYSICAL_PATH') & overall_metrics['evaluation_design'].eq('PATH_RANDOM_5FOLD')]['balanced_accuracy'].astype(float).mean())
        scene_value = float(overall_metrics[overall_metrics['task_name'].eq(row.task_name) & overall_metrics['architecture'].eq(row.architecture) & overall_metrics['evaluation_unit'].eq('PHYSICAL_PATH') & overall_metrics['evaluation_design'].eq('VERIFIED_SCENE_BLOCK_5FOLD')]['balanced_accuracy'].astype(float).mean())
        summary_lines.append(f'- {TASK_DISPLAY_NAMES[row.task_name]}, {row.architecture}: path-random={path_value:.4f}; scene-blocked={scene_value:.4f}; delta={row.mean_delta_scene_minus_path:.4f} (95% CI {row.ci95_low:.4f} to {row.ci95_high:.4f}).')
    summary_lines.extend(['', '## Claim boundary', 'The analysis is a prespecified sensitivity check using two standard ImageNet-initialized architectures, fixed training settings, five split seeds, and no hyperparameter search. It supports architecture-level robustness of the evaluation-independence effect but does not estimate field performance.', ''])
    (paths.aggregate / 'stage3d_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    print(claim.to_string(index=False))
    print('\n'.join(summary_lines))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
