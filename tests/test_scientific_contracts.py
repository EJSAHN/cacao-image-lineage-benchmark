"""Small regression tests of production functions and statements, not new analyses."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from run_design_confirmation import align_probabilities, choose_training_rows, split_iterator
from prepare_design_confirmation import crossing_count
from aggregate_control_analyses import matched_null_inference, METRICS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / 'src/cacao_image_benchmark/workflow'


def test_probability_columns_follow_requested_labels():
    raw = np.array([[.2, .7, .1], [.6, .1, .3]])
    got = align_probabilities(raw, ['c', 'a', 'b'], ['a', 'b', 'c'])
    np.testing.assert_allclose(got, raw[:, [1, 2, 0]])
    np.testing.assert_allclose(got.sum(axis=1), 1)


def test_missing_probability_class_raises():
    with pytest.raises(RuntimeError, match='missing class'):
        align_probabilities(np.array([[.3, .7]]), ['a', 'b'], ['a', 'b', 'c'])


def sample_table():
    return pd.DataFrame({'sample_id': [f'i{i}' for i in range(10)],
                         'final_strict_lineage_id': ['l0'] * 3 + ['l1'] * 2 + ['l2'] * 5,
                         'is_lineage_representative': ['YES','NO','NO','YES','NO','YES','NO','NO','NO','NO']})


def test_lineage_weights_equalize_training_lineage_totals():
    frame = sample_table()
    selected, weights, audit = choose_training_rows(frame, np.arange(10), 'LINEAGE_WEIGHTED')
    sums = pd.Series(weights).groupby(frame.final_strict_lineage_id).sum().to_numpy()
    np.testing.assert_allclose(sums, [10 / 3] * 3)
    assert weights.flags.writeable and audit['train_rows_after_mode'] == 10


def test_representative_selection_stays_inside_training_partition():
    frame = sample_table()
    train = np.array([1, 2, 4, 7, 8])
    selected, weights, _ = choose_training_rows(frame, train, 'LINEAGE_REPRESENTATIVE')
    assert set(selected) <= set(train) and len(selected) == 3 and weights is None
    assert frame.iloc[selected].final_strict_lineage_id.nunique() == 3


def test_repeated_split_assignments_are_disjoint_and_exhaustive(tmp_path):
    samples = sample_table()
    assignment = samples[['sample_id']].copy()
    assignment['fold'] = np.arange(10) % 5
    path = tmp_path / 'assignment.tsv'
    assignment.to_csv(path, sep='\t', index=False)
    seen = []
    for fold, train, test in split_iterator(pd.Series({'analysis_family':'REPEATED_CV', 'assignment_path':str(path)}), samples):
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(10))
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(10))


def test_split_crossing_counter_detects_cross_fold_group():
    f = pd.DataFrame({'group': ['a','a','b','b'], 'fold':[0,0,1,1]})
    assert crossing_count(f, 'group') == 0
    f.loc[1,'fold']=1
    assert crossing_count(f, 'group') == 1


def null_fixture(benefit, losses):
    rows=[]
    for typ, val, loss in [('OBSERVED_NAIVE',.8,.2), ('OBSERVED_SCENE_SAFE',.6,.4)]:
        rows.append(dict(scenario_id='test', evaluation_weighting='PATH', config_type=typ, null_design='',
                         balanced_accuracy=val, macro_f1=val, log_loss=loss, multiclass_brier=loss, ece_15bin=loss))
    for val,loss in zip(benefit,losses):
        rows.append(dict(scenario_id='test',evaluation_weighting='PATH',config_type='MATCHED_NULL',null_design='STRUCTURE',
                         balanced_accuracy=val, macro_f1=val, log_loss=loss, multiclass_brier=loss, ece_15bin=loss))
    return pd.DataFrame(rows)


def test_matched_null_plus_one_floor_both_tail_directions():
    got=matched_null_inference(null_fixture([.7,.75,.8],[.3,.25,.2]))
    np.testing.assert_allclose(got.empirical_one_sided_p, .25)
    assert set(got.test_direction)=={'lower_than_null','higher_than_null'}


def test_matched_null_includes_ties():
    got=matched_null_inference(null_fixture([.6,.7,.8],[.4,.3,.2]))
    np.testing.assert_allclose(got.empirical_one_sided_p, .5)


def test_matched_null_requires_observed_baseline():
    frame=null_fixture([.7,.8],[.3,.2])
    with pytest.raises(RuntimeError,match='missing'):
        matched_null_inference(frame[frame.config_type!='OBSERVED_NAIVE'].copy())


def production_assignments(filename, names):
    tree=ast.parse((WORKFLOW/filename).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes)==len(names)
    order={name:i for i,name in enumerate(names)}
    return sorted(nodes,key=lambda n:order[n.targets[0].id])


def test_production_end_to_end_partition_statements_keep_validation_separate():
    # Execute the actual production assignment AST, avoiding imports or CNN training.
    statements=production_assignments('train_end_to_end_sensitivity.py', ['validation_fold','train_frame','validation_frame','test_frame'])
    code=compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])), '<production partitions>', 'exec')
    samples=pd.DataFrame({'sample_id':range(20),'fold':np.arange(20)%5})
    for test_fold in range(5):
        env={'samples':samples,'test_fold':test_fold}
        exec(code,env)
        groups=[set(env[k].sample_id) for k in ['train_frame','validation_frame','test_frame']]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
        assert set.union(*groups)==set(range(20))
        assert len(env['train_frame'])==12


def test_production_pipeline_fit_statement_uses_training_features_only():
    # Instantiate and execute the production Pipeline and fit call on a small fixture.
    nodes=production_assignments('run_design_confirmation.py',['estimator'])
    tree=ast.parse((WORKFLOW/'run_design_confirmation.py').read_text())
    fits=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
          and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
          and n.value.func.value.id=='estimator' and n.value.func.attr=='fit']
    assert len(fits)==1
    X=np.array([[0.],[1.],[2.],[3.],[1000.],[2000.]])
    env={'Pipeline':Pipeline,'StandardScaler':StandardScaler,'LogisticRegression':LogisticRegression,
         'TASK_CLASS_WEIGHT':{'fixture':None},'config':{'task_name':'fixture'},'seed':1,'fold':0,
         'X':X,'y':np.array(['a','a','b','b','a','b']),'selected_idx':np.array([0,1,2,3]),'fit_kwargs':{}}
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes+fits,type_ignores=[])),'<production fit>','exec')
    exec(code,env)
    np.testing.assert_allclose(env['estimator'].named_steps['scale'].mean_, [1.5])
    assert env['estimator'].named_steps['scale'].n_samples_seen_==4
