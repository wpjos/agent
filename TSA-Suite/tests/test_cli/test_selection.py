# -*- coding: utf-8 -*-

"""特征选择器 CLI 测试。"""

import json

import pandas as pd

from tsas.engine.operator.cli import __main__ as cli_main
from tsas.engine.operator.cli import feature_selection


def test_selection_cli_run_column_selector(tmp_path) -> None:
    """测试目的：验证 CLI 可运行单个静态列选择器并写出主输出与 EO。"""
    input_path = tmp_path / 'input.csv'
    config_path = tmp_path / 'config.json'
    output_path = tmp_path / 'output.csv'
    eo_path = tmp_path / 'eo.json'
    pd.DataFrame({'a': [1], 'b': [2], 'c': [3]}).to_csv(input_path, index=False)
    config_path.write_text(json.dumps({'operator': 'column_selector', 'config': {'input_columns': ['c', 'a']}}),
                           encoding='utf-8')

    feature_selection.main(
        ['run', '--input', str(input_path), '--config', str(config_path), '--output', str(output_path), '--eo-output',
         str(eo_path)])

    assert pd.read_csv(output_path).to_dict(orient='list') == {'c': [3], 'a': [1]}
    assert json.loads(eo_path.read_text(encoding='utf-8')) == {'selected_indices': [2, 0]}


def test_unified_cli_uses_feature_selection_entry(capsys) -> None:
    """测试目的：验证统一 CLI 入口暴露的特征选择器模块名为 feature_selection。"""
    cli_main.main(['help'])

    usage = capsys.readouterr().out
    assert 'feature_selection' in usage
    assert '  selection             特征选择器算子' not in usage


def test_selection_cli_fit_and_run_loaded_variance_selector(tmp_path) -> None:
    """测试目的：验证 CLI 可训练并加载运行方差阈值选择器。"""
    input_path = tmp_path / 'input.csv'
    config_path = tmp_path / 'config.json'
    model_dir = tmp_path / 'model'
    output_path = tmp_path / 'output.csv'
    eo_path = tmp_path / 'eo.json'
    pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]}).to_csv(input_path, index=False)
    config_path.write_text(json.dumps({'operator': 'variance_threshold_selector', 'config': {'threshold': 0.1}}),
                           encoding='utf-8')

    feature_selection.main(
        ['fit', '--input', str(input_path), '--config', str(config_path), '--model-dir', str(model_dir)])
    feature_selection.main(
        ['run', '--input', str(input_path), '--config', str(config_path), '--load', str(model_dir), '--output',
         str(output_path), '--eo-output', str(eo_path)])

    assert list(pd.read_csv(output_path).columns) == ['b']
    assert json.loads(eo_path.read_text(encoding='utf-8'))['selected_indices'] == [1]
