# -*- coding: utf-8 -*-

"""
评价指标算子命令行入口

提供评价指标算子的命令行接口，支持 help 和 run 两个子命令。
支持一次性调用多个评价算子，输出为 JSON 格式。

子命令:
    - ``help [算子名称]``: 查看所有算子列表或指定算子的详细帮助
    - ``run``: 执行评价指标计算

调用方式::

    python -m tsas.engine.operator.cli evaluation help
    python -m tsas.engine.operator.cli evaluation help binary_classification_metric
    python -m tsas.engine.operator.cli evaluation run --input predictions.csv --output result.json --config eval_pipeline.yaml

配置文件示例 (YAML)::

    operators:
      - name: "binary_classification_metric"
        truth_columns: ["label"]
        predict_columns: ["predict"]
        config:
          positive_label: 1
      - name: "point_adjust"
        truth_columns: ["label"]
        predict_columns: ["predict"]

输出 JSON 示例::

    {
      "results": {
        "binary_classification_metric": {
          "result": {"f1": 0.85, "far": 0.12},
          "main_scores": {"f1": 0.85, "far": 0.12}
        }
      }
    }
"""

import argparse
import sys

import numpy as np
import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.cli.common import (
    extract_encoding_arg, build_help_subparser,
    handle_help, instantiate_operator,
)
from tsas.engine.operator.cli.config_loader import load_config
from tsas.engine.operator.cli.io import load_data, save_json, ensure_encoding
from tsas.engine.operator.cli.registry import OperatorRegistry

__all__ = [
    'main',
    'create_registry',
]


def create_registry() -> OperatorRegistry:
    """
    创建评价指标算子注册中心

    扫描 ``tsas.engine.operator.evaluation`` 包，
    注册所有非抽象的 BaseMetricOperator 子类。

    Returns:
        OperatorRegistry: 已完成 discover 的注册中心实例
    """
    from tsas.engine.operator.evaluation.base import BaseMetricOperator

    registry = OperatorRegistry(
        base_class=BaseMetricOperator,
        scan_packages=['tsas.engine.operator.evaluation'],
    )
    registry.discover()
    return registry


def _build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好子命令的解析器
    """
    parser = argparse.ArgumentParser(
        prog='python -m tsas.engine.operator.cli evaluation',
        description='评价指标算子命令行接口',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- help 子命令（委托公共函数构建）----
    build_help_subparser(subparsers)

    # ---- run 子命令 ----
    run_parser = subparsers.add_parser('run', help='执行评价指标计算')
    run_parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    run_parser.add_argument('--output', '-o', required=True, help='输出结果 JSON 文件路径')
    run_parser.add_argument('--config', '-c', required=True, help='评价管线配置文件路径')

    return parser


def _handle_help(registry: OperatorRegistry, operator_name: str | None) -> None:
    """处理 help 子命令

    委托公共函数 ``handle_help`` 完成帮助文档输出。

    Args:
        registry (OperatorRegistry): 算子注册中心
        operator_name (str | None): 算子名称，``None`` 时列出全部
    """
    handle_help(registry, operator_name)


def _resolve_output_key(
    op_spec: dict, op_name: str, used_keys: set[str]
) -> str:
    """
    解析输出 JSON 中的 key

    优先使用 alias 字段，否则使用算子 name()。
    如果 key 重复，自动追加 ``_1``, ``_2`` 等后缀。

    Args:
        op_spec (dict): 算子规格字典
        op_name (str): 算子名称（来自 name()）
        used_keys (set[str]): 已使用的 key 集合

    Returns:
        str: 唯一的输出 key
    """
    key = op_spec.get('alias', op_name)

    # 自动去重
    if key in used_keys:
        suffix = 1
        while f"{key}_{suffix}" in used_keys:
            suffix += 1
        key = f"{key}_{suffix}"

    used_keys.add(key)
    return key


def _result_to_dict(result) -> dict | float:
    """
    将算子运行结果转换为可 JSON 序列化的字典

    支持 float、Pydantic BaseModel、ndarray 等类型。

    Args:
        result: 算子运行结果

    Returns:
        dict | float: 可序列化的结果
    """
    if isinstance(result, float | int):
        return result
    elif isinstance(result, BaseModel):
        return result.model_dump()
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, dict):
        return {k: _result_to_dict(v) for k, v in result.items()}
    else:
        return str(result)


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 run 子命令

    加载数据和配置，实例化并执行多个评价算子，合并结果输出为 JSON。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    operators_config = config.get('operators', [])
    if not operators_config:
        raise ValueError("配置文件中 'operators' 不能为空")

    results = {}
    used_keys: set[str] = set()

    for i, op_spec in enumerate(operators_config):
        # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
        op_instance = instantiate_operator(op_spec, registry)
        op_name = op_instance.name()

        # 准备输入数据
        truth_columns = op_spec.get('truth_columns', [])
        predict_columns = op_spec.get('predict_columns', [])
        input_columns = op_spec.get('input_columns', [])

        if truth_columns and predict_columns:
            # 双输入: (y_truth, y_predict)
            y_truth = df[truth_columns].values
            y_predict = df[predict_columns].values
            # 如果是单列，squeeze 为一维
            if y_truth.shape[1] == 1:
                y_truth = y_truth.ravel()
            if y_predict.shape[1] == 1:
                y_predict = y_predict.ravel()
            op_input = (y_truth, y_predict)
        elif input_columns:
            # 单输入
            op_input = df[input_columns].values
            if op_input.shape[1] == 1:
                op_input = op_input.ravel()
        else:
            # 使用全部列
            op_input = df.values

        # 执行算子
        run_result = op_instance.run(op_input)

        # 构造结果条目
        entry = {
            'result': _result_to_dict(run_result),
        }

        # 提取 main_scores（如果算子支持）
        if hasattr(op_instance, 'scores'):
            try:
                scores = op_instance.scores(op_input)
                if scores is not None:
                    entry['main_scores'] = scores
            except Exception:
                pass

        # 确定输出 key
        output_key = _resolve_output_key(op_spec, op_name, used_keys)
        results[output_key] = entry

    # 输出
    output_data = {'results': results}
    save_json(output_data, args.output)
    print(f"评价完成，结果已保存至: {args.output}")


def main(args: list[str] | None = None) -> None:
    """
    评价指标 CLI 主函数

    Args:
        args (list[str] | None): 命令行参数列表。None 时使用 ``sys.argv[1:]``
    """
    # 提取全局 --encoding 参数（委托公共函数）
    encoding, filtered_args = extract_encoding_arg(args)

    # 应用编码设置
    ensure_encoding(encoding)

    parser = _build_parser()
    parsed = parser.parse_args(filtered_args)

    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    # 创建注册中心
    registry = create_registry()

    if parsed.command == 'help':
        _handle_help(registry, parsed.operator_name)
    elif parsed.command == 'run':
        _handle_run(registry, parsed)


if __name__ == '__main__':
    main()
