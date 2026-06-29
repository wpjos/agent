# -*- coding: utf-8 -*-

"""
特征构造算子命令行入口

提供特征构造算子的命令行接口，支持 help、run、fit 三个子命令。

子命令:
    - ``help [算子名称]``: 查看所有算子列表或指定算子的详细帮助
    - ``run``: 执行特征构造管线
    - ``fit``: 训练可学习特征算子并可选保存模型

调用方式::

    # 通过统一入口
    python -m tsas.engine.operator.cli feature_construction help
    python -m tsas.engine.operator.cli feature_construction help square_feature
    python -m tsas.engine.operator.cli feature_construction run --input data.csv --output features.csv --config pipeline.yaml

    # 直接调用
    python -m tsas.engine.operator.cli.feature_construction help

配置文件示例 (YAML)::

    operators:
      - name: "square_feature"
        config:
          input_columns: ["col_a", "col_b"]
      - name: "polynomial_feature"
        config:
          input_columns: ["col_c"]
          degrees: [2, 3]
    keep_original: true
"""

import argparse
import sys

import numpy as np
import pandas as pd

from tsas.engine.operator.base import BaseOperator, LearnableOperatorMixin
from tsas.engine.operator.cli.common import (
    extract_encoding_arg, build_help_subparser,
    handle_help, instantiate_operator,
)
from tsas.engine.operator.cli.config_loader import load_config
from tsas.engine.operator.cli.io import load_data, save_data, ensure_encoding
from tsas.engine.operator.cli.registry import OperatorRegistry

__all__ = [
    'main',
    'create_registry',
]


def create_registry() -> OperatorRegistry:
    """
    创建特征构造算子注册中心

    扫描 ``tsas.engine.operator.feature.construction`` 包，
    注册所有非抽象的 BaseFeatureMixin 子类。

    Returns:
        OperatorRegistry: 已完成 discover 的注册中心实例
    """
    from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

    registry = OperatorRegistry(
        base_class=BaseFeatureMixin,
        scan_packages=['tsas.engine.operator.feature.construction'],
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
        prog='python -m tsas.engine.operator.cli feature_construction',
        description='特征构造算子命令行接口',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- help 子命令（委托公共函数构建）----
    build_help_subparser(subparsers)

    # ---- run 子命令 ----
    run_parser = subparsers.add_parser('run', help='执行特征构造')
    run_parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    run_parser.add_argument('--output', '-o', required=True, help='输出数据文件路径')
    run_parser.add_argument('--config', '-c', required=True, help='管线配置文件路径')
    run_parser.add_argument('--load', default=None, help='加载已训练模型的目录路径')

    # ---- fit 子命令 ----
    fit_parser = subparsers.add_parser('fit', help='训练可学习特征算子')
    fit_parser.add_argument('--input', '-i', required=True, help='训练数据文件路径')
    fit_parser.add_argument('--config', '-c', required=True, help='管线配置文件路径')
    fit_parser.add_argument('--save', default=None, help='保存训练模型的目录路径')

    return parser


def _handle_help(registry: OperatorRegistry, operator_name: str | None) -> None:
    """处理 help 子命令

    委托公共函数 ``handle_help`` 完成帮助文档输出。

    Args:
        registry (OperatorRegistry): 算子注册中心
        operator_name (str | None): 算子名称，``None`` 时列出全部
    """
    handle_help(registry, operator_name)


def _instantiate_operators(
    config: dict, registry: OperatorRegistry
) -> tuple[list[tuple[BaseOperator, dict]], bool]:
    """根据配置文件实例化算子列表

    从配置中读取 ``operators`` 列表，委托公共函数 ``instantiate_operator``
    完成每个算子的核心实例化。

    Args:
        config (dict): 解析后的配置字典，须包含 ``operators`` 字段
        registry (OperatorRegistry): 算子注册中心

    Returns:
        tuple[list[tuple[BaseOperator, dict]], bool]: (算子实例和规格列表, keep_original 标志)，
            每个元素为 (算子实例, 算子规格字典)

    Raises:
        ValueError: 配置文件中 ``operators`` 为空时
    """
    operators_config = config.get('operators', [])
    if not operators_config:
        raise ValueError("配置文件中 'operators' 不能为空")

    keep_original = config.get('keep_original', True)
    result = []

    for op_spec in operators_config:
        # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
        op_instance = instantiate_operator(op_spec, registry)
        result.append((op_instance, op_spec))

    return result, keep_original


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 run 子命令

    加载数据和配置，实例化算子，逐个执行并拼接结果。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    operators_and_specs, keep_original = _instantiate_operators(config, registry)

    # 加载已训练模型（如指定）
    if args.load:
        _load_operators_state(operators_and_specs, args.load)

    # 逐个执行算子并收集结果
    result_parts = []
    if keep_original:
        result_parts.append(df)

    for op_instance, op_spec in operators_and_specs:
        # 选择输入列
        op_input = _select_input_columns(df, op_instance)

        # 执行
        output = op_instance.run(op_input)

        # 处理输出（可能是 tuple[DataFrame/ndarray, EO]）
        if isinstance(output, tuple):
            output = output[0]

        # 转换为 DataFrame
        if isinstance(output, np.ndarray):
            output = pd.DataFrame(output)

        result_parts.append(output)

    # 拼接结果（按索引对齐）
    result = pd.concat(result_parts, axis=1)

    # 保存
    save_data(result, args.output)
    print(f"特征构造完成，结果已保存至: {args.output}")


def _handle_fit(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 fit 子命令

    加载数据和配置，实例化算子，逐个训练可学习算子。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    operators_and_specs, _ = _instantiate_operators(config, registry)

    # 逐个训练
    for op_instance, op_spec in operators_and_specs:
        if isinstance(op_instance, LearnableOperatorMixin):
            op_input = _select_input_columns(df, op_instance)
            op_instance.fit(op_input)
            print(f"算子 '{op_instance.name()}' 训练完成")
        else:
            print(f"算子 '{op_instance.name()}' 不需要训练，跳过")

    # 保存模型（如指定）
    if args.save:
        _save_operators_state(operators_and_specs, args.save)
        print(f"模型已保存至: {args.save}")


def _select_input_columns(df: pd.DataFrame, op_instance: BaseOperator) -> pd.DataFrame:
    """
    根据算子的 config 中的 input_columns 选择输入列

    如果算子 config 中指定了 input_columns，则只取对应列；
    否则使用全部列。

    Args:
        df (pd.DataFrame): 原始输入数据
        op_instance (BaseOperator): 算子实例

    Returns:
        pd.DataFrame: 选择后的数据子集
    """
    config = op_instance.config
    if config is not None and hasattr(config, 'input_columns'):
        input_columns = config.input_columns
        if input_columns:
            return df[input_columns]
    return df


def _save_operators_state(
    operators_and_specs: list[tuple[BaseOperator, dict]], save_dir: str
) -> None:
    """
    保存所有算子的状态到指定目录

    每个算子保存在以其 name() + 序号命名的子目录中。

    Args:
        operators_and_specs (list[tuple[BaseOperator, dict]]): 算子实例和规格列表
        save_dir (str): 保存目录根路径
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for i, (op_instance, _) in enumerate(operators_and_specs):
        op_dir = save_path / f"{i}_{op_instance.name()}"
        op_instance.save(op_dir)


def _load_operators_state(
    operators_and_specs: list[tuple[BaseOperator, dict]], load_dir: str
) -> None:
    """
    从指定目录加载所有算子的状态

    匹配已实例化算子的保存目录，调用各算子的 load 方法替换实例。

    Args:
        operators_and_specs (list[tuple[BaseOperator, dict]]): 算子实例和规格列表
        load_dir (str): 加载目录根路径
    """
    from pathlib import Path
    load_path = Path(load_dir)

    for i, (op_instance, op_spec) in enumerate(operators_and_specs):
        op_dir = load_path / f"{i}_{op_instance.name()}"
        if op_dir.exists():
            loaded = type(op_instance).load(op_dir)
            # 替换列表中的实例
            operators_and_specs[i] = (loaded, op_spec)


def main(args: list[str] | None = None) -> None:
    """
    特征构造 CLI 主函数

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
    elif parsed.command == 'fit':
        _handle_fit(registry, parsed)


if __name__ == '__main__':
    main()
