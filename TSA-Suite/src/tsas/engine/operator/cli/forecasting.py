# -*- coding: utf-8 -*-

"""
时序预测算子命令行入口

提供时序预测算子的命令行接口，支持 help、run、fit 三个子命令。

子命令:
    - ``help [算子名称]``: 查看所有算子列表或指定算子的详细帮助
    - ``run``: 执行时序预测
    - ``fit``: 训练预测器并可选保存模型

调用方式::

    python -m tsas.engine.operator.cli forecasting help
    python -m tsas.engine.operator.cli forecasting help itransformer_forecaster
    python -m tsas.engine.operator.cli forecasting fit --input train.csv --target target --config forecaster.yaml --save model_dir/
    python -m tsas.engine.operator.cli forecasting run --input window.csv --config forecaster.yaml --load model_dir/ --output pred.csv

配置文件示例 (YAML)::

    operator:
      name: "itransformer_forecaster"
      input_columns: ["feat_0", "feat_1", "feat_2", "target"]
      target_column: "target"
      config:
        seq_len: 100
        pred_len: 20
        d_model: 128
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
from tsas.engine.operator.forecasting.base import BaseForecaster

__all__ = [
    'main',
    'create_registry',
]


def create_registry() -> OperatorRegistry:
    """
    创建时序预测算子注册中心

    扫描 ``tsas.engine.operator.forecasting`` 包，
    注册所有 ``BaseForecaster`` 非抽象子类。

    Returns:
        OperatorRegistry: 已完成 discover 的注册中心实例
    """
    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['tsas.engine.operator.forecasting'],
        filter_fn=lambda cls: issubclass(cls, BaseForecaster),
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
        prog='python -m tsas.engine.operator.cli forecasting',
        description='时序预测算子命令行接口',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- help 子命令（委托公共函数构建）----
    build_help_subparser(subparsers)

    # ---- run 子命令 ----
    run_parser = subparsers.add_parser('run', help='执行时序预测')
    run_parser.add_argument('--input', '-i', required=True, help='输入数据文件路径（单个预测窗口）')
    run_parser.add_argument('--output', '-o', required=True, help='输出预测结果文件路径')
    run_parser.add_argument('--config', '-c', required=True, help='算子配置文件路径')
    run_parser.add_argument('--load', default=None, help='加载已训练模型的目录路径')

    # ---- fit 子命令 ----
    fit_parser = subparsers.add_parser('fit', help='训练预测器')
    fit_parser.add_argument('--input', '-i', required=True, help='训练数据文件路径')
    fit_parser.add_argument('--target', '-t', required=True, help='目标变量列名')
    fit_parser.add_argument('--config', '-c', required=True, help='算子配置文件路径')
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


def _instantiate_operator(
    config: dict, registry: OperatorRegistry
) -> tuple[BaseOperator, list[str] | None, str | None]:
    """根据配置文件实例化单个预测算子

    从配置字典的 ``operator`` 字段中提取算子规格，委托公共函数
    ``instantiate_operator`` 完成核心实例化，并额外提取 ``input_columns``
    和 ``target_column``。

    Args:
        config (dict): 解析后的配置字典，须包含 ``operator`` 字段
        registry (OperatorRegistry): 算子注册中心

    Returns:
        tuple[BaseOperator, list[str] | None, str | None]:
            (算子实例, 输入列名列表, 目标列名)，
            输入列名为 ``None`` 时表示使用全部列

    Raises:
        ValueError: 配置格式不正确时（缺少 ``operator`` 字段）
    """
    op_config = config.get('operator', {})
    if not op_config:
        raise ValueError("配置文件中缺少 'operator' 字段")

    # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
    op_instance = instantiate_operator(op_config, registry)

    # 提取输入列名和目标列名（forecasting 模块特有逻辑）
    input_columns = op_config.get('input_columns', None)
    target_column = op_config.get('target_column', None)
    return op_instance, input_columns, target_column


def _select_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """
    根据列名列表选择 DataFrame 的子集

    Args:
        df (pd.DataFrame): 原始数据
        columns (list[str] | None): 列名列表，None 时返回全部列

    Returns:
        pd.DataFrame: 选择后的数据
    """
    if columns:
        return df[columns]
    return df


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 run 子命令

    加载数据和配置，实例化算子，执行预测并保存结果。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    op_instance, input_columns, _ = _instantiate_operator(config, registry)

    # 加载已训练模型
    if args.load:
        from pathlib import Path
        op_instance = type(op_instance).load(Path(args.load))

    # 选择输入列并执行预测
    op_input = _select_columns(df, input_columns)
    output = op_instance.run(op_input)

    # 处理输出
    if isinstance(output, tuple):
        main_output = output[0]
    else:
        main_output = output

    # 转换为 DataFrame
    if isinstance(main_output, np.ndarray):
        if main_output.ndim == 1:
            main_output = pd.DataFrame({'forecast': main_output})
        else:
            main_output = pd.DataFrame(main_output)
    elif not isinstance(main_output, pd.DataFrame):
        main_output = pd.DataFrame({'forecast': [main_output]})

    # 保存
    save_data(main_output, args.output)
    print(f"预测完成，结果已保存至: {args.output}")


def _handle_fit(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 fit 子命令

    加载数据和配置，实例化算子，执行训练。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    op_instance, input_columns, target_column = _instantiate_operator(config, registry)

    if not isinstance(op_instance, LearnableOperatorMixin):
        print(f"算子 '{op_instance.name()}' 不需要训练")
        return

    # 选择输入列
    op_input = _select_columns(df, input_columns)

    # 构造目标列
    if target_column and target_column in df.columns:
        op_target = df[[target_column]]
    elif args.target and args.target in df.columns:
        op_target = df[[args.target]]
    else:
        raise ValueError(
            f"未找到目标列 '{target_column or args.target}'，"
            f"可用列: {list(df.columns)}"
        )

    # 训练
    op_instance.fit(op_input, op_target)
    print(f"算子 '{op_instance.name()}' 训练完成")

    # 保存模型
    if args.save:
        from pathlib import Path
        save_path = Path(args.save)
        op_instance.save(save_path)
        print(f"模型已保存至: {args.save}")


def main(args: list[str] | None = None) -> None:
    """
    时序预测 CLI 主函数

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
