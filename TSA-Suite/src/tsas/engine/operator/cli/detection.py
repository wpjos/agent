# -*- coding: utf-8 -*-

"""
异常检测算子命令行入口

提供异常检测算子的命令行接口，支持 help、run、fit 三个子命令。
支持单算子模式，包括 Predictor、Scorer 和 Detector（含 Composite）类型。

子命令:
    - ``help [算子名称]``: 查看所有算子列表或指定算子的详细帮助
    - ``run``: 执行异常检测
    - ``fit``: 训练检测器并可选保存模型

调用方式::

    python -m tsas.engine.operator.cli detection help
    python -m tsas.engine.operator.cli detection help knn_detector
    python -m tsas.engine.operator.cli detection fit --input train.csv --config detector.yaml --save model_dir/
    python -m tsas.engine.operator.cli detection run --input test.csv --config detector.yaml --load model_dir/ --output result.csv

配置文件示例 (YAML)::

    operator:
      name: "knn_detector"
      input_columns: ["sensor_1", "sensor_2", "sensor_3"]
      config:
        n_neighbors: 5
        percentile: 95.0
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
    创建异常检测算子注册中心

    扫描 ``tsas.engine.operator.detection`` 包，
    注册所有 Predictor、Scorer 和 Detector 类型的非抽象算子。

    Returns:
        OperatorRegistry: 已完成 discover 的注册中心实例
    """
    from tsas.engine.operator.detection.base import (
        BasePredictorMixin, BaseScorerMixin, BaseDeciderMixin, BaseDetector,
    )

    def _filter_detection_operator(cls: type) -> bool:
        """注册所有检测管线角色: Predictor、Scorer、Decider、Detector"""
        return issubclass(cls, (BasePredictorMixin, BaseScorerMixin, BaseDeciderMixin, BaseDetector))

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['tsas.engine.operator.detection'],
        filter_fn=_filter_detection_operator,
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
        prog='python -m tsas.engine.operator.cli detection',
        description='异常检测算子命令行接口',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- help 子命令（委托公共函数构建）----
    build_help_subparser(subparsers)

    # ---- run 子命令 ----
    run_parser = subparsers.add_parser('run', help='执行异常检测')
    run_parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    run_parser.add_argument('--output', '-o', required=True, help='输出数据文件路径')
    run_parser.add_argument('--config', '-c', required=True, help='算子配置文件路径')
    run_parser.add_argument('--load', default=None, help='加载已训练模型的目录路径')

    # ---- fit 子命令 ----
    fit_parser = subparsers.add_parser('fit', help='训练检测器')
    fit_parser.add_argument('--input', '-i', required=True, help='训练数据文件路径')
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
) -> tuple[BaseOperator, list[str] | None]:
    """根据配置文件实例化单个检测算子

    从配置字典的 ``operator`` 字段中提取算子规格，委托公共函数
    ``instantiate_operator`` 完成核心实例化，并额外提取 ``input_columns``。

    Args:
        config (dict): 解析后的配置字典，须包含 ``operator`` 字段
        registry (OperatorRegistry): 算子注册中心

    Returns:
        tuple[BaseOperator, list[str] | None]: (算子实例, 输入列名列表)，
            输入列名为 ``None`` 时表示使用全部列

    Raises:
        ValueError: 配置格式不正确时（缺少 ``operator`` 字段）
    """
    op_config = config.get('operator', {})
    if not op_config:
        raise ValueError("配置文件中缺少 'operator' 字段")

    # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
    op_instance = instantiate_operator(op_config, registry)

    # 提取输入列名（detection 模块特有逻辑）
    input_columns = op_config.get('input_columns', None)
    return op_instance, input_columns


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

    加载数据和配置，实例化算子，执行检测并保存结果。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    op_instance, input_columns = _instantiate_operator(config, registry)

    # 加载已训练模型
    if args.load:
        from pathlib import Path
        op_instance = type(op_instance).load(Path(args.load))

    # 选择输入列并执行
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
            main_output = pd.DataFrame({'result': main_output})
        else:
            main_output = pd.DataFrame(main_output)
    elif not isinstance(main_output, pd.DataFrame):
        main_output = pd.DataFrame({'result': [main_output]})

    # 合并原始数据和检测结果
    result = pd.concat([df, main_output], axis=1)

    # 保存
    save_data(result, args.output)
    print(f"异常检测完成，结果已保存至: {args.output}")


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
    op_instance, input_columns = _instantiate_operator(config, registry)

    # 训练
    if not isinstance(op_instance, LearnableOperatorMixin):
        print(f"算子 '{op_instance.name()}' 不需要训练")
        return

    op_input = _select_columns(df, input_columns)
    op_instance.fit(op_input)
    print(f"算子 '{op_instance.name()}' 训练完成")

    # 保存模型
    if args.save:
        from pathlib import Path
        save_path = Path(args.save)
        op_instance.save(save_path)
        print(f"模型已保存至: {args.save}")


def main(args: list[str] | None = None) -> None:
    """
    异常检测 CLI 主函数

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
