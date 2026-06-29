# -*- coding: utf-8 -*-

"""
CLI 公共工具模块

抽取三个 CLI 子模块（detection、evaluation、feature_construction）中重复的
"脚手架"代码，提供统一的公共函数接口。

核心能力:
    - ``extract_encoding_arg``: 从命令行参数中提取全局 ``--encoding`` 参数
    - ``build_help_subparser``: 构建标准的 ``help`` 子命令解析器
    - ``handle_help``: 统一处理 ``help`` 子命令（列表模式 / 详情模式）
    - ``instantiate_operator``: 从配置字典中实例化单个算子

设计原则:
    - 仅抽取三个模块中 **完全一致** 或 **高度同质** 的逻辑
    - 各模块差异化的 ``run`` / ``fit`` 处理逻辑保留在各自模块中
    - 公共函数不引入额外的抽象层，保持简单直接

使用示例::

    from tsas.engine.operator.cli.common import (
        extract_encoding_arg, build_help_subparser,
        handle_help, instantiate_operator,
    )

    # 提取全局编码参数
    encoding, remaining = extract_encoding_arg(args)

    # 构建 help 子解析器
    help_parser = build_help_subparser(subparsers)

    # 处理 help 命令
    handle_help(registry, operator_name)

    # 从配置字典实例化单个算子
    op = instantiate_operator(op_spec, registry)
"""

import argparse

from tsas.engine.operator.base import BaseOperator
from tsas.engine.operator.cli.help_generator import generate_detail, generate_list
from tsas.engine.operator.cli.registry import OperatorRegistry

__all__ = [
    'extract_encoding_arg',
    'build_help_subparser',
    'handle_help',
    'instantiate_operator',
]


def extract_encoding_arg(args: list[str] | None) -> tuple[str | None, list[str]]:
    """从命令行参数列表中提取全局 ``--encoding`` 参数

    扫描参数列表 **头部** 的 ``--encoding ENCODING`` 对（位于子命令之前），
    提取后从参数列表中移除，返回剩余的参数列表。

    仅提取第一个 ``--encoding``，后续出现的同名参数保留在剩余列表中
    （由各子模块自行处理冲突检测）。

    Args:
        args (list[str] | None):
            原始命令行参数列表。为 ``None`` 时返回 ``(None, [])``

    Returns:
        tuple[str | None, list[str]]:
            - (str | None): 提取到的编码值，未指定时为 ``None``
            - (list[str]): 移除 ``--encoding`` 对后的剩余参数列表
    """
    if args is None:
        return None, []

    encoding: str | None = None
    remaining: list[str] = []
    i = 0

    while i < len(args):
        # 仅匹配 --encoding 且后面有值参数
        if args[i] == '--encoding' and i + 1 < len(args):
            encoding = args[i + 1]
            i += 2  # 跳过 --encoding 和其值
        else:
            remaining.append(args[i])
            i += 1

    return encoding, remaining


def build_help_subparser(
        subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """构建标准的 ``help`` 子命令解析器

    创建一个 ``help`` 子解析器，接受一个可选的位置参数 ``operator_name``：
    不指定时列出所有可用算子，指定时显示该算子的详细帮助文档。

    三个 CLI 子模块（detection、evaluation、feature_construction）的
    ``help`` 子解析器结构完全一致，统一由此函数构建。

    Args:
        subparsers (argparse._SubParsersAction):
            由 ``argparse.ArgumentParser.add_subparsers()`` 创建的
            子解析器容器

    Returns:
        argparse.ArgumentParser: 配置好的 ``help`` 子解析器
    """
    help_parser = subparsers.add_parser('help', help='查看算子帮助信息')
    help_parser.add_argument(
        'operator_name',
        nargs='?',
        default=None,
        help='算子名称（不指定时列出所有可用算子）',
    )
    return help_parser


def handle_help(
        registry: OperatorRegistry,
        operator_name: str | None,
) -> None:
    """统一处理 ``help`` 子命令

    根据是否指定算子名称，分发到列表模式或详情模式：

    - ``operator_name`` 为 ``None`` 时：调用 ``generate_list`` 输出所有算子列表
    - ``operator_name`` 非 ``None`` 时：调用 ``generate_detail`` 输出指定算子的详细帮助

    Args:
        registry (OperatorRegistry):
            已执行 ``discover()`` 的算子注册中心实例
        operator_name (str | None):
            目标算子名称。为 ``None`` 时输出所有算子的列表概览
    """
    if operator_name is None:
        # 列表模式：输出所有算子的名称和简介
        operators = registry.list_all()
        print(generate_list(operators))
    else:
        # 详情模式：输出指定算子的完整帮助文档
        cls = registry.get(operator_name)
        print(generate_detail(cls))


def instantiate_operator(
        op_spec: dict,
        registry: OperatorRegistry,
) -> BaseOperator:
    """从配置字典中实例化单个算子

    从 ``op_spec`` 中提取算子名称和配置参数，通过注册中心查找对应的算子类，
    使用算子类的 ``_config_type`` 构造配置实例后创建算子对象。

    此函数封装了三个 CLI 子模块中重复的 "查找 → 构造 config → 实例化" 逻辑。

    配置字典格式::

        {
            "name": "knn_scorer",          # 必填，算子名称
            "config": {"n_neighbors": 5},  # 可选，实例参数字典
        }

    Args:
        op_spec (dict):
            单个算子的配置字典，必须包含 ``"name"`` 键，
            可选包含 ``"config"`` 键（实例参数字典）
        registry (OperatorRegistry):
            已执行 ``discover()`` 的算子注册中心实例

    Returns:
        BaseOperator: 实例化后的算子对象

    Raises:
        ValueError: 当 ``op_spec`` 中缺少 ``"name"`` 键时
        KeyError: 当注册中心中找不到指定名称的算子时
    """
    # 提取算子名称（必填字段）
    op_name = op_spec.get('name')
    if not op_name:
        raise ValueError("算子配置中缺少 'name' 字段")

    # 通过注册中心查找算子类
    op_cls = registry.get(op_name)

    # 提取实例参数字典（可选字段）
    cls_config = op_spec.get('config', {})

    # 按算子类的 _config_type 构造配置实例
    if op_cls._config_type and cls_config:
        # 有 Config 类型定义且提供了配置参数 → 构造 Pydantic 实例
        config_instance = op_cls._config_type(**cls_config)
        op_instance = op_cls(config=config_instance)
    else:
        # 无 Config 类型或未提供配置参数 → 无参实例化
        op_instance = op_cls()

    return op_instance
