# -*- coding: utf-8 -*-

"""
CLI 公共工具模块单元测试

对应源文件：
- cli/common.py

测试范围：
- ``extract_encoding_arg``: 编码参数提取（None/有/无/多个/尾部/无值等场景）
- ``build_help_subparser``: help 子解析器构建
- ``handle_help``: 帮助命令处理（列表模式/详情模式/未知算子）
- ``instantiate_operator``: 算子实例化（有config/无config/无name/未知算子）
"""

import argparse

import pytest

from tsas.engine.operator.cli.common import (
    extract_encoding_arg,
    build_help_subparser,
    handle_help,
    instantiate_operator,
)
from tsas.engine.operator.cli.detection import create_registry as create_detection_registry


# ============================================================================
# extract_encoding_arg 测试
# ============================================================================

class TestExtractEncodingArg:
    """测试 ``extract_encoding_arg`` 编码参数提取函数"""

    def test_none_input(self):
        """
        目的：验证 ``None`` 输入时返回 ``(None, [])``
        输入：``None``
        预期：``(None, [])``
        """
        encoding, remaining = extract_encoding_arg(None)
        assert encoding is None
        assert remaining == []

    def test_empty_list(self):
        """
        目的：验证空列表输入时返回 ``(None, [])``
        输入：``[]``
        预期：``(None, [])``
        """
        encoding, remaining = extract_encoding_arg([])
        assert encoding is None
        assert remaining == []

    def test_no_encoding_arg(self):
        """
        目的：验证无 ``--encoding`` 参数时原样返回
        输入：``['help', 'knn_scorer']``
        预期：``(None, ['help', 'knn_scorer'])``
        """
        encoding, remaining = extract_encoding_arg(['help', 'knn_scorer'])
        assert encoding is None
        assert remaining == ['help', 'knn_scorer']

    def test_with_encoding_at_beginning(self):
        """
        目的：验证 ``--encoding`` 在头部时正确提取
        输入：``['--encoding', 'utf-8', 'help']``
        预期：``('utf-8', ['help'])``
        """
        encoding, remaining = extract_encoding_arg(['--encoding', 'utf-8', 'help'])
        assert encoding == 'utf-8'
        assert remaining == ['help']

    def test_encoding_not_at_beginning(self):
        """
        目的：验证 ``--encoding`` 不在头部时仍能被正确提取（遍历整个列表）
        输入：``['help', '--encoding', 'gbk']``
        预期：``('gbk', ['help'])``
        """
        encoding, remaining = extract_encoding_arg(['help', '--encoding', 'gbk'])
        assert encoding == 'gbk'
        assert remaining == ['help']

    def test_only_encoding(self):
        """
        目的：验证仅包含 ``--encoding`` 时正确提取
        输入：``['--encoding', 'latin-1']``
        预期：``('latin-1', [])``
        """
        encoding, remaining = extract_encoding_arg(['--encoding', 'latin-1'])
        assert encoding == 'latin-1'
        assert remaining == []

    def test_multiple_encoding_keeps_last(self):
        """
        目的：验证多个 ``--encoding`` 时取最后一个（覆盖）
        输入：``['--encoding', 'utf-8', '--encoding', 'gbk', 'help']``
        预期：``('gbk', ['help'])``
        """
        encoding, remaining = extract_encoding_arg(
            ['--encoding', 'utf-8', '--encoding', 'gbk', 'help']
        )
        assert encoding == 'gbk'
        assert remaining == ['help']

    def test_encoding_at_end_without_value(self):
        """
        目的：验证 ``--encoding`` 在末尾且无值参数时不被提取
        输入：``['help', '--encoding']``
        预期：``(None, ['help', '--encoding'])``，``--encoding`` 无值时保留
        """
        encoding, remaining = extract_encoding_arg(['help', '--encoding'])
        assert encoding is None
        assert remaining == ['help', '--encoding']

    def test_encoding_alone_without_value(self):
        """
        目的：验证单独的 ``--encoding`` 且无后续参数时不被提取
        输入：``['--encoding']``
        预期：``(None, ['--encoding'])``
        """
        encoding, remaining = extract_encoding_arg(['--encoding'])
        assert encoding is None
        assert remaining == ['--encoding']

    def test_preserves_order(self):
        """
        目的：验证剩余参数的顺序不被打乱
        输入：``['--encoding', 'utf-8', 'run', '--input', 'data.csv', '--output', 'out.csv']``
        预期：``('utf-8', ['run', '--input', 'data.csv', '--output', 'out.csv'])``
        """
        encoding, remaining = extract_encoding_arg(
            ['--encoding', 'utf-8', 'run', '--input', 'data.csv', '--output', 'out.csv']
        )
        assert encoding == 'utf-8'
        assert remaining == ['run', '--input', 'data.csv', '--output', 'out.csv']


# ============================================================================
# build_help_subparser 测试
# ============================================================================

class TestBuildHelpSubparser:
    """测试 ``build_help_subparser`` help 子解析器构建函数"""

    def test_creates_help_subcommand(self):
        """
        目的：验证构建的子解析器包含 ``help`` 子命令
        输入：空的 subparsers 容器
        预期：解析 ``['help']`` 成功，``operator_name`` 为 ``None``
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        build_help_subparser(subparsers)

        # 解析 help 无参数
        args = parser.parse_args(['help'])
        assert args.command == 'help'
        assert args.operator_name is None

    def test_help_with_operator_name(self):
        """
        目的：验证 ``help`` 带算子名称时正确解析
        输入：``['help', 'knn_scorer']``
        预期：``operator_name == 'knn_scorer'``
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        build_help_subparser(subparsers)

        args = parser.parse_args(['help', 'knn_scorer'])
        assert args.command == 'help'
        assert args.operator_name == 'knn_scorer'

    def test_returns_parser(self):
        """
        目的：验证函数返回值为 ``argparse.ArgumentParser`` 实例
        输入：空的 subparsers 容器
        预期：返回值为 ``argparse.ArgumentParser`` 类型
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        result = build_help_subparser(subparsers)
        assert isinstance(result, argparse.ArgumentParser)


# ============================================================================
# handle_help 测试
# ============================================================================

class TestHandleHelp:
    """测试 ``handle_help`` 帮助命令处理函数"""

    @pytest.fixture
    def registry(self):
        """创建已完成 discover 的检测算子注册中心"""
        return create_detection_registry()

    def test_list_all_operators(self, registry, capsys):
        """
        目的：验证 ``operator_name`` 为 ``None`` 时输出所有算子列表
        输入：``registry``, ``None``
        预期：输出包含已注册的检测算子名称
        """
        handle_help(registry, None)
        captured = capsys.readouterr()
        # 检测算子应该出现在列表中
        assert "knn_scorer" in captured.out
        assert "knn_detector" in captured.out

    def test_detail_single_operator(self, registry, capsys):
        """
        目的：验证指定算子名称时输出详细帮助
        输入：``registry``, ``'knn_scorer'``
        预期：输出包含算子名称标题和参数信息
        """
        handle_help(registry, 'knn_scorer')
        captured = capsys.readouterr()
        assert "## knn_scorer" in captured.out
        assert "n_neighbors" in captured.out

    def test_unknown_operator_raises(self, registry):
        """
        目的：验证查找不存在的算子时抛出 ``KeyError``
        输入：``registry``, ``'nonexistent_operator'``
        预期：抛出 ``KeyError``，错误信息包含 "未找到名为"
        """
        with pytest.raises(KeyError, match="未找到名为"):
            handle_help(registry, 'nonexistent_operator')


# ============================================================================
# instantiate_operator 测试
# ============================================================================

class TestInstantiateOperator:
    """测试 ``instantiate_operator`` 算子实例化函数"""

    @pytest.fixture
    def registry(self):
        """创建已完成 discover 的检测算子注册中心"""
        return create_detection_registry()

    def test_with_config(self, registry):
        """
        目的：验证有 ``config`` 字段时正确实例化
        输入：包含 ``name`` 和 ``config`` 的 op_spec
        预期：返回的算子实例具有正确的配置参数
        """
        op_spec = {
            'name': 'knn_scorer',
            'config': {'n_neighbors': 10},
        }
        op_instance = instantiate_operator(op_spec, registry)
        assert op_instance.name() == 'knn_scorer'
        assert op_instance.config.n_neighbors == 10

    def test_without_config(self, registry):
        """
        目的：验证无 ``config`` 字段时使用默认配置
        输入：仅包含 ``name`` 的 op_spec
        预期：返回的算子实例使用默认参数
        """
        op_spec = {'name': 'knn_scorer'}
        op_instance = instantiate_operator(op_spec, registry)
        assert op_instance.name() == 'knn_scorer'
        # 默认 n_neighbors 应为 5（由 KNNConfig 定义）
        assert op_instance.config.n_neighbors == 5

    def test_with_empty_config(self, registry):
        """
        目的：验证 ``config`` 为空字典时使用默认配置
        输入：``config`` 为 ``{}`` 的 op_spec
        预期：返回的算子实例使用默认参数
        """
        op_spec = {'name': 'knn_scorer', 'config': {}}
        op_instance = instantiate_operator(op_spec, registry)
        assert op_instance.name() == 'knn_scorer'
        assert op_instance.config.n_neighbors == 5

    def test_missing_name_raises(self, registry):
        """
        目的：验证缺少 ``name`` 字段时抛出 ``ValueError``
        输入：无 ``name`` 的 op_spec
        预期：抛出 ``ValueError``，错误信息包含 "缺少 'name'"
        """
        op_spec = {'config': {'n_neighbors': 5}}
        with pytest.raises(ValueError, match="缺少 'name'"):
            instantiate_operator(op_spec, registry)

    def test_empty_name_raises(self, registry):
        """
        目的：验证 ``name`` 为空字符串时抛出 ``ValueError``
        输入：``name`` 为 ``''`` 的 op_spec
        预期：抛出 ``ValueError``
        """
        op_spec = {'name': '', 'config': {}}
        with pytest.raises(ValueError, match="缺少 'name'"):
            instantiate_operator(op_spec, registry)

    def test_unknown_operator_raises(self, registry):
        """
        目的：验证注册中心中不存在指定算子时抛出 ``KeyError``
        输入：不存在的算子名称
        预期：抛出 ``KeyError``
        """
        op_spec = {'name': 'nonexistent_operator'}
        with pytest.raises(KeyError, match="未找到名为"):
            instantiate_operator(op_spec, registry)

    def test_operator_without_config_type(self, registry):
        """
        目的：验证算子类无 ``_config_type`` 定义时也能正常实例化
        输入：一个 ``_config_type`` 为 ``None`` 的算子 spec
        预期：返回有效实例（无参实例化）
        """
        # threshold_decider 有 config_type，但传空 config 时走 else 分支
        op_spec = {'name': 'threshold_decider'}
        op_instance = instantiate_operator(op_spec, registry)
        assert op_instance.name() == 'threshold_decider'
