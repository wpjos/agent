# -*- coding: utf-8 -*-

"""
CLI 统一分发入口单元测试

对应源文件：
- cli/__main__.py: main, _print_usage

测试范围：
- 无参数时输出帮助并退出
- help 子命令
- 未知模块名称报错
- 正确转发到子模块
"""

import pytest

from tsas.engine.operator.cli.__main__ import main


class TestMainDispatcher:
    """测试统一分发入口"""

    def test_no_args_exits(self):
        """
        目的：验证无参数时 sys.exit(1)
        输入：空参数列表
        预期：SystemExit(1)
        """
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_help_flag(self, capsys):
        """
        目的：验证 --help 输出使用说明
        输入：['--help']
        预期：输出包含可用模块列表
        """
        main(['--help'])
        captured = capsys.readouterr()
        assert "feature_construction" in captured.out
        assert "detection" in captured.out
        assert "evaluation" in captured.out

    def test_help_command(self, capsys):
        """
        目的：验证 help 子命令输出使用说明
        输入：['help']
        预期：输出包含可用模块列表
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "feature_construction" in captured.out

    def test_unknown_module_exits(self):
        """
        目的：验证未知模块名称时 sys.exit(1)
        输入：['nonexistent_module']
        预期：SystemExit(1)
        """
        with pytest.raises(SystemExit) as exc_info:
            main(['nonexistent_module'])
        assert exc_info.value.code == 1

    def test_forward_to_feature_construction(self, capsys):
        """
        目的：验证正确转发到 feature_construction 子模块
        输入：['feature_construction', 'help']
        预期：输出包含特征算子列表
        """
        main(['feature_construction', 'help'])
        captured = capsys.readouterr()
        assert "square_feature" in captured.out

    def test_forward_to_detection(self, capsys):
        """
        目的：验证正确转发到 detection 子模块
        输入：['detection', 'help']
        预期：输出包含检测算子列表
        """
        main(['detection', 'help'])
        captured = capsys.readouterr()
        assert "knn_scorer" in captured.out

    def test_forward_to_evaluation(self, capsys):
        """
        目的：验证正确转发到 evaluation 子模块
        输入：['evaluation', 'help']
        预期：输出包含评价指标算子列表
        """
        main(['evaluation', 'help'])
        captured = capsys.readouterr()
        assert "binary_classification" in captured.out
