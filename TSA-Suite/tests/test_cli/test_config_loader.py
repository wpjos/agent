# -*- coding: utf-8 -*-

"""
配置文件加载模块单元测试

对应源文件：
- cli/config_loader.py: load_config

测试范围：
- JSON 配置加载
- YAML 配置加载（.yaml 和 .yml）
- JSON5 配置加载
- 不支持的格式报错
- 文件不存在报错
"""

import json

import pytest

from tsas.engine.operator.cli.config_loader import load_config


class TestLoadConfigJSON:
    """测试 JSON 格式的配置加载"""

    def test_load_json(self, tmp_path):
        """
        目的：验证 JSON 配置文件能正确加载
        输入：标准 JSON 配置文件
        预期：返回对应的字典
        """
        config = {"operators": [{"name": "test", "config": {"a": 1}}]}
        json_path = tmp_path / "config.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        result = load_config(json_path)
        assert result == config

    def test_load_json_string_path(self, tmp_path):
        """
        目的：验证支持字符串路径
        输入：字符串格式的文件路径
        预期：正常加载
        """
        config = {"key": "value"}
        json_path = tmp_path / "config.json"
        with open(json_path, 'w') as f:
            json.dump(config, f)

        result = load_config(str(json_path))
        assert result == config


class TestLoadConfigYAML:
    """测试 YAML 格式的配置加载"""

    def test_load_yaml(self, tmp_path):
        """
        目的：验证 .yaml 配置文件能正确加载
        输入：YAML 格式的配置文件
        预期：返回对应的字典
        """
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "operators:\n"
            "  - name: test\n"
            "    config:\n"
            "      a: 1\n",
            encoding='utf-8',
        )

        result = load_config(yaml_path)
        assert result == {"operators": [{"name": "test", "config": {"a": 1}}]}

    def test_load_yml(self, tmp_path):
        """
        目的：验证 .yml 后缀也能正确加载
        输入：.yml 后缀的 YAML 文件
        预期：正常解析
        """
        yml_path = tmp_path / "config.yml"
        yml_path.write_text("key: value\n", encoding='utf-8')

        result = load_config(yml_path)
        assert result == {"key": "value"}


class TestLoadConfigJSON5:
    """测试 JSON5 格式的配置加载"""

    def test_load_json5(self, tmp_path):
        """
        目的：验证 JSON5 配置文件能正确加载（支持注释和尾逗号）
        输入：带注释和尾逗号的 JSON5 文件
        预期：正常解析为字典
        """
        json5_path = tmp_path / "config.json5"
        json5_path.write_text(
            '{\n'
            '  // 这是注释\n'
            '  "key": "value",\n'
            '  "number": 42,\n'
            '}\n',
            encoding='utf-8',
        )

        result = load_config(json5_path)
        assert result == {"key": "value", "number": 42}


class TestLoadConfigErrors:
    """测试配置加载的错误场景"""

    def test_file_not_found(self):
        """
        目的：验证文件不存在时抛出 FileNotFoundError
        输入：不存在的文件路径
        预期：抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_config("/nonexistent/config.json")

    def test_unsupported_format(self, tmp_path):
        """
        目的：验证不支持的格式抛出 ValueError
        输入：后缀为 .toml 的文件
        预期：抛出 ValueError
        """
        path = tmp_path / "config.toml"
        path.write_text("key = 'value'")

        with pytest.raises(ValueError, match="不支持的配置文件格式"):
            load_config(path)
