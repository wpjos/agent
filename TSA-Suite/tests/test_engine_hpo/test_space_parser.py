# -*- coding: utf-8 -*-

"""
SpaceParser 单元测试。

通过 mock OpenAI 客户端验证自然语言解析、重试、模板和格式化输出，
不依赖真实 LLM API。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tsas.engine.hpo.backends.hebo.space_parser import SpaceParser


class MockCompletion:
    """模拟 OpenAI chat.completions.create 返回值。"""

    def __init__(self, content: str):
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        self.choices = [choice]


class TestSpaceParserTemplates:
    """测试预定义模板。"""

    def test_list_templates(self):
        """返回可用模板列表包含 itransformer_forecasting。"""
        templates = SpaceParser.list_templates()
        assert 'itransformer_forecasting' in templates

    def test_from_template_itransformer(self):
        """iTransformer 模板结构完整。"""
        parsed = SpaceParser.from_template('itransformer_forecasting')
        assert 'params_config' in parsed
        assert 'objectives' in parsed
        assert 'constraints' in parsed

        names = [p['name'] for p in parsed['params_config']]
        assert 'seq_len' in names
        assert 'd_model' in names
        assert 'lr' in names
        assert 'dropout' in names

        # lr 应为 pow 类型
        lr_config = next(p for p in parsed['params_config'] if p['name'] == 'lr')
        assert lr_config['type'] == 'pow'

    def test_from_template_unknown_raises(self):
        """未知模板抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知模板"):
            SpaceParser.from_template('not_exist')


class TestSpaceParserFormatProposal:
    """测试建议书格式化。"""

    def test_format_proposal_contains_tables(self):
        """建议书包含参数空间表和优化目标表。"""
        parsed = SpaceParser.from_template('itransformer_forecasting')
        text = SpaceParser.format_proposal(parsed)

        assert '参数空间' in text
        assert '优化目标' in text
        assert 'seq_len' in text
        assert 'rmse' in text

    def test_format_proposal_no_constraints(self):
        """无约束时显示无约束提示。"""
        parsed = SpaceParser.from_template('itransformer_forecasting')
        text = SpaceParser.format_proposal(parsed)
        assert '（无约束）' in text


class TestSpaceParserWithMockLLM:
    """使用 mock LLM 测试解析流程。"""

    @pytest.fixture
    def parser(self):
        """带有 mock 客户端的解析器。"""
        client = MagicMock()
        return SpaceParser(client=client)

    def test_parse_success(self, parser):
        """正常 JSON 响应解析成功。"""
        parsed_response = {
            'params_config': [
                {'name': 'temperature', 'type': 'num', 'lb': 200, 'ub': 400},
            ],
            'objectives': [
                {'name': 'conversion_rate', 'direction': 'max', 'description': '转化率'},
            ],
        }
        parser.client.chat.completions.create.return_value = MockCompletion(
            json.dumps(parsed_response, ensure_ascii=False)
        )

        result = parser.parse('优化催化剂合成')

        assert result['params_config'][0]['name'] == 'temperature'
        assert result['objectives'][0]['name'] == 'conversion_rate'
        assert result['constraints'] == []

    def test_parse_with_code_fence(self, parser):
        """LLM 返回被代码块包裹的 JSON 时正确剥离。"""
        parsed_response = {
            'params_config': [
                {'name': 'x', 'type': 'num', 'lb': 0, 'ub': 1},
            ],
            'objectives': [
                {'name': 'y', 'direction': 'min'},
            ],
        }
        content = '```json\n' + json.dumps(parsed_response, ensure_ascii=False) + '\n```'
        parser.client.chat.completions.create.return_value = MockCompletion(content)

        result = parser.parse('test')
        assert result['params_config'][0]['name'] == 'x'

    def test_parse_retry_on_invalid_json(self, parser):
        """JSON 解析失败时重试。"""
        valid_response = {
            'params_config': [{'name': 'x', 'type': 'num', 'lb': 0, 'ub': 1}],
            'objectives': [{'name': 'y', 'direction': 'min'}],
        }
        parser.client.chat.completions.create.side_effect = [
            MockCompletion('这不是 JSON'),
            MockCompletion(json.dumps(valid_response)),
        ]

        result = parser.parse('test', max_retries=3)
        assert result['params_config'][0]['name'] == 'x'
        assert parser.client.chat.completions.create.call_count == 2

    def test_parse_retry_exhausted_raises(self, parser):
        """重试次数耗尽后抛出 ValueError。"""
        parser.client.chat.completions.create.return_value = MockCompletion('invalid')

        with pytest.raises(ValueError, match="JSON 解析失败"):
            parser.parse('test', max_retries=2)

    def test_parse_missing_params_config_raises(self, parser):
        """缺少 params_config 时抛出 ValueError。"""
        response = {'objectives': [{'name': 'y', 'direction': 'min'}]}
        parser.client.chat.completions.create.return_value = MockCompletion(
            json.dumps(response)
        )

        with pytest.raises(ValueError, match="params_config"):
            parser.parse('test')

    def test_parse_missing_objectives_raises(self, parser):
        """缺少 objectives 时抛出 ValueError。"""
        response = {'params_config': [{'name': 'x', 'type': 'num', 'lb': 0, 'ub': 1}]}
        parser.client.chat.completions.create.return_value = MockCompletion(
            json.dumps(response)
        )

        with pytest.raises(ValueError, match="objectives"):
            parser.parse('test')


class TestSpaceParserClientCreation:
    """测试客户端延迟创建。"""

    def test_client_creation_without_openai_raises(self):
        """openai 未安装时访问 client 抛出 ImportError。"""
        parser = SpaceParser()
        with pytest.raises(ImportError):
            # 触发延迟导入，但当前环境若已安装 openai 则不会抛
            # 这里通过 mock 让 import 失败
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == 'openai':
                    raise ImportError('No module named openai')
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                _ = parser.client
            finally:
                builtins.__import__ = original_import

    def test_client_creation_without_api_key(self, monkeypatch):
        """未设置 OPENAI_API_KEY 时抛出 EnvironmentError。"""
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        parser = SpaceParser()
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            _ = parser.client
