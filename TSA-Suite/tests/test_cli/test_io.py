# -*- coding: utf-8 -*-

"""
数据IO模块单元测试

对应源文件：
- cli/io.py: load_data, save_data, save_json

测试范围：
- CSV 加载和保存
- TSV 加载和保存
- 预留格式报错（MAT、HDF5）
- 不支持的格式报错
- 文件不存在报错
- JSON 保存
- 自动创建目录
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tsas.engine.operator.cli.io import load_data, save_data, save_json


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def sample_df():
    """测试用 DataFrame（3列4行）"""
    return pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [5.0, 6.0, 7.0, 8.0],
        'c': [9.0, 10.0, 11.0, 12.0],
    })


# ============================================================================
# CSV 测试
# ============================================================================

class TestLoadDataCSV:
    """测试 CSV 格式的数据加载"""

    def test_load_csv(self, sample_df, tmp_path):
        """
        目的：验证 CSV 文件能正确加载为 DataFrame
        输入：包含 3 列 4 行数据的 CSV 文件
        预期：加载后 DataFrame 形状和值与原始一致
        """
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        result = load_data(csv_path)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_load_csv_string_path(self, sample_df, tmp_path):
        """
        目的：验证支持字符串路径
        输入：字符串格式的文件路径
        预期：正常加载
        """
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))
        pd.testing.assert_frame_equal(result, sample_df)


class TestLoadDataErrors:
    """测试数据加载的错误场景"""

    def test_file_not_found(self):
        """
        目的：验证文件不存在时抛出 FileNotFoundError
        输入：一个不存在的文件路径
        预期：抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError, match="输入文件不存在"):
            load_data("/nonexistent/path/data.csv")

    def test_unsupported_format(self, tmp_path):
        """
        目的：验证不支持的文件后缀抛出 ValueError
        输入：后缀为 .xyz 的文件
        预期：抛出 ValueError，提示不支持
        """
        path = tmp_path / "data.xyz"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_data(path)

    def test_reserved_format_mat(self, tmp_path):
        """
        目的：验证预留格式（.mat）抛出 ValueError 并提示尚未实现
        输入：后缀为 .mat 的文件
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "data.mat"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="尚未实现"):
            load_data(path)

    def test_reserved_format_h5(self, tmp_path):
        """
        目的：验证预留格式（.h5）抛出 ValueError 并提示尚未实现
        输入：后缀为 .h5 的文件
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "data.h5"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="尚未实现"):
            load_data(path)


class TestSaveData:
    """测试数据保存"""

    def test_save_csv(self, sample_df, tmp_path):
        """
        目的：验证 DataFrame 能正确保存为 CSV
        输入：3列4行的 DataFrame
        预期：保存后重新加载，数据一致
        """
        csv_path = tmp_path / "output.csv"
        save_data(sample_df, csv_path)

        result = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_save_tsv(self, sample_df, tmp_path):
        """
        目的：验证 DataFrame 能正确保存为 TSV
        输入：3列4行的 DataFrame
        预期：保存后重新加载，数据一致
        """
        tsv_path = tmp_path / "output.tsv"
        save_data(sample_df, tsv_path)

        result = pd.read_csv(tsv_path, sep='\t')
        pd.testing.assert_frame_equal(result, sample_df)

    def test_save_creates_directory(self, sample_df, tmp_path):
        """
        目的：验证保存时自动创建不存在的目录
        输入：指定一个不存在的目录路径
        预期：目录自动创建，文件正常保存
        """
        csv_path = tmp_path / "subdir" / "nested" / "output.csv"
        save_data(sample_df, csv_path)
        assert csv_path.exists()

    def test_save_unsupported_format(self, sample_df, tmp_path):
        """
        目的：验证不支持的格式抛出 ValueError
        输入：后缀为 .xyz 的路径
        预期：抛出 ValueError
        """
        path = tmp_path / "output.xyz"
        with pytest.raises(ValueError, match="不支持的文件格式"):
            save_data(sample_df, path)

    def test_save_reserved_format(self, sample_df, tmp_path):
        """
        目的：验证预留格式抛出 ValueError
        输入：后缀为 .hdf5 的路径
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "output.hdf5"
        with pytest.raises(ValueError, match="尚未实现"):
            save_data(sample_df, path)


class TestLoadDataTSV:
    """测试 TSV 格式的数据加载"""

    def test_load_tsv(self, sample_df, tmp_path):
        """
        目的：验证 TSV 文件能正确加载
        输入：制表符分隔的数据文件
        预期：加载后数据一致
        """
        tsv_path = tmp_path / "test.tsv"
        sample_df.to_csv(tsv_path, sep='\t', index=False)

        result = load_data(tsv_path)
        pd.testing.assert_frame_equal(result, sample_df)


class TestSaveJson:
    """测试 JSON 保存"""

    def test_save_json_basic(self, tmp_path):
        """
        目的：验证字典能正确保存为 JSON
        输入：简单字典
        预期：文件内容为正确的 JSON
        """
        data = {"f1": 0.85, "far": 0.12}
        json_path = tmp_path / "result.json"

        save_json(data, json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_json_chinese(self, tmp_path):
        """
        目的：验证中文字符不被转义
        输入：包含中文的字典
        预期：文件中包含原始中文字符
        """
        data = {"名称": "测试"}
        json_path = tmp_path / "result.json"

        save_json(data, json_path)

        content = json_path.read_text(encoding='utf-8')
        assert "测试" in content
        assert "\\u" not in content

    def test_save_json_creates_directory(self, tmp_path):
        """
        目的：验证保存 JSON 时自动创建目录
        输入：指定不存在的目录路径
        预期：目录自动创建
        """
        json_path = tmp_path / "sub" / "result.json"
        save_json({"a": 1}, json_path)
        assert json_path.exists()
