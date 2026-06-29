#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts-prep-profiler: 时间序列数据探查识别工具

读取原始数据文件，自动识别分片形态、列类型、时间列、标注列、单位信息、数据规模。
输出 Markdown 格式的探查报告。

用户提示信息通过可选的元信息文件提供（--metadata），格式见 references/metadata_spec.md。

用法:
    python profiler.py <data_path> [--output <report_path>] [--metadata <metadata_path>]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# 输出编码统一为 utf-8（兼容 Windows GBK 终端）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# 依赖自检：启动时立刻检查，缺失给出友好提示
def _check_deps():
    missing = []
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"[FATAL] 缺少依赖: {', '.join(missing)}", file=sys.stderr)
        print(f"        请确保当前 Python 环境已安装 pandas 和 numpy", file=sys.stderr)
        print(f"        或换用一个已配置的 Python 解释器启动本脚本", file=sys.stderr)
        print(f"        当前 Python: {sys.executable}", file=sys.stderr)
        sys.exit(1)


_check_deps()

import pandas as pd
import numpy as np


# ============================================================
# 元信息解析
# ============================================================

def parse_metadata(metadata_path):
    """
    解析元信息 MD 文件，提取用户提示。

    元信息文件格式见 references/metadata_spec.md。
    支持以下小节（均可选）：
        ## 时间列      → 单个列名
        ## 标注列      → 逗号分隔的列名列表
        ## 类别列      → 逗号分隔的列名列表
        ## 跳过的文件  → 逗号分隔的文件名列表

    返回:
        user_hints: dict
            - time_column: str | None
            - label_columns: set[str]
            - category_columns: set[str]
        skip_files: set[str]
    """
    user_hints = {
        "time_column": None,
        "label_columns": set(),
        "category_columns": set(),
    }
    skip_files = set()

    if metadata_path is None:
        return user_hints, skip_files

    if not os.path.exists(metadata_path):
        print(f"[FATAL] 元信息文件不存在: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _parse_list_section(section_name):
        """解析逗号分隔列表小节"""
        m = re.search(
            rf'## {section_name}\s*\n(.*?)(?=\n## |\Z)',
            content, re.DOTALL
        )
        if m:
            # 取小节标题后的第一个非空行
            for line in m.group(1).strip().split("\n"):
                line = line.strip()
                if line:
                    # 去除反引号、逗号分隔
                    items = [item.strip().strip("`") for item in line.split(",")]
                    items = [item for item in items if item]
                    return items
        return []

    def _parse_single_section(section_name):
        """解析单值小节"""
        m = re.search(
            rf'## {section_name}\s*\n(.*?)(?=\n## |\Z)',
            content, re.DOTALL
        )
        if m:
            for line in m.group(1).strip().split("\n"):
                line = line.strip()
                if line:
                    return line.strip().strip("`")
        return None

    user_hints["time_column"] = _parse_single_section("时间列")
    user_hints["label_columns"] = set(_parse_list_section("标注列"))
    user_hints["category_columns"] = set(_parse_list_section("类别列"))
    skip_files = set(_parse_list_section("跳过的文件"))

    return user_hints, skip_files

# ============================================================
# 常量定义
# ============================================================

NA_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "-", "--", "nil"}

TIME_COL_HINTS = {
    "time", "date", "timestamp", "datetime", "index", "no",
    "sequence_no", "seq", "seq_no", "ts", "tick", "step"
}

LABEL_COL_HINTS = {"label", "labels", "target", "class"}

KNOWN_UNITS = [
    # 电学
    "a", "amps", "amp", "ma", "ua", "v", "mv", "w", "kw", "mw", "va", "kva",
    "ah", "mah", "hz", "khz", "mhz", "ohm", "f", "uf", "pf", "h",
    # 力学/运动
    "n", "kn", "pa", "kpa", "mpa", "gpa", "bar", "mbar",
    "m/s", "m/s2", "km/h", "rpm", "g", "mg", "kg", "t",
    "mm", "cm", "m", "km", "um", "nm",
    # 温度
    "celsius", "fahrenheit", "°c", "℃", "°f", "℉", "k",
    # 时间
    "s", "ms", "us", "ns", "min", "h", "d",
    # 其他
    "%", "ppm", "ppb", "db", "lux", "ph", "l", "ml",
    "mol", "mmol", "bar", "atm", "rms"
]


# ============================================================
# 工具函数
# ============================================================

def is_na_token(val):
    """判断值是否是 NA 标记"""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in NA_TOKENS


def try_parse_complex(val):
    """尝试解析为复数 (a+bj 或 a+bi)。
    只接受确实包含虚部标记 j/i 的字符串，纯数字不算复数。"""
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # 必须包含 j 或 i 才可能是复数
    if 'j' not in s and 'i' not in s:
        return None
    # 把虚部标记 i 替换为 j（仅当 i 前面是数字或 +/- 时）
    s = re.sub(r'(\d)i$', r'\1j', s)
    s = re.sub(r'(\d)i([+\-])', r'\1j\2', s)
    if 'i' in s and 'j' not in s:
        s = s.replace('i', 'j')
    try:
        result = complex(s)
        return result
    except (ValueError, TypeError):
        return None


def try_parse_number(val):
    """尝试解析为数值（int 或 float）"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return None
        return val
    s = str(val).strip()
    if not s or s.lower() in NA_TOKENS:
        return None
    # 尝试 int
    try:
        return int(s)
    except ValueError:
        pass
    # 尝试 float
    try:
        f = float(s)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except ValueError:
        pass
    # 科学计数法
    try:
        f = float(s.replace(",", ""))
        return f
    except ValueError:
        return None


def try_parse_int(val):
    """尝试解析为整数"""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if val == int(val) and not np.isnan(val):
            return int(val)
        return None
    s = str(val).strip()
    if not s or s.lower() in NA_TOKENS:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
            if f == int(f):
                return int(f)
        except ValueError:
            pass
    return None


# 时间格式候选列表（从最常见到最少见）
TIME_FORMATS = [
    # ISO 标准
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    # 斜杠分隔
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M",
    # 含/不含前导零的变体
    "%Y-%-m-%-d %H:%M:%S",
    "%Y-%-m-%-d %H:%M",
    "%Y-%-m-%-d",
    # 紧凑格式
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y%m%d",
    # 带时区
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    # 只有时间
    "%H:%M:%S.%f",
    "%H:%M:%S",
    "%H:%M",
]


def try_parse_datetime(val):
    """尝试解析为 datetime，返回 datetime 对象或 None"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    s = str(val).strip()
    if not s or s.lower() in NA_TOKENS:
        return None

    # 先尝试 pandas 的智能解析（覆盖面最广）
    try:
        ts = pd.to_datetime(s, errors="raise")
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        pass

    # 逐一尝试预设格式
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    return None


def extract_unit_from_parentheses(col_name):
    """从列名括号中提取单位"""
    # 半角括号
    m = re.search(r'\(([^)]+)\)\s*$', col_name)
    if m:
        return m.group(1).strip()
    # 全角括号
    m = re.search(r'[（(]([^）)]+)[）)]\s*$', col_name)
    if m:
        return m.group(1).strip()
    return None


def extract_unit_from_suffix(col_name):
    """从列名后缀提取单位（匹配 _unit 形式的已知单位）"""
    # 去除可能的括号
    base = re.sub(r'[（(].*?[）)]', '', col_name).strip()
    lower = base.lower()
    # 按长度降序匹配，避免 "a" 匹配到 "amps" 的情况
    for unit in sorted(KNOWN_UNITS, key=len, reverse=True):
        if lower.endswith('_' + unit):
            return unit
    return None


# ============================================================
# 列分析器
# ============================================================

class ColumnProfile:
    """单列的剖析结果"""

    def __init__(self, name):
        self.name = name
        self.judged_type = "unknown"      # 最终判分类型
        self.na_count = 0
        self.total_count = 0
        self.unique_values = None         # set of raw values (非NA)
        self.unique_count = 0
        self.sample_values = []           # 前5个非NA原始值
        self.unit = None
        self.is_monotonic_increasing = False
        self.value_range = None           # (min, max) for numeric
        # 时间列相关
        self.is_time_column = False
        self.time_type = None             # "timestamp" | "sequence"
        self.time_format_sample = None
        self.time_min = None
        self.time_max = None
        # 标注列相关
        self.is_label_column = False
        self.label_encoding = None        # "0/1" | "-1/1"
        # 类别列相关
        self.category_values = None       # sorted list

    def to_dict(self):
        return {
            "name": self.name,
            "judged_type": self.judged_type,
            "na_count": self.na_count,
            "total_count": self.total_count,
            "unique_count": self.unique_count,
            "sample_values": [str(v) for v in self.sample_values],
            "unit": self.unit,
            "is_time_column": self.is_time_column,
            "time_type": self.time_type,
            "time_format_sample": str(self.time_format_sample) if self.time_format_sample else None,
            "time_min": str(self.time_min) if self.time_min else None,
            "time_max": str(self.time_max) if self.time_max else None,
            "is_label_column": self.is_label_column,
            "label_encoding": self.label_encoding,
            "category_values": self.category_values,
            "value_range": [str(v) for v in self.value_range] if self.value_range else None,
        }


def profile_column(series, col_name, user_hints=None):
    """分析单列，返回 ColumnProfile"""
    user_hints = user_hints or {}
    profile = ColumnProfile(col_name)
    profile.total_count = len(series)

    # 提取非 NA 值
    non_na_raw = []
    na_count = 0
    for val in series:
        if is_na_token(val):
            na_count += 1
        else:
            non_na_raw.append(val)
    profile.na_count = na_count
    profile.sample_values = non_na_raw[:5]

    # 唯一值
    unique_raw = set(str(v) for v in non_na_raw)
    profile.unique_values = unique_raw
    profile.unique_count = len(unique_raw)

    # 检查单位
    profile.unit = extract_unit_from_parentheses(col_name)
    if not profile.unit:
        profile.unit = extract_unit_from_suffix(col_name)

    # 如果没有非 NA 值，标记为 unknown
    if not non_na_raw:
        profile.judged_type = "unknown"
        return profile

    # ---- 标注列判定 ----
    is_user_label = col_name in user_hints.get("label_columns", set())

    # 检查是否是 {0,1} 或 {1,-1}
    int_values = []
    for v in non_na_raw:
        iv = try_parse_int(v)
        if iv is not None:
            int_values.append(iv)

    non_na_set = set(int_values) if len(int_values) == len(non_na_raw) else set()

    label_match_01 = non_na_set == {0, 1} or non_na_set == {0} or non_na_set == {1}
    label_match_neg1_1 = non_na_set == {1, -1} or non_na_set == {1} or non_na_set == {-1}

    name_is_label = col_name.lower().strip() == "label"

    if is_user_label:
        profile.is_label_column = True
        if non_na_set == {1, -1} or (non_na_set == {-1}):
            profile.label_encoding = "-1/1"
        else:
            profile.label_encoding = "0/1"
        profile.judged_type = "label"
    elif name_is_label and (label_match_01 or label_match_neg1_1):
        profile.is_label_column = True
        profile.label_encoding = "-1/1" if label_match_neg1_1 else "0/1"
        profile.judged_type = "label"
        return profile

    # ---- 复数列判定 ----
    complex_count = sum(1 for v in non_na_raw if try_parse_complex(v) is not None)
    if complex_count / len(non_na_raw) >= 0.9:
        profile.judged_type = "complex"
        return profile

    # ---- 类别列（整数型）判定 ----
    if len(int_values) == len(non_na_raw):
        # 所有值可解析为整数
        unique_int = set(int_values)
        is_category = (len(unique_int) <= 20 and
                       len(unique_int) / len(non_na_raw) < 0.05 if len(non_na_raw) > 0 else False)
        if is_category:
            profile.judged_type = "category_int"
            profile.category_values = sorted(unique_int)
            return profile

    # ---- 类别列（字符串型）判定 ----
    str_values = [str(v) for v in non_na_raw]
    unique_str = set(str_values)
    if (len(unique_str) <= 20 and
            len(unique_str) / len(non_na_raw) < 0.05):
        profile.judged_type = "category_str"
        profile.category_values = sorted(unique_str)
        return profile

    # ---- 数值列（整数） ----
    if len(int_values) == len(non_na_raw):
        profile.judged_type = "numeric_int"
        int_array = np.array(int_values)
        profile.value_range = (int(int_array.min()), int(int_array.max()))
        profile.is_monotonic_increasing = all(
            int_values[i] <= int_values[i + 1] for i in range(len(int_values) - 1)
        )
        return profile

    # ---- 数值列（浮点） ----
    num_count = sum(1 for v in non_na_raw if try_parse_number(v) is not None)
    if num_count / len(non_na_raw) >= 0.9:
        profile.judged_type = "numeric_float"
        nums = [try_parse_number(v) for v in non_na_raw if try_parse_number(v) is not None]
        if nums:
            profile.value_range = (min(nums), max(nums))
        return profile

    # ---- 字符串列 ----
    profile.judged_type = "string"
    return profile


# ============================================================
# 时间列识别
# ============================================================

def identify_time_column(profiles, user_hint=None):
    """从列剖析结果中识别时间列"""
    if user_hint:
        for p in profiles:
            if p.name == user_hint:
                p.is_time_column = True
                p.time_type = "timestamp"
                return p.name
        # 用户指定了但找不到 → 后续报错
        return None

    timestamp_candidates = []
    sequence_candidates = []

    for p in profiles:
        # 时间戳列判定：解析成功率 >= 90%
        # 重新读取该列数据来做时间解析
        pass  # 在 analyze_file 中处理

    return None


def check_time_column(series, col_name, col_profile):
    """检查一列是否是时间列"""
    non_na = [v for v in series if not is_na_token(v)]
    if not non_na:
        return False, None

    # ---- 时间戳判定 ----
    dt_success = 0
    first_dt = None
    for v in non_na[:100]:  # 采样前100个
        dt = try_parse_datetime(v)
        if dt is not None:
            dt_success += 1
            if first_dt is None:
                first_dt = v

    sample_success_rate = dt_success / min(len(non_na), 100)

    if sample_success_rate >= 0.9:
        col_profile.is_time_column = True
        col_profile.time_type = "timestamp"
        col_profile.time_format_sample = first_dt
        # 解析全列获取 min/max
        dts = [try_parse_datetime(v) for v in non_na]
        dts_valid = [d for d in dts if d is not None]
        if dts_valid:
            col_profile.time_min = min(dts_valid)
            col_profile.time_max = max(dts_valid)
        return True, "timestamp"

    # ---- 整数序号判定 ----
    # 条件1：单调递增 且 唯一值占比 > 80%
    if col_profile.judged_type in ("numeric_int", "category_int"):
        int_vals = [try_parse_int(v) for v in non_na if try_parse_int(v) is not None]
        if len(int_vals) == len(non_na):
            is_mono = all(int_vals[i] <= int_vals[i + 1] for i in range(len(int_vals) - 1))
            unique_ratio = len(set(int_vals)) / len(int_vals) if int_vals else 0
            col_profile.is_monotonic_increasing = is_mono

            name_hint = col_name.lower() in TIME_COL_HINTS
            is_first_col = profiles_index.get(col_name, -1) == 0

            if is_mono and unique_ratio > 0.8:
                col_profile.is_time_column = True
                col_profile.time_type = "sequence"
                col_profile.time_min = int_vals[0] if int_vals else None
                col_profile.time_max = int_vals[-1] if int_vals else None
                return True, "sequence"
            elif name_hint and is_first_col:
                col_profile.is_time_column = True
                col_profile.time_type = "sequence"
                col_profile.time_min = int_vals[0] if int_vals else None
                col_profile.time_max = int_vals[-1] if int_vals else None
                return True, "sequence"

    return False, None


# 全局变量：记录列的索引位置（用于整数序号判定）
profiles_index = {}


# ============================================================
# 文件分析
# ============================================================

class FileProfile:
    """单文件的剖析结果"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.filesize = 0
        self.row_count = 0
        self.col_count = 0
        self.columns = []                 # List[ColumnProfile]
        self.time_column = None           # ColumnProfile of time col
        self.skipped = False
        self.skip_reason = None
        self.errors = []
        self.warnings = []

    def get_col_names(self):
        return [c.name for c in self.columns]


def analyze_file(filepath, user_hints):
    """分析单个文件"""
    global profiles_index

    fp = FileProfile(filepath)
    fp.filesize = os.path.getsize(filepath)

    # 判断是否空文件
    if fp.filesize == 0:
        fp.skipped = True
        fp.skip_reason = "空文件"
        return fp

    # 读取 CSV
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False, na_values=[])
    except Exception as e:
        fp.skipped = True
        fp.skip_reason = f"读取失败: {e}"
        return fp

    fp.row_count = len(df)
    fp.col_count = len(df.columns)

    # 判断是否纯表头
    if fp.row_count == 0:
        fp.skipped = True
        fp.skip_reason = "纯表头（无数据行）"
        return fp

    # 建立列名→索引映射
    profiles_index = {col: i for i, col in enumerate(df.columns)}

    # 检查单位行（表头下1~3行）
    unit_row_index = None
    if fp.row_count >= 3:
        for i in range(min(3, fp.row_count)):
            row = df.iloc[i]
            # 如果整行都是非数字且短，可能是单位行
            non_num_count = sum(1 for v in row if v.strip() and not is_na_token(v) and try_parse_number(v) is None)
            if non_num_count >= fp.col_count * 0.7:
                unit_row_index = i
                break

    # 分析每一列
    for i, col_name in enumerate(df.columns):
        if unit_row_index is not None:
            series = df.iloc[unit_row_index + 1:, i]
        else:
            series = df.iloc[:, i]

        col_profile = profile_column(series, col_name, user_hints)

        # 记录单位行信息
        if unit_row_index is not None:
            unit_val = str(df.iloc[unit_row_index, i]).strip()
            if unit_val and not is_na_token(unit_val):
                if not col_profile.unit:
                    col_profile.unit = unit_val

        fp.columns.append(col_profile)

    # 识别时间列
    user_time_col = user_hints.get("time_column")
    timestamp_cols = []
    sequence_cols = []

    for p in fp.columns:
        is_time, time_type = check_time_column(df[p.name], p.name, p)
        if is_time:
            if time_type == "timestamp":
                timestamp_cols.append(p)
            else:
                sequence_cols.append(p)

    # 处理用户指定
    if user_time_col:
        found = False
        for p in fp.columns:
            if p.name == user_time_col:
                p.is_time_column = True
                if p.time_type is None:
                    p.time_type = "timestamp"
                fp.time_column = p
                found = True
                break
        if not found:
            fp.errors.append(f"用户指定的时间列 '{user_time_col}' 不存在")
    else:
        # 时间戳优先
        if len(timestamp_cols) > 1:
            fp.errors.append(
                f"检测到多个时间戳候选列: {[c.name for c in timestamp_cols]}，请用户指定")
        elif len(timestamp_cols) == 1:
            fp.time_column = timestamp_cols[0]
        elif len(sequence_cols) >= 1:
            fp.time_column = sequence_cols[0]
        else:
            fp.errors.append("未检测到任何时间列候选，可声明用行号作为序号")

    # 更新时间列的 judged_type
    if fp.time_column:
        fp.time_column.judged_type = "time"

    return fp


# ============================================================
# 分片形态判定
# ============================================================

def normalize_col_name(col_name, is_time_col=False):
    """列名归一化：时间列统一映射为 'time'，其余去除单位括号"""
    if is_time_col:
        return "time"
    # 去除单位括号（半角/全角）
    base = re.sub(r'[（(].*?[）)]', '', col_name).strip()
    return base if base else col_name


def determine_shard_pattern(file_profiles):
    """判定分片形态 A/B/C/D/E/F"""
    valid_fps = [fp for fp in file_profiles if not fp.skipped]

    if len(valid_fps) <= 1:
        return "A", "单一文件"

    # 构建归一化后的列名集合（时间列统一为 'time'）
    col_sets = []
    for fp in valid_fps:
        normalized = set()
        for col in fp.columns:
            normalized.add(normalize_col_name(col.name, col.is_time_column))
        col_sets.append(normalized)

    all_same_cols = all(s == col_sets[0] for s in col_sets)

    if all_same_cols:
        # 列完全相同（归一后）→ B（按行划分）
        time_ranges = []
        for fp in valid_fps:
            if fp.time_column:
                time_ranges.append((fp.time_column.time_min, fp.time_column.time_max))

        if time_ranges:
            has_overlap = False
            for i in range(len(time_ranges)):
                for j in range(i + 1, len(time_ranges)):
                    a_min, a_max = time_ranges[i]
                    b_min, b_max = time_ranges[j]
                    try:
                        if a_min < b_max and b_min < a_max:
                            has_overlap = True
                            break
                    except TypeError:
                        pass
                if has_overlap:
                    break

            return "B", f"按行划分（列完全相同，时间区间{'有重叠' if has_overlap else '无重叠'}）"
        return "B", "按行划分（列完全相同）"

    # 列不完全相同（归一后）
    # 判断除 time 外的业务列是否有交集
    business_col_sets = []
    for fp in valid_fps:
        biz_cols = set()
        for col in fp.columns:
            if not col.is_time_column and not col.is_label_column:
                biz_cols.add(normalize_col_name(col.name))
        business_col_sets.append(biz_cols)

    biz_intersect = set.intersection(*business_col_sets) if business_col_sets else set()

    # 提取时间范围
    time_ranges = []
    for fp in valid_fps:
        if fp.time_column:
            time_ranges.append((fp.time_column.time_min, fp.time_column.time_max))

    # 计算时间重叠率
    time_overlap_ratio = 0.0
    if time_ranges and len(time_ranges) >= 2:
        try:
            all_mins = [r[0] for r in time_ranges]
            all_maxs = [r[1] for r in time_ranges]
            overlap_start = max(all_mins)
            overlap_end = min(all_maxs)
            if overlap_start <= overlap_end:
                durations = [(r[1] - r[0]) for r in time_ranges]
                overlap_duration = overlap_end - overlap_start
                min_duration = min(durations)
                if min_duration.total_seconds() > 0:
                    time_overlap_ratio = overlap_duration.total_seconds() / min_duration.total_seconds()
        except (TypeError, AttributeError, ZeroDivisionError):
            pass

    if biz_intersect:
        # 业务列有交集 → 可能是 D/E
        if time_overlap_ratio >= 0.5:
            return "C", f"按列划分（业务列有交集，时间高度重叠 {time_overlap_ratio:.0%}）"
        col_set_counter = Counter(frozenset(s) for s in business_col_sets)
        if len(col_set_counter) < len(business_col_sets):
            return "D", "同时按行列划分且对齐（有行分组）"
        return "E", "同时按行列划分但不对齐（业务列有交集）"

    # 业务列无交集（各文件是不同物理量）
    if time_overlap_ratio >= 0.5:
        return "C", f"按列划分（各文件不同物理量，时间高度重叠 {time_overlap_ratio:.0%}）"

    return "E", "同时按行列划分但不对齐（业务列无交集，时间不重叠）"


# ============================================================
# 报告生成
# ============================================================

def generate_report(file_profiles, shard_pattern, shard_reason, output_path):
    """生成 Markdown 探查报告"""
    valid_fps = [fp for fp in file_profiles if not fp.skipped]
    skipped_fps = [fp for fp in file_profiles if fp.skipped]

    lines = []
    lines.append("# 探查报告（tsas-num-prep-profiler）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 概览
    lines.append("## 1. 概览\n")
    lines.append(f"- **文件数量**：{len(file_profiles)} 个（有效 {len(valid_fps)}，跳过 {len(skipped_fps)}）")
    lines.append(f"- **分片形态判定**：{shard_pattern}（{shard_reason}）")
    total_rows = sum(fp.row_count for fp in valid_fps)
    total_size = sum(fp.filesize for fp in valid_fps)
    lines.append(f"- **数据规模**：总行数 {total_rows}，总大小 {format_size(total_size)}")
    lines.append("")

    # 各文件详情
    lines.append("## 2. 各文件详情\n")
    for fp in file_profiles:
        lines.append(f"### 文件：{fp.filename}\n")
        if fp.skipped:
            lines.append(f"- **状态**：已跳过（{fp.skip_reason}）\n")
            continue

        lines.append(f"- **行数 / 列数**：{fp.row_count} / {fp.col_count}")
        lines.append(f"- **文件大小**：{format_size(fp.filesize)}")

        # 时间列
        if fp.time_column:
            tc = fp.time_column
            lines.append(f"- **时间列**：`{tc.name}`（类型：{tc.time_type}）")
            if tc.time_format_sample:
                lines.append(f"- **时间格式样例**：`{tc.time_format_sample}`")
            if tc.time_min and tc.time_max:
                lines.append(f"- **时间范围**：{tc.time_min} ~ {tc.time_max}")
        else:
            lines.append("- **时间列**：未识别")
            if fp.errors:
                for err in fp.errors:
                    lines.append(f"- **错误**：{err}")

        # 列清单表
        lines.append("\n| 列名 | 数据类型 | 空值数量 | 非空数量 | 唯一值数 | 单位 | 样例值 |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in fp.columns:
            samples = ", ".join(str(v)[:30] for v in p.sample_values[:3])
            unit = p.unit or "-"
            non_na = p.total_count - p.na_count
            lines.append(
                f"| {p.name} | {p.judged_type} | {p.na_count} | {non_na} | {p.unique_count} | {unit} | {samples} |"
            )

        # 单位信息
        units_found = {p.name: p.unit for p in fp.columns if p.unit}
        if units_found:
            lines.append("\n**单位信息**：")
            for col, unit in units_found.items():
                lines.append(f"- `{col}` → {unit}")

        # 错误/警告
        if fp.warnings:
            lines.append("\n**警告**：")
            for w in fp.warnings:
                lines.append(f"- {w}")

        lines.append("")

    # 标注列汇总
    lines.append("## 3. 标注列\n")
    label_cols = []
    for fp in valid_fps:
        for p in fp.columns:
            if p.is_label_column:
                label_cols.append((fp.filename, p.name, p.label_encoding))
    if label_cols:
        lines.append("| 文件 | 列名 | 编码方式 |")
        lines.append("|---|---|---|")
        for fname, cname, enc in label_cols:
            lines.append(f"| {fname} | {cname} | {enc} |")
    else:
        lines.append("未识别到标注列。")
    lines.append("")

    # 类别列汇总
    lines.append("## 4. 类别列\n")
    cat_cols = []
    for fp in valid_fps:
        for p in fp.columns:
            if p.judged_type in ("category_int", "category_str"):
                cat_cols.append((fp.filename, p.name, p.judged_type, p.category_values))
    if cat_cols:
        lines.append("| 文件 | 列名 | 类型 | 候选值 |")
        lines.append("|---|---|---|---|")
        for fname, cname, ctype, cvals in cat_cols:
            vals_str = ", ".join(str(v) for v in cvals) if cvals else "-"
            lines.append(f"| {fname} | {cname} | {ctype} | {vals_str} |")
    else:
        lines.append("未识别到类别列。")
    lines.append("")

    # 全量列汇总（除时间列外所有列）
    lines.append("## 5. 全量列汇总\n")
    lines.append("> 包含所有非时间列的完整信息，按文件名+列名排列。\n")
    lines.append("| 列名 | 来源文件 | 数据类型 | 单位 | 时间范围 | 非空数量 | 空值数量 | 唯一值数 | 样例值 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for fp in valid_fps:
        time_range_str = ""
        if fp.time_column and fp.time_column.time_min and fp.time_column.time_max:
            time_range_str = f"{fp.time_column.time_min} ~ {fp.time_column.time_max}"
        for p in fp.columns:
            if p.is_time_column:
                continue
            samples = ", ".join(str(v)[:20] for v in p.sample_values[:3])
            unit = p.unit or "-"
            non_na = p.total_count - p.na_count
            lines.append(
                f"| {p.name} | {fp.filename} | {p.judged_type} | {unit} | {time_range_str} | "
                f"{non_na} | {p.na_count} | {p.unique_count} | {samples} |"
            )
    lines.append("")

    # 跳过的文件
    lines.append("## 6. 跳过的文件\n")
    if skipped_fps:
        lines.append("| 文件名 | 原因 |")
        lines.append("|---|---|")
        for fp in skipped_fps:
            lines.append(f"| {fp.filename} | {fp.skip_reason} |")
    else:
        lines.append("无跳过的文件。")
    lines.append("")

    # 错误汇总
    lines.append("## 7. 错误/警告\n")
    all_errors = []
    all_warnings = []
    for fp in file_profiles:
        for err in fp.errors:
            all_errors.append(f"[{fp.filename}] {err}")
        for w in fp.warnings:
            all_warnings.append(f"[{fp.filename}] {w}")
    if all_errors:
        lines.append("### 错误（致命）\n")
        for err in all_errors:
            lines.append(f"- {err}")
    if all_warnings:
        lines.append("\n### 警告\n")
        for w in all_warnings:
            lines.append(f"- {w}")
    if not all_errors and not all_warnings:
        lines.append("无错误和警告。")
    lines.append("")

    # 写入文件
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="时间序列数据探查识别工具")
    parser.add_argument("data_path", help="数据文件或目录路径")
    parser.add_argument("--metadata", default=None,
                        help="元信息文件路径（可选，包含用户提示，格式见 references/metadata_spec.md）")
    parser.add_argument("--output", "-o", default=None,
                        help="报告输出路径（缺省为 data_path 同目录下的 raw_data_profiler.md）")

    args = parser.parse_args()

    # 解析元信息
    user_hints, skip_files = parse_metadata(args.metadata)
    if args.metadata:
        print(f"读取元信息：{args.metadata}")
        print(f"  时间列：{user_hints['time_column'] or '(自动推断)'}")
        print(f"  标注列：{user_hints['label_columns'] or '(自动推断)'}")
        print(f"  类别列：{user_hints['category_columns'] or '(自动推断)'}")
        print(f"  跳过文件：{skip_files or '无'}")

    # 发现文件
    data_path = Path(args.data_path)
    if data_path.is_dir():
        files = sorted(data_path.rglob("*.csv"), key=lambda p: str(p).lower())
    else:
        files = [data_path]

    if not files:
        print("错误：未找到任何 CSV 文件", file=sys.stderr)
        sys.exit(1)

    print(f"发现 {len(files)} 个文件，开始探查...")

    # 分析每个文件
    file_profiles = []
    for filepath in files:
        fname = os.path.basename(filepath)
        if fname in skip_files:
            fp = FileProfile(filepath)
            fp.skipped = True
            fp.skip_reason = "用户指定跳过"
            file_profiles.append(fp)
            print(f"  跳过：{fname}")
            continue

        print(f"  分析中：{fname}")
        fp = analyze_file(filepath, user_hints)
        file_profiles.append(fp)

    # 判定分片形态
    shard_pattern, shard_reason = determine_shard_pattern(file_profiles)
    print(f"分片形态判定：{shard_pattern}（{shard_reason}）")

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_dir = data_path.parent if data_path.is_file() else data_path
        output_path = str(output_dir / "raw_data_profiler.md")

    # 生成报告
    report = generate_report(file_profiles, shard_pattern, shard_reason, output_path)
    print(f"\n报告已生成：{output_path}")

    # 检查是否有致命错误
    has_errors = any(fp.errors for fp in file_profiles)
    if has_errors:
        print("\n⚠️ 存在致命错误，请查看报告。", file=sys.stderr)

    return 0 if not has_errors else 2


if __name__ == "__main__":
    sys.exit(main())
