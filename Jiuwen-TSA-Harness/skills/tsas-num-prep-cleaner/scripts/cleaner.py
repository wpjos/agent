#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tsas-num-prep-cleaner: 时间序列数据单元格治理工具

读取合并后的 CSV + 元信息 MD，执行：
1. 显式缺失处理（NA 标记 → NaN）
2. 类型归一（低格降级 / 高格升格）
3. 单位归一（同物理量单位统一）
4. 类别编码（首次出现顺序 → 整数）
5. 标注编码（强制 0/1）

输出 cleaned_data.csv + cleaned_data_report.md。

用法:
    python cleaner.py <merged_data.csv> \
        --metadata <metadata.md> \
        [--output-dir <dir>] \
        [--output <csv文件名>] \
        [--report-output <报告文件名>] \
        [--target-units <json>] \
        [--category-maps <json>] \
        [--label-cols <cols>] \
        [--enable-outlier-detection] \
        [--value-range <json>] \
        [--sentinels <json>]
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
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
# 常量定义
# ============================================================

NA_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "-", "--", "nil"}

# 类型层级（从低到高）
TYPE_HIERARCHY = ["int", "float", "complex", "str"]

# 已知单位 → 基准物理量和换算因子
# 格式: "单位字符串": (物理量类别, 基准单位, 换算因子)
UNIT_CONVERSION = {
    # 电流
    "a": ("current", "a", 1.0),
    "amp": ("current", "a", 1.0),
    "amps": ("current", "a", 1.0),
    "ma": ("current", "a", 0.001),
    "ua": ("current", "a", 0.000001),
    # 电压
    "v": ("voltage", "v", 1.0),
    "mv": ("voltage", "v", 0.001),
    "kv": ("voltage", "v", 1000.0),
    # 功率
    "w": ("power", "w", 1.0),
    "kw": ("power", "w", 1000.0),
    "mw": ("power", "w", 0.001),
    # 温度（特殊：不是线性比例，需特殊处理）
    "celsius": ("temperature", "celsius", 1.0),
    "c": ("temperature", "celsius", 1.0),
    "fahrenheit": ("temperature", "celsius", None),  # 特殊换算
    "f": ("temperature", "celsius", None),
    "k": ("temperature", "celsius", None),
    # 频率
    "hz": ("frequency", "hz", 1.0),
    "khz": ("frequency", "hz", 1000.0),
    "mhz": ("frequency", "hz", 1000000.0),
    # 压力
    "pa": ("pressure", "pa", 1.0),
    "kpa": ("pressure", "pa", 1000.0),
    "mpa": ("pressure", "pa", 1000000.0),
    "bar": ("pressure", "pa", 100000.0),
    "mbar": ("pressure", "pa", 100.0),
    # 转速
    "rpm": ("rpm", "rpm", 1.0),
    # 质量
    "g": ("mass", "g", 1.0),
    "mg": ("mass", "g", 0.001),
    "kg": ("mass", "g", 1000.0),
    "t": ("mass", "g", 1000000.0),
    # 长度
    "mm": ("length", "mm", 1.0),
    "cm": ("length", "mm", 10.0),
    "m": ("length", "mm", 1000.0),
    "km": ("length", "mm", 1000000.0),
    "um": ("length", "mm", 0.001),
    # 时间间隔
    "s": ("duration", "s", 1.0),
    "ms": ("duration", "s", 0.001),
    "us": ("duration", "s", 0.000001),
    # 其他（无换算需求）
    "rms": ("dimensionless", "rms", 1.0),
    "%": ("dimensionless", "%", 1.0),
    "db": ("dimensionless", "db", 1.0),
}

# 已知单位别名映射（统一到标准写法）
UNIT_ALIASES = {
    "°c": "celsius", "℃": "celsius",
    "°f": "fahrenheit", "℉": "fahrenheit",
    "amp": "amps",
}


# ============================================================
# 工具函数
# ============================================================

def is_na_token(val):
    """判断值是否为 NA 标记"""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in NA_TOKENS


def try_parse_int(val):
    """尝试解析为整数（严格匹配：不含小数点/指数符号的整数字面量）"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # 整数字面量：可选正负号 + 纯数字
    if re.match(r'^[+-]?\d+$', s):
        return int(s)
    return None


def try_parse_float(val):
    """尝试解析为浮点数"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def try_parse_complex(val):
    """尝试解析为复数（仅 j/i 后缀形式）"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # 替换 i 为 j（仅当在复数末尾时）
    if s.endswith('i') and not s.endswith('inf') and not s.endswith('nan'):
        s = s[:-1] + 'j'
    if 'j' not in s:
        return None
    try:
        return complex(s)
    except ValueError:
        return None


def detect_cell_type(val):
    """检测单个单元格的类型（已排除NA后）"""
    if val is None or is_na_token(val):
        return None
    if try_parse_int(val) is not None:
        return "int"
    if try_parse_float(val) is not None:
        return "float"
    c = try_parse_complex(val)
    if c is not None:
        return "complex"
    return "str"


def get_type_level(t):
    """获取类型在层级中的位置"""
    if t in TYPE_HIERARCHY:
        return TYPE_HIERARCHY.index(t)
    return len(TYPE_HIERARCHY)


def normalize_unit(unit_str):
    """归一化单位字符串"""
    if not unit_str:
        return None
    s = str(unit_str).strip().lower()
    if not s or s == "-":
        return None
    return UNIT_ALIASES.get(s, s)


# ============================================================
# 元信息解析
# ============================================================

def parse_metadata(metadata_path):
    """解析元信息 MD 文件，提取列类型、单位、角色、标注列、类别列等。

    元信息格式见 references/metadata_spec.md。
    核心表格是「列清单」：| 列名 | 数据类型 | 单位 | 角色 |
    可选小节：标注列、类别列、目标单位、类别映射、有效值域、哨兵值。

    返回:
        info: dict
            - columns: dict[列名] -> {type, unit, role}
            - label_cols: list[str]
            - category_cols: list[str]
    """
    if metadata_path is None:
        raise ValueError("元信息文件路径不能为空")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元信息文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    info = {
        "columns": {},       # 列名 -> {type, unit, role}
        "label_cols": [],    # 标注列名
        "category_cols": [], # 类别列名
    }

    # 解析 ## 列清单 表格
    # | 列名 | 数据类型 | 单位 | 角色 |
    col_table = re.search(
        r'## 列清单\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    if not col_table:
        raise ValueError("元信息中未找到「列清单」表格")

    for line in col_table.group(1).split("\n"):
        parts = [p.strip() for p in line.split("|")]
        # 需要 4 列内容（首尾空 + 4 = 6）
        if len(parts) >= 6:
            col_name = parts[1].strip().strip("`")
            col_type = parts[2].strip().strip("`")
            col_unit = parts[3].strip().strip("`")
            col_role = parts[4].strip().strip("`")
            # 跳过表头行、分隔线行
            if col_name and col_name != "列名" and not col_name.startswith("---"):
                if col_unit == "-" or not col_unit:
                    col_unit = None
                else:
                    col_unit = normalize_unit(col_unit)
                info["columns"][col_name] = {
                    "type": col_type,
                    "unit": col_unit,
                    "role": col_role,
                }

    if not info["columns"]:
        raise ValueError("元信息中「列清单」表格没有解析到任何列")

    # 解析 ## 标注列（逗号分隔列表）
    label_section = re.search(
        r'## 标注列\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    if label_section:
        for line in label_section.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                items = [item.strip().strip("`") for item in line.split(",")]
                info["label_cols"].extend([item for item in items if item])

    # 解析 ## 类别列（逗号分隔列表）
    cat_section = re.search(
        r'## 类别列\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    if cat_section:
        for line in cat_section.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                items = [item.strip().strip("`") for item in line.split(",")]
                info["category_cols"].extend([item for item in items if item])

    # 解析 ## 目标单位（每行一个：列名: 目标单位）
    unit_section = re.search(
        r'## 目标单位\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    target_units_from_meta = {}
    if unit_section:
        for line in unit_section.group(1).strip().split("\n"):
            line = line.strip()
            if ":" in line:
                col, u = line.split(":", 1)
                col = col.strip()
                u = u.strip()
                if col and u:
                    target_units_from_meta[col] = u

    # 解析 ## 类别映射（JSON）
    map_section = re.search(
        r'## 类别映射\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    cat_maps_from_meta = {}
    if map_section:
        text = map_section.group(1).strip()
        # 尝试去除可能的代码块标记
        text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        try:
            cat_maps_from_meta = json.loads(text)
        except json.JSONDecodeError:
            pass  # 非 JSON 格式就忽略

    # 解析 ## 有效值域（JSON）
    range_section = re.search(
        r'## 有效值域\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    value_ranges_from_meta = {}
    if range_section:
        text = range_section.group(1).strip()
        text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        try:
            value_ranges_from_meta = json.loads(text)
        except json.JSONDecodeError:
            pass

    # 解析 ## 哨兵值（JSON）
    sentinel_section = re.search(
        r'## 哨兵值\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    sentinels_from_meta = {}
    if sentinel_section:
        text = sentinel_section.group(1).strip()
        text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        try:
            sentinels_from_meta = json.loads(text)
        except json.JSONDecodeError:
            pass

    info["target_units"] = target_units_from_meta
    info["category_maps"] = cat_maps_from_meta
    info["value_ranges"] = value_ranges_from_meta
    info["sentinels"] = sentinels_from_meta

    return info


# ============================================================
# 治理逻辑
# ============================================================

class CleanRecord:
    """单列治理记录"""
    def __init__(self, col_name, role, input_type):
        self.col_name = col_name
        self.role = role  # 时间列/标注列/业务数据
        self.input_type = input_type  # 元信息中声明的类型
        self.final_type = None  # 治理后类型
        self.na_count = 0  # NA 标记数量
        self.type_fixes = []  # 类型修改明细 [(行号, 原值, 新值, 操作)]
        self.unit_info = None  # 原始单位
        self.unit_converted = False  # 是否做了单位换算
        self.unit_from = None
        self.unit_to = None
        self.category_map = None  # 类别编码映射
        self.label_converted = False  # 标注编码是否转换
        self.label_from = None  # 原始编码方式
        self.outliers = []  # 异常值/哨兵值 [(行号, 原值)]
        self.errors = []
        self.warnings = []


# ============================================================
# 1. 显式缺失处理
# ============================================================

def process_na_values(df, records):
    """将 NA 标记字符串转为统一空值"""
    for col in df.columns:
        if col == "time":
            continue
        na_mask = df[col].apply(is_na_token)
        na_count = na_mask.sum()
        if na_count > 0:
            # 替换为空字符串（保持 dtype=str，下游统一处理）
            df.loc[na_mask, col] = ""
            records[col].na_count = na_count


# ============================================================
# 2. 类型归一
# ============================================================

def normalize_column_type(col_values, col_name, record):
    """
    对单列做类型归一。
    返回: (治理后的值列表, 是否有修改)
    """
    # 收集每个非空单元格的类型
    cell_types = []
    for v in col_values:
        t = detect_cell_type(v)
        if t is not None:
            cell_types.append(t)

    if not cell_types:
        return col_values, False

    # 统计类型分布
    type_counter = {t: 0 for t in TYPE_HIERARCHY}
    for t in cell_types:
        type_counter[t] = type_counter.get(t, 0) + 1

    total = len(cell_types)
    dominant_type = max(type_counter, key=type_counter.get)
    dominant_ratio = type_counter[dominant_type] / total

    has_changes = False
    result = list(col_values)

    if dominant_ratio >= 0.9:
        # 主导类型占比 >= 90%，降级少量高格数据
        target_type = dominant_type
        for i, v in enumerate(col_values):
            if is_na_token(v) or v == "":
                continue
            cell_t = detect_cell_type(v)
            if cell_t != target_type:
                # 尝试降级
                new_val = try_downgrade(v, target_type)
                if new_val is not None:
                    result[i] = new_val
                    record.type_fixes.append((i, v, new_val, f"降级为{target_type}"))
                    has_changes = True
                else:
                    # 降级失败 → 置空
                    result[i] = ""
                    record.type_fixes.append((i, v, "", "降级失败置空"))
                    has_changes = True
    else:
        # 高格数据占比 > 10%，升格
        # 找到最高类型
        max_level = max(get_type_level(t) for t in cell_types)
        target_type = TYPE_HIERARCHY[max_level]

        for i, v in enumerate(col_values):
            if is_na_token(v) or v == "":
                continue
            cell_t = detect_cell_type(v)
            if cell_t != target_type and cell_t is not None:
                new_val = try_upgrade(v, target_type)
                if new_val is not None:
                    result[i] = new_val
                    record.type_fixes.append((i, v, new_val, f"升格为{target_type}"))
                    has_changes = True
                else:
                    # 升格失败 → 置空
                    result[i] = ""
                    record.type_fixes.append((i, v, "", "升格失败置空"))
                    has_changes = True

    record.final_type = target_type
    return result, has_changes


def try_downgrade(val, target_type):
    """尝试将值降级到目标类型"""
    if target_type == "int":
        v = try_parse_int(val)
        return str(v) if v is not None else None
    if target_type == "float":
        v = try_parse_float(val)
        return str(v) if v is not None else None
    if target_type == "complex":
        v = try_parse_complex(val)
        return str(v).replace("(", "").replace(")", "") if v is not None else None
    if target_type == "str":
        return str(val).strip()
    return None


def try_upgrade(val, target_type):
    """尝试将值升格到目标类型"""
    return try_downgrade(val, target_type)


# ============================================================
# 3. 单位归一
# ============================================================

def normalize_column_unit(col_values, col_name, unit, user_target_units, record):
    """
    对单列做单位归一。
    user_target_units: 用户指定的目标单位 {列名: 目标单位}
    """
    if not unit:
        return col_values, False

    # 用户指定目标单位
    target_unit = user_target_units.get(col_name)

    if not target_unit:
        # 无用户指定，且只有一个来源单位 → 无需转换
        record.unit_info = unit
        return col_values, False

    # 检查是否需要转换
    source_info = UNIT_CONVERSION.get(unit)
    target_info = UNIT_CONVERSION.get(target_unit)

    if not source_info or not target_info:
        # 未知的单位 → 记录警告，不做转换
        record.warnings.append(f"单位 '{unit}' 或目标单位 '{target_unit}' 不在已知换算表中")
        record.unit_info = unit
        return col_values, False

    src_phys, src_base, src_factor = source_info
    tgt_phys, tgt_base, tgt_factor = target_info

    if src_phys != tgt_phys:
        record.errors.append(
            f"单位不兼容：列 '{col_name}' 当前单位 '{unit}'({src_phys}) "
            f"无法转换到目标单位 '{target_unit}'({tgt_phys})"
        )
        return col_values, False

    if unit == target_unit:
        record.unit_info = unit
        return col_values, False

    # 温度特殊处理
    if src_phys == "temperature" and (src_factor is None or tgt_factor is None):
        return _convert_temperature(col_values, unit, target_unit, col_name, record)

    # 线性比例换算
    factor = src_factor / tgt_factor
    result = []
    has_changes = False
    for v in col_values:
        if is_na_token(v) or v == "":
            result.append(v)
            continue
        num = try_parse_float(v)
        if num is not None:
            new_val = num * factor
            # 格式化：尽量保持精度但避免浮点噪声
            if abs(new_val - round(new_val, 6)) < 1e-9:
                new_val = round(new_val, 6)
            result.append(str(new_val))
            has_changes = True
        else:
            result.append(v)

    record.unit_info = target_unit
    record.unit_converted = True
    record.unit_from = unit
    record.unit_to = target_unit
    return result, has_changes


def _convert_temperature(col_values, from_unit, to_unit, col_name, record):
    """温度单位换算（℃/℉/K）"""
    result = []
    has_changes = False
    for v in col_values:
        if is_na_token(v) or v == "":
            result.append(v)
            continue
        num = try_parse_float(v)
        if num is None:
            result.append(v)
            continue

        # 先转到摄氏度
        from_lower = from_unit.lower()
        if from_lower in ("fahrenheit", "f"):
            celsius = (num - 32) * 5 / 9
        elif from_lower == "k":
            celsius = num - 273.15
        else:  # celsius, c
            celsius = num

        # 再从摄氏度转到目标
        to_lower = to_unit.lower()
        if to_lower in ("fahrenheit", "f"):
            new_val = celsius * 9 / 5 + 32
        elif to_lower == "k":
            new_val = celsius + 273.15
        else:  # celsius, c
            new_val = celsius

        if abs(new_val - round(new_val, 6)) < 1e-9:
            new_val = round(new_val, 6)
        result.append(str(new_val))
        has_changes = True

    record.unit_info = to_unit
    record.unit_converted = True
    record.unit_from = from_unit
    record.unit_to = to_unit
    return result, has_changes


# ============================================================
# 4. 类别编码
# ============================================================

def encode_category(col_values, col_name, user_category_maps, record):
    """
    对类别列做编码：首次出现顺序 → 整数。
    user_category_maps: {列名: {原始值: 整数码}}
    """
    user_map = user_category_maps.get(col_name)

    if user_map:
        # 使用用户指定映射
        result = []
        for v in col_values:
            if is_na_token(v) or v == "":
                result.append(v)
                continue
            code = user_map.get(str(v).strip())
            if code is not None:
                result.append(str(code))
            else:
                result.append(v)
                record.warnings.append(f"值 '{v}' 不在用户指定的类别映射中，保持原值")
        record.category_map = user_map
        return result

    # 自动编码：首次出现顺序
    code_map = OrderedDict()
    next_code = 0
    result = []
    for v in col_values:
        if is_na_token(v) or v == "":
            result.append(v)
            continue
        s = str(v).strip()
        if s not in code_map:
            code_map[s] = next_code
            next_code += 1
        result.append(str(code_map[s]))

    record.category_map = dict(code_map)
    return result


# ============================================================
# 5. 标注编码
# ============================================================

def encode_label(col_values, col_name, record):
    """
    标注列强制统一为 0/1。
    -1/1 表示法：1→0（正常），-1→1（异常）。
    """
    # 检测当前编码方式
    unique_vals = set()
    for v in col_values:
        if not is_na_token(v) and v != "":
            unique_vals.add(str(v).strip())

    # 判断编码方式
    if unique_vals <= {"0", "1"}:
        # 已是 0/1，无需转换
        record.label_from = "0/1"
        return col_values

    if unique_vals <= {"-1", "1"}:
        # -1/1 表示法 → 转换
        result = []
        for v in col_values:
            if is_na_token(v) or v == "":
                result.append(v)
                continue
            s = str(v).strip()
            if s == "1":
                result.append("0")
            elif s == "-1":
                result.append("1")
            else:
                result.append(v)
        record.label_converted = True
        record.label_from = "-1/1"
        return result

    # 其他编码方式 → 报错
    record.errors.append(
        f"标注列 '{col_name}' 的取值 {unique_vals} 不符合任何已知编码方式（0/1 或 -1/1）"
    )
    record.label_from = f"未知({unique_vals})"
    return col_values


# ============================================================
# 6. 异常值/哨兵值（默认关闭）
# ============================================================

def detect_outliers(col_values, col_name, value_ranges, sentinels, record):
    """
    异常值/哨兵值检测（仅在用户开启时执行）。
    value_ranges: {列名: (min, max)}
    sentinels: {列名: [哨兵值列表]}
    """
    if col_name not in value_ranges and col_name not in sentinels:
        return col_values

    result = list(col_values)
    rng = value_ranges.get(col_name)
    sentinel_list = sentinels.get(col_name, [])

    for i, v in enumerate(col_values):
        if is_na_token(v) or v == "":
            continue
        num = try_parse_float(v)
        if num is None:
            continue

        is_outlier = False
        if rng and (num < rng[0] or num > rng[1]):
            is_outlier = True
        if num in sentinel_list:
            is_outlier = True

        if is_outlier:
            record.outliers.append((i, v))
            result[i] = ""  # 标记为空

    return result


# ============================================================
# 报告生成
# ============================================================

def generate_report(records, col_order, output_path, input_csv, 
                    total_na, total_type_fixes, total_unit_conversions,
                    total_label_conversions, total_categories, total_outliers):
    """生成治理报告"""
    lines = []
    lines.append("# 治理报告（tsas-num-prep-cleaner）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 概览
    lines.append("## 1. 概览\n")
    lines.append(f"- **输入文件**：{os.path.basename(input_csv)}")
    lines.append(f"- **治理后列数**：{len(col_order)}")
    lines.append(f"- **显式缺失处理**：{total_na} 个单元格")
    lines.append(f"- **类型归一修改**：{total_type_fixes} 个单元格")
    lines.append(f"- **单位归一转换**：{total_unit_conversions} 列")
    lines.append(f"- **标注编码转换**：{total_label_conversions} 列")
    lines.append(f"- **类别编码**：{total_categories} 列")
    if total_outliers is not None:
        lines.append(f"- **异常值/哨兵值处理**：{total_outliers} 个单元格")
    else:
        lines.append(f"- **异常值/哨兵值处理**：关闭（默认）")
    lines.append("")

    # 各列治理详情
    lines.append("## 2. 各列治理详情\n")
    for col in col_order:
        if col == "time":
            continue
        rec = records.get(col)
        if not rec:
            continue

        lines.append(f"### 列：{col}\n")
        lines.append(f"- **角色**：{rec.role}")
        lines.append(f"- **输入类型**：{rec.input_type}")
        lines.append(f"- **治理后类型**：{rec.final_type or '无变化'}")
        lines.append(f"- **显式缺失**：{rec.na_count} 个")

        if rec.unit_info:
            if rec.unit_converted:
                lines.append(f"- **单位归一**：{rec.unit_from} → {rec.unit_to}（已转换）")
            else:
                lines.append(f"- **单位归一**：{rec.unit_info}（无需转换）")

        if rec.label_converted:
            lines.append(f"- **标注编码转换**：{rec.label_from} → 0/1")
        elif rec.role == "标注列":
            lines.append(f"- **标注编码**：{rec.label_from or '0/1'}（无需转换）")

        if rec.category_map:
            map_str = ", ".join(f"{k}→{v}" for k, v in rec.category_map.items())
            lines.append(f"- **类别编码映射**：{map_str}")

        if rec.type_fixes:
            lines.append(f"- **类型修改明细**（前 20 条）：")
            for idx, orig, new, op in rec.type_fixes[:20]:
                lines.append(f"  - 行{idx}: '{orig}' → '{new}'（{op}）")
            if len(rec.type_fixes) > 20:
                lines.append(f"  - ... 共 {len(rec.type_fixes)} 条")

        if rec.outliers:
            lines.append(f"- **异常值/哨兵值**（前 20 条）：")
            for idx, orig in rec.outliers[:20]:
                lines.append(f"  - 行{idx}: '{orig}'")
            if len(rec.outliers) > 20:
                lines.append(f"  - ... 共 {len(rec.outliers)} 个")

        if rec.errors:
            lines.append(f"- **错误**：")
            for err in rec.errors:
                lines.append(f"  - {err}")
        if rec.warnings:
            lines.append(f"- **警告**：")
            for w in rec.warnings:
                lines.append(f"  - {w}")

        lines.append("")

    # 列顺序
    lines.append("## 3. 治理后列清单\n")
    lines.append("| 列名 | 角色 | 类型 | 说明 |")
    lines.append("|---|---|---|---|")
    for col in col_order:
        rec = records.get(col, None)
        if col == "time":
            lines.append(f"| {col} | 时间列 | datetime | 时间列不参与单元格治理 |")
        elif rec:
            unit_note = f"，单位={rec.unit_info}" if rec.unit_info else ""
            lines.append(f"| {col} | {rec.role} | {rec.final_type or rec.input_type}{unit_note} | - |")
    lines.append("")

    # 错误/警告汇总
    lines.append("## 4. 错误/警告汇总\n")
    all_errors = []
    all_warnings = []
    for col, rec in records.items():
        for err in rec.errors:
            all_errors.append(f"[{col}] {err}")
        for w in rec.warnings:
            all_warnings.append(f"[{col}] {w}")

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

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="时间序列数据单元格治理工具")
    parser.add_argument("input_csv", help="合并后的 CSV 文件路径（merged_data.csv）")
    parser.add_argument("--metadata", "-m", required=True,
                        help="元信息 MD 文件路径（包含列清单、可选标注列/类别列/目标单位等，格式见 references/metadata_spec.md）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（缺省为输入文件同目录）")
    parser.add_argument("--output", default=None,
                        help="治理后数据输出文件名（缺省 cleaned_data.csv）")
    parser.add_argument("--report-output", default=None,
                        help="治理报告输出文件名（缺省 cleaned_data_report.md）")
    parser.add_argument("--target-units", default=None,
                        help='目标单位 JSON，如 \'{"current_amps": "ma"}\'（覆盖元信息中的同名字段）')
    parser.add_argument("--category-maps", default=None,
                        help='类别编码映射 JSON，如 \'{"status": {"Y": 0, "N": 1}}\'（覆盖元信息中的同名字段）')
    parser.add_argument("--label-cols", default=None,
                        help="标注列名（逗号分隔，覆盖元信息中的标注列字段）")
    parser.add_argument("--enable-outlier-detection", action="store_true",
                        help="开启异常值/哨兵值检测（默认关闭）")
    parser.add_argument("--value-range", default=None,
                        help='有效值域 JSON，如 \'{"current_amps": [0, 100]}\'（覆盖元信息中的同名字段）')
    parser.add_argument("--sentinels", default=None,
                        help='哨兵值 JSON，如 \'{"current_amps": [-99, -100]}\'（覆盖元信息中的同名字段）')

    args = parser.parse_args()

    # 解析元信息
    print(f"读取元信息：{args.metadata}")
    try:
        meta = parse_metadata(args.metadata)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"错误：元信息 JSON 解析失败: {e}", file=sys.stderr)
        return 1

    # 解析 CLI JSON 参数（覆盖元信息中的同名字段）
    try:
        target_units = json.loads(args.target_units) if args.target_units else meta.get("target_units", {})
        category_maps = json.loads(args.category_maps) if args.category_maps else meta.get("category_maps", {})
        value_ranges = json.loads(args.value_range) if args.value_range else meta.get("value_ranges", {})
        sentinels = json.loads(args.sentinels) if args.sentinels else meta.get("sentinels", {})
    except json.JSONDecodeError as e:
        print(f"错误：CLI JSON 参数解析失败: {e}", file=sys.stderr)
        return 1
    user_label_cols = args.label_cols.split(",") if args.label_cols else []

    # 确定输出目录
    input_path = Path(args.input_csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = args.output or "cleaned_data.csv"
    report_name = args.report_output or "cleaned_data_report.md"

    # 读取合并数据
    print(f"\n读取合并数据：{args.input_csv}")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, na_values=[])
    print(f"  行数：{len(df)}，列数：{len(df.columns)}")

    # 确定每列角色
    # 1) 元信息「列清单」中角色为「标注列」「类别列」的列名
    label_cols_from_meta = [
        col for col, info in meta["columns"].items()
        if info.get("role") in ("标注列", "label")
    ]
    category_cols_from_meta = [
        col for col, info in meta["columns"].items()
        if info.get("role") in ("类别列", "category")
    ]
    # 2) 元信息中独立的「标注列」「类别列」小节
    label_cols_from_meta.extend(meta.get("label_cols", []))
    category_cols_from_meta.extend(meta.get("category_cols", []))
    # 3) CLI 参数覆盖（如提供）
    if user_label_cols:
        label_cols = set(user_label_cols)
    else:
        label_cols = set(label_cols_from_meta)
    category_cols = set(category_cols_from_meta)

    # 构建 records
    records = OrderedDict()
    col_order = list(df.columns)

    for col in col_order:
        if col == "time":
            continue

        col_meta = meta["columns"].get(col, {})

        # 角色
        role = col_meta.get("role", "业务数据")
        if col in label_cols:
            role = "标注列"
        elif col in category_cols:
            role = "类别列"

        # 类型与单位
        p_type = col_meta.get("type", "unknown")
        unit = col_meta.get("unit")

        records[col] = CleanRecord(col, role, p_type)
        records[col].unit_info = unit

    print(f"\n开始单元格治理...")
    print(f"  标注列：{label_cols or '无'}")
    print(f"  类别列：{category_cols or '无'}")

    # 1. 显式缺失处理
    print("\n[1/5] 显式缺失处理...")
    process_na_values(df, records)
    total_na = sum(r.na_count for r in records.values())
    print(f"  处理 {total_na} 个 NA 标记")

    # 2. 类型归一（非时间列、非标注列、非类别列）
    print("[2/5] 类型归一...")
    total_type_fixes = 0
    for col in col_order:
        if col == "time":
            continue
        if col in label_cols or col in category_cols:
            continue
        new_vals, changed = normalize_column_type(df[col].tolist(), col, records[col])
        df[col] = new_vals
        total_type_fixes += len(records[col].type_fixes)
    print(f"  类型修改 {total_type_fixes} 个单元格")

    # 3. 单位归一
    print("[3/5] 单位归一...")
    total_unit_conversions = 0
    for col in col_order:
        if col == "time" or col in label_cols:
            continue
        unit = meta["columns"].get(col, {}).get("unit")
        if not unit:
            continue
        new_vals, changed = normalize_column_unit(
            df[col].tolist(), col, unit, target_units, records[col]
        )
        df[col] = new_vals
        if records[col].unit_converted:
            total_unit_conversions += 1
    print(f"  单位转换 {total_unit_conversions} 列")

    # 4. 类别编码
    print("[4/5] 类别编码...")
    total_categories = 0
    for col in category_cols:
        if col in df.columns:
            new_vals = encode_category(df[col].tolist(), col, category_maps, records[col])
            df[col] = new_vals
            records[col].final_type = "int"
            total_categories += 1
    print(f"  类别编码 {total_categories} 列")

    # 5. 标注编码
    print("[5/5] 标注编码...")
    total_label_conversions = 0
    for col in label_cols:
        if col in df.columns:
            new_vals = encode_label(df[col].tolist(), col, records[col])
            df[col] = new_vals
            records[col].final_type = "int"
            if records[col].label_converted:
                total_label_conversions += 1
    print(f"  标注转换 {total_label_conversions} 列")

    # 6. 异常值检测（可选）
    total_outliers = None
    if args.enable_outlier_detection:
        print("\n[可选] 异常值/哨兵值检测...")
        total_outliers = 0
        for col in col_order:
            if col == "time" or col in label_cols:
                continue
            new_vals = detect_outliers(df[col].tolist(), col, value_ranges, sentinels, records[col])
            df[col] = new_vals
            total_outliers += len(records[col].outliers)
        print(f"  检出异常值 {total_outliers} 个")

    # 检查致命错误
    has_fatal = any(r.errors for r in records.values())

    # 写入输出
    csv_path = str(output_dir / csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n治理后数据已写入：{csv_path}")
    print(f"  行数：{len(df)}，列数：{len(df.columns)}")

    # 生成报告
    report_path = str(output_dir / report_name)
    generate_report(records, col_order, report_path, args.input_csv,
                    total_na, total_type_fixes, total_unit_conversions,
                    total_label_conversions, total_categories, total_outliers)
    print(f"治理报告已生成：{report_path}")

    if has_fatal:
        print("\n⚠️ 存在致命错误，请查看报告。", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
