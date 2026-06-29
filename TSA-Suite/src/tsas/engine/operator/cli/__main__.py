# -*- coding: utf-8 -*-

"""
算子 CLI 统一分发入口

通过 ``python -m tsas.engine.operator.cli <模块> <子命令> [参数]`` 调用。

支持的模块:
    - ``feature_construction``: 特征构造算子
    - ``feature_selection``: 特征选择器算子
    - ``detection``: 异常检测算子
    - ``evaluation``: 评价指标算子

示例::

    python -m tsas.engine.operator.cli feature_construction help
    python -m tsas.engine.operator.cli feature_selection help
    python -m tsas.engine.operator.cli detection run --input data.csv --config det.yaml
    python -m tsas.engine.operator.cli evaluation run --input data.csv --config eval.json
    python -m tsas.engine.operator.cli --encoding utf-8 detection run --input data.csv --config det.yaml
"""

import sys

from tsas.engine.operator.cli.io import ensure_encoding

# 模块名称到子模块的映射
_MODULE_MAP = {
    'feature_construction': 'tsas.engine.operator.cli.feature_construction',
    'feature_selection': 'tsas.engine.operator.cli.feature_selection',
    'detection': 'tsas.engine.operator.cli.detection',
    'forecasting': 'tsas.engine.operator.cli.forecasting',
    'evaluation': 'tsas.engine.operator.cli.evaluation',
}


def _print_usage() -> None:
    """输出统一入口的使用说明"""
    print("用法: python -m tsas.engine.operator.cli [--encoding ENCODING] <模块> <子命令> [参数]")
    print()
    print("全局参数:")
    print("  --encoding ENCODING  指定终端输出编码，默认 UTF-8")
    print()
    print("可用模块:")
    print("  feature_construction  特征构造算子")
    print("  feature_selection     特征选择器算子")
    print("  detection             异常检测算子")
    print("  forecasting           时序预测算子")
    print("  evaluation            评价指标算子")
    print()
    print("示例:")
    print("  python -m tsas.engine.operator.cli feature_construction help")
    print("  python -m tsas.engine.operator.cli feature_selection help")
    print("  python -m tsas.engine.operator.cli detection run --input data.csv --config det.yaml")
    print("  python -m tsas.engine.operator.cli forecasting fit --input train.csv --target target --config fc.yaml")
    print("  python -m tsas.engine.operator.cli evaluation run --input data.csv --config eval.json")

def main(args: list[str] | None = None) -> None:
    """
    统一分发入口主函数

    解析全局参数（如 ``--encoding``），将剩余参数转发给对应子模块。
    当全局和模块两级同时指定 ``--encoding`` 时报错。

    Args:
        args (list[str] | None): 命令行参数列表。None 时使用 ``sys.argv[1:]``
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        _print_usage()
        sys.exit(1)

    # 提取全局 --encoding 参数（位于模块名之前）
    global_encoding = None
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == '--encoding' and i + 1 < len(args):
            global_encoding = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1

    if not remaining:
        _print_usage()
        sys.exit(1)

    module_name = remaining[0]

    if module_name in ('-h', '--help', 'help'):
        _print_usage()
        return

    if module_name not in _MODULE_MAP:
        print(f"错误: 未知模块 '{module_name}'")
        print(f"可用模块: {', '.join(sorted(_MODULE_MAP.keys()))}")
        sys.exit(1)

    # 冲突检测：全局和模块两级同时指定 --encoding 时报错
    if global_encoding is not None and '--encoding' in remaining:
        print("错误: 不允许在全局和模块两级同时指定 --encoding 参数")
        sys.exit(1)

    # 应用全局编码设置
    ensure_encoding(global_encoding)

    # 动态导入子模块并调用其 main 函数
    import importlib
    sub_module = importlib.import_module(_MODULE_MAP[module_name])
    sub_module.main(remaining[1:])


if __name__ == '__main__':
    main()
