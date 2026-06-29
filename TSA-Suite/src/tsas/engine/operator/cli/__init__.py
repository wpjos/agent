# -*- coding: utf-8 -*-

"""
算子命令行接口模块

提供统一的命令行入口，支持 feature.construction、detection、evaluation 三个模块的
算子发现、帮助文档自动生成、数据IO和配置文件加载。

调用方式::

    # 统一分发入口
    python -m tsas.engine.operator.cli feature_construction help
    python -m tsas.engine.operator.cli detection run --input data.csv --config det.yaml
    python -m tsas.engine.operator.cli evaluation run --input data.csv --config eval.yaml
"""

__all__: list[str] = []
