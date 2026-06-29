# -*- coding: utf-8 -*-

"""
HPO（超参数优化）模块

基于 Optuna 实现异常检测算子的自动化超参数搜索，支持 Detector/Scorer
（含 Composite）的自动超参寻优。

核心设计:
    搜索空间声明基于 Pydantic 原生 Field(ge/le) 和 Enum/Literal 类型注解，
    零侵入。仅在需要 log 采样、非1步长等特殊语义时通过 ``Annotated`` 注入
    ``SearchHint`` 标记。

模块组成:
    - search_hint: SearchHint 标记 + 搜索空间提取
    - trainer: HPOTrainer 优化编排器
    - result: HPOResult + TrialInfo 结果容器

使用示例::

    from tsas.engine.hpo import HPOTrainer
    from tsas.engine.operator.detection.knn import KNNDetector
    from tsas.engine.operator.evaluation.binary_classification import (
        BinaryClassificationMetric,
    )

    trainer = HPOTrainer(KNNDetector, BinaryClassificationMetric(),
                         n_trials=50, top_k=3)
    result = trainer.fit(train_data, val_labels=val_labels, val_split=0.3)
    print(result.best_params)
    print(result.best_score)
"""

from tsas.engine.hpo.result import HPOResult, TrialInfo
from tsas.engine.hpo.search_hint import (
    SearchHint,
    config_to_optuna_suggestions,
    extract_search_space,
    extract_search_space_from_operator,
)
from tsas.engine.hpo.trainer import HPOTrainer

__all__ = [
    'SearchHint',
    'HPOTrainer',
    'HPOResult',
    'TrialInfo',
    'extract_search_space',
    'extract_search_space_from_operator',
    'config_to_optuna_suggestions',
]
