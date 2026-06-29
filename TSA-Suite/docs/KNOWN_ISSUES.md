# TSA-Suite 已知问题

> 记录项目中尚未解决的已知问题，便于开发者排查。

---

## 1. Evaluation 模块测试无法收集 ✅ 已解决

**影响范围**: `tests/test_engine_operator/test_evaluation/` 下所有测试文件

**现象**: pytest 收集测试时抛出 `ModuleNotFoundError: No module named 'tsas.engine.operator.evaluation.point_adjust'`

**原因**: `evaluation/__init__.py` 中 import 了 `point_adjust` 模块，但该文件尚未创建

**解决**: commit `bdbfd0f` 移除了 `evaluation/__init__.py` 中的所有 import 语句，pytest 不再报错

**状态**: 已解决（2026-04-29）

---

## 2. test_base.py 文件重名导致收集冲突

**影响范围**: `tests/test_engine_operator/test_feature/test_construction/test_base.py`

**现象**: pytest 报 `import file mismatch`，两个 `test_base.py` 因 `__pycache__` 冲突

**原因**: 两个不同目录下存在同名测试文件：
- `tests/test_engine_operator/test_base.py`
- `tests/test_engine_operator/test_feature/test_construction/test_base.py`

**状态**: 需重命名其中一个文件（如改为 `test_feature_base.py`）
