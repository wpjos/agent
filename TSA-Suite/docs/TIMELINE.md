# CICADA 合入开发记录

> 2026-04-24 | 项目调研与方案设计
> 2026-04-25 | CICADA reconstruct 改造 + Predictor 适配层实现 + 测试 + 文档

---

## 目录

1. [开发背景](#一开发背景)
2. [方案设计](#二方案设计)
3. [CICADA 侧改造](#三cicada-侧改造)
4. [TSA-Suite 侧实现](#四tsa-suite-侧实现)
5. [测试覆盖](#五测试覆盖)
6. [变更清单](#六变更清单)
7. [时间线](#七时间线)
8. [验证](#八验证)

---

## 一、开发背景

### 1.1 目标

将 CICADA（Continual Learning via Incremental Component Adaptive Architecture）算法作为重构型 Predictor 合入 TSA-Suite 异常检测框架的四层架构。

### 1.2 CICADA 简介

CICADA 是基于 Mixture-of-Experts + MAML 元学习 + 动态架构扩展的重构型异常检测算法，核心特点：

- 14 种异构专家编码器（GradPCA, GradKPCA, GradSFA, MLP, CNN, PatchTST 等）
- MAML 元学习实现快速适应
- 训练过程中动态分裂专家实现架构扩展
- 输出重构值，用于下游残差评分

### 1.3 框架约束

TSA-Suite 的检测模块采用四层架构：

```
Predictor（预测器） → Scorer（评分器） → Decider（决策器） → Detector（检测器）
```

CICADA 定位为 **Predictor**，输出重构值，后续可串联 ResidualScorer → Decider 组成完整检测管线。

---

## 二、方案设计

### 2.1 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 集成版本 | model.py 独立版 | supervised.py / reg.py 依赖 d2l，不可独立运行 |
| 定位 | Predictor（非 DirectScorer） | 输出重构值，下游评分路径灵活可组合 |
| EO | None | 首期不加额外输出，后续按需扩展 |
| 持久化 | torch.save 整体序列化 | MAML 包装 + 动态架构扩展，state_dict 难以还原 |
| CICADA 导入 | 延迟导入（方法体内 import） | tsas 不强依赖 cicada 包 |
| num_channels | Config 中默认 None，fit 时自动推断 | 避免用户手动指定输入维度 |

### 2.2 超参数统一

CICADA 构造函数约 30 个参数，全部收编至 `CICADAPredictorConfig`（Pydantic frozen BaseModel），包括：

- 专家配置（name）
- 窗口/形状（win_size, stride, num_channels, batch_size）
- 编码器架构（latent_space_size, n_components, normalization, ar_order）
- 注意力（attn_bucket_heads, decoder_all_heads, forward_expansion）
- 元学习（train_init_meta_lr, test_meta_lr, meta_split_threshold, lr_split_factor, ml_lambda, penalty_rate）
- 优化器（lr, ttlr, gamma）
- 动态扩展（adaptive_add, epoch_add, close_epochs）
- 推理（infer_mode, th）

### 2.3 继承结构

```python
class CICADAPredictor(
    UnsupervisedNumericOperatorMixin[None],              # 无监督训练
    BasePredictor[None, CICADAPredictorConfig, None],    # 重构型 Predictor
)
```

---

## 三、CICADA 侧改造

### 3.1 新增 reconstruct() 方法

在 CICADA 类中新增 `reconstruct(data)` 方法，暴露重构值：

```
raw (N, C)
  → ReconstructDataset (z-normalize + 滑动窗口, stride=win_size)
  → DataLoader → batches of (B, win_size, C)
  → self.model(X) → Xhat, _, _         ← 纯 forward, 无 MAML adapt
  → 反归一化: recon * std + mean
  → 右填充至原始长度
  → 返回 (N, C)
```

设计要点：

| 项目 | 说明 |
|------|------|
| 无 MAML adapt | 纯 forward pass + `torch.no_grad()` |
| 反归一化 | `train_mean`/`train_std` 在 `fit()` 中已保存 |
| 填充对齐 | 与 `decision_function` 一致 |
| 向后兼容 | `fit()` 和 `decision_function()` 完全不动 |

---

## 四、TSA-Suite 侧实现

### 4.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/tsas/engine/operator/detection/cicada.py` | ~290 | CICADAPredictorConfig + CICADAPredictor |
| `tests/test_engine_operator/test_detection/test_cicada.py` | ~270 | 22 个单元测试 |

### 4.2 核心实现

- `_fit_data`: 延迟创建 CICADA 实例，从 `x.shape[1]` 推断 num_channels，调用 `self._model.fit()`
- `_run_data`: 调用 `self._model.reconstruct()` 返回重构值
- `save`: `super().save()` + `torch.save(model)` + JSON 元信息
- `load`: `super().load()` + `torch.load()` + 恢复 `_fitted = True`

### 4.3 文档更新

- `detection.md`: 目录结构新增 `cicada.py`，算子列表新增 CICADAPredictor

---

## 五、测试覆盖

### 5.1 测试配置

所有 CICADA 测试使用最小参数加速：

```python
CICADAPredictor(name=["MLP"], win_size=10, num_channels=3,
                batch_size=32, epochs=1, latent_space_size=8, n_components=4)
```

### 5.2 测试用例

| 测试类 | 用例数 | 覆盖内容 |
|--------|--------|---------|
| TestCICADAPredictorConfig | 6 | 默认值、frozen、win_size/epochs/infer_mode 校验、自定义参数 |
| TestCICADAPredictorFit | 6 | 模型创建、num_channels 推断/显式指定、数据过短/1D 报错、DataFrame |
| TestCICADAPredictorRun | 5 | 输出形状、值有限、未训练报错、DataFrame/ndarray 双类型 |
| TestCICADAPredictorSaveLoad | 5 | roundtrip、fitted 恢复、模型可用、文件完整性、num_channels 持久化 |

---

## 六、变更清单

| Commit | 日期 | 文件 | 变更类型 | 说明 |
|--------|------|------|---------|------|
| `053cd9e` | 04-25 | `detection/cicada.py`, `test_cicada.py` | 新增 | CICADAPredictor 适配层 + 22 个单元测试 |
| `7d5233d` | 04-25 | `detection/detection.md` | 更新 | 目录结构和算子列表补充 CICADA |
| `0076939` | 04-25 | `detection/TIMELINE.md` | 新增 | CICADA 合入开发记录 |
| `902e2f1` | 04-26 | `KNOWN_ISSUES.md` | 新增 | 项目已知问题记录 |
| `0a509a6` | 04-26 | `detection/TIMELINE.md` | 更新 | 补充合入和已知问题记录 |
| `bc581a2` | 04-29 | 多文件 | refactor | 移除旧实现代码及测试 |
| `8d83151` | 04-29 | `hpo/` | 新增 | HPO 超参数优化模块（+3542 行） |
| `0c63716` | 04-29 | `detection/knn.py`, `zscore.py` | 新增 | KNN/Z-Score 异常检测算子（+1011 行） |
| `80cb61a` | 04-29 | `detection/composite.py` | 新增 | 组合异常检测算子（+1670 行） |
| `bdbfd0f` | 04-29 | `evaluation/__init__.py` | 修复 | 移除不存在模块的 import |
| `1c5ec57` | 04-29 | `cli/` | 新增 | 统一算子 CLI 接口（+4467 行） |
| `9072988` | 04-29 | `feature-construction.md` | 更新 | 刷新特征构造模块文档 |
| `a1051ed` | 04-29 | 多文件 | docs | 文档统一迁移至 docs/ 目录 |
| `0bf1277` | 04-29 | `.gitignore` | chore | 添加 .DS_Store |
| - | 04-30 | `feature/construction/signal_feature.py` | 新增 | 31 个预测性维护信号特征算子（+680 行） |
| - | 04-30 | `feature/construction/__init__.py` | 更新 | 补充信号特征导出 |
| - | 04-30 | `feature/construction/signal_feature.py` | 更新 | 新增 SpeedRpmFeature（HPS+ 转速估计），完成 32 维特征全覆盖 |
| - | 04-30 | `test_signal_feature.py` | 新增 | 132 个信号特征测试用例，覆盖全部 26 个特征类 |
| - | 04-30 | `pyproject.toml` | 新增 | 项目配置（依赖管理、pytest、ruff） |

---

## 七、时间线

```
TSA-Suite 项目
  │
  ├─ 04-24  调研阶段
  │         阅读项目全部 .md 文档和源码
  │         确认四层架构、Mixin 组合模式、NumericOperator 管线
  │
  ├─ 04-25  方案设计
  │         讨论 CICADA 定位：Predictor vs DirectScorer
  │         确认集成版本：model.py 独立版
  │         确认超参统一策略：全部收编至 Pydantic Config
  │         发现 tsas.basic.util.random 缺失 bug（历史遗留）
  │
  ├─ 04-25  CICADA 侧改造（CICADA 项目）
  │         新增 reconstruct() 方法
  │         新增 4 个专项测试
  │         109 个全量测试通过
  │
  ├─ 04-25  分支管理
  │         创建 feat/CICADA 分支
  │         同事修复 tsas.basic 模块推至 master
  │         git rebase origin/master（线性合并）
  │
  ├─ 053cd9e  实现阶段
  │           创建 cicada.py（CICADAPredictorConfig + CICADAPredictor）
  │           创建 test_cicada.py（22 个测试）
  │           修复：输入校验从 _validate_ndarray_input 移至 _fit_data（fit 管线不经过 run 校验）
  │
  ├─ 7d5233d  文档阶段
  │           更新 detection.md 目录结构和算子列表
  │
  ├─ 0076939  记录阶段
  │           创建 TIMELINE.md 开发记录
  │
  ├─ 04-25  合入 master
  │         fast-forward merge feat/CICADA → master
  │         删除远程 feat/CICADA 分支（可选）
  │
  ├─ 902e2f1  04-26
  │           新增 KNOWN_ISSUES.md 记录历史遗留问题
  │           - evaluation 模块 point_adjust 缺失
  │           - test_base.py 重名导致 __pycache__ 冲突
  │
  ├─ bdbfd0f  04-29
  │           修复 KNOWN_ISSUES 问题1：移除 evaluation/__init__.py 中不存在的 import
  │
  ├─ 8d83151  04-29
  │           新增 HPO 超参数优化模块（+3542 行）
  │           - 基于 Optuna 实现，支持单目标/多目标优化
  │           - 零侵入搜索空间声明（Pydantic Field + Enum/Literal）
  │           - SearchHint 支持 log 采样和非1步长
  │           - HPOTrainer 支持验证集切分与交叉验证
  │
  ├─ 0c63716  04-29
  │           新增 KNN 和 Z-Score 异常检测算子（+1011 行）
  │           - KNNScorer / KNNDetector：K 近邻距离评分 + PercentileDecider
  │           - ZScoreScorer / ZScoreDetector：Z-Score 评分 + ThresholdDecider
  │
  ├─ 80cb61a  04-29
  │           新增组合异常检测算子（+1670 行）
  │           - CompositeScorer：Predictor + 多 Scorer 组合评分
  │           - CompositeDetector：Predictor + 多 Scorer + Decider 组合检测
  │
  ├─ 1c5ec57  04-29
  │           新增统一算子 CLI 接口（+4467 行）
  │           - 支持 feature_construction / detection / evaluation 三模块
  │           - 配置文件加载（JSON / YAML / JSON5）
  │           - 帮助文档自动生成（Markdown）
  │
  ├─ 9072988  04-29
  │           刷新特征构造模块文档
  │
  ├─ a1051ed  04-29
  │           将项目文档统一迁移至 docs/ 目录
  │
  ├─ 0bf1277  04-29
  │           添加 .DS_Store 到 .gitignore
  │
  ├─ 04-30  信号特征构造算子合入
  │         新增 signal_feature.py（+680 行）
  │         合入预测性维护 32 维特征中的 31 个（speed_rpm 日后处理）
  │         - Group A: 11 个简单统计特征（mean_square, variance, rms 等）
  │         - Group B: 3 个需采样率特征（spectral_entropy, roughness, sharpness）
  │         - Group C: 5 个频域特征（spectral_centroid, msf, rmsf, freq_var, freq_std）
  │         - Group D: 3 个复合特征（envelope_rms, average_kurtosis, hnr）
  │         - Group E: 3 类频带特征 × 3 频带 = 9 个（band_kurtosis/rms/hnr）
  │         全部为 IndependentMapFeature + Base 模式
  │         更新 feature/construction/__init__.py 导出
  │         与 ops 库完成数值交叉验证（31/31 精确匹配）
  │
  ├─ 04-30  SpeedRpmFeature 合入
  │         新增 Group F: HPS+ 转速估计特征
  │         - SpeedRpmConfig: sample_rate + speed_min/speed_max + n_harmonics/speed_delta/std_min
  │         - 纯 NumPy 实现 HPS+ 算法（频谱归一化 → HPS → 谐波验证选峰）
  │         - 与 ops RECIPES speed_rpm 交叉验证精确匹配
  │         32 维特征全覆盖完成
  │
  ├─ 04-30  测试补全
  │         新增 test_signal_feature.py（+1097 行）
  │         - 移植 ops 测试模式和参考值，适配 IndependentMapFeature 框架
  │         - 132 个测试用例覆盖全部 26 个特征类
  │         - Oracle 对比 + 手工参考值 + 域知识断言 + 边界情况
  │
  ├─ 04-30  项目配置
  │         新增 pyproject.toml
  │         - Python >= 3.11，核心依赖 5 个，可选依赖 hpo/knn
  │         - CICADA 以注释保留版本约束 cicada-ad>=0.1.0（本地包）
  │         - pytest pythonpath=["src"]，无需手动 PYTHONPATH
  │         - CICADA 22 个测试全部通过
  │
  ├─ 06-16  CICADA 接口对齐 bq_cicada 1.0.0
  │         上游 bq_cicada 升级到 1.0.0，重构了多个参数命名，导致旧封装无法导入。
  │         本次按"最小修复"方案打通 semi 变体接口（reg / sup / semi_class 留待后续）：
  │         - 包名：cicada → bq_cicada（pyproject 新增 cicada extra，本地 path 依赖）
  │         - Config 字段：name → experts
  │         - Loss 权重：ml_lambda → lambda_self；penalty_rate → lambda_lr；
  │           新增 lambda_recon / lambda_mse（与 bq_cicada 默认值对齐）
  │         - 测试同步两处字段名；CICADA 22 个测试全绿（detection 81/81）
  │         - 待办：reg / sup / semi_class 三变体的 Predictor 适配，
  │           以及 4 变体共有的新参数（mask_*, top_k, loss_components 等）暴露
  │
  └─ master 为当前开发版本
```

---

## 八、验证

### 导入测试

```python
from tsas.engine.operator.detection.cicada import CICADAPredictor, CICADAPredictorConfig
```

### 端到端测试

```python
import numpy as np
from tsas.engine.operator.detection.cicada import CICADAPredictor

train = np.random.randn(200, 3).astype(np.float32)
test = np.random.randn(100, 3).astype(np.float32)

predictor = CICADAPredictor(name=["MLP"], win_size=10, num_channels=3,
                             batch_size=32, epochs=5)
predictor.fit(train)
recon = predictor.run(test)
assert recon.shape == test.shape
```

### 单元测试

```bash
PYTHONPATH=src python -m pytest tests/test_engine_operator/test_detection/test_cicada.py -v
# 22 passed
```

### 全量回归

```bash
PYTHONPATH=src python -m pytest \
  tests/test_engine_operator/test_detection/ \
  tests/test_engine_operator/test_base.py \
  tests/test_engine_operator/test_feature/test_construction/test_simple_feature.py \
  -v
# 185 passed
```
