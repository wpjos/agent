# 算子模块统一命令行工具 (CLI) 使用指南

本文档旨在介绍如何使用 `tsas.engine.operator.cli` 提供的命令行工具来执行特征构造、特征选择、时序异常检测和评价指标计算，以及如何将新开发的算子接入到 CLI 中生效。

## 1. 命令行调用入口

CLI 工具既支持通过 `tsas.engine.operator.cli` 作为统一入口（然后通过指定子模块名分发），也支持直接调用具体的模块入口。

**方式一：统一调用入口**

```bash
python -m tsas.engine.operator.cli [--encoding ENCODING] <模块名> <子命令> [参数]
```

**方式二：指定模块调用入口**

```bash
python -m tsas.engine.operator.cli.<模块名> [--encoding ENCODING] <子命令> [参数]
```

支持的模块名（`<模块名>`）：

- `feature_construction`：特征构造模块
- `feature_selection`：特征选择器模块
- `detection`：时序异常检测模块
- `evaluation`：评价指标模块

支持的子命令：

- `help`：查看可用算子列表或指定算子的详细参数文档
- `run`：执行算子的推理/计算
- `fit`：执行算子的训练（仅支持有状态的可学习算子）

### 全局参数 `--encoding`

所有模块均支持 `--encoding` 参数，用于指定终端输出编码（解决 Windows 终端中文乱码问题）。

- **不指定时**：自动检测当前终端编码，若非 UTF-8 则尝试设置为 UTF-8（Windows 下自动执行 `chcp 65001`）。
- **指定时**：强制使用指定的编码（如 `utf-8`、`gbk`）。
- **冲突检测**：不允许在统一入口和模块入口两级同时指定 `--encoding`，否则报错。

```bash
# 统一入口指定编码
python -m tsas.engine.operator.cli --encoding utf-8 detection help

# 模块入口指定编码
python -m tsas.engine.operator.cli.detection --encoding utf-8 help
```

---

## 2. 工具使用说明

### 2.1 获取帮助 (help)

`help` 命令提供针对 AI Agent 和开发者友好的结构化 Markdown 文档，参数说明自动提取自代码中 Pydantic 模型的定义。

- **查看所有可用算子（以detection为例）：**
  ```bash
  python -m tsas.engine.operator.cli detection help
  或
    python -m tsas.engine.operator.cli.detection help

  ```
  **输出样例：**
  ```markdown
  ## 可用算子列表

  ### 管线组件算子

  | 名称                | 类型           | 可训练 | 简介                                                     |
  | ------------------- | -------------- | ------ | --------------------------------------------------------- |
  | cicada_predictor    | Predictor      | 是     | CICADA 重构型预测器                                       |
  | knn_scorer          | Scorer(Single) | 是     | KNN 直接评分器                                            |
  | percentile_decider  | Decider        | 是     | 百分位阈值决策器（第 3 层 Decider）                       |
  ...

  ### 端到端检测器算子

  | 名称           | 类型     | 可训练 | 简介                                           |
  | -------------- | -------- | ------ | ---------------------------------------------- |
  | knn_detector   | Detector | 是     | KNN 检测器 — 组合 KNNScorer + PercentileDecider |
  ...

  ### 组合管线算子

  | 名称               | 类型    | 可训练 | 简介     |
  | ------------------ | ------- | ------ | -------- |
  | composite_detector | Decider | 是     | 组合检测器 |
  ...

  共 18 个算子。使用 `help <算子名称>` 查看详细信息。
  ```

- **查看指定算子的详细使用说明（以detection为例）：**
  ```bash
  python -m tsas.engine.operator.cli detection help knn_detector
  ```
  **输出样例：**
  ```markdown
  ## knn_detector

  KNN 检测器 — 组合 KNNScorer + PercentileDecider

  **版本**：1.0.0

  ### 基础分类

  **类型**: Detector
  **可训练**: 是 - 无监督
  **支持分批推理**: 否

  ### 输入

  x: 特征矩阵，形状 (n_samples, n_features)

  ### 主输出 (pandas.DataFrame | numpy.ndarray)

  0/1 异常标签，形状 (n_samples,)，1 表示异常。
  标签由内部 KNNScorer 输出的异常分数经 PercentileDecider 的百分位阈值决策得到

  ### 实例参数 (KNNDetectorConfig)

  | 参数名          | 类型                        | 默认值    | 值域/候选             | 说明         |
  | --------------- | --------------------------- | --------- | --------------------- | ------------ |
  | n_neighbors     | int                         | 5         | [1, 20]               | 近邻数量 K   |
  | distance_metric | enum(euclidean, manhattan)  | euclidean | euclidean, manhattan  | 距离度量方式 |
  | score_method    | enum(maximum, mean, median) | maximum   | maximum, mean, median | 分数合并方式 |
  | percentile      | float                       | 95.0      | [50.0, 99.9]          | 百分位阈值   |

  ### 训练参数
  （无）

  ### 运行参数
  （无）
  ```

  对于主输出类型为 `BaseModel` 子类的算子（典型场景是 evaluation 模块的指标算子，
  主输出 `MR` 是结构化结果对象），CLI Help 会在标题中带类型名，并自动追加
  `**结构**：` 字段表，例如：

  ```markdown
  ### 主输出 (BinaryClassificationResult)

  **结构**：

  | 字段名           | 类型            | 说明 |
  | ---------------- | --------------- | ---- |
  | accuracy         | float           |      |
  | precision        | float           |      |
  | recall           | float           |      |
  | f1               | float           |      |
  | ...              | ...             |      |
  | confusion_matrix | list[list[int]] |      |
  ```

  字段表来自算子类继承链中泛型参数 `O`（`BaseOperator[I, O, C, RP]`）的自动推断
  （多层泛型追踪），开发者无需手动维护。详见 `base.md` 的
  "5.1.2 主输入/输出类型推断" 小节。

### 2.2 执行特征构造 (feature_construction)

特征构造模块支持一次编排多个特征算子，输出拼接后的新特征列（可选保留原始列）。

**配置文件 (`feature.yaml` / `feature.json`) 示例：**

*YAML 格式:*

```yaml
operators:
  - name: "square_feature"
    config:
      input_columns: [ "sensor_1" ]
  - name: "polynomial_feature"
    config:
      input_columns: [ "sensor_2" ]
      degrees: [ 2, 3 ]
keep_original: true
```

*JSON 格式:*

```json
{
  "operators": [
    {
      "name": "square_feature",
      "config": {
        "input_columns": [
          "sensor_1"
        ]
      }
    },
    {
      "name": "polynomial_feature",
      "config": {
        "input_columns": [
          "sensor_2"
        ],
        "degrees": [
          2,
          3
        ]
      }
    }
  ],
  "keep_original": true
}
```

**输出说明：**
输出的数据（如 `result.csv`）为表格格式。如果 `keep_original` 为 `true`，输出将包含所有输入数据的原始列，并在其后追加各算子新生成的特征列（如 `sensor_1_square`、`sensor_2_poly_2` 等）。如果为 `false`，则仅包含新生成的特征列及索引列。

**执行命令：**

```bash
python -m tsas.engine.operator.cli feature_construction run --input data.csv --config feature.yaml --output result.csv
或
python -m tsas.engine.operator.cli.feature_construction run --input data.csv --config feature.yaml --output result.csv
```

### 2.3 执行时序异常检测 (detection)

检测模块支持单个预测器 (`Predictor`)、评分器 (`Scorer`)、决策器 (`Decider`) 或端到端检测器 (`Detector`)，也可通过 `composite_scorer`/`composite_detector` 编排管线。

**配置文件 (`det.yaml` / `det.json`) 示例：**

*YAML 格式:*

```yaml
operator:
  name: "knn_detector"
  input_columns: [ "sensor_1", "sensor_2" ]
  config:
    n_neighbors: 5
    percentile: 95.0
```

*JSON 格式:*

```json
{
  "operator": {
    "name": "knn_detector",
    "input_columns": [
      "sensor_1",
      "sensor_2"
    ],
    "config": {
      "n_neighbors": 5,
      "percentile": 95.0
    }
  }
}
```

**输出说明：**
输出的数据（如 `result.csv`）将保留所有的原始输入列，并在其后追加检测结果列。

- 对于评分器 (`Scorer`)，追加 `anomaly_score` 列（或多列如 `sensor_1_score` 等，取决于具体算子）。
- 对于决策器 (`Decider`)，追加 `anomaly_label` 列（0 为正常，1 为异常）。
- 对于端到端检测器 (`Detector`)，同时追加 `anomaly_score` 和 `anomaly_label` 列。

**执行训练（fit，仅对需训练的算子）：**

```bash
python -m tsas.engine.operator.cli detection fit --input train.csv --config det.yaml --save model_dir/
```

**执行检测（run）：**

```bash
# 如果算子已被训练，通过 --load 指定模型目录加载状态
python -m tsas.engine.operator.cli detection run --input test.csv --config det.yaml --load model_dir/ --output result.csv
```

### 2.4 执行评价指标 (evaluation)

评价模块支持一次调用多个指标，结果汇总输出为 JSON 格式。

**配置文件 (`eval.yaml` / `eval.json`) 示例：**

*YAML 格式:*

```yaml
operators:
  - name: "binary_classification_metric"
    alias: "metric_1"
    truth_columns: [ "label" ]
    predict_columns: [ "predict" ]
    config:
      positive_label: 1
```

*JSON 格式:*

```json
{
  "operators": [
    {
      "name": "binary_classification_metric",
      "alias": "metric_1",
      "truth_columns": [
        "label"
      ],
      "predict_columns": [
        "predict"
      ],
      "config": {
        "positive_label": 1
      }
    }
  ]
}
```

**执行评价（run）：**

```bash
python -m tsas.engine.operator.cli evaluation run --input data.csv --config eval.yaml --output result.json
```

**输出说明与样例：**
评价指标输出为 JSON 格式。返回的对象以算子名称（或配置的 `alias`）作为 Key，其下包含了算子的明细结果 (`result`) 和关键得分摘要 (`main_scores`)。

*输出 JSON 样例 (`result.json`):*

```json
{
  "results": {
    "metric_1": {
      "result": {
        "f1": 0.85,
        "far": 0.12,
        "mcc": 0.73,
        "confusion_matrix": [
          [
            80,
            10
          ],
          [
            5,
            15
          ]
        ]
      },
      "main_scores": {
        "f1": 0.85,
        "far": 0.12
      }
    }
  }
}
```

---

## 3. 支持的数据格式与配置文件

**算子输入数据要求：**

- 数据需以表格结构提供。第一行必须为**表头（列名）**，后续行为数据。
- 所有的算子在使用时，需要通过配置文件中的 `input_columns`（或 `truth_columns` / `predict_columns`）来显式指定其要处理的列名称。CLI 会在内部通过列名映射将所需数据提取并传给算子。
- 缺失值：请根据具体的算子要求对空值进行预处理或由算子本身支持包含 NaN 的计算。

**数据文件读写格式：**
CLI 工具会自动根据参数中 `--input` 和 `--output` 文件的后缀名选择合适的读取/写入方式。

- 目前首选格式：CSV (`.csv`)
- 已预留对 TSV (`.tsv`), MAT (`.mat`), HDF5 (`.h5`, `.hdf5`) 等格式的扩展支持。

**配置文件格式：**
配置文件格式通过 `--config` 指定的后缀名自动分发，支持：

- `.json`: 标准 JSON
- `.json5`: 支持注释的 JSON5
- `.yaml` / `.yml`: YAML 格式

---

## 4. 算子注册与生效机制

为了让 CLI 可以使用新开发的算子，算子需要通过注册中心进行注册。当前的 CLI 采用**动态包扫描自动注册机制**。

### 4.1 自动发现原理

当 CLI 启动时，相应的模块（如 `detection` CLI）会实例化一个 `OperatorRegistry`。注册中心会使用 `pkgutil.walk_packages` 自动递归扫描指定的包路径（如 `tsas.engine.operator.detection`）。
只要你的算子类满足以下条件，它就会在 CLI 中自动生效，**无需手动在某处注册**：

1. 继承了对应的基类或 Mixin：
    - 特征算子需继承 `BaseFeatureMixin`
    - 检测算子需继承 `BasePredictorMixin`、`BaseScorerMixin`、`BaseDeciderMixin` 或其组合（实际的 Detector 类通常组合 `BaseDeciderMixin` + `NumericOperator`）
    - 评价算子需继承 `BaseMetricOperator`
2. 实现了 `name()` 类方法，并返回了独一无二的名称。
3. 存放在对应模块的包或子包下。

### 4.2 开发与注册步骤指南

如果你要开发一个新的算子（例如在 `detection` 模块），你只需：

**Step 1. 编写算子代码及 Pydantic Config：**

```python
# src/tsas/engine/operator/detection/my_detector.py
from tsas.engine.operator.detection.base import BaseDeciderMixin
from tsas.engine.operator.base import NumericOperator
from pydantic import BaseModel, Field


class MyDetectorConfig(BaseModel):
    # 使用 Pydantic Field 详细定义参数类型、默认值、值域（ge/le等）、以及 description
    # CLI help 将直接从这里提取信息生成文档
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0, description="检测灵敏度参数")


class MyDetector(BaseDeciderMixin[None, MyDetectorConfig, None], NumericOperator):

    @classmethod
    def name(cls) -> str:
        return "my_detector"  # 这是在 CLI 和 Config 中使用的名字

    def _run_data(self, ...):
        # 你的实现
        pass
```

**Step 2. 编写完善的 Docstring：**
类级别的 Docstring 将被 CLI 提取为该算子的“功能描述”。

```python
class MyDetector(...):
    """
    MyDetector 是一种基于自定义逻辑的异常检测器。
    
    它通过分析数据的敏感度波动来识别异常。
    """
```

**Step 3. 验证生效：**
确保文件放在相应的目录下（例如 `src/tsas/engine/operator/detection` ）。运行以下命令：

```bash
python -m tsas.engine.operator.cli detection help my_detector
```

如果能正常看到说明、`sensitivity` 参数以及默认值/值域，即说明注册成功并已生效！