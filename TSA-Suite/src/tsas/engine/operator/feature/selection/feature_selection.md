# 特征选择器算子说明

本模块实现特征选择器（Selector）基础类型与首批示例算子。Selector 继承当前数值算子管线，支持 `DataFrame` 与 `ndarray` 输入，主输出为选择后的特征数据，附加输出 `EO` 强制包含 `selected_indices`。

## 设计约定

- 包路径为 `tsas.engine.operator.feature.selection`。
- 除包目录名外，代码中的基础类型和具体算子统一使用 `Selector` 命名。
- `input_columns=None` 表示完整输入的全部列都是候选特征。
- `input_columns` 支持 `list[str]` 或 `list[int]`：
    - `list[str]` 仅适用于 `DataFrame` 输入，按列名选择候选列。
    - `list[int]` 适用于 `DataFrame` 与 `ndarray` 输入，按完整输入列位置选择候选列。
    - 不允许 `str` 与 `int` 混用，也不允许重复项。
- 非候选列不透传。
- 所有 Selector 都必须返回 `EO`，且 `EO.selected_indices` 按输出列顺序记录每个输出列对应完整输入中的原始列位置。

## 已实现算子

### `ColumnSelector`

静态列选择器，不需要训练，直接将 `input_columns` 解析出的候选列作为输出。

```python
from tsas.engine.operator.feature.selection import ColumnSelector, ColumnSelectorConfig

selector = ColumnSelector(config=ColumnSelectorConfig(input_columns=["a", "c"]))
selected, eo = selector.run(data)
print(eo.selected_indices)
```

### `VarianceThresholdSelector`

无监督训练型选择器，在 `fit()` 阶段基于候选特征方差学习保留列，`run()` 阶段复用训练得到的选择结果。

```python
from tsas.engine.operator.feature.selection import VarianceThresholdSelector, VarianceThresholdSelectorConfig

selector = VarianceThresholdSelector(config=VarianceThresholdSelectorConfig(threshold=0.1))
selector.fit(train_data)
selected, eo = selector.run(test_data)
print(eo.selected_indices, eo.variances)
```

如果没有任何特征方差严格大于阈值，算子不抛错，返回零列数据并记录 `WARNING`，此时 `eo.selected_indices == []`。

## CLI

特征选择器提供独立的 `feature_selection` 子命令，第一轮仅支持单个选择器。

```bash
python -m tsas.engine.operator.cli feature_selection help
python -m tsas.engine.operator.cli feature_selection run --input data.csv --config selector.json --output selected.csv --eo-output eo.json
python -m tsas.engine.operator.cli feature_selection fit --input train.csv --config selector.json --model-dir selector_model
python -m tsas.engine.operator.cli feature_selection run --input test.csv --config selector.json --load selector_model --output selected.csv --eo-output eo.json
```

配置文件示例：

```json
{
  "operator": "variance_threshold_selector",
  "config": {
    "input_columns": [0, 1, 2],
    "threshold": 0.1
  }
}
```