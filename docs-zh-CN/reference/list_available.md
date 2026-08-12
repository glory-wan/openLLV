# openLLV.list_available()

`openLLV.list_available()` 按类别返回每个已注册公开组件类的去重元数据。

## Function Form

```python
openLLV.list_available() -> Dict[str, List[Dict[str, Any]]]
```

## Parameters

该函数没有参数。

## Returns

```python
{
    "models": [{"name": str, "aliases": List[str]}, ...],
    "algorithms": [{"name": str, "aliases": List[str]}, ...],
    "metrics": [{"name": str, "aliases": List[str]}, ...],
    "losses": [{"name": str, "aliases": List[str]}, ...],
    "datasets": [{"name": str, "aliases": List[str]}, ...],
}
```

每个实现类在各类别中只出现一次。`name` 是 Python 类名；`aliases` 复制类的 `aliases` 属性（单个字符串会规范化为单元素列表）。

## Behavior Details

- 各类别中的类按类名不区分大小写排序。
- 该函数不同于 `list_models()`、`list_algorithms()`、`list_losses()`、`list_datasets()`：后者返回每个可接受查找键，本函数按实现类去重注册表值。
- 此处算法 `name` 也是类名，不一定是 `LLVEnhancer.name` 注册标签。例如 DCP 实现记录的类名为 `DarkChannel`，别名为 `dcp`。因此每条记录是展示元数据，不保证完整列出全部可接受查找键；完整键应使用对应的平铺 `list_*()` 函数。

## Example

```python
import openLLV as llv

available = llv.list_available()
for model in available["models"]:
    print(model["name"], model["aliases"])
```

## Related

- 模型：[`openLLV.list_models()`](list_models.md)
- 算法：[`openLLV.list_algorithms()`](list_algorithms.md)
- 指标：[`openLLV.list_metrics()`](list_metrics.md)
- Loss：[`openLLV.list_losses()`](list_losses.md)
- 数据集：[`openLLV.list_datasets()`](list_datasets.md)
