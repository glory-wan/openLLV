# openLLV.list_metrics()

`openLLV.list_metrics()` 返回所有已注册图像质量指标的简名。

## Function Form

```python
openLLV.list_metrics() -> List[str]
```

## Parameters

该函数没有参数。

## Returns

按指标注册/插入顺序返回的 `List[str]`（未显式排序）。移除类名的 `Metric` 后缀，因此 `PSNRMetric` 返回为 `"PSNR"`。

## Behavior Details

- 查注册表前会导入指标实现。
- `evaluate()` 匹配名称时不区分大小写，也接受 `Metric` 后缀。

## Example

```python
import openLLV as llv

metrics = llv.list_metrics()
results = llv.evaluate("enhanced", "reference", metrics=metrics[:2])
```

## Related

- 评估：[`openLLV.evaluate()`](evaluate.md)
- 所有组件类别：[`openLLV.list_available()`](list_available.md)
