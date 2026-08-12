# openLLV.list_algorithms()

`openLLV.list_algorithms()` 返回每个可接受的已注册传统算法查找名，包括别名。

## Function Form

```python
openLLV.list_algorithms() -> List[str]
```

## Parameters

该函数没有参数。

## Returns

排序后的 `List[str]`，包含各具体类名、显式声明的 `name` 和别名所贡献的小写注册表键。

## Behavior Details

- 查找去除首尾空白后不区分大小写。
- 带别名的实现会贡献多个字符串；若需要每个类一条去重记录，使用 `list_available()`。

## Example

```python
import openLLV as llv

print(llv.list_algorithms())
enhanced, path = llv.predict("clahe", "input.jpg")
```

## Related

- 算法 reference：[`algorithms/`](algorithms/)
- 模型：[`openLLV.list_models()`](list_models.md)
